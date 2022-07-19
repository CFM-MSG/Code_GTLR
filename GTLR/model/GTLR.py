# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn


from GTLR.model.position_encoding import build_position_encoding
from GTLR.model.graph_transformer import build_graph_transformer

from GTLR.utils import rnns

from GTLR.model.position_encoding import PositionEmbeddingSine

import pdb

def create_gtlr(cfg, backbone=None):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223

    transformer = build_graph_transformer(cfg['arch_params'])
    position_embedding, txt_position_embedding = build_position_encoding(cfg['arch_params'])

    model = GTLR(
        transformer,
        position_embedding,
        txt_position_embedding,
        cfg['arch_params']
    )
    return model

class TextEncoder(nn.Module):
    def __init__(self, input_dim = 300, hidden_dim=256, n_layers=2):
        super().__init__()
        self.query_gru = nn.GRU(input_size  = input_dim,
                                        hidden_size  = hidden_dim,
                                        num_layers   = n_layers,
                                        bias         = True,
                                        dropout      = 0.5,
                                        bidirectional= True,
                                        batch_first = True)

    def forward(self, tokens, tokens_lengths):
        
        packed_tokens = nn.utils.rnn.pack_padded_sequence(tokens, tokens_lengths.data.tolist(), batch_first=True, enforce_sorted=False)
        word_level, _  = self.query_gru(packed_tokens)
        word_level= nn.utils.rnn.pad_packed_sequence(word_level, batch_first=True)[0]

        H = word_level.shape[-1]
        sentence_level = torch.cat((word_level[:,-1,:H//2], word_level[:,0,H//2:]), dim = -1)
        return sentence_level, word_level

class GTLR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, cfg):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.proposal_embed = PositionEmbeddingSine(num_pos_feats=cfg.get("dec_query_idim", 512))
        self.text_encoder = TextEncoder(hidden_dim=cfg.get("dec_query_idim", 512)//2)
        self.cfg = cfg
        self.num_queries = cfg.get("num_queries", 10)
        self.span_loss_type = cfg.get("span_loss_type", "l1")
        self.max_v_l = cfg.get("max_v_l", 75)
        self.use_txt_pos = cfg.get("use_txt_pos", False)
        self.n_input_proj = cfg.get("n_input_proj", 2)
        self.contrastive_align_loss = cfg.get("contrastive_align_loss", True)
        self.contrastive_hdim = cfg.get("contrastive_hdim", 64)
        self.aux_loss = cfg.get("aux_loss", False)
        self.input_dropout = cfg.get("input_dropout", 0.)
        self.use_word_emb_rec = cfg.get("use_word_emb_rec", True)

        span_pred_dim = 2 if self.span_loss_type == "l1" else self.max_v_l * 2
        word_pred_dim = cfg.get("word_pred_dim", 1500)
        hidden_dim = transformer.d_model
        txt_dim = cfg.get("txt_dim", 300)
        vid_dim = cfg.get("vid_dim", 4096)

        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.word_predict = MLP(hidden_dim, hidden_dim, word_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 1)  # 0: background, 1: foreground
        # self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[self.n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[2])
        ][:self.n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=self.input_dropout, relu=relu_args[2])
        ][:self.n_input_proj])
        if self.contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, self.contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, self.contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, self.contrastive_hdim)


        # self.saliency_proj = nn.Linear(hidden_dim, 1)

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def forward(self, net_inps):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        enc_vid = net_inps["videoFeat"]
        enc_vid_mask = self.get_mask_from_sequence_lengths(net_inps["videoFeat_lengths"], int(enc_vid.shape[1]))
        enc_text = net_inps['textEmbedding']
        enc_text_mask = self.get_mask_from_sequence_lengths(net_inps['textEmbedding_lengths'], int(enc_text.shape[1]))
        
        # dec_sentence = net_inps['sentences_emb']
        # dec_sentence_mask = [self.get_mask_from_sequence_lengths(s_len, int(s_emb.shape[1])) for s_len, s_emb in zip(net_inps['sentences_lengths'], net_inps['sentences_emb'])]
        # dec_sentence_length = net_inps['sentences_lengths']
        # for s_len, s_emb in zip(net_inps['sentences_lengths'], net_inps['sentences_emb']):
        #     dec_sentence_mask.append(self.get_mask_from_sequence_lengths(s_len, int(s_emb.shape[1])))
        dec_sentence, rec_words = self.text_encoder(net_inps['sentences_emb'], net_inps['sentences_lengths'])
        proposal_num = net_inps['proposal_num']
        dec_query = torch.split(dec_sentence, proposal_num.int().tolist(), dim=0) #tuple[tensor(#num_proposal, dim)]
        
        batch_dec_query, _ = rnns.pad_sequence(dec_query, instant_padding=False, padding_num=256)
        batch_dec_mask = self.get_mask_from_sequence_lengths(proposal_num, int(batch_dec_query.shape[1]))
        # dec_query_mask = self.get_mask_from_sequence_lengths(net_inps['sentences_lengths'], int(enc_text.shape[1]))
        
        src_vid = self.input_vid_proj(enc_vid)
        src_txt = self.input_txt_proj(enc_text)

        # query_sen_word = torch.cat((batch_dec_query, src_txt), dim=1)
        # query_mask = torch.cat((batch_dec_mask, dec_query_mask), dim=-1)

        # dec_pos_embed = self.proposal_embed(query_sen_word, query_mask)
        dec_pos_embed = self.proposal_embed(batch_dec_query, batch_dec_mask)
        
        
        # src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        # mask = torch.cat([enc_vid_mask, enc_text_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        # src = src_vid  # (bsz, L_vid+L_txt, d)
        # mask = enc_vid_mask.bool()  # (bsz, L_vid+L_txt)

        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, enc_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt, enc_text_mask) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions

        # pos = torch.cat([pos_vid, pos_txt], dim=1)
        # pos = pos_vid
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        trans_inputs = {}
        # trans_inputs["enc_input"] = src
        # trans_inputs["enc_mask"] = ~mask
        # trans_inputs["enc_pos_embed"] = pos

        trans_inputs["enc_vid_input"] = src_vid
        trans_inputs["enc_vid_mask"] = ~enc_vid_mask.bool()
        trans_inputs["enc_vid_pos_embed"] = pos_vid

        trans_inputs["enc_txt_input"] = src_txt
        trans_inputs["enc_txt_mask"] = ~enc_text_mask.bool()
        trans_inputs["enc_txt_pos_embed"] = pos_txt

        trans_inputs["dec_query"] = batch_dec_query
        trans_inputs["dec_query_mask"] = ~batch_dec_mask.bool()
        trans_inputs["dec_pos_embed"] = dec_pos_embed
        
        trans_inputs["rec_query"] = src_txt if self.use_word_emb_rec else torch.zeros_like(src_txt)
        trans_inputs["rec_query_mask"] = ~enc_text_mask.bool()
        trans_inputs["rec_pos_embed"] = pos_txt

        # trans_inputs["dec_query"] = query_sen_word
        # trans_inputs["dec_query_mask"] = ~query_mask.bool()
        # trans_inputs["dec_pos_embed"] = dec_pos_embed

        # hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        hs, rec_hs, attn = self.transformer(trans_inputs)
        hs_loc = hs[-1, :, :, :]
        hs_rec = rec_hs[-1, :, : , :]
        attn = attn[:, :, :, :src_vid.shape[1]]

        # outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs_loc)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        outputs_words = self.word_predict(hs_rec)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_spans': outputs_coord, 'pred_mask':batch_dec_mask, 'pred_attn': attn, "pred_words": outputs_words}
        # out = {}
        # txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        # vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        # pdb.set_trace()

        if self.contrastive_align_loss:
            # proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                # proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        

        # out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze()  # (bsz, L_vid)

        # if self.aux_loss:
        #     # assert proj_queries and proj_txt_mem
        #     out['aux_outputs'] = [
        #         {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        #     if self.contrastive_align_loss:
        #         assert proj_queries is not None
        #         for idx, d in enumerate(proj_queries[:-1]):
        #             out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

