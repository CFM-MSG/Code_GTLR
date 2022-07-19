import copy
import pdb
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from GTLR.model.graph_transformer_layer import GraphSelfAttention, GraphEncoderDecoderAttention
from GTLR.model.transformer import TransformerDecoderLayer, TransformerDecoder


class Graph_Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False, return_intermediate_dec=False, use_mme=True, use_rec=True):
        super().__init__()

        encoder_layer = GraphTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = GraphTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder_layer = GraphTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model) 
        # self.decoder = GraphTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, "relu", normalize_before, return_attn=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, )



        # reconstructer_layer = GraphTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        # reconstructer_norm = nn.LayerNorm(d_model) 
        # self.reconstructer = GraphTransformerDecoder(reconstructer_layer, num_decoder_layers, reconstructer_norm, return_intermediate=return_intermediate_dec)

        reconstructer_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, "relu", normalize_before, return_attn=True)
        reconstructer_norm = nn.LayerNorm(d_model)
        self.reconstructer = TransformerDecoder(reconstructer_layer, num_decoder_layers, reconstructer_norm,
                                          return_intermediate=return_intermediate_dec, )


        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        self.use_mme = use_mme
        self.use_rec = use_rec
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs):
        # src = inputs["enc_input"]
        # pos_embed = inputs["enc_pos_embed"]
        # src_mask = inputs["enc_mask"]

        src_vid = inputs["enc_vid_input"]
        vid_pos_embed = inputs["enc_vid_pos_embed"]
        vid_mask = inputs["enc_vid_mask"]

        src_txt = inputs["enc_txt_input"]
        txt_pos_embed = inputs["enc_txt_pos_embed"]
        txt_mask = inputs["enc_txt_mask"]

        queries = inputs["dec_query"]
        query_embed = inputs["dec_pos_embed"]
        query_mask = inputs["dec_query_mask"]

        rec_query = inputs["rec_query"]
        rec_mask = inputs["rec_query_mask"]
        rec_embed = inputs["rec_pos_embed"]

        # bs, c, t = src.shape
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.permute(1, 0, 2)
        tgt = queries.permute(1, 0, 2)
        rec_embed = rec_embed.permute(1, 0, 2)
        rec_query = rec_query.permute(1, 0, 2)
        # tgt = torch.zeros_like(query_embed)
        # encoder_mask = None
        # pdb.set_trace()

        if self.use_mme:
            src = torch.cat((src_vid, src_txt), dim = 1).permute(1, 0, 2)
            src_mask = torch.cat((vid_mask, txt_mask), dim=-1)
            pos_embed = torch.cat((vid_pos_embed, txt_pos_embed), dim = 1).permute(1, 0, 2)
            memory = self.encoder(src, src_key_padding_mask=src_mask, pos=pos_embed)
            memory_vid = memory[:src_vid.shape[1],:,:]
            memory_src = memory[src_vid.shape[1]:,:,:]

            # tgt_mask = None
            # hs, edge = self.decoder(tgt, ctx.permute(1,0,2), tgt_key_padding_mask=query_mask, memory_key_padding_mask=src_mask, pos=pos_embed, query_pos=query_embed)
            # tgt = memory_src.mean(dim=0).unsqueeze(dim=0)
            ret_dec = self.decoder(tgt, memory_vid, tgt_mask = None, tgt_key_padding_mask=None, memory_key_padding_mask=vid_mask, pos=None, query_pos=query_embed)  # (#layers, #queries, batch_size, d), (#layers, batch_size, #queries, #memory)
            if isinstance(ret_dec, Tuple):
                hs, attn = ret_dec
            else:
                hs = ret_dec
                attn = None

            # rec_hs, rec_edge = self.reconstructer(rec_query, hs[-1].permute(1,0,2), tgt_key_padding_mask=rec_mask, memory_key_padding_mask=query_mask, pos=query_embed, query_pos=rec_embed)
            rec_hs = hs


            if self.use_rec:
                memory_rec = hs[-1]
                # memory_rec = memory_rec[query_mask.permute(1,0)]
                rec_hs, _ = self.reconstructer(memory_src, memory_rec, tgt_mask = None, tgt_key_padding_mask=rec_mask, memory_key_padding_mask=query_mask, pos=None, query_pos=rec_embed)  # (#layers, #queries, batch_size, d), (#layers, batch_size, #queries, #memory)
        else:
            src = src_vid.permute(1, 0, 2)
            src_mask = vid_mask
            pos_embed = vid_pos_embed.permute(1, 0, 2)

            memory = self.encoder(src, src_key_padding_mask=src_mask, pos=pos_embed)

            # pdb.set_trace()
            # tgt_mask = None
            # hs, edge = self.decoder(tgt, ctx.permute(1,0,2), tgt_key_padding_mask=query_mask, memory_key_padding_mask=src_mask, pos=pos_embed, query_pos=query_embed)
            ret_dec = self.decoder(tgt, memory, tgt_mask = None, tgt_key_padding_mask=query_mask, memory_key_padding_mask=src_mask, pos=None, query_pos=query_embed)  # (#layers, #queries, batch_size, d), (#layers, batch_size, #queries, #memory)
            if isinstance(ret_dec, Tuple):
                hs, attn = ret_dec
            else:
                hs = ret_dec
                attn = None

            # rec_hs, rec_edge = self.reconstructer(rec_query, hs[-1].permute(1,0,2), tgt_key_padding_mask=rec_mask, memory_key_padding_mask=query_mask, pos=query_embed, query_pos=rec_embed)

            rec_hs = hs
            if self.use_rec:

                rec_hs, _ = self.reconstructer(rec_query, hs[-1], tgt_mask = None, tgt_key_padding_mask=rec_mask, memory_key_padding_mask=query_mask, pos=None, query_pos=rec_embed)  # (#layers, #queries, batch_size, d), (#layers, batch_size, #queries, #memory)
                


        # pdb.set_trace()
        return hs.transpose(1, 2), rec_hs.transpose(1, 2), attn



class GraphTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class GraphTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output, edge = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate), edge
        return output.unsqueeze(0), edge


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()

        self.self_attn = GraphSelfAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src_mask = ~src_key_padding_mask
        if src_mask is not None:
            graph = q.permute(1,0,2) * src_mask[:,:,None]
            adj = src_mask[:,:,None].float() @ src_mask[:,None,:].float()
        else:
            graph = q.permute(1,0,2)
            adj =  (torch.ones((q.size(1),q.size(0),q.size(0))))
            adj = adj.to(q.device)
        # pdb.set_trace()
        
        src2, _ = self.self_attn(graph, src, adj)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        if src_mask:
            graph = q.permute(1,0,2) * src_mask[:,:,None]
        else:
            graph = q.permute(1,0,2)

        adj =  (torch.ones((q.size(1),q.size(0),q.size(0))))
        adj = adj.to(q.device)
        src2, _ = self.self_attn(graph, pos, adj).permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class GraphTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False): 
        super().__init__()

        self.self_attn = GraphSelfAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GraphEncoderDecoderAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt_mask = ~tgt_key_padding_mask
        memory_mask = ~memory_key_padding_mask
        if tgt_mask is not None:
            graph = q.permute(1,0,2) * tgt_mask[:,:,None]
            adj = tgt_mask[:,:,None].float() @ tgt_mask[:,None,:].float()
        else:
            graph = q.permute(1,0,2)
            adj = torch.ones((q.size(1),q.size(0),q.size(0)))
            adj = adj.to(q.device)

        tgt2, edge = self.self_attn(graph, tgt, adj)

        tgt2 = tgt2.permute(1,0,2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if memory_mask is not None:
            st_adj = tgt_mask[:,:,None].float() @ memory_mask[:,None,:].float() 
        st_adj = torch.ones((q.size(1),q.size(0),memory.size(1))).to(q.device)

        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).permute(1,0,2), self.with_pos_embed(memory,pos.permute(1,0,2)), memory, tgt, st_adj).permute(1,0,2)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, edge

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        adj = torch.ones((q.size(0),q.size(0)))
        adj = adj.to(q.device)
        tgt2, edge = self.self_attn(q.permute(1,0,2), adj).permute(1,0,2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        st_adj = torch.ones((q.size(0),memory.size(1))).to(q.device)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).permute(1,0,2), self.with_pos_embed(memory,pos.permute(1,0,2)), memory, tgt, st_adj).permute(1,0,2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, edge

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_graph_transformer(args):
     return Graph_Transformer(
        d_model=args["hidden_dim"],
        dropout=args["dropout"],
        nhead=args["nheads"],
        dim_feedforward=args["dim_feedforward"],
        num_encoder_layers=args["enc_layers"],
        num_decoder_layers=args["dec_layers"],
        normalize_before=args["pre_norm"],
        return_intermediate_dec=True,
        use_mme=args["use_mme"],
        use_rec = args["use_rec"]
       )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "elu":
        return F.elu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/elu, not {activation}.")
