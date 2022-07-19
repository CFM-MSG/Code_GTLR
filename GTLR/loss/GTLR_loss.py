import torch.nn as nn
import torch
from torch.autograd import Variable
import pdb

from .span_utils import generalized_temporal_iou, span_cxw_to_xx
import torch.nn.functional as F


def create_GTLR_Loss(loss_args):
    return _GTLR_Loss(loss_args)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score

class _GTLR_Loss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg, margin=0, measure=False, max_violation=False):
        super(_GTLR_Loss, self).__init__()
        self.cfg = cfg
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

        self.use_loc_loss = cfg.get("use_loc_loss", True)
        self.use_iou_loss = cfg.get("use_iou_loss", True)
        self.use_tag_loss = cfg.get("use_tag_loss", True)
        self.use_rec_loss = cfg.get("use_rec_loss", True)

        self.loc_loss_weight = cfg.get("loc_loss_weight", 1)
        self.iou_loss_weight = cfg.get("iou_loss_weight", 1)
        self.tag_loss_weight = cfg.get("tag_loss_weight", 1)
        self.rec_loss_weight = cfg.get("rec_loss_weight", 0.01)


    def loc_loss(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        src_spans = outputs['pred_spans']
        tgt_spans = torch.stack([targets['time_start']/targets['duration'].unsqueeze(dim=-1), targets['time_end']/targets['duration'].unsqueeze(dim=-1)], dim=-1)
        loss_span = (F.l1_loss(src_spans, tgt_spans, reduction='none') * outputs['pred_mask'].unsqueeze(dim=-1)).sum(dim=-1).sum(dim=-1)

        return self.loc_loss_weight*(loss_span.mean())

    def iou_loss(self, outputs, targets):
        assert 'pred_spans' in outputs
        src_spans = outputs['pred_spans']
        tgt_spans = torch.stack([targets['time_start']/targets['duration'].unsqueeze(dim=-1), targets['time_end']/targets['duration'].unsqueeze(dim=-1)], dim=-1)

        total_spans = src_spans.reshape(-1,2)
        total_spans_start = torch.where(total_spans[:, 1] >= total_spans[:, 0], total_spans[:,0], total_spans[:,1])
        total_spans = torch.stack([total_spans_start, total_spans[:,1]],dim=-1)
        # valid_src_spans = total_spans[outputs['pred_mask'].reshape(-1).bool(),:]
        valid_src_spans = total_spans

        # valid_tgt_spans = tgt_spans.reshape(-1,2)[outputs['pred_mask'].reshape(-1).bool(),:]

        valid_tgt_spans = tgt_spans.reshape(-1,2)
        # pdb.set_trace()
        loss_giou = 1 - torch.diag(generalized_temporal_iou(valid_src_spans, valid_tgt_spans))
        if torch.isnan(loss_giou.mean()):
            pdb.set_trace()
        return self.iou_loss_weight*(loss_giou.mean())

    def attention_guided_loss(self, outputs, targets):
        assert 'pred_attn' in outputs
        assert 'pred_mask' in outputs
        assert 'feature_mask' in targets


        src_attn = outputs['pred_attn']
        feat_mask = targets['feature_mask']
        query_mask = outputs['pred_mask']

        
        # src_attn = src_attn*feat_mask.unsqueeze(dim=1).unsqueeze(dim=0)
        # loss = -(((src_attn.sum(dim=-1)+1e-9).log()*query_mask.unsqueeze(dim=0)).sum(dim=-1)/query_mask.sum(dim=-1).unsqueeze(dim=0)).mean()

        src_attn = src_attn.mean(dim=0)[query_mask.bool()]
        loss = -(((src_attn+1e-9).log()*feat_mask).sum(dim=-1)/(feat_mask.sum(dim=-1)+1e-9)).sum()

        return self.tag_loss_weight*(loss/query_mask.shape[0])

    def reconstruct_loss(self, outputs, targets):
        assert 'pred_words' in outputs
        assert 'word_labels' in targets
        pred_words = outputs["pred_words"]
        word_labels = targets["word_labels"]
        pred_mask = outputs['pred_mask']

        # pdb.set_trace()
        # pred_words = pred_words[pred_mask.bool().unsqueeze()]

        
        loss = F.nll_loss(F.log_softmax(pred_words.reshape(-1, pred_words.shape[-1]), dim=-1), word_labels.reshape(-1), reduction="sum")
        return self.rec_loss_weight*(loss/pred_mask.sum().sum())
    
    def forward(self, output, target=None):
        loss = {"total_loss": 0}
        if self.use_loc_loss:
            loss["loc_loss"] = self.loc_loss(output, target)
            loss["total_loss"] += loss["loc_loss"]
        if self.use_iou_loss:
            loss["iou_loss"] = self.iou_loss(output, target)
            loss["total_loss"] += loss["iou_loss"]
        if self.use_tag_loss:
            loss["tag_loss"] = self.attention_guided_loss(output, target)
            loss["total_loss"] += loss["tag_loss"]

        if self.use_rec_loss:
            loss["rec_loss"] = self.reconstruct_loss(output, target)
            loss["total_loss"] += loss["rec_loss"]


        return loss
        
