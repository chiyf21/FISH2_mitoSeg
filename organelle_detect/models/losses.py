"""Detection losses for YOLO / DETR / expert heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import bbox2dist3d, bbox_ciou_2d, box_giou_3d, box_iou_3d


TAL_TOPK = 10
TAL_ALPHA = 0.5
TAL_BETA = 6.0
GAIN_BOX, GAIN_CLS, GAIN_DFL = 7.5, 0.5, 1.5


TAL_STRIDE_VAL = 16
YX = [1, 2, 4, 5]


@torch.no_grad()
def _tal_assign_3d(pred_probs, pred_boxes, anchors_zyx, gt_boxes, gt_labels, bg_id: int = 0,
                   topk: int = TAL_TOPK, alpha: float = TAL_ALPHA, beta: float = TAL_BETA, eps: float = 1e-9,
                   slice2d: bool = False):
    """TaskAlignedAssigner (Ultralytics tal.py) on 3D anchors.

    Candidates are anchors whose center lies inside the GT, with GT y/x sides shorter
    than TAL_STRIDE_VAL enlarged to it about the center (select_candidates_in_gts).
    Among them each GT takes the top-k by score^alpha * IoU^beta. An anchor claimed by
    several GTs keeps the max-IoU one. Target score = align metric rescaled so each GT's
    best anchor gets that GT's best IoU.

    slice2d: boxes are thin per-slice boxes (z, y1, x1, z+1, y2, x2); IoU is 2D CIoU
    on (y, x) and only same-slice anchors are candidates. Otherwise 3D IoU.

    pred_probs [B,A,nc] softmax, pred_boxes [B,A,6], anchors_zyx [3,A].
    Returns target_cls [B,A] long, target_box [B,A,6], target_score [B,A], fg [B,A] bool.
    """
    b, a, _ = pred_boxes.shape
    device = pred_boxes.device
    target_cls = torch.full((b, a), bg_id, dtype=torch.long, device=device)
    target_box = torch.zeros_like(pred_boxes)
    target_score = torch.zeros((b, a), dtype=pred_boxes.dtype, device=device)
    fg = torch.zeros((b, a), dtype=torch.bool, device=device)
    anc = anchors_zyx.T.to(pred_boxes.dtype)  # [A,3]
    ar = torch.arange(a, device=device)
    for i in range(b):
        gtb, gtl = gt_boxes[i].to(device=device, dtype=pred_boxes.dtype), gt_labels[i].long().to(device)
        if gtb.numel() == 0:
            continue
        g = gtb.shape[0]
        lo, hi = gtb[:, :3].clone(), gtb[:, 3:].clone()
        ctr, half = (lo + hi) / 2, (hi - lo) / 2
        half[:, 1:] = half[:, 1:].clamp(min=TAL_STRIDE_VAL / 2)
        lo[:, 1:], hi[:, 1:] = ctr[:, 1:] - half[:, 1:], ctr[:, 1:] + half[:, 1:]
        inside = ((anc[None] - lo[:, None] > eps) & (hi[:, None] - anc[None] > eps)).all(-1)  # [G,A]
        if slice2d:
            iou = bbox_ciou_2d(gtb[:, None, YX], pred_boxes[i][None, :, YX]).clamp(min=0) * inside
        else:
            iou = box_iou_3d(gtb, pred_boxes[i]).clamp(min=0) * inside  # [G,A]
        score = pred_probs[i][:, gtl].T  # [G,A]
        metric = score.pow(alpha) * iou.pow(beta) * inside
        k = min(topk, a)
        top_idx = metric.masked_fill(~inside, -1.0).topk(k, dim=1).indices
        sel = torch.zeros((g, a), dtype=torch.bool, device=device)
        sel.scatter_(1, top_idx, True)
        sel &= inside
        multi = sel.sum(0) > 1
        if multi.any():
            onehot = torch.zeros_like(sel)
            onehot[iou.argmax(0), ar] = True
            sel[:, multi] = onehot[:, multi]
        pos = sel.any(0)
        if not pos.any():
            continue
        gi = sel.float().argmax(0)  # [A]
        m = metric * sel
        norm = (m * (iou * sel).amax(1, keepdim=True) / (m.amax(1, keepdim=True) + eps)).amax(0)
        fg[i] = pos
        target_cls[i, pos] = gtl[gi[pos]]
        target_box[i, pos] = gtb[gi[pos]]
        target_score[i] = norm * pos
    return target_cls, target_box, target_score, fg


def _dfl_loss(pred_dist, target):
    """pred_dist [N,S,R] logits, target [N,S] in stride units -> [N] (Ultralytics DFLoss)."""
    r = pred_dist.shape[-1]
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1 - wl
    pd = pred_dist.reshape(-1, r)
    ll = F.cross_entropy(pd, tl.reshape(-1), reduction="none").view_as(target)
    lr = F.cross_entropy(pd, tr.clamp(max=r - 1).reshape(-1), reduction="none").view_as(target)
    return (ll * wl + lr * wr).mean(-1)


def yolo_loss(
    out: dict,
    gt_boxes,
    gt_labels,
    crop: int,
    bg_id: int = 0,
    hard_neg_boxes=None,
    hard_neg_iou: float = 0.3,
    hard_neg_weight: float = 3.0,
) -> dict[str, torch.Tensor]:
    """YOLOv8/11 detection loss: TAL assignment, IoU-aware soft class targets, score-weighted IoU + DFL.

    Class prob is softmax over nc (channel bg_id = background); for nc=2 the soft
    cross-entropy equals Ultralytics' BCE on the fg-vs-bg logit difference.
    Slice heads (box_mode "slice2d") expect per-slice thin GT boxes and use 2D CIoU
    plus DFL on (top, left, bottom, right); 3D heads use 3D GIoU plus DFL on 6 sides.
    """
    pred_boxes = out["boxes"].float()
    logits = out["logits"].float()
    b, a, nc = logits.shape
    anchors = out["anchors"].float()
    slice2d = out.get("box_mode") == "slice2d"
    logp = logits.log_softmax(-1)
    target_cls, target_box, target_score, fg = _tal_assign_3d(
        logp.detach().exp(), pred_boxes.detach(), anchors, gt_boxes, gt_labels, bg_id, slice2d=slice2d
    )
    tsum = target_score.sum().clamp(min=1.0)

    q = torch.zeros_like(logp)
    q[..., bg_id] = 1.0 - target_score
    q.scatter_add_(-1, target_cls.unsqueeze(-1), target_score.unsqueeze(-1))
    q[..., bg_id] = q[..., bg_id].clamp(max=1.0)
    cls_per = -(q * logp).sum(-1)  # [B,A]
    if hard_neg_boxes is not None:
        w = torch.ones_like(cls_per)
        for i in range(b):
            nb = hard_neg_boxes[i]
            if nb is None or nb.numel() == 0:
                continue
            iou = box_iou_3d(pred_boxes[i].detach(), nb.to(pred_boxes.device))
            hi = (iou.max(dim=1).values >= hard_neg_iou).float()
            w[i] = 1.0 + (hard_neg_weight - 1.0) * hi
        cls_per = cls_per * w
    cls_loss = cls_per.sum() / tsum

    if fg.any():
        wt = target_score[fg]
        if slice2d:
            iou = bbox_ciou_2d(pred_boxes[fg][:, YX], target_box[fg][:, YX]).nan_to_num(0.0)
        else:
            iou = box_giou_3d(pred_boxes[fg], target_box[fg]).nan_to_num(0.0)
        box_loss = ((1.0 - iou) * wt).sum() / tsum
        pred_dist = out.get("pred_dist")
        if pred_dist is not None:
            reg_max = pred_dist.shape[2]
            st = out["anchor_strides"].float()  # [A,3]
            st6 = torch.cat([st, st], 1)
            bi, ai = fg.nonzero(as_tuple=True)
            tgt_d = bbox2dist3d(anchors.T[ai], target_box[fg], st6[ai], reg_max)
            if slice2d:
                tgt_d = tgt_d[:, YX]  # (top, left, bottom, right)
            pdist = pred_dist.float().permute(0, 3, 1, 2)[bi, ai]  # [N,S,R]
            dfl = (_dfl_loss(pdist, tgt_d) * wt).sum() / tsum
        else:
            dfl = pred_boxes.sum() * 0.0
    else:
        box_loss = pred_boxes.sum() * 0.0
        dfl = pred_boxes.sum() * 0.0
        if out.get("pred_dist") is not None:
            box_loss = box_loss + out["pred_dist"].sum() * 0.0
    total = GAIN_BOX * box_loss + GAIN_CLS * cls_loss + GAIN_DFL * dfl
    return {"loss": total, "cls": cls_loss.detach(), "giou": box_loss.detach(), "dfl": dfl.detach(),
            "l1": dfl.detach(), "n_pos": fg.sum().detach()}


def expert_loss(outs: list[dict], gt_boxes, gt_labels, crop: int, num_experts: int) -> dict[str, torch.Tensor]:
    total = 0.0
    parts = []
    for e, out in enumerate(outs):
        e_boxes, e_labels = [], []
        cls_id = e + 1  # organelle class ids are 1..m
        for b in range(len(gt_boxes)):
            keep = gt_labels[b] == cls_id
            e_boxes.append(gt_boxes[b][keep])
            # expert uses {0:bg, 1:foreground}
            e_labels.append(torch.ones(int(keep.sum()), dtype=torch.long, device=gt_labels[b].device))
        parts.append(yolo_loss(out, e_boxes, e_labels, crop, bg_id=0)["loss"])
        total = total + parts[-1]
    loss = total / max(num_experts, 1)
    return {"loss": loss, "cls": loss.detach(), "giou": loss.detach(), "l1": loss.detach()}


def hungarian_detr(pred_logits, pred_boxes_norm, gt_boxes, gt_labels, crop: int, eos_coef=0.1):
    """pred_boxes_norm in [0,1]; gt_boxes in voxels. no-object class is last logit."""
    bs, nq, nc = pred_logits.shape
    device = pred_logits.device
    losses = []
    for i in range(bs):
        logits = pred_logits[i].float()
        pbox = pred_boxes_norm[i].float() * crop
        gtb, gtl = gt_boxes[i].float(), gt_labels[i].long()
        target_cls = torch.full((nq,), nc - 1, dtype=torch.long, device=device)  # no-object
        target_box = torch.zeros((nq, 6), device=device)
        if gtb.numel() > 0:
            prob = logits.softmax(-1)
            cost_cls = -prob[:, gtl]
            cost_bbox = torch.cdist(pbox, gtb, p=1) / crop
            cost_giou = -box_iou_3d(pbox, gtb)
            C = 2.0 * cost_bbox + 1.0 * cost_cls + 2.0 * cost_giou
            ri, ci = linear_sum_assignment(C.detach().cpu().numpy())
            ri = torch.as_tensor(ri, device=device)
            ci = torch.as_tensor(ci, device=device)
            target_cls[ri] = gtl[ci]
            target_box[ri] = gtb[ci]
        weight = torch.ones(nc, device=device)
        weight[-1] = eos_coef
        cls_loss = F.cross_entropy(logits, target_cls, weight=weight)
        pos = target_cls != (nc - 1)
        if pos.any():
            box_loss = (1.0 - box_giou_3d(pbox[pos], target_box[pos]).nan_to_num(0.0)).mean()
            l1 = F.l1_loss(pbox[pos].clamp(0, crop) / crop, target_box[pos] / crop)
        else:
            box_loss = pbox.sum() * 0.0
            l1 = pbox.sum() * 0.0
        losses.append(cls_loss + 2.0 * box_loss + 5.0 * l1)
    loss = torch.stack(losses).mean()
    return {"loss": loss, "cls": loss.detach(), "giou": loss.detach(), "l1": loss.detach()}
