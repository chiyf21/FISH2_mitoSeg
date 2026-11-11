"""Losses for the unified seg+det 3D model.

Mask losses follow Mask2Former: Hungarian matching on (class, mask), then
per-matched-query sigmoid CE + Dice on masks, and CE on class logits with a
down-weighted no-object class. Detection uses the existing YOLO loss on the
instance classes (mito/lyso/ld). ER is supervised only by its mask head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_giou_3d, box_iou_3d


def dice_loss_3d(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / max(num_masks, 1.0)


def sigmoid_ce_loss_3d(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs.flatten(1), targets.flatten(1), reduction="none")
    return loss.mean(1).sum() / max(num_masks, 1.0)


def _gt_to_pred_size(gt_masks, pred_size):
    """Nearest-neighbour resize of GT masks [G,D,H,W] to the prediction's
    spatial size when they differ (e.g. head upsamples 125->128). No-op when
    sizes already match."""
    out = []
    for g in gt_masks:
        if g.shape[0] > 0 and g.ndim >= 3 and tuple(g.shape[-3:]) != tuple(pred_size):
            g = F.interpolate(g[None].float(), size=pred_size, mode="nearest").squeeze(0) > 0.5
        out.append(g)
    return out


def _match_masks(pred_logits, pred_masks, gt_labels, gt_masks, cost_class=1.0, cost_mask=5.0, cost_dice=5.0):
    """Hungarian match per batch. Returns list of (pred_idx, tgt_idx)."""
    bs, q = pred_logits.shape[:2]
    indices = []
    for b in range(bs):
        prob = pred_logits[b].float().softmax(-1)  # [Q, C+1]
        tgt_ids = gt_labels[b].long()
        if tgt_ids.numel() == 0:
            indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        cost_class_m = -prob[:, tgt_ids]
        out_mask = pred_masks[b].float().flatten(1)  # [Q, V]
        tgt_mask = gt_masks[b].float().flatten(1)  # [G, V]
        # subsample voxels for tractable matching
        v = out_mask.shape[1]
        step = max(1, v // 4096)
        out_s = out_mask[:, ::step]  # [Q, Vp]
        tgt_s = tgt_mask[:, ::step]  # [G, Vp]
        out_e = out_s[:, None, :].expand(-1, tgt_s.shape[0], -1)  # [Q,G,Vp]
        ones = torch.ones_like(out_e)
        zeros = torch.zeros_like(out_e)
        pos = F.binary_cross_entropy_with_logits(out_e, ones, reduction="none").mean(-1)  # [Q,G]
        neg = F.binary_cross_entropy_with_logits(out_e, zeros, reduction="none").mean(-1)  # [Q,G]
        gt_frac = tgt_s.mean(1)[None]  # [1,G]
        cost_mask_m = pos * gt_frac + neg * (1 - gt_frac)  # [Q,G]
        num = 2 * (out_s.sigmoid() @ tgt_s.T)
        den = out_s.sigmoid().sum(1)[:, None] + tgt_s.sum(1)[None]
        cost_dice_m = 1 - (num + 1) / (den + 1)
        C = cost_class * cost_class_m + cost_mask * cost_mask_m + cost_dice * cost_dice_m
        C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=1e4)
        ri, ci = linear_sum_assignment(C.detach().cpu().numpy())
        indices.append(
            (torch.as_tensor(ri, dtype=torch.long), torch.as_tensor(ci, dtype=torch.long))
        )
    return indices


def mask2former_loss_3d(
    pred_logits, pred_masks, gt_labels, gt_masks, eos_coef=0.1, num_classes=None, cls_weight=None
):
    """pred_logits [B,Q,C+1], pred_masks [B,Q,D,H,W]; gt lists per batch.

    cls_weight: optional [C] per-foreground-class multipliers (e.g. inverse
    frequency) to counter class collapse onto the dominant organelle.
    """
    if num_classes is None:
        num_classes = pred_logits.shape[-1] - 1
    device = pred_logits.device
    gt_masks = _gt_to_pred_size(gt_masks, pred_masks.shape[-3:])
    indices = _match_masks(pred_logits, pred_masks, gt_labels, gt_masks)
    losses_cls, losses_mask, losses_dice = [], [], []
    for b in range(pred_logits.shape[0]):
        logits = pred_logits[b].float()
        tgt = torch.full((logits.shape[0],), num_classes, dtype=torch.long, device=device)
        src_idx, tgt_idx = indices[b]
        src_idx = src_idx.to(device)
        tgt_idx = tgt_idx.to(device)
        if src_idx.numel() > 0:
            tgt[src_idx] = gt_labels[b].long()[tgt_idx]
        weight = torch.ones(num_classes + 1, device=device)
        if cls_weight is not None:
            weight[:num_classes] = cls_weight.to(device)[:num_classes]
        weight[-1] = eos_coef
        losses_cls.append(F.cross_entropy(logits, tgt, weight=weight))
        if src_idx.numel() > 0:
            pm = pred_masks[b][src_idx].float()
            gm = gt_masks[b][tgt_idx].float()
            n = float(src_idx.numel())
            losses_mask.append(sigmoid_ce_loss_3d(pm, gm, n))
            losses_dice.append(dice_loss_3d(pm, gm, n))
    cls = torch.stack(losses_cls).mean()
    if losses_mask:
        mask = torch.stack(losses_mask).mean()
        dice = torch.stack(losses_dice).mean()
    else:
        mask = pred_masks.sum() * 0.0
        dice = pred_masks.sum() * 0.0
    total = cls + 5.0 * mask + 5.0 * dice
    return {"loss": total, "cls": cls.detach(), "mask": mask.detach(), "dice": dice.detach()}


def mask2former_binary_loss_3d(
    pred_logits,
    pred_masks,
    gt_labels,
    gt_masks,
    num_classes,
    queries_per_class,
    eos_coef=0.1,
    cls_weight=None,
    sup_mask=None,
    neg_mask=None,
):
    """Per-class binary-head Mask2Former loss.

    pred_logits [B, Q_total, 2] with channel order (bg, fg); query group c
    (rows c*qpc : (c+1)*qpc) belongs to foreground class c and is Hungarian-
    matched ONLY against GT entries of class c. Matched queries -> fg,
    unmatched -> bg (down-weighted by eos_coef, like Mask2Former's no-object).
    gt_labels are 0-indexed foreground classes. cls_weight (optional [C])
    scales the fg class of each head's CE.

    Because each class has its own queries/head/matching, a rare class gets
    the same classification gradient magnitude as a dominant one — there is
    no cross-class competition.

    sup_mask: optional bool [B, C] (0-indexed fg classes). Samples with
    sup_mask[b, c] == False do NOT annotate class c (it may be visible but
    unlabeled in that sample's source volume) — they are excluded from class
    c's loss entirely instead of being supervised as background.
    """
    device = pred_logits.device
    gt_masks = _gt_to_pred_size(gt_masks, pred_masks.shape[-3:])
    cls_terms, mask_terms, dice_terms = [], [], []
    for c in range(num_classes):
        s = c * queries_per_class
        e = s + queries_per_class
        if sup_mask is not None or neg_mask is not None:
            bidx = []
            for b in range(pred_logits.shape[0]):
                pos = sup_mask is not None and bool(sup_mask[b, c])
                neg = neg_mask is not None and bool(neg_mask[b, c])
                if pos or neg:
                    bidx.append(b)
        else:
            bidx = list(range(pred_logits.shape[0]))
        if not bidx:
            continue  # no sample in this batch supervises class c
        keep_b = torch.as_tensor(bidx, dtype=torch.long, device=device)
        logits_c = pred_logits[keep_b, s:e].float()  # [Bs, Qc, 2]
        masks_c = pred_masks[keep_b, s:e]  # [Bs, Qc, D, H, W]
        # single-class GT (label 1 = fg) so _match_masks cost_class = -p_fg
        sub_labels, sub_masks = [], []
        for b in bidx:
            keep = gt_labels[b] == c
            sub_labels.append(torch.ones(int(keep.sum()), dtype=torch.long, device=device))
            sub_masks.append(gt_masks[b][keep])
        indices = _match_masks(logits_c, masks_c, sub_labels, sub_masks)
        fg_w = 1.0 if cls_weight is None else float(cls_weight[c])
        weight = torch.tensor([eos_coef, fg_w], device=device)
        for b in range(logits_c.shape[0]):
            tgt = torch.zeros(logits_c.shape[1], dtype=torch.long, device=device)
            src_idx, tgt_idx = indices[b]
            src_idx = src_idx.to(device)
            tgt_idx = tgt_idx.to(device)
            if src_idx.numel() > 0:
                tgt[src_idx] = 1
            cls_terms.append(F.cross_entropy(logits_c[b], tgt, weight=weight))
            if src_idx.numel() > 0:
                pm = masks_c[b][src_idx].float()
                gm = sub_masks[b][tgt_idx].float()
                n = float(src_idx.numel())
                mask_terms.append(sigmoid_ce_loss_3d(pm, gm, n))
                dice_terms.append(dice_loss_3d(pm, gm, n))
    cls = torch.stack(cls_terms).mean()
    if mask_terms:
        mask = torch.stack(mask_terms).mean()
        dice = torch.stack(dice_terms).mean()
    else:
        mask = pred_masks.sum() * 0.0
        dice = pred_masks.sum() * 0.0
    total = cls + 5.0 * mask + 5.0 * dice
    return {"loss": total, "cls": cls.detach(), "mask": mask.detach(), "dice": dice.detach()}


def prompt_mask_loss(pred_masks, gt_masks, pred_ious=None):
    """Supervise box-prompt masks 1:1 with GT instance masks (aligned order)."""
    losses_mask, losses_dice = [], []
    for pm, gm in zip(pred_masks, gt_masks):
        if pm.numel() == 0 or gm.numel() == 0:
            continue
        n = min(pm.shape[0], gm.shape[0])
        pm = pm[:n].float()
        gm = gm[:n].float()
        if gm.shape[0] > 0 and gm.shape[-3:] != pm.shape[-3:]:
            gm = F.interpolate(gm[None], size=pm.shape[-3:], mode="nearest").squeeze(0)
        losses_mask.append(sigmoid_ce_loss_3d(pm, gm, float(n)))
        losses_dice.append(dice_loss_3d(pm, gm, float(n)))
    if not losses_mask:
        z = sum(p.sum() for p in pred_masks) * 0.0
        return {"loss": z, "mask": z.detach(), "dice": z.detach()}
    mask = torch.stack(losses_mask).mean()
    dice = torch.stack(losses_dice).mean()
    return {"loss": 5.0 * mask + 5.0 * dice, "mask": mask.detach(), "dice": dice.detach()}
