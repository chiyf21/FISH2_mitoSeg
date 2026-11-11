"""Unified organelle model: YOLO backbone+neck, dual segmentation paths.

Classes: 0 bg, 1 er, 2 mito, 3 lyso, 4 ld, 5 golgi

m2f / m2f_bin / det_sam / det_sam_mc

det_sam: per-class binary YOLO + per-class prompt (num_prompt_classes=1).
det_sam_mc: one YOLO nc=3 (bg, mito, lyso) + one prompt with num_prompt_classes=2;
  bbox geometry and class softmax are fused in PromptMask3D before mask decode.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from data.organelle_classes import INSTANCE_IDS, NUM_CLASSES, NUM_PROMPT_CLASSES  # noqa: F401
from .backbone_yolo2d import YOLO2DBackbone, YOLO2DNeck
from .backbone_yolo3d import YOLO3DBackbone, YOLO3DNeck
from .head_mask2former2d import Mask2Former2DHead
from .head_mask2former2d_binary import Mask2Former2DBinaryHead
from .head_mask2former3d import Mask2Former3DHead
from .head_mask2former3d_binary import Mask2Former3DBinaryHead
from .head_prompt2d import PromptMask2D
from .head_prompt3d import PromptMask3D
from .head_yolo import Detect2D, Detect3D, SliceDetect3D
from .box_ops import link_slice_boxes, nms_3d

NMS_IOU = 0.6
NMS_PRE = 100
SLICE_CAND_THR = 0.05
SLICE_PRE = 1000


def det_cfg_from_state(state: dict) -> dict:
    """Detection-head kwargs for OrganelleSegDet3D inferred from a checkpoint.

    6-channel box conv = legacy raw-distance 3D head; nested depthwise class branch =
    per-slice 2D head (4*reg_max box channels); otherwise 3D DFL head (6*reg_max).
    """
    keys = [k[len("module."):] if k.startswith("module.") else k for k in state]
    n_heads = len({k.split(".")[1] for k in keys if k.startswith("det_heads.")})
    extra = {"det_classes": INSTANCE_IDS[:n_heads]} if n_heads else {}
    if any(k.endswith("pe_gaussian") for k in keys):
        extra["prompt_pe"] = "rff"
    for prefix in ("det_heads.0.", "det_head."):
        w = state.get(prefix + "cv_box.0.2.weight", state.get("module." + prefix + "cv_box.0.2.weight"))
        if w is None:
            continue
        if any(k.startswith(prefix + "cv_cls.0.0.0.") for k in keys):
            return {"det_box": "slice2d", "reg_max": w.shape[0] // 4, **extra}
        return {"det_box": "3d", "reg_max": max(1, w.shape[0] // 6), **extra}
    return {}


def _nms_select(boxes: torch.Tensor, scores: torch.Tensor, topk: int, labels: torch.Tensor | None = None):
    """Top-NMS_PRE by score, class-aware 3D NMS, then keep topk. Returns indices into boxes."""
    pre = scores.topk(min(NMS_PRE, scores.shape[0])).indices
    b = boxes[pre].float()
    if labels is not None:
        b = b + labels[pre].float().unsqueeze(-1) * 1e4
    keep = nms_3d(b, scores[pre].float(), NMS_IOU)
    return pre[keep[:topk]]


def _slice_detections(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor | None = None):
    """Per-slice 2D detections of one sample: score >= SLICE_CAND_THR, per-slice (class-aware) NMS.

    Thin boxes on different slices never overlap, so one 3D NMS pass is per-slice NMS.
    Returns indices into boxes, sorted by score.
    """
    cand = (scores >= SLICE_CAND_THR).nonzero(as_tuple=True)[0]
    if cand.numel() > SLICE_PRE:
        cand = cand[scores[cand].topk(SLICE_PRE).indices]
    b = boxes[cand].float().cpu()
    if labels is not None:
        b = b + labels[cand].float().cpu().unsqueeze(-1) * 1e4
    keep = nms_3d(b, scores[cand].float().cpu(), NMS_IOU)
    return cand[keep.to(cand.device)]


def _link_select(boxes, scores, payload, topk, labels=None):
    """Per-slice NMS -> link slices into 3D instances (per class) -> 3D NMS -> topk.

    Returns 3D boxes [T,6], payload [T,C], and the per-slice detections (boxes, scores).
    """
    idx = _slice_detections(boxes, scores, labels)
    sb, ss, sp = boxes[idx].detach().float().cpu(), scores[idx].detach().float().cpu(), payload[idx].detach().float().cpu()
    groups = [torch.ones_like(ss, dtype=torch.bool)] if labels is None else [
        (labels[idx].cpu() == c) for c in labels[idx].unique().tolist()
    ]
    tb, ts, tp = [], [], []
    for m in groups:
        b3, s3, p3, _ = link_slice_boxes(sb[m], ss[m], sp[m])
        tb.append(b3)
        ts.append(s3)
        tp.append(p3)
    tb, ts, tp = torch.cat(tb), torch.cat(ts), torch.cat(tp)
    if tb.shape[0]:
        keep = nms_3d(tb, ts, NMS_IOU)[:topk]
        tb, tp = tb[keep], tp[keep]
    dev = boxes.device
    return tb.to(dev), tp.to(dev), (sb.to(dev), ss.to(dev))

from data.organelle_classes import (  # noqa: E402,F401
    CLASS_NAMES,
    CLASS_TO_ID,
    ER_ID,
)


class OrganelleSegDet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        neck_ch: int = 128,
        num_queries: int = 100,
        mode: str = "det_sam",
        dim: str = "3d",
        reg_max: int = 16,
        det_box: str = "slice2d",
        det_classes: tuple[int, ...] = (2, 3),
        prompt_pe: str = "legacy",
        prompt_jitter: float = 0.0,
    ):
        super().__init__()
        self.reg_max = reg_max
        # training only: GT prompt box sides get N(0, (prompt_jitter * side)^2) noise, capped at 20 voxels
        self.prompt_jitter = prompt_jitter
        if det_box not in ("slice2d", "3d"):
            raise ValueError(f"unknown det_box {det_box}")
        self.det_box = det_box if dim == "3d" else "2d"
        if mode not in ("m2f", "m2f_bin", "det_sam", "det_sam_mc"):
            raise ValueError(f"unknown mode {mode}")
        if dim not in ("3d", "2d"):
            raise ValueError(f"unknown dim {dim}")
        self.mode = mode
        self.dim = dim
        # det_sam: one (detect, prompt) head pair per class id in det_classes (2 mito, 3 lyso)
        self.instance_ids = tuple(det_classes)

        if dim == "3d":
            self.backbone = YOLO3DBackbone(in_ch=in_ch, base_ch=base_ch, num_layers=6)
            self.neck = YOLO3DNeck(self.backbone.out_channels, out_channels=neck_ch)
            SegHead, SegBinHead, DetHead, PromptHead = (
                Mask2Former3DHead,
                Mask2Former3DBinaryHead,
                Detect3D,
                PromptMask3D,
            )
        else:
            self.backbone = YOLO2DBackbone(in_ch=in_ch, base_ch=base_ch, num_layers=6)
            self.neck = YOLO2DNeck(self.backbone.out_channels, out_channels=neck_ch)
            SegHead, SegBinHead, DetHead, PromptHead = (
                Mask2Former2DHead,
                Mask2Former2DBinaryHead,
                Detect2D,
                PromptMask2D,
            )

        ch = self.neck.out_channels
        strides = self.neck.out_strides
        if dim == "3d":
            DetHead = partial(SliceDetect3D if det_box == "slice2d" else Detect3D, reg_max=reg_max)

        if mode in ("m2f_bin", "det_sam", "det_sam_mc"):
            self.seg_head = SegBinHead(in_channels=ch, num_classes=NUM_CLASSES, num_queries=num_queries)
        else:
            self.seg_head = SegHead(in_channels=ch, num_classes=NUM_CLASSES, num_queries=num_queries)

        if mode == "det_sam":
            self.det_heads = nn.ModuleList(DetHead(nc=2, ch=ch, strides=strides) for _ in self.instance_ids)
            pe_kw = {"pe": prompt_pe} if dim == "3d" else {}
            self.prompt_decoders = nn.ModuleList(
                PromptHead(feat_ch=ch[0], mask_dim=neck_ch, stride=strides[0], num_prompt_classes=1, **pe_kw)
                for _ in self.instance_ids
            )
            self.er_head = SegHead(in_channels=ch, num_classes=1, num_queries=max(8, num_queries // 4))
        elif mode == "det_sam_mc":
            self.mc_prompt_classes = 2
            self.det_head = DetHead(nc=3, ch=ch, strides=strides)
            self.prompt_decoder = PromptHead(
                feat_ch=ch[0], mask_dim=neck_ch, stride=strides[0], num_prompt_classes=self.mc_prompt_classes
            )
            self.er_head = SegHead(in_channels=ch, num_classes=1, num_queries=max(8, num_queries // 4))

    def forward(self, x, gt_boxes=None, gt_box_labels=None):
        feats = self.backbone(x)
        p3, p4, p5 = self.neck(feats)
        py = [p3, p4, p5]
        out = {"seg": self.seg_head(py)}
        if self.mode == "det_sam":
            out["det"] = [head(py) for head in self.det_heads]
            out["er"] = self.er_head(py)
            out["prompt"] = self._run_prompts(p3, out["det"], gt_boxes, gt_box_labels)
        elif self.mode == "det_sam_mc":
            out["det"] = self.det_head(py)
            out["er"] = self.er_head(py)
            out["prompt"] = self._run_prompt_mc(p3, out["det"], gt_boxes, gt_box_labels)
        return out

    def _run_prompts(self, p3, det_outs, gt_boxes, gt_box_labels):
        prompts = []
        for i, cls_id in enumerate(self.instance_ids):
            scores = slices = None
            if gt_boxes is not None and gt_box_labels is not None:
                boxes = [b[lab.long() == cls_id] for b, lab in zip(gt_boxes, gt_box_labels)]
                if self.training and self.prompt_jitter > 0:
                    boxes = [self._jitter_boxes(b, p3.shape[2:], self.prompt_decoders[i].stride) for b in boxes]
            else:
                boxes, scores, slices = self._top_boxes_binary(det_outs[i], topk=20, return_slices=True)
            cls_probs = [torch.ones(bb.shape[0], 1, device=bb.device, dtype=bb.dtype) for bb in boxes]
            pr = self.prompt_decoders[i](p3, boxes, cls_probs)
            pr["boxes"], pr["box_probs"], pr["slice_dets"] = boxes, scores, slices
            prompts.append(pr)
        return prompts

    def _jitter_boxes(self, b: torch.Tensor, feat_shape, stride) -> torch.Tensor:
        if b.numel() == 0:
            return b
        b = b.float()
        side = (b[:, 3:] - b[:, :3]).clamp(min=1.0).repeat(1, 2)
        noise = (torch.randn_like(b) * self.prompt_jitter * side).clamp(-20.0, 20.0)
        j = b + noise
        hi = torch.tensor([n * s for n, s in zip(feat_shape, stride)], device=b.device, dtype=b.dtype)
        lo3 = torch.minimum(j[:, :3], j[:, 3:] - 1).clamp(min=0)
        hi3 = torch.maximum(j[:, 3:], lo3 + 1).minimum(hi)
        return torch.cat([lo3, hi3], dim=1)

    @staticmethod
    def _box_labels_to_mc_probs(labels: torch.Tensor) -> torch.Tensor:
        cp = torch.zeros(labels.shape[0], 2, device=labels.device, dtype=torch.float32)
        cp[labels == 2, 0] = 1.0
        cp[labels == 3, 1] = 1.0
        return cp

    @staticmethod
    def _remap_box_labels_mc(labels: torch.Tensor) -> torch.Tensor:
        out = labels.clone()
        out[labels == 2] = 1
        out[labels == 3] = 2
        return out

    def _run_prompt_mc(self, p3, det_out, gt_boxes, gt_box_labels):
        if gt_boxes is not None and gt_box_labels is not None:
            boxes_list, probs_list = [], []
            for b, lab in zip(gt_boxes, gt_box_labels):
                lab = lab.long()
                keep = (lab == 2) | (lab == 3)
                boxes_list.append(b[keep])
                probs_list.append(self._box_labels_to_mc_probs(lab[keep]))
            return self.prompt_decoder(p3, boxes_list, probs_list)
        boxes_list, probs_list, slices = self._top_boxes_mc(det_out, topk=20, return_slices=True)
        pr = self.prompt_decoder(p3, boxes_list, probs_list)
        pr["boxes"], pr["box_probs"], pr["slice_dets"] = boxes_list, probs_list, slices
        return pr

    @staticmethod
    def _top_boxes_mc(det_out, topk=20, return_slices=False):
        """3D prompt boxes [K,6] + class probs [K,2] per sample.

        Slice heads: per-slice NMS, z-linking into 3D instances, then 3D NMS.
        return_slices additionally gives the per-slice 2D detections (boxes, scores).
        """
        boxes = det_out["boxes"]
        prob = det_out["logits"].float().softmax(-1)
        fg = prob[..., 1:3]
        scores, labels = fg.max(-1)
        out_boxes, out_probs, out_slices = [], [], []
        for b in range(boxes.shape[0]):
            if det_out.get("box_mode") == "slice2d":
                tb, tp, sl = _link_select(boxes[b], scores[b], fg[b], topk, labels[b])
                out_boxes.append(tb)
                out_probs.append(tp)
                out_slices.append(sl)
                continue
            idx = _nms_select(boxes[b], scores[b], topk, labels[b])
            out_boxes.append(boxes[b][idx].detach())
            out_probs.append(fg[b][idx].detach())
        return (out_boxes, out_probs, out_slices) if return_slices else (out_boxes, out_probs)

    @staticmethod
    def _top_boxes_binary(det_out, topk=20, return_slices=False):
        boxes = det_out["boxes"]
        fg = det_out["logits"].float().softmax(-1)[..., 1]
        out_boxes, out_probs, out_slices = [], [], []
        for b in range(boxes.shape[0]):
            if det_out.get("box_mode") == "slice2d":
                tb, tp, sl = _link_select(boxes[b], fg[b], fg[b].unsqueeze(-1), topk)
                out_boxes.append(tb)
                out_probs.append(tp)
                out_slices.append(sl)
                continue
            idx = _nms_select(boxes[b], fg[b], topk)
            out_boxes.append(boxes[b][idx].detach())
            out_probs.append(fg[b][idx].detach().unsqueeze(-1))
        return (out_boxes, out_probs, out_slices) if return_slices else (out_boxes, out_probs)

    def _top_boxes(self, det_out, topk=20):
        if isinstance(det_out, list):
            all_boxes, all_probs = [], []
            bs = det_out[0]["boxes"].shape[0]
            device = det_out[0]["boxes"].device
            box_dim = det_out[0]["boxes"].shape[-1]
            for b in range(bs):
                boxes_b, probs_b = [], []
                for i, d in enumerate(det_out):
                    bb, pp = self._top_boxes_binary(d, topk=topk)
                    onehot = torch.zeros(bb[b].shape[0], NUM_PROMPT_CLASSES, device=device)
                    onehot[:, i] = pp[b].squeeze(-1)
                    boxes_b.append(bb[b])
                    probs_b.append(onehot)
                all_boxes.append(torch.cat(boxes_b, 0) if boxes_b else torch.zeros(0, box_dim, device=device))
                all_probs.append(
                    torch.cat(probs_b, 0) if probs_b else torch.zeros(0, NUM_PROMPT_CLASSES, device=device)
                )
            return all_boxes, all_probs
        boxes = det_out["boxes"]
        fg = det_out["logits"].softmax(-1)[..., 1:]
        scores = fg.max(-1).values
        out_boxes, out_probs = [], []
        for b in range(boxes.shape[0]):
            k = min(topk, boxes.shape[1])
            idx = scores[b].topk(k).indices
            out_boxes.append(boxes[b][idx].detach())
            out_probs.append(fg[b][idx].detach())
        return out_boxes, out_probs


OrganelleSegDet = OrganelleSegDet3D
