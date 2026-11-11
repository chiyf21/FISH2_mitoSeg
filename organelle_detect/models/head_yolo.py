"""Decoupled 3D YOLO detection head.

Class layout (user spec): k = n_organelle_types + 1 background.
  channel 0 = background, 1..m = organelle classes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_yolo3d import Conv3dBN
from .backbone_yolo2d import Conv2dBN
from .box_ops import dist2bbox2d, dist2bbox3d, make_anchors_2d, make_anchors_3d


class Detect3D(nn.Module):
    """YOLOv8-style decoupled 3D head.

    reg_max > 1: each of the 6 box sides is a distribution over reg_max bins in
    stride units (DFL), decoded as the expectation times the per-axis stride.
    reg_max == 1: legacy raw voxel-distance regression (old checkpoints).
    Channel 0 of the class logits is background; class prob = softmax over nc.
    """

    def __init__(
        self,
        nc: int,
        ch: tuple[int, ...],
        strides: tuple[int, ...],
        reg_max: int = 16,
        prior_crop: int = 188,
        prior_depth: int = 10,
    ):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.strides = strides
        self.reg_max = reg_max
        self.prior_crop = prior_crop
        self.prior_depth = prior_depth
        c2 = max(16, ch[0] // 4, reg_max * 4) if reg_max > 1 else max(16, ch[0] // 4)
        c3 = max(ch[0], min(nc, 64))
        self.cv_box = nn.ModuleList(
            nn.Sequential(Conv3dBN(c, c2, 3, 1), Conv3dBN(c2, c2, 3, 1), nn.Conv3d(c2, 6 * reg_max, 1)) for c in ch
        )
        self.cv_cls = nn.ModuleList(
            nn.Sequential(Conv3dBN(c, c3, 3, 1), Conv3dBN(c3, c3, 3, 1), nn.Conv3d(c3, nc, 1)) for c in ch
        )
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        if reg_max > 1:
            self.bias_init()

    def bias_init(self):
        """Ultralytics init: DFL bins flat, foreground logit at ~5 objects per crop prior."""
        for box_seq, cls_seq, s in zip(self.cv_box, self.cv_cls, self.strides):
            sz, sxy = (float(s[0]), float(s[-1])) if isinstance(s, (tuple, list)) else (float(s), float(s))
            n_anchor = (self.prior_crop / sxy) ** 2 * max(self.prior_depth / sz, 1.0)
            box_seq[-1].bias.data[:] = 1.0
            cls_seq[-1].bias.data[:] = 0.0
            cls_seq[-1].bias.data[1:] = math.log(5 / max(self.nc - 1, 1) / n_anchor)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = feats[0].shape[0]
        boxes, scores = [], []
        for i in range(self.nl):
            boxes.append(self.cv_box[i](feats[i]).flatten(2))  # [B,6*R,A_i]
            scores.append(self.cv_cls[i](feats[i]).flatten(2))  # [B,nc,A_i]
        box = torch.cat(boxes, dim=2)
        cls = torch.cat(scores, dim=2)
        anchors, strides = make_anchors_3d(feats, self.strides)
        if self.reg_max > 1:
            a = box.shape[-1]
            pred_dist = box.float().view(bs, 6, self.reg_max, a)
            dist_units = (pred_dist.softmax(2) * self.proj.view(1, 1, -1, 1)).sum(2)  # [B,6,A]
            dist = dist_units * torch.cat([strides, strides], dim=1).T[None]  # stride units -> voxels
        else:
            pred_dist = None
            dist = box.float().clamp(min=0, max=4096)
        decoded = dist2bbox3d(dist, anchors)
        return {
            "boxes": decoded,
            "logits": cls.permute(0, 2, 1).contiguous(),  # [B,A,nc]
            "raw_dist": box,
            "pred_dist": pred_dist,  # [B,6,R,A] or None (legacy)
            "reg_max": self.reg_max,
            "anchors": anchors,
            "anchor_strides": strides,
            "bs": bs,
        }


class SliceDetect3D(nn.Module):
    """YOLO11-style anchor-free DFL head: 3D features in, one 2D box per z-slice out.

    Coarser-in-z levels are trilinearly resized in z to the finest level's depth, so
    every level has one anchor per (z-slice, y, x) cell. Each anchor regresses the 4
    in-plane distances (top, left, bottom, right) as reg_max-bin distributions in
    stride units. Boxes are returned as thin 3D boxes (z, y1, x1, z+1, y2, x2) so
    3D IoU between same-slice boxes equals 2D IoU and is 0 across slices.
    Box branch = Ultralytics cv2, class branch = YOLO11 cv3 (depthwise-separable).
    """

    box_mode = "slice2d"

    def __init__(
        self,
        nc: int,
        ch: tuple[int, ...],
        strides: tuple,
        reg_max: int = 16,
        prior_crop: int = 188,
    ):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.strides = tuple((1, int(s[1]), int(s[2])) for s in strides)
        self.reg_max = reg_max
        self.prior_crop = prior_crop
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv_box = nn.ModuleList(
            nn.Sequential(Conv3dBN(c, c2, 3, 1), Conv3dBN(c2, c2, 3, 1), nn.Conv3d(c2, 4 * reg_max, 1)) for c in ch
        )
        self.cv_cls = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv3dBN(c, c, 3, 1, g=c), Conv3dBN(c, c3, 1, 1)),
                nn.Sequential(Conv3dBN(c3, c3, 3, 1, g=c3), Conv3dBN(c3, c3, 1, 1)),
                nn.Conv3d(c3, nc, 1),
            )
            for c in ch
        )
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)
        self.bias_init()

    def bias_init(self):
        """Ultralytics Detect.bias_init: box bins 2.0, fg logit at ~5 objects per slice image."""
        for box_seq, cls_seq, s in zip(self.cv_box, self.cv_cls, self.strides):
            box_seq[-1].bias.data[:] = 2.0
            cls_seq[-1].bias.data[:] = 0.0
            cls_seq[-1].bias.data[1:] = math.log(5 / max(self.nc - 1, 1) / (self.prior_crop / s[-1]) ** 2)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = feats[0].shape[0]
        d = max(f.shape[2] for f in feats)
        feats = [
            f if f.shape[2] == d else F.interpolate(f, size=(d, *f.shape[3:]), mode="trilinear", align_corners=False)
            for f in feats
        ]
        box = torch.cat([self.cv_box[i](feats[i]).flatten(2) for i in range(self.nl)], dim=2)  # [B,4R,A]
        cls = torch.cat([self.cv_cls[i](feats[i]).flatten(2) for i in range(self.nl)], dim=2)  # [B,nc,A]
        anchors, strides = make_anchors_3d(feats, self.strides)  # [3,A] (z+.5, y, x), [A,3]
        a = box.shape[-1]
        pred_dist = box.float().view(bs, 4, self.reg_max, a)
        dist = (pred_dist.softmax(2) * self.proj.view(1, 1, -1, 1)).sum(2)  # [B,4,A] stride units
        syx = strides[:, 1:].T  # [2,A]
        dist = dist * torch.cat([syx, syx], 0)[None]
        z, y, x = anchors
        zero = torch.zeros_like(dist[:, 0])
        decoded = torch.stack(
            [zero + z - 0.5, y - dist[:, 0], x - dist[:, 1], zero + z + 0.5, y + dist[:, 2], x + dist[:, 3]], dim=-1
        )  # [B,A,6]
        return {
            "boxes": decoded,
            "logits": cls.permute(0, 2, 1).contiguous(),  # [B,A,nc]
            "raw_dist": box,
            "pred_dist": pred_dist,  # [B,4,R,A]
            "reg_max": self.reg_max,
            "box_mode": self.box_mode,
            "anchors": anchors,
            "anchor_strides": strides,
            "bs": bs,
        }


class Detect2D(nn.Module):
    """2D decoupled YOLO head: boxes are (y1,x1,y2,x2)."""

    def __init__(self, nc: int, ch: tuple[int, ...], strides: tuple[int, ...]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.strides = strides
        c2 = max(16, ch[0] // 4)
        c3 = max(ch[0], min(nc, 64))
        self.cv_box = nn.ModuleList(
            nn.Sequential(Conv2dBN(c, c2, 3, 1), Conv2dBN(c2, c2, 3, 1), nn.Conv2d(c2, 4, 1)) for c in ch
        )
        self.cv_cls = nn.ModuleList(
            nn.Sequential(Conv2dBN(c, c3, 3, 1), Conv2dBN(c3, c3, 3, 1), nn.Conv2d(c3, nc, 1)) for c in ch
        )

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = feats[0].shape[0]
        boxes, scores = [], []
        for i in range(self.nl):
            boxes.append(self.cv_box[i](feats[i]).flatten(2))  # [B,4,A_i]
            scores.append(self.cv_cls[i](feats[i]).flatten(2))  # [B,nc,A_i]
        box = torch.cat(boxes, dim=2)
        cls = torch.cat(scores, dim=2)
        anchors, strides = make_anchors_2d(feats, self.strides)
        decoded = dist2bbox2d(box.float().clamp(min=0, max=4096), anchors)
        return {
            "boxes": decoded,
            "logits": cls.permute(0, 2, 1).contiguous(),
            "raw_dist": box,
            "anchors": anchors,
            "anchor_strides": strides,
            "bs": bs,
        }


class ExpertDetect3D(nn.Module):
    """m parallel 2-class YOLO heads (background vs one organelle)."""

    def __init__(self, num_experts: int, ch: tuple[int, ...], strides: tuple[int, ...]):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(Detect3D(nc=2, ch=ch, strides=strides) for _ in range(num_experts))

    def forward(self, feats: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [head(feats) for head in self.experts]
