"""Shared 6-layer 3D YOLO backbone with one of three heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone_yolo3d import YOLO3DBackbone
from .head_detr import DETR3DHead
from .head_yolo import Detect3D, ExpertDetect3D


class OrganelleDetector(nn.Module):
    def __init__(
        self,
        head: str,
        num_organelles: int,
        in_ch: int = 1,
        base_ch: int = 32,
        num_layers: int = 6,
        num_queries: int = 100,
    ):
        super().__init__()
        if head not in {"yolo", "detr", "experts"}:
            raise ValueError(head)
        self.head_name = head
        self.num_organelles = num_organelles
        self.backbone = YOLO3DBackbone(in_ch=in_ch, base_ch=base_ch, num_layers=num_layers)
        ch = self.backbone.out_channels
        strides = self.backbone.out_strides
        if head == "yolo":
            # k = m + 1 background
            self.head = Detect3D(nc=num_organelles + 1, ch=ch, strides=strides)
        elif head == "detr":
            self.head = DETR3DHead(feat_ch=ch[-1], num_classes=num_organelles + 1)
        else:
            self.head = ExpertDetect3D(num_experts=num_organelles, ch=ch, strides=strides)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        return self.head(feats)
