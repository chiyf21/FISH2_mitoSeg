"""Per-class binary Mask2Former-style 3D head.

Unlike the standard Mask2Former classification head (one softmax over C
classes + a shared no-object channel, where rare classes get out-competed by
dominant ones), here each organelle class owns:

  - its own group of object queries (num_queries split evenly over classes)
  - its own binary classification head: 2 logits = (background, foreground)

Matching and supervision are done independently per class, so classes never
compete for queries or softmax mass — a lyso query only has to beat "not
lyso", not ER/mitochondria queries. The transformer encoder/decoder, mask
embedding and pixel decoder are shared (mask geometry is class-agnostic);
only the classification head is per-class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_yolo3d import Conv3dBN


class Mask2Former3DBinaryHead(nn.Module):
    """num_classes independent binary (fg/bg) mask classifiers over P3/P4/P5.

    Output pred_logits: [B, Q_total, 2] with channel order (bg, fg) and query
    groups laid out contiguously as [class0 | class1 | ... | classC-1];
    pred_masks: [B, Q_total, D, H, W].
    """

    def __init__(
        self,
        in_channels: tuple[int, ...],
        num_classes: int,
        mask_dim: int = 128,
        hidden_dim: int = 256,
        num_queries: int = 100,  # total, split evenly over classes
        nheads: int = 8,
        dec_layers: int = 4,
        dim_ff: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.queries_per_class = max(4, num_queries // num_classes)
        self.num_queries = self.queries_per_class * num_classes
        self.num_levels = len(in_channels)
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleList(
            nn.Conv3d(c, hidden_dim, 1) if c != hidden_dim else nn.Identity() for c in in_channels
        )
        self.level_embed = nn.Embedding(self.num_levels, hidden_dim)
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nheads, dim_ff, dropout=0.1, batch_first=True),
            num_layers=2,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, nheads, dim_ff, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # one binary (bg, fg) classification head per organelle class
        self.class_embed = nn.ModuleList(
            nn.Linear(hidden_dim, 2) for _ in range(num_classes)
        )
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, mask_dim)
        )
        # Pixel feature branch: upsample P4/P5 to P3 and fuse → mask features at stride 8.
        self.mask_fuse = nn.Sequential(
            Conv3dBN(hidden_dim * self.num_levels, mask_dim, 3, 1),
            Conv3dBN(mask_dim, mask_dim, 3, 1),
        )
        # upsample the fused stride-(z1,xy8) map back to the input resolution
        self.mask_up = nn.Upsample(scale_factor=(1, 8, 8), mode="trilinear", align_corners=False)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        b = feats[0].shape[0]
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        srcs = []
        for i, p in enumerate(proj):
            t = p.flatten(2).transpose(1, 2)  # [B, N_i, C]
            t = t + self.level_embed.weight[i][None, None, :]
            srcs.append(t)
        src = torch.cat(srcs, dim=1)  # [B, N, C]
        memory = self.encoder(src)

        q = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        qpos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(q + qpos, memory)  # [B, Q_total, C]

        # per-class binary logits, then reassemble into [B, Q_total, 2]
        per_cls = []
        for c in range(self.num_classes):
            s = c * self.queries_per_class
            e = s + self.queries_per_class
            per_cls.append(self.class_embed[c](hs[:, s:e]))  # [B, Qc, 2]
        logits = torch.cat(per_cls, dim=1)

        mask_embed = self.mask_embed(hs)  # [B, Q_total, mask_dim]

        p3 = proj[0]
        up = [F.interpolate(p, size=p3.shape[2:], mode="trilinear", align_corners=False) for p in proj]
        mask_features = self.mask_fuse(torch.cat(up, dim=1))
        mask_features = self.mask_up(mask_features)  # full input resolution
        masks = torch.einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)
        return {
            "pred_logits": logits,
            "pred_masks": masks,
            "mask_features": mask_features,
            "mask_size": mask_features.shape[2:],
            "num_classes": self.num_classes,
            "queries_per_class": self.queries_per_class,
        }
