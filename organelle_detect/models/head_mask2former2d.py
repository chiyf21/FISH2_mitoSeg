"""2D Mask2Former-style segmentation head on the YOLO2D neck pyramid.

Mirror of head_mask2former3d.py with Conv2d / bilinear upsample.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_yolo2d import Conv2dBN


class Mask2Former2DHead(nn.Module):
    """Transformer-decoder mask head over P3/P4/P5 (2D).

    num_classes counts foreground classes only; channel num_classes is the
    no-object / background query class.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...],
        num_classes: int,
        mask_dim: int = 128,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dec_layers: int = 4,
        dim_ff: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_levels = len(in_channels)
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleList(
            nn.Conv2d(c, hidden_dim, 1) if c != hidden_dim else nn.Identity() for c in in_channels
        )
        self.level_embed = nn.Embedding(self.num_levels, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nheads, dim_ff, dropout=0.1, batch_first=True),
            num_layers=2,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, nheads, dim_ff, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, mask_dim)
        )
        self.mask_fuse = nn.Sequential(
            Conv2dBN(hidden_dim * self.num_levels, mask_dim, 3, 1),
            Conv2dBN(mask_dim, mask_dim, 3, 1),
        )
        self.mask_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        b = feats[0].shape[0]
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        srcs = []
        for i, p in enumerate(proj):
            t = p.flatten(2).transpose(1, 2)  # [B, N_i, C]
            t = t + self.level_embed.weight[i][None, None, :]
            srcs.append(t)
        memory = self.encoder(torch.cat(srcs, dim=1))

        q = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        qpos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(q + qpos, memory)

        logits = self.class_embed(hs)
        mask_embed = self.mask_embed(hs)

        p3 = proj[0]
        up = [F.interpolate(p, size=p3.shape[2:], mode="bilinear", align_corners=False) for p in proj]
        mask_features = self.mask_fuse(torch.cat(up, dim=1))
        mask_features = self.mask_up(mask_features)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return {
            "pred_logits": logits,
            "pred_masks": masks,
            "mask_features": mask_features,
            "mask_size": mask_features.shape[2:],
        }
