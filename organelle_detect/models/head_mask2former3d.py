"""3D Mask2Former-style segmentation head on the YOLO3D neck pyramid.

Queries are class-bearing object queries (class logits include a final
"no-object" channel, Mask2Former convention). Masks are produced by a dot
product between per-query mask embeddings and a high-resolution pixel feature
map, exactly like Mask2Former, but in 3D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_yolo3d import Conv3dBN


class Mask2Former3DHead(nn.Module):
    """Transformer-decoder mask head over P3/P4/P5.

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
            nn.Conv3d(c, hidden_dim, 1) if c != hidden_dim else nn.Identity() for c in in_channels
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

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 = no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, mask_dim)
        )
        # Pixel feature branch: upsample P4/P5 to P3 and fuse → mask features at stride 8.
        self.mask_fuse = nn.Sequential(
            Conv3dBN(hidden_dim * self.num_levels, mask_dim, 3, 1),
            Conv3dBN(mask_dim, mask_dim, 3, 1),
        )
        # upsample the fused stride-(z8,xy8) map back to the input resolution;
        # z stride is 1 (anisotropic backbone), xy stride is 8
        self.mask_up = nn.Upsample(scale_factor=(1, 8, 8), mode="trilinear", align_corners=False)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        b = feats[0].shape[0]
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        # flatten pyramid into one token stream with level embeddings
        srcs, sizes = [], []
        for i, p in enumerate(proj):
            d, h, w = p.shape[2:]
            sizes.append((d, h, w))
            t = p.flatten(2).transpose(1, 2)  # [B, N_i, C]
            t = t + self.level_embed.weight[i][None, None, :]
            srcs.append(t)
        src = torch.cat(srcs, dim=1)  # [B, N, C]
        memory = self.encoder(src)

        q = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        qpos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(q + qpos, memory)

        logits = self.class_embed(hs)  # [B, Q, C+1]
        mask_embed = self.mask_embed(hs)  # [B, Q, mask_dim]

        p3 = proj[0]
        up = [F.interpolate(p, size=p3.shape[2:], mode="trilinear", align_corners=False) for p in proj]
        mask_features = self.mask_fuse(torch.cat(up, dim=1))  # [B, mask_dim, D8, H8, W8]
        mask_features = self.mask_up(mask_features)  # full input resolution
        masks = torch.einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)
        return {
            "pred_logits": logits,
            "pred_masks": masks,
            "mask_features": mask_features,
            "mask_size": mask_features.shape[2:],
        }
