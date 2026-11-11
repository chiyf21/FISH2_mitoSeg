"""Sparse DETR-style 3D detection head on the coarsest YOLO feature map."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inp: int, hid: int, out: int, layers: int = 3):
        super().__init__()
        mods = []
        for i in range(layers):
            mods.append(nn.Linear(inp if i == 0 else hid, out if i == layers - 1 else hid))
            if i < layers - 1:
                mods.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class DETR3DHead(nn.Module):
    """num_classes includes background / no-object as the last channel (DETR convention)."""

    def __init__(
        self,
        feat_ch: int,
        num_classes: int,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_ff: int = 512,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.input_proj = nn.Conv3d(feat_ch, d_model, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 6, 3)

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        x = feats[-1]
        b, c, d, h, w = x.shape
        src = self.input_proj(x).flatten(2).permute(0, 2, 1)  # [B, DHW, C]
        memory = self.encoder(src)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(q, memory)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()  # normalized z1,y1,x1,z2,y2,x2 in [0,1]
        return {"logits": logits, "boxes_norm": boxes}
