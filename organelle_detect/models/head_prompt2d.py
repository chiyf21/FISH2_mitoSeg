"""SAM2-style 2D prompt mask decoder (mirror of head_prompt3d.py).

Boxes are (y1,x1,y2,x2). Each box is decoded individually; class softmax is
fused into the prompt embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoWayBlock2D(nn.Module):
    def __init__(self, d_model: int, nheads: int, dim_ff: int = 512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross1 = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.cross2 = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor):
        q = self.norm1(queries + self.self_attn(queries, queries, queries)[0])
        q = self.norm2(q + self.cross1(q, keys, keys)[0])
        q = self.norm3(q + self.mlp(q))
        k = self.norm4(keys + self.cross2(keys, q, q)[0])
        return q, k


class PromptMask2D(nn.Module):
    """Box-prompt mask decoder over the finest neck feature map (stride 8)."""

    def __init__(
        self,
        feat_ch: int,
        mask_dim: int = 128,
        hidden_dim: int = 256,
        nheads: int = 8,
        layers: int = 2,
        stride: int = 8,
        num_prompt_classes: int = 3,
    ):
        super().__init__()
        self.stride = int(stride) if not isinstance(stride, (tuple, list)) else int(stride[-1])
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(feat_ch, hidden_dim, 1) if feat_ch != hidden_dim else nn.Identity()
        self.mask_dim = mask_dim
        self.num_prompt_classes = num_prompt_classes

        self.point_embed = nn.Embedding(3, hidden_dim)
        self.mask_token = nn.Embedding(1, hidden_dim)
        self.iou_token = nn.Embedding(1, hidden_dim)
        self.prompt_fuse = nn.Linear(hidden_dim + num_prompt_classes, hidden_dim)

        self.blocks = nn.ModuleList(TwoWayBlock2D(hidden_dim, nheads) for _ in range(layers))
        self.final_cross = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, mask_dim)
        )
        self.iou_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pixel_proj = nn.Conv2d(hidden_dim, mask_dim, 1)
        self.mask_up = nn.Upsample(scale_factor=self.stride, mode="bilinear", align_corners=False)

    def _encode_boxes(
        self,
        boxes: torch.Tensor,
        cls_probs: torch.Tensor | None,
        feat_shape,
        device,
    ) -> torch.Tensor:
        """boxes: [N, 4] (y1,x1,y2,x2); returns [N, 3, C] prompt tokens."""
        n = boxes.shape[0]
        h, w = feat_shape
        s = self.stride
        scale = torch.tensor([s, s, s, s], device=device, dtype=boxes.dtype)
        bnorm = boxes / scale
        corners = torch.stack([bnorm[:, :2], bnorm[:, 2:]], dim=1)  # [N,2,2]
        center = 0.5 * (bnorm[:, :2] + bnorm[:, 2:])  # [N,2]

        def pe(coords):  # [...,2] -> [...,C]
            freqs = torch.arange(0, self.hidden_dim // 4, device=device, dtype=boxes.dtype)
            ang = coords[..., None] * (2.0 ** freqs) * torch.pi
            emb = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
            if emb.shape[-1] < self.hidden_dim:
                emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
            return emb[..., : self.hidden_dim]

        corner_pe = pe(corners)
        center_pe = pe(center)
        types = self.point_embed.weight
        geo = torch.cat(
            [
                corner_pe[:, 0:1] + types[0][None, None, :],
                corner_pe[:, 1:2] + types[1][None, None, :],
                center_pe[:, None, :] + types[2][None, None, :],
            ],
            dim=1,
        )
        if cls_probs is None:
            cls_probs = torch.zeros(n, self.num_prompt_classes, device=device, dtype=geo.dtype)
        cls = cls_probs.to(geo.dtype)[:, None, :].expand(-1, geo.shape[1], -1)
        return self.prompt_fuse(torch.cat([geo, cls], dim=-1))

    def forward(
        self,
        feat: torch.Tensor,
        boxes: list[torch.Tensor],
        cls_probs: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """feat: [B,C,H,W]; boxes: list of [N_i,4]. Each box decoded alone."""
        b, c, h, w = feat.shape
        x = self.up(self.input_proj(feat))
        full_size = (h * self.stride, w * self.stride)
        img_tokens = x.flatten(2).transpose(1, 2)
        pix = self.mask_up(self.pixel_proj(x))

        all_masks, all_ious = [], []
        for i in range(b):
            bx = boxes[i]
            cp = cls_probs[i] if cls_probs is not None else None
            if bx.numel() == 0:
                all_masks.append(torch.zeros((0, *full_size), device=feat.device))
                all_ious.append(torch.zeros((0,), device=feat.device))
                continue
            n = bx.shape[0]
            img_tok_i = img_tokens[i : i + 1]
            pix_i = pix[i : i + 1].flatten(2).transpose(1, 2).squeeze(0)
            masks_i, ious_i = [], []
            for j in range(n):
                prompts = self._encode_boxes(
                    bx[j : j + 1],
                    cp[j : j + 1] if cp is not None else None,
                    (h, w),
                    feat.device,
                )
                queries = torch.cat(
                    [
                        self.mask_token.weight.expand(1, 1, -1),
                        self.iou_token.weight.expand(1, 1, -1),
                        prompts,
                    ],
                    dim=1,
                )
                keys = img_tok_i
                for blk in self.blocks:
                    queries, keys = blk(queries, keys)
                out = self.norm_out(queries + self.final_cross(queries, keys, keys)[0])
                mask_emb = self.mask_mlp(out[:, 0])
                masks_i.append((mask_emb @ pix_i.T).view(1, *full_size))
                ious_i.append(self.iou_mlp(out[:, 1]).squeeze(-1))
            all_masks.append(torch.cat(masks_i, dim=0))
            all_ious.append(torch.cat(ious_i, dim=0))
        return {"pred_masks": all_masks, "pred_iou": all_ious, "mask_size": (h, w)}
