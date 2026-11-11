"""SAM2-style 3D prompt mask decoder.

Detected 3D boxes are used as box prompts. Each box is encoded as two corner
tokens (top-left-front / bottom-right-back) plus a foreground token, then a
two-way transformer (prompt <-> image tokens, like SAM's mask decoder) refines
them and produces a binary mask per box.

Two user-mandated behaviors:
  1. Each detected box is decoded against the image individually (no multi-box
     prompt batching through the transformer).
  2. The detected box's class (softmax -> one-hot) is concatenated with the
     box's geometric encoding before entering the prompt embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoWayBlock3D(nn.Module):
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


class PromptMask3D(nn.Module):
    """Box-prompt mask decoder over the finest neck feature map (stride 8).

    num_prompt_classes: number of foreground classes a box can belong to
    (mito/lyso/ld = 3). The box class is supplied as softmax probabilities and
    fused with the geometric encoding before the prompt embedding.
    """

    PROMPT_CHUNK = 32

    def __init__(
        self,
        feat_ch: int,
        mask_dim: int = 128,
        hidden_dim: int = 256,
        nheads: int = 8,
        layers: int = 2,
        stride=(1, 8, 8),
        num_prompt_classes: int = 3,
        pe: str = "legacy",
        pe_scale: float = 1.0,
    ):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if pe not in ("legacy", "rff"):
            raise ValueError(f"unknown prompt pe {pe}")
        self.stride = stride  # (z, y, x)
        self.hidden_dim = hidden_dim
        self.pe_mode = pe
        if pe == "rff":
            # SAM-style random Fourier features on box coords normalized to [-1, 1] of the crop
            self.register_buffer("pe_gaussian", pe_scale * torch.randn(3, hidden_dim // 2))
        self.input_proj = nn.Conv3d(feat_ch, hidden_dim, 1) if feat_ch != hidden_dim else nn.Identity()
        self.mask_dim = mask_dim
        self.num_prompt_classes = num_prompt_classes

        # prompt encodings: 2 corners + 1 foreground point
        self.point_embed = nn.Embedding(3, hidden_dim)
        self.not_a_box_embed = nn.Embedding(1, hidden_dim)
        self.mask_token = nn.Embedding(1, hidden_dim)
        self.iou_token = nn.Embedding(1, hidden_dim)
        # fuse geometric encoding + class one-hot into the prompt embedding
        self.prompt_fuse = nn.Linear(hidden_dim + num_prompt_classes, hidden_dim)

        self.blocks = nn.ModuleList(TwoWayBlock3D(hidden_dim, nheads) for _ in range(layers))
        self.final_cross = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, mask_dim)
        )
        self.iou_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.up = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pixel_proj = nn.Conv3d(hidden_dim, mask_dim, 1)
        self.mask_up = nn.Upsample(scale_factor=self.stride, mode="trilinear", align_corners=False)

    def _encode_boxes(
        self,
        boxes: torch.Tensor,
        cls_probs: torch.Tensor | None,
        feat_shape,
        device,
    ) -> torch.Tensor:
        """boxes: [N, 6] in crop voxels; cls_probs: [N, num_prompt_classes] softmax.

        Returns corner+fg tokens [N, 3, C] with class fused into every token.
        """
        n = boxes.shape[0]
        d, h, w = feat_shape
        sz, sy, sx = self.stride
        # normalize crop-voxel boxes to feature-map coords (per-axis stride)
        if self.pe_mode == "rff":
            full = torch.tensor([d * sz, h * sy, w * sx] * 2, device=device, dtype=torch.float32)
            bnorm = 2.0 * boxes.float() / full - 1.0

            def pe(coords):  # [...,3] in [-1,1] -> [...,C]
                ang = 2 * torch.pi * coords @ self.pe_gaussian.float()
                return torch.cat([ang.sin(), ang.cos()], dim=-1)
        else:
            scale = torch.tensor([sz, sy, sx, sz, sy, sx], device=device, dtype=boxes.dtype)
            # pe() uses frequencies up to 2^(hidden_dim/6): the trained decoder only saw integer-voxel
            # GT boxes, and fractional (predicted) coordinates give unseen high-frequency codes -> broken masks
            bnorm = boxes.round() / scale

            def pe(coords):  # [...,3] -> [...,C]
                freqs = torch.arange(0, self.hidden_dim // 6, device=device, dtype=boxes.dtype)
                ang = coords[..., None] * (2.0 ** freqs) * torch.pi
                emb = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
                if emb.shape[-1] < self.hidden_dim:
                    emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
                return emb[..., : self.hidden_dim]
        corners = torch.stack([bnorm[:, :3], bnorm[:, 3:]], dim=1)  # [N,2,3]
        center = 0.5 * (bnorm[:, :3] + bnorm[:, 3:])  # [N,3]

        corner_pe = pe(corners)  # [N,2,C]
        center_pe = pe(center)  # [N,C]
        types = self.point_embed.weight  # [3,C]
        geo = torch.cat(
            [
                corner_pe[:, 0:1] + types[0][None, None, :],
                corner_pe[:, 1:2] + types[1][None, None, :],
                center_pe[:, None, :] + types[2][None, None, :],
            ],
            dim=1,
        )  # [N,3,C]
        # fuse the box class (softmax one-hot) into each prompt token
        if cls_probs is None:
            cls_probs = torch.zeros(n, self.num_prompt_classes, device=device, dtype=geo.dtype)
        cls = cls_probs.to(geo.dtype)[:, None, :].expand(-1, geo.shape[1], -1)  # [N,3,K]
        tokens = self.prompt_fuse(torch.cat([geo, cls], dim=-1))
        return tokens  # [N,3,C]

    def forward(
        self,
        feat: torch.Tensor,
        boxes: list[torch.Tensor],
        cls_probs: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """feat: [B,C,D,H,W] (stride-8 map); boxes: list of [N_i,6] per batch.

        cls_probs: optional list of [N_i, K] softmax class probs per box.
        Each box is decoded individually (one prompt at a time).
        """
        b, c, d, h, w = feat.shape
        x = self.input_proj(feat)
        x = self.up(x)
        sz, sy, sx = self.stride
        full_size = (d * sz, h * sy, w * sx)
        img_tokens = x.flatten(2).transpose(1, 2)  # [B, DHW, C]
        pix = self.mask_up(self.pixel_proj(x))  # [B, mask_dim, Df, Hf, Wf]

        all_masks, all_ious = [], []
        for i in range(b):
            bx = boxes[i]
            cp = cls_probs[i] if cls_probs is not None else None
            if bx.numel() == 0:
                all_masks.append(torch.zeros((0, *full_size), device=feat.device))
                all_ious.append(torch.zeros((0,), device=feat.device))
                continue
            n = bx.shape[0]
            img_tok_i = img_tokens[i : i + 1]  # [1, DHW, C]
            pix_i = pix[i : i + 1].flatten(2).transpose(1, 2).squeeze(0)  # [DHW_full, mask_dim]
            masks_i, ious_i = [], []
            # prompts are decoded independently (each has its own copy of the image tokens), batched in chunks
            for j in range(0, n, self.PROMPT_CHUNK):
                bj = bx[j : j + self.PROMPT_CHUNK]
                k = bj.shape[0]
                cj = cp[j : j + k] if cp is not None else None
                prompts = self._encode_boxes(bj, cj, (d, h, w), feat.device)  # [k,3,C]
                queries = torch.cat(
                    [
                        self.mask_token.weight.expand(k, 1, -1),
                        self.iou_token.weight.expand(k, 1, -1),
                        prompts,
                    ],
                    dim=1,
                )  # [k, 5, C]
                keys = img_tok_i.expand(k, -1, -1)
                for blk in self.blocks:
                    queries, keys = blk(queries, keys)
                out = self.norm_out(queries + self.final_cross(queries, keys, keys)[0])
                mask_emb = self.mask_mlp(out[:, 0])  # [k, mask_dim]
                masks_i.append((mask_emb @ pix_i.T).view(k, *full_size))
                ious_i.append(self.iou_mlp(out[:, 1]).squeeze(-1))
            all_masks.append(torch.cat(masks_i, dim=0))
            all_ious.append(torch.cat(ious_i, dim=0))
        return {"pred_masks": all_masks, "pred_iou": all_ious, "mask_size": (d, h, w)}
