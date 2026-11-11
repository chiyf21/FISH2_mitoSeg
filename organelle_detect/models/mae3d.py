"""3D Masked Autoencoder (He et al., CVPR 2022) adapted to volumetric EM.

Official recipe we keep:
  * random shuffle masking (default 75%)
  * encoder sees visible patches only
  * mask tokens inserted in the decoder
  * MSE reconstruction on masked patches, with per-patch pixel normalization
Reference: https://github.com/facebookresearch/mae (models_mae.py)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3-axis sincos"
    omega = np.arange(embed_dim // 6, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 6)))
    gz, gy, gx = [np.arange(grid_size, dtype=np.float32) for _ in range(3)]
    grid = np.stack(np.meshgrid(gz, gy, gx, indexing="ij"), axis=0).reshape(3, -1)
    out = []
    for axis in range(3):
        out.append(np.sin(grid[axis][:, None] * omega[None]))
        out.append(np.cos(grid[axis][:, None] * omega[None]))
    return np.concatenate(out, axis=1).astype(np.float32)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**3
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MAE3D(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=384,
        depth=8,
        num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        norm_pix_loss=True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans)
        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.initialize_weights()

    def initialize_weights(self):
        pos = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size)
        self.pos_embed.data[:, 1:].copy_(torch.from_numpy(pos).float().unsqueeze(0))
        dpos = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size)
        self.decoder_pos_embed.data[:, 1:].copy_(torch.from_numpy(dpos).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_size
        b, c, d, h, w = imgs.shape
        gd, gh, gw = d // p, h // p, w // p
        x = imgs.reshape(b, c, gd, p, gh, p, gw, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        return x.reshape(b, gd * gh * gw, p**3 * c)

    def random_masking(self, x, mask_ratio):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))
        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, d))
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
