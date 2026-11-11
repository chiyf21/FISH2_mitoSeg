"""3D YOLO-style CSP backbone with a configurable number of stages (default 6).

Input:  (B, 1, D, H, W)
Stages downsample by 2 each time after the stem, matching Ultralytics YOLO's
P1..P6 pyramid. Detection heads consume the last three maps (P3/P4/P5 when
num_layers=6 with stem included as stage 0, or the last 3 of `num_layers`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _autopad(k: int, p=None) -> int:
    return k // 2 if p is None else p


class Conv3dBN(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p=None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck3D(nn.Module):
    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        hid = c
        self.cv1 = Conv3dBN(c, hid, 3, 1)
        self.cv2 = Conv3dBN(hid, c, 3, 1)
        self.use_add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.use_add else y


class C2f3D(nn.Module):
    """3D analogue of Ultralytics C2f: split, n bottlenecks, concat all intermediates."""

    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv3dBN(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv3dBN((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck3D(self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class YOLO3DBackbone(nn.Module):
    """Six-stage 3D YOLO backbone, anisotropic in z.

    FIB-SEM crops are 6 z-layers but wide in xy, so z is downsampled less
    aggressively than xy: the stem and early stages use stride (1,2,2), and z
    is only downsampled in the deeper stages. This keeps a non-degenerate
    z dimension through the stride-8/16/32 pyramid that the neck consumes.

    Channel schedule (base=32): 32, 64, 128, 256, 384, 512
    xy strides:                   2,  4,   8,  16,  32,  64
    z  strides:                   1,  1,   1,   2,   2,   2
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_layers: int = 6,
        depth: int = 2,
        num_out: int = 3,
    ):
        super().__init__()
        if num_layers != 6:
            raise ValueError("This implementation is the agreed 6-layer 3D YOLO backbone.")
        chs = [
            base_ch,
            base_ch * 2,
            base_ch * 4,
            base_ch * 8,
            base_ch * 12,
            base_ch * 16,
        ]
        # per-stage (z, xy) stride; z downsampled only in the last two stages
        z_stride = [1, 1, 1, 2, 2]
        # layer 0: stem, stride (1,2,2)
        self.stem = Conv3dBN(in_ch, chs[0], 3, (1, 2, 2))
        self.stages = nn.ModuleList()
        # layer 1: no extra downsample (still stride xy2 / z1)
        self.stages.append(C2f3D(chs[0], chs[0], n=depth))
        # layers 2-6: downsample (z_stride, 2, 2) then C2f
        for i in range(1, num_layers - 1):
            zs = z_stride[i - 1]
            block = nn.Sequential(
                Conv3dBN(chs[i - 1], chs[i], 3, (zs, 2, 2)),
                C2f3D(chs[i], chs[i], n=depth),
            )
            self.stages.append(block)
        self.num_layers = num_layers
        self.feat_channels = [chs[0], chs[0], chs[1], chs[2], chs[3], chs[4]]
        self.out_indices = tuple(range(num_layers - num_out, num_layers))
        self.out_channels = tuple(self.feat_channels[i] for i in self.out_indices)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.stem(x)
        feats.append(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return [feats[i] for i in self.out_indices]


class SPPF3D(nn.Module):
    """3D SPPF: stacked max-pool pyramid on the deepest feature map."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hid = c1 // 2
        self.cv1 = Conv3dBN(c1, hid, 1, 1)
        self.cv2 = Conv3dBN(hid * 4, c2, 1, 1)
        self.pool = nn.MaxPool3d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, dim=1))


class YOLO3DNeck(nn.Module):
    """FPN + PAN over the last three backbone maps, YOLO-style.

    Anisotropic: z is upsampled/downsampled by 1 (kept), xy by 2, matching the
    backbone's anisotropic strides.
    """

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 128, depth: int = 1):
        super().__init__()
        self.lateral = nn.ModuleList(Conv3dBN(c, out_channels, 1, 1) for c in in_channels)
        self.sppf = SPPF3D(in_channels[-1], out_channels)
        # P5->P4 halves z (z stride 2), P4->P3 keeps z (z stride 1); xy always x2
        self.up54 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
        self.up43 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.down34 = Conv3dBN(out_channels, out_channels, 3, (1, 2, 2))
        self.down45 = Conv3dBN(out_channels, out_channels, 3, (2, 2, 2))
        self.fpn = nn.ModuleList(C2f3D(out_channels * 2, out_channels, n=depth) for _ in range(2))
        self.pan = nn.ModuleList(C2f3D(out_channels * 2, out_channels, n=depth) for _ in range(2))
        self.out_channels = (out_channels,) * 3
        # (z, y, x) strides of the three output maps
        self.out_strides = ((1, 8, 8), (1, 16, 16), (2, 32, 32))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = [self.lateral[i](f) for i, f in enumerate(feats)]
        p5 = self.sppf(feats[-1]) + p5
        # top-down
        p4 = self.fpn[0](torch.cat([self.up54(p5), p4], dim=1))
        p3 = self.fpn[1](torch.cat([self.up43(p4), p3], dim=1))
        # bottom-up
        p4 = self.pan[0](torch.cat([self.down34(p3), p4], dim=1))
        p5 = self.pan[1](torch.cat([self.down45(p4), p5], dim=1))
        return [p3, p4, p5]
