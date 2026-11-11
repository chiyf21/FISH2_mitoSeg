"""2D YOLO-style CSP backbone + FPN/PAN neck (mirror of backbone_yolo3d).

Input: (B, 1, H, W). Strides P3/P4/P5 = 8/16/32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _autopad(k: int, p=None) -> int:
    return k // 2 if p is None else p


class Conv2dBN(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p=None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck2D(nn.Module):
    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = Conv2dBN(c, c, 3, 1)
        self.cv2 = Conv2dBN(c, c, 3, 1)
        self.use_add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.use_add else y


class C2f2D(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv2dBN(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv2dBN((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck2D(self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class YOLO2DBackbone(nn.Module):
    """Six-stage 2D YOLO backbone. Channel / stride schedule mirrors 3D xy path.

    Channel schedule (base=32): 32, 64, 128, 256, 384, 512
    strides:                    2,  4,   8,  16,  32,  64
    Outputs the last 3 maps (P3/P4/P5 at strides 8/16/32).
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
            raise ValueError("2D backbone mirrors the agreed 6-layer YOLO schedule.")
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 12, base_ch * 16]
        self.stem = Conv2dBN(in_ch, chs[0], 3, 2)
        self.stages = nn.ModuleList()
        self.stages.append(C2f2D(chs[0], chs[0], n=depth))
        for i in range(1, num_layers - 1):
            self.stages.append(
                nn.Sequential(Conv2dBN(chs[i - 1], chs[i], 3, 2), C2f2D(chs[i], chs[i], n=depth))
            )
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


class SPPF2D(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hid = c1 // 2
        self.cv1 = Conv2dBN(c1, hid, 1, 1)
        self.cv2 = Conv2dBN(hid * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, dim=1))


class YOLO2DNeck(nn.Module):
    """FPN + PAN over the last three backbone maps."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 128, depth: int = 1):
        super().__init__()
        self.lateral = nn.ModuleList(Conv2dBN(c, out_channels, 1, 1) for c in in_channels)
        self.sppf = SPPF2D(in_channels[-1], out_channels)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.down = nn.ModuleList(Conv2dBN(out_channels, out_channels, 3, 2) for _ in range(2))
        self.fpn = nn.ModuleList(C2f2D(out_channels * 2, out_channels, n=depth) for _ in range(2))
        self.pan = nn.ModuleList(C2f2D(out_channels * 2, out_channels, n=depth) for _ in range(2))
        self.out_channels = (out_channels,) * 3
        self.out_strides = (8, 16, 32)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = [self.lateral[i](f) for i, f in enumerate(feats)]
        p5 = self.sppf(feats[-1]) + p5
        p4 = self.fpn[0](torch.cat([self.up(p5), p4], dim=1))
        p3 = self.fpn[1](torch.cat([self.up(p4), p3], dim=1))
        p4 = self.pan[0](torch.cat([self.down[0](p3), p4], dim=1))
        p5 = self.pan[1](torch.cat([self.down[1](p4), p5], dim=1))
        return [p3, p4, p5]
