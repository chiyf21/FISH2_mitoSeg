"""Exact network classes from original nuclear UNet training script."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)

class UNet2D(nn.Module):

    def __init__(self, base=32, in_ch=1, out_ch=1):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([ConvBlock(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2) for i in range(4, 0, -1)])
        self.dec = nn.ModuleList([ConvBlock(chs[i], chs[i - 1]) for i in range(4, 0, -1)])
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 4:
                skips.append(x)
                x = self.pool(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = u(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False)
            x = d(torch.cat([x, s], dim=1))
        return self.head(x)
