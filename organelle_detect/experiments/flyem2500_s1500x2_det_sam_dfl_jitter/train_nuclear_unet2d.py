"""Train a 2D UNet for nucleus segmentation on nuclear2d_patches.

Data : outputs/nuclear2d_patches/{images,masks}.npy + split.json (0.8/0.2)
Loss : BCEWithLogits (binary CE) + soft Dice
Out  : outputs/train_nuclear_unet2d/{best.pt,last.pt,history.json}
       outputs/logs/nuclear_unet2d/{curve.png,pred_viz.png}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("/path/to/JaneliaEM/outputs/nuclear2d_patches")
OUT_DIR = Path("/path/to/JaneliaEM/outputs/train_nuclear_unet2d")
VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/nuclear_unet2d")


# ---------------- model ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

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
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], dim=1))
        return self.head(x)


# ---------------- data ----------------
class PatchDataset(Dataset):
    def __init__(self, indices):
        self.img = np.load(DATA_DIR / "images.npy", mmap_mode="r")
        self.msk = np.load(DATA_DIR / "masks.npy", mmap_mode="r")
        self.idx = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = torch.from_numpy(np.ascontiguousarray(self.img[j])).float().div_(255.0).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(self.msk[j])).float().unsqueeze(0)
        return x, y


# ---------------- losses / metrics ----------------
def dice_loss_from_logits(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def dice_iou(logits, target, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum()
    union = p.sum() + target.sum() - inter
    dice = (2 * inter + 1e-6) / (p.sum() + target.sum() + 1e-6)
    return float(dice), float((inter + 1e-6) / (union + 1e-6))


def run_epoch(model, loader, opt, scaler, device, train: bool):
    model.train() if train else model.eval()
    tot_loss = tot_bce = tot_dice = tot_dsc = tot_iou = n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                bce = F.binary_cross_entropy_with_logits(logits, y)
                dl = dice_loss_from_logits(logits.float(), y)
                loss = bce + dl
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            dsc, iou = dice_iou(logits.float(), y)
            bs = x.shape[0]
            tot_loss += float(loss) * bs
            tot_bce += float(bce) * bs
            tot_dice += float(dl) * bs
            tot_dsc += dsc * bs
            tot_iou += iou * bs
            n += bs
    return dict(loss=tot_loss / n, bce=tot_bce / n, dice_loss=tot_dice / n,
                dice=tot_dsc / n, iou=tot_iou / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA_DIR / "split.json").read_text())
    tr_ds, te_ds = PatchDataset(split["train"]), PatchDataset(split["test"])
    print(f"[data] train={len(tr_ds)} test={len(te_ds)}", flush=True)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=True)

    device = torch.device("cuda")
    model = UNet2D(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet2D base={args.base} params={n_params/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_dice = -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tr_ld, opt, scaler, device, train=True)
        te = run_epoch(model, te_ld, opt, scaler, device, train=False)
        sched.step()
        history.append(dict(epoch=ep, **{f"tr_{k}": v for k, v in tr.items()},
                            **{f"te_{k}": v for k, v in te.items()}))
        flag = ""
        if te["dice"] > best_dice:
            best_dice = te["dice"]
            torch.save(dict(model=model.state_dict(), epoch=ep, dice=best_dice,
                            args=vars(args)), OUT_DIR / "best.pt")
            flag = " *best*"
        torch.save(dict(model=model.state_dict(), epoch=ep, dice=te["dice"],
                        args=vars(args)), OUT_DIR / "last.pt")
        print(f"[ep {ep:03d}] tr loss={tr['loss']:.4f} (bce {tr['bce']:.4f} dice {tr['dice_loss']:.4f}) "
              f"dice={tr['dice']:.4f} | te loss={te['loss']:.4f} dice={te['dice']:.4f} "
              f"iou={te['iou']:.4f} | {time.time()-t0:.0f}s {flag}", flush=True)
        (OUT_DIR / "history.json").write_text(json.dumps(history))

    print(f"[done] best test dice={best_dice:.4f} in {time.time()-t0:.0f}s", flush=True)

    # -------- curves --------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = history
    eps = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(eps, [r["tr_loss"] for r in h], label="train")
    axes[0].plot(eps, [r["te_loss"] for r in h], label="test")
    axes[0].set_title("CE + Dice loss"); axes[0].legend(); axes[0].set_xlabel("epoch")
    axes[1].plot(eps, [r["tr_dice"] for r in h], label="train")
    axes[1].plot(eps, [r["te_dice"] for r in h], label="test")
    axes[1].set_title("Dice"); axes[1].legend(); axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "curve.png", dpi=120)

    # -------- prediction viz on test patches --------
    ckpt = torch.load(OUT_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = np.random.default_rng(0)
    show_idx = rng.choice(len(te_ds), size=min(8, len(te_ds)), replace=False)
    fig, axes = plt.subplots(len(show_idx), 4, figsize=(13, 3.1 * len(show_idx)))
    for r, i in enumerate(show_idx):
        x, y = te_ds[int(i)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x.unsqueeze(0).to(device))
        p = torch.sigmoid(logit.float()).cpu().numpy()[0, 0]
        im, gt = x.numpy()[0], y.numpy()[0]
        dsc, iou = dice_iou(logit.float().cpu(), y.unsqueeze(0))
        axes[r, 0].imshow(im, cmap="gray"); axes[r, 0].set_title("EM")
        axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("GT")
        axes[r, 2].imshow(p, cmap="gray", vmin=0, vmax=1); axes[r, 2].set_title("pred prob")
        axes[r, 3].imshow(im, cmap="gray")
        axes[r, 3].imshow(np.ma.masked_where(p < 0.5, p > 0.5), cmap="autumn", alpha=0.45)
        axes[r, 3].set_title(f"overlay dice={dsc:.3f}")
        for c in range(4):
            axes[r, c].axis("off")
    fig.suptitle(f"nuclear UNet2D test preds (best ep {ckpt['epoch']}, dice {ckpt['dice']:.4f})")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "pred_viz.png", dpi=110)
    print(f"[viz] {VIZ_DIR/'curve.png'} {VIZ_DIR/'pred_viz.png'}", flush=True)


if __name__ == "__main__":
    main()
