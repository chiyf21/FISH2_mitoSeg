#!/usr/bin/env python3
"""Pretrain a 3D MAE backbone on CellMap EM crops.

Follows He et al. 2022 (https://github.com/facebookresearch/mae):
  mask_ratio=0.75, pixel-normalized MSE on masked tokens only.
Window size defaults to the crop chosen by visualize_instances.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import MAECropDataset, build_instance_index  # noqa: E402
from models.mae3d import MAE3D  # noqa: E402


def load_crop_size(default: int = 128) -> int:
    p = ROOT / "outputs" / "viz" / "chosen_crop.json"
    if p.exists():
        return int(json.loads(p.read_text())["chosen_crop"])
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=0, help="0 = use viz-chosen size")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "mae")
    args = parser.parse_args()
    crop = args.crop_size or load_crop_size()
    if crop % args.patch_size:
        raise ValueError(f"crop {crop} must be divisible by patch {args.patch_size}")
    args.out.mkdir(parents=True, exist_ok=True)

    records = build_instance_index()
    ds = MAECropDataset(records, crop_size=crop, samples_per_epoch=args.steps * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    model = MAE3D(img_size=crop, patch_size=args.patch_size).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for imgs in tqdm(loader, desc=f"mae e{epoch}"):
            imgs = imgs.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n += 1
        mean = running / max(n, 1)
        print(f"epoch {epoch} mae_loss={mean:.4f} crop={crop}", flush=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "crop_size": crop,
            "patch_size": args.patch_size,
            "loss": mean,
        }
        torch.save(ckpt, args.out / "mae3d_last.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, args.out / f"mae3d_e{epoch}.pt")


if __name__ == "__main__":
    main()
