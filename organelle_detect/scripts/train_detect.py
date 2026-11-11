#!/usr/bin/env python3
"""Train 3D organelle detectors sharing a 6-layer YOLO backbone.

Heads (run three times, once per --head):
  yolo     : k-class softmax head, k = n_organelles + 1 background
  detr     : DETR decoder, last class is no-object
  experts  : m parallel 2-class YOLO heads, one organelle each
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

from data.datasets import (  # noqa: E402
    NUM_ORGANELLES,
    DetectionCropDataset,
    build_instance_index,
    detection_collate,
)
from models.detector import OrganelleDetector  # noqa: E402
from models.losses import expert_loss, hungarian_detr, yolo_loss  # noqa: E402


def load_crop_size(default: int = 128) -> int:
    p = ROOT / "outputs" / "viz" / "chosen_crop.json"
    if p.exists():
        return int(json.loads(p.read_text())["chosen_crop"])
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=["yolo", "detr", "experts"], required=True)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    crop = args.crop_size or load_crop_size()
    out = args.out or (ROOT / "outputs" / "detect" / args.head)
    out.mkdir(parents=True, exist_ok=True)

    records = build_instance_index()
    ds = DetectionCropDataset(records, crop_size=crop, samples_per_epoch=args.steps * args.batch_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate,
    )
    model = OrganelleDetector(head=args.head, num_organelles=NUM_ORGANELLES).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log_path = out / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, boxes, labels in tqdm(loader, desc=f"{args.head} e{epoch}"):
            images = images.to(args.device, non_blocking=True)
            boxes = [b.to(args.device) for b in boxes]
            labels = [l.to(args.device) for l in labels]
            opt.zero_grad(set_to_none=True)
            out_m = model(images)
            if args.head == "yolo":
                losses = yolo_loss(out_m, boxes, labels, crop)
            elif args.head == "detr":
                detr_labels = [lab - 1 for lab in labels]
                losses = hungarian_detr(out_m["logits"], out_m["boxes_norm"], boxes, detr_labels, crop)
            else:
                losses = expert_loss(out_m, boxes, labels, crop, NUM_ORGANELLES)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                print("skip non-finite loss", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n += 1
        mean = running / max(n, 1)
        print(f"epoch {epoch} head={args.head} loss={mean:.4f} crop={crop} k={NUM_ORGANELLES}+1", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "loss": mean, "head": args.head, "crop": crop}) + "\n")
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "head": args.head,
            "crop_size": crop,
            "num_organelles": NUM_ORGANELLES,
            "loss": mean,
        }
        torch.save(ckpt, out / "last.pt")
        if epoch % 5 == 0:
            torch.save(ckpt, out / f"e{epoch}.pt")


if __name__ == "__main__":
    main()
