#!/usr/bin/env python3
"""Train the unified 3D seg+det model (YOLO3D backbone+neck, three heads).

Supervision:
  seg   : Mask2Former3D over merged classes {er,mito,lyso,ld} (Hungarian, no-object bg)
  det   : YOLO3D boxes+class on instance organelles {mito,lyso,ld}
  prompt: SAM2-style two-way transformer masks from GT boxes (train) / pred boxes (eval)
  er    : ER-only Mask2Former3D (mask supervision only, no boxes)
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

from data.dataset_segdet import (  # noqa: E402
    NUM_CLASSES,
    SegDetCropDataset,
    segdet_collate,
)
from data.datasets import build_instance_index  # noqa: E402
from models.losses import yolo_loss  # noqa: E402
from models.losses_segdet import mask2former_loss_3d, prompt_mask_loss  # noqa: E402
from models.model_segdet3d import OrganelleSegDet3D  # noqa: E402


def load_crop_size(default: int = 128) -> int:
    p = ROOT / "outputs" / "viz" / "chosen_crop.json"
    if p.exists():
        return int(json.loads(p.read_text())["chosen_crop"])
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    crop = args.crop_size or load_crop_size()
    out = args.out or (ROOT / "outputs" / "segdet3d")
    out.mkdir(parents=True, exist_ok=True)

    records = build_instance_index()
    ds = SegDetCropDataset(records, crop_size=crop, samples_per_epoch=args.steps * args.batch_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=segdet_collate
    )
    model = OrganelleSegDet3D().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log_path = out / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "seg": 0.0, "det": 0.0, "prompt": 0.0, "er": 0.0}
        n = 0
        for images, masks, mask_labels, er_masks, boxes, box_labels, inst_masks in tqdm(
            loader, desc=f"segdet e{epoch}"
        ):
            images = images.to(args.device, non_blocking=True)
            masks = [m.to(args.device) for m in masks]
            mask_labels = [l.to(args.device) for l in mask_labels]
            er_masks = [m.to(args.device) for m in er_masks]
            boxes = [b.to(args.device) for b in boxes]
            box_labels = [l.to(args.device) for l in box_labels]
            inst_masks = [m.to(args.device) for m in inst_masks]

            opt.zero_grad(set_to_none=True)
            out_m = model(images, gt_boxes=boxes)

            # direct segmentation over all merged classes
            seg = mask2former_loss_3d(
                out_m["seg"]["pred_logits"], out_m["seg"]["pred_masks"], mask_labels, masks
            )
            # detection on instance organelles only (remap to 1..3 for the det head)
            det_labels = [l - 1 for l in box_labels]  # mito2->1, lyso3->2, ld4->3
            det = yolo_loss(out_m["det"], boxes, det_labels, crop)
            # prompt masks supervised by GT instance masks
            prompt = prompt_mask_loss(out_m["prompt"]["pred_masks"], inst_masks)
            # ER-only head
            er_labels = [torch.zeros(m.shape[0], dtype=torch.long, device=m.device) for m in er_masks]
            er = mask2former_loss_3d(
                out_m["er"]["pred_logits"], out_m["er"]["pred_masks"], er_labels, er_masks, num_classes=1
            )

            loss = seg["loss"] + det["loss"] + prompt["loss"] + er["loss"]
            if not torch.isfinite(loss):
                print("skip non-finite loss", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running["loss"] += float(loss.item())
            running["seg"] += float(seg["loss"].item())
            running["det"] += float(det["loss"].item())
            running["prompt"] += float(prompt["loss"].item())
            running["er"] += float(er["loss"].item())
            n += 1
        mean = {k: v / max(n, 1) for k, v in running.items()}
        print(
            f"epoch {epoch} loss={mean['loss']:.4f} seg={mean['seg']:.4f} det={mean['det']:.4f} "
            f"prompt={mean['prompt']:.4f} er={mean['er']:.4f} crop={crop}",
            flush=True,
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "crop": crop, **mean}) + "\n")
        ckpt = {"epoch": epoch, "model": model.state_dict(), "crop_size": crop, "loss": mean["loss"]}
        torch.save(ckpt, out / "last.pt")
        if epoch % 5 == 0:
            torch.save(ckpt, out / f"e{epoch}.pt")


if __name__ == "__main__":
    main()
