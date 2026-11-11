#!/usr/bin/env python3
"""Box-tightness eval of det_sam detection heads on multi-scale test crops.

Per scale x class: GT recall at IoU 0.5/0.7 among post-NMS top-20 boxes, mean best IoU
per GT, and precision/recall of boxes above a score threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viz_val as V  # noqa: E402
from data.dataset_npz import NPZCropDataset, npz_collate  # noqa: E402
from models.box_ops import box_iou_3d  # noqa: E402

NPZ = Path("/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_ms1500_3000x10")
SCALES = ("s1500", "s3000")
CLASSES = {2: "mito", 3: "lyso"}
HEAD_OF = {2: 0, 3: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-per-scale", type=int, default=200)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = NPZCropDataset(NPZ, split="test", dim="3d")
    model = V.load_model("det_sam", args.ckpt)
    rng = np.random.default_rng(0)
    res = {}
    for scale in SCALES:
        cand = [i for i in range(len(ds)) if f"/{scale}_" in ds.region_of[i]]
        pick = rng.choice(cand, size=min(args.n_per_scale, len(cand)), replace=False).tolist()
        dl = DataLoader(Subset(ds, pick), batch_size=8, num_workers=4, collate_fn=npz_collate)
        acc = {c: {"best_iou": [], "n_pred": 0, "n_pred_tp": 0, "n_gt_hit_thr": 0, "pred_sizes": [], "gt_sizes": []}
               for c in CLASSES}
        with torch.no_grad():
            for batch in dl:
                img = batch[0].to(V.DEV)
                out = model(img)
                gtb_all, gtl_all = batch[4], batch[5]
                for c, h in HEAD_OF.items():
                    boxes, probs = model._top_boxes_binary(out["det"][h], topk=20)
                    for b in range(img.shape[0]):
                        gt = gtb_all[b][gtl_all[b].long() == c].to(V.DEV).float()
                        pb, ps = boxes[b].float(), probs[b][:, 0].float()
                        a = acc[c]
                        conf = ps > args.score_thr
                        a["n_pred"] += int(conf.sum())
                        a["pred_sizes"] += (pb[conf][:, 3:] - pb[conf][:, :3]).cpu().tolist()
                        if gt.shape[0] == 0:
                            continue
                        a["gt_sizes"] += (gt[:, 3:] - gt[:, :3]).cpu().tolist()
                        iou = box_iou_3d(gt, pb)  # [G,K]
                        a["best_iou"] += (iou.max(1).values if iou.shape[1] else torch.zeros(gt.shape[0])).tolist()
                        if conf.any():
                            ic = iou[:, conf]
                            a["n_pred_tp"] += int((ic.max(0).values >= 0.5).sum())
                            a["n_gt_hit_thr"] += int((ic.max(1).values >= 0.5).sum())
        for c, name in CLASSES.items():
            a = acc[c]
            bi = np.asarray(a["best_iou"])
            ps = np.asarray(a["pred_sizes"]).reshape(-1, 3)
            gs = np.asarray(a["gt_sizes"]).reshape(-1, 3)
            res[f"{scale}/{name}"] = {
                "n_gt": int(bi.size),
                "top20_recall@0.5": float((bi >= 0.5).mean()) if bi.size else None,
                "top20_recall@0.7": float((bi >= 0.7).mean()) if bi.size else None,
                "mean_best_iou": float(bi.mean()) if bi.size else None,
                f"n_pred@{args.score_thr}": a["n_pred"],
                f"precision@{args.score_thr},iou0.5": a["n_pred_tp"] / max(a["n_pred"], 1),
                f"recall@{args.score_thr},iou0.5": a["n_gt_hit_thr"] / max(bi.size, 1),
                "median_pred_zyx": np.median(ps, 0).round(1).tolist() if len(ps) else None,
                "median_gt_zyx": np.median(gs, 0).round(1).tolist() if len(gs) else None,
            }
            print(scale, name, json.dumps(res[f"{scale}/{name}"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"ckpt": str(args.ckpt), **res}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
