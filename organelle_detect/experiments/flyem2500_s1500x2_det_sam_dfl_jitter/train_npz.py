#!/usr/bin/env python3
"""Train the organelle model on exported .npz crops.

Experiments:
  --mode m2f      YOLO backbone+neck + Mask2Former over all classes
  --mode m2f_bin  same, but per-class binary (fg/bg) heads + per-class query
                  groups; classes never compete for queries/softmax mass
  --mode det_sam  YOLO box -> per-box SAM2 prompt mask (mito/lyso/ld/golgi)
                  + per-class binary Mask2Former seg head + Mask2Former for ER

  --dim 3d        volumetric [B,1,6,H,W] (default)
  --dim 2d        each z-slice as independent sample [B,1,H,W]

Multi-GPU via DistributedDataParallel; gradient accumulation keeps the
effective batch at `--effective-batch` even when the per-step batch shrinks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.dataset_npz import (  # noqa: E402
    NPZCropDataset,
    RareAwareRatioBatchSampler,
    compute_class_presence,
    npz_collate,
)
from models.box_ops import slice_boxes_from_masks  # noqa: E402
from models.losses import yolo_loss  # noqa: E402
from losses_segdet_tversky import (  # noqa: E402
    mask2former_binary_loss_3d,
    mask2former_loss_3d,
    prompt_mask_loss,
)
from data.organelle_classes import INSTANCE_IDS, NUM_CLASSES  # noqa: E402
from models.model_segdet3d import OrganelleSegDet3D  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["m2f", "m2f_bin", "det_sam", "det_sam_mc"], required=True)
    p.add_argument("--dim", choices=["3d", "2d"], default="3d")
    p.add_argument("--det-box", choices=["slice2d", "3d"], default="slice2d",
                   help="slice2d: per-z-slice 2D ltrb boxes; 3d: one 6-side 3D box per anchor")
    p.add_argument(
        "--npz-root",
        type=Path,
        nargs="+",
        default=[ROOT.parent / "outputs" / "npz_crops_v6"],
        help="one or more NPZ roots (e.g. cellmap v6 + fibsem + heinrich)",
    )
    p.add_argument("--scales", nargs="+", default=None,
                   help="keep only crops with these meta scales (e.g. s1500); default all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16, help="per-step batch per GPU")
    p.add_argument("--effective-batch", type=int, default=16, help="target effective batch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--prompt-pe", choices=["legacy", "rff"], default="legacy",
                   help="box-prompt positional encoding (det_sam 3d): legacy 2^k Fourier, or SAM-style "
                        "random Fourier features on normalized coords")
    p.add_argument("--prompt-jitter", type=float, default=0.0,
                   help="std of GT prompt box side noise as a fraction of the side length (training only)")
    p.add_argument("--eos-coef", type=float, default=0.1,
                   help="weight for unmatched/background segmentation queries")
    p.add_argument("--bf16", action="store_true", help="use CUDA bfloat16 autocast for model forward")
    p.add_argument("--resume", type=Path, default=None, help="resume model weights and epoch from a checkpoint")
    p.add_argument("--mask-loss", choices=["tversky", "ce"], default="tversky")
    p.add_argument("--tversky-alpha", type=float, default=0.5)
    p.add_argument("--tversky-beta", type=float, default=0.5)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def compute_class_weights(ds, num_classes=NUM_CLASSES):
    """Inverse-sqrt frequency weights over foreground classes,
    normalised to mean 1 and clipped. Cached to <root>/class_weights.json
    (mask_labels are 1..C -> channels 0..C-1)."""
    cache_name = f"class_weights_c{num_classes}"
    if len(ds.roots) > 1:
        cache_name += "_combined"
    cache = ds.root / f"{cache_name}.json"
    if cache.exists():
        w = json.loads(cache.read_text())["weights"]
        return torch.tensor(w, dtype=torch.float32)
    counts = np.ones(num_classes, dtype=np.float64)  # laplace smoothing
    for f in ds.files:
        with np.load(f, allow_pickle=True) as z:
            for ml in z["mask_labels"]:
                ids, cnt = np.unique(np.asarray(ml), return_counts=True)
                for i, c in zip(ids, cnt):
                    if 1 <= i <= num_classes:
                        counts[i - 1] += c
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    w = np.clip(w, 0.5, 3.0)
    cache.write_text(json.dumps({"weights": w.tolist(), "counts": counts.tolist()}, indent=2))
    return torch.tensor(w, dtype=torch.float32)


_DET_SHARED = ("anchors", "anchor_strides")


def _slice_det(det: dict, keep_t: torch.Tensor, B: int) -> dict:
    return {k: (v[keep_t] if torch.is_tensor(v) and k not in _DET_SHARED and v.shape[0] == B else v)
            for k, v in det.items()}


def compute_losses(args, model, batch, device, cls_weight=None):
    images, masks, mask_labels, er_masks, boxes, box_labels, inst_masks, sup_masks, neg_masks = batch
    images = images.to(device, non_blocking=True)
    masks = [m.to(device) for m in masks]
    mask_labels = [l.to(device) for l in mask_labels]
    er_masks = [m.to(device) for m in er_masks]
    boxes = [b.to(device) for b in boxes]
    box_labels = [l.to(device) for l in box_labels]
    inst_masks = [m.to(device) for m in inst_masks]
    # sup_masks [B, C] bool (0-indexed fg classes): False = class not annotated
    # in that sample's source volume -> excluded from that class's loss entirely
    # (kept on CPU: only used for per-sample indexing decisions)
    sup_masks = sup_masks.cpu()
    neg_masks = neg_masks.cpu()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
        out_m = model(images, gt_boxes=boxes, gt_box_labels=box_labels)
    # seg head foreground channels are 0-indexed (0..3); mask_labels are 1..4 -> shift
    seg_labels = [l - 1 for l in mask_labels]
    if args.mode in ("m2f_bin", "det_sam", "det_sam_mc"):
        seg = mask2former_binary_loss_3d(
            out_m["seg"]["pred_logits"],
            out_m["seg"]["pred_masks"],
            seg_labels,
            masks,
            num_classes=out_m["seg"]["num_classes"],
            queries_per_class=out_m["seg"]["queries_per_class"],
            eos_coef=args.eos_coef,
            cls_weight=cls_weight,
            sup_mask=sup_masks,
            neg_mask=neg_masks,
            mask_loss=args.mask_loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
        )
    else:
        seg = mask2former_loss_3d(
            out_m["seg"]["pred_logits"], out_m["seg"]["pred_masks"], seg_labels, masks, cls_weight=cls_weight,
            mask_loss=args.mask_loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta
        )
    loss = seg["loss"]
    det_l = prompt_l = er_l = torch.zeros((), device=device)
    if args.mode == "det_sam":
        # per-class binary det + prompt (one head pair per mito/lyso/ld/golgi)
        B = images.shape[0]
        crop = images.shape[-1]
        det_terms, prompt_terms = [], []
        for i, cls_id in enumerate(getattr(model, "module", model).instance_ids):
            # supervise this class only on samples whose source volume annotates it
            cidx = cls_id - 1
            bidx = [b for b in range(B) if bool(sup_masks[b, cidx]) or bool(neg_masks[b, cidx])]
            if not bidx:
                continue
            bidx_pos = [b for b in bidx if bool(sup_masks[b, cidx])]
            keep_t = torch.as_tensor(bidx, dtype=torch.long, device=device)
            e_boxes, e_labels, e_masks = [], [], []
            slice2d = out_m["det"][i].get("box_mode") == "slice2d"
            for b in bidx:
                lab, mlab, msk = box_labels[b], mask_labels[b], masks[b]
                keep_m = mlab.long() == cls_id
                e_masks.append(msk[keep_m])
                if slice2d:
                    sb, _, _ = slice_boxes_from_masks(msk[keep_m], mlab[keep_m])
                    e_boxes.append(sb)
                    e_labels.append(torch.ones(sb.shape[0], dtype=torch.long, device=device))
                else:
                    keep_b = lab.long() == cls_id
                    e_boxes.append(boxes[b][keep_b])
                    e_labels.append(torch.ones(int(keep_b.sum()), dtype=torch.long, device=device))
            det_sub = _slice_det(out_m["det"][i], keep_t, B)
            det_terms.append(yolo_loss(det_sub, e_boxes, e_labels, crop)["loss"])
            if bidx_pos:
                pm_sub = [out_m["prompt"][i]["pred_masks"][b] for b in bidx_pos]
                e_masks_pos = [masks[b][(mask_labels[b].long() == cls_id)] for b in bidx_pos]
                prompt_terms.append(prompt_mask_loss(pm_sub, e_masks_pos, mask_loss=args.mask_loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta)["loss"])
        if det_terms:
            det_l = torch.stack(det_terms).mean()
        if prompt_terms:
            prompt_l = torch.stack(prompt_terms).mean()
        # ER supervised only where annotated (class id 1 -> channel 0)
        bidx_er = [b for b in range(B) if bool(sup_masks[b, 0])]
        if bidx_er:
            keep_er = torch.as_tensor(bidx_er, dtype=torch.long, device=device)
            er_labels = [torch.zeros(er_masks[b].shape[0], dtype=torch.long, device=device) for b in bidx_er]
            er = mask2former_loss_3d(
                out_m["er"]["pred_logits"][keep_er], out_m["er"]["pred_masks"][keep_er],
                er_labels, [er_masks[b] for b in bidx_er], num_classes=1,
            )
            er_l = er["loss"]
        loss = loss + det_l + prompt_l + er_l
    elif args.mode == "det_sam_mc":
        # single YOLO nc=3 (bg, mito, lyso) + one prompt (class dim 2)
        B = images.shape[0]
        crop = images.shape[-1]
        bidx = [
            b for b in range(B)
            if bool(sup_masks[b, 1]) or bool(sup_masks[b, 2])
            or bool(neg_masks[b, 1]) or bool(neg_masks[b, 2])
        ]
        if bidx:
            keep_t = torch.as_tensor(bidx, dtype=torch.long, device=device)
            e_boxes, e_labels = [], []
            bidx_pos = [b for b in bidx if bool(sup_masks[b, 1]) or bool(sup_masks[b, 2])]
            slice2d = out_m["det"].get("box_mode") == "slice2d"
            for b in bidx:
                if slice2d:
                    mlab = mask_labels[b].long()
                    keep = (mlab == 2) | (mlab == 3)
                    bb, lab, _ = slice_boxes_from_masks(masks[b][keep], mlab[keep])
                else:
                    lab = box_labels[b].long()
                    keep = (lab == 2) | (lab == 3)
                    bb, lab = boxes[b][keep], lab[keep]
                e_boxes.append(bb)
                e_labels.append(OrganelleSegDet3D._remap_box_labels_mc(lab))
            det_sub = _slice_det(out_m["det"], keep_t, B)
            det_l = yolo_loss(det_sub, e_boxes, e_labels, crop)["loss"]
            if bidx_pos:
                pm_sub = [out_m["prompt"]["pred_masks"][b] for b in bidx_pos]
                e_masks_pos = []
                for b in bidx_pos:
                    mlab = mask_labels[b].long()
                    msk = masks[b]
                    keep = (mlab == 2) | (mlab == 3)
                    e_masks_pos.append(msk[keep])
                prompt_l = prompt_mask_loss(pm_sub, e_masks_pos, mask_loss=args.mask_loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta)["loss"]
            loss = loss + det_l + prompt_l
        bidx_er = [b for b in range(B) if bool(sup_masks[b, 0])]
        if bidx_er:
            keep_er = torch.as_tensor(bidx_er, dtype=torch.long, device=device)
            er_labels = [torch.zeros(er_masks[b].shape[0], dtype=torch.long, device=device) for b in bidx_er]
            er = mask2former_loss_3d(
                out_m["er"]["pred_logits"][keep_er], out_m["er"]["pred_masks"][keep_er],
                er_labels, [er_masks[b] for b in bidx_er], num_classes=1,
            )
            er_l = er["loss"]
            loss = loss + er_l
    return loss, seg["loss"], det_l, prompt_l, er_l


@torch.no_grad()
def evaluate(args, model, test_loader, device, cls_weight=None, max_batches: int | None = None):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals = {"loss": 0.0, "seg": 0.0, "det": 0.0, "prompt": 0.0, "er": 0.0}
    n = 0
    for bi, batch in enumerate(test_loader):
        if max_batches is not None and bi >= max_batches:
            break
        loss, seg_l, det_l, prompt_l, er_l = compute_losses(args, raw, batch, device, cls_weight)
        if not torch.isfinite(loss):
            continue
        totals["loss"] += float(loss.item())
        totals["seg"] += float(seg_l.item())
        totals["det"] += float(det_l.item())
        totals["prompt"] += float(prompt_l.item())
        totals["er"] += float(er_l.item())
        n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    tag = f"{args.mode}" if args.dim == "3d" else f"{args.mode}_{args.dim}"
    out = args.out or (ROOT.parent / "outputs" / f"train_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    use_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world = 1

    npz_roots = args.npz_root if isinstance(args.npz_root, list) else [args.npz_root]
    ds = NPZCropDataset(npz_roots, split="train", dim=args.dim, scales=args.scales)
    test_ds = NPZCropDataset(npz_roots, split="test", dim=args.dim, scales=args.scales)

    # class frequency stats + rare-class floor sampling (freq < 1/10 → ≥1/10 of each batch)
    if use_dist:
        presence_box = [compute_class_presence(ds, num_classes=NUM_CLASSES) if rank == 0 else None]
        dist.broadcast_object_list(presence_box, src=0)
        presence = presence_box[0]
    else:
        presence = compute_class_presence(ds, num_classes=NUM_CLASSES)
    rare = [
        presence["class_names"][c]
        for c, f in enumerate(presence["frequency"])
        if f < 0.1 and len(presence["has_class"].get(str(c), presence["has_class"].get(c, []))) > 0
    ]
    batch_sampler = RareAwareRatioBatchSampler(
        ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        presence=presence,
        rare_thr=0.1,
        num_replicas=world if use_dist else 1,
        rank=rank if use_dist else 0,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=npz_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=npz_collate,
    )
    if rank == 0:
        print(f"dim={args.dim} train samples={len(ds)}  test samples={len(test_ds)}", flush=True)
        freq_msg = ", ".join(
            f"{n}={f:.4f}" for n, f in zip(presence["class_names"], presence["frequency"])
        )
        print(f"[{tag}] class_freq(sum_counts/N)=[{freq_msg}]", flush=True)
        print(
            f"[{tag}] rare(freq<0.1)={rare or 'none'} → each batch ≥{batch_sampler.min_rare}/{args.batch_size} "
            f"samples contain each rare class",
            flush=True,
        )

    model = OrganelleSegDet3D(mode=args.mode, dim=args.dim, det_box=args.det_box,
                              prompt_pe=args.prompt_pe, prompt_jitter=args.prompt_jitter).to(device)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        del checkpoint
        if rank == 0:
            print(f"[resume] loaded {args.resume}; continuing at epoch {start_epoch}", flush=True)
    if use_dist and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if rank == 0:
        cls_weight = compute_class_weights(ds).to(device)
    else:
        cls_weight = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
    if use_dist:
        dist.broadcast(cls_weight, src=0)
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume CHECKPOINT")
        print(json.dumps(evaluate(args, model, test_loader, device, cls_weight), indent=2), flush=True)
        return
    if rank == 0:
        print(f"[{tag}] cls_weight(er/mito/lyso/ld/golgi)={cls_weight.tolist()} eos_coef={args.eos_coef}", flush=True)

    per_step_global = args.batch_size * world
    accum = max(1, args.effective_batch // max(1, per_step_global))
    if rank == 0:
        print(
            f"[{tag}] world={world} batch/step/gpu={args.batch_size} accum={accum} "
            f"global_effective={per_step_global * accum} samples={len(ds)}",
            flush=True,
        )

    log_path = out / f"train_log_{tag}.jsonl"
    keys = ["loss", "seg", "det", "prompt", "er"]
    best_val = float("inf")
    if args.resume and rank == 0:
        best_path = out / f"best_{tag}.pt"
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_val = float(best_checkpoint.get("val_loss", best_val))
            del best_checkpoint
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batch_sampler.set_epoch(epoch - 1)
        running = {k: 0.0 for k in keys}
        n = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"{tag} e{epoch}", disable=rank != 0)
        for step, batch in enumerate(pbar):
            loss, seg_l, det_l, prompt_l, er_l = compute_losses(args, model, batch, device, cls_weight)
            if not torch.isfinite(loss):
                continue
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad(set_to_none=True)

            running["loss"] += float(loss.item())
            running["seg"] += float(seg_l.item())
            running["det"] += float(det_l.item())
            running["prompt"] += float(prompt_l.item())
            running["er"] += float(er_l.item())
            n += 1
        mean = {k: v / max(n, 1) for k, v in running.items()}
        val = {}
        if rank == 0 and epoch % args.eval_every == 0:
            val = evaluate(args, model, test_loader, device, cls_weight)
        if rank == 0:
            msg = (
                f"[{tag}] epoch {epoch} loss={mean['loss']:.4f} seg={mean['seg']:.4f} "
                f"det={mean['det']:.4f} prompt={mean['prompt']:.4f} er={mean['er']:.4f}"
            )
            if val:
                msg += f" | val_loss={val['val_loss']:.4f} val_seg={val['val_seg']:.4f}"
                if args.mode in ("det_sam", "det_sam_mc"):
                    msg += (
                        f" val_det={val['val_det']:.4f} val_prompt={val['val_prompt']:.4f}"
                        f" val_er={val['val_er']:.4f}"
                    )
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "dim": args.dim, **mean, **val}) + "\n")
            state = model.module.state_dict() if use_dist else model.state_dict()
            torch.save(
                {"epoch": epoch, "model": state, "mode": args.mode, "dim": args.dim,
                 "eos_coef": args.eos_coef, **mean},
                out / f"last_{tag}.pt",
            )
            if val and val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save(
                    {"epoch": epoch, "model": state, "mode": args.mode, "dim": args.dim,
                     "eos_coef": args.eos_coef, **mean, **val},
                    out / f"best_{tag}.pt",
                )
                print(f"[{tag}] new best val_loss={best_val:.4f} @ epoch {epoch}", flush=True)
        # Rank 0 evaluates and checkpoints alone. Keep the other ranks out of
        # the next DDP forward (which broadcasts buffers) until it is finished.
        if rank == 0 and epoch % 5 == 0:
            torch.save(
                {"epoch": epoch, "model": state, "mode": args.mode, "dim": args.dim,
                 "eos_coef": args.eos_coef, **mean},
                out / f"e{epoch}_{tag}.pt",
            )
        if use_dist:
            dist.barrier()

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
