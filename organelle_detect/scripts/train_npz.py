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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset_npz import (  # noqa: E402
    NPZCropDataset,
    RareAwareRatioBatchSampler,
    compute_class_presence,
    npz_collate,
)
from models.box_ops import slice_boxes_from_masks  # noqa: E402
from models.losses import GAIN_BOX, GAIN_CLS, GAIN_DFL, yolo_loss  # noqa: E402
from models.losses_segdet import (  # noqa: E402
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
    p.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-6)
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
SUBLOSS_KEYS = (
    "seg_cls", "seg_mask", "seg_dice", "det_cls", "det_box", "det_dfl",
    "prompt_mask", "prompt_dice", "er_cls", "er_mask", "er_dice",
)
METRIC_KEYS = ("loss", "seg", "det", "prompt", "er", *SUBLOSS_KEYS)


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
        )
    else:
        seg = mask2former_loss_3d(
            out_m["seg"]["pred_logits"], out_m["seg"]["pred_masks"], seg_labels, masks, cls_weight=cls_weight
        )
    loss = seg["loss"]
    det_l = prompt_l = er_l = torch.zeros((), device=device)
    parts = {key: torch.zeros((), device=device) for key in SUBLOSS_KEYS}
    parts["seg_cls"] = seg["cls"]
    parts["seg_mask"] = 5.0 * seg["mask"]
    parts["seg_dice"] = 5.0 * seg["dice"]
    if args.mode == "det_sam":
        # per-class binary det + prompt (one head pair per mito/lyso/ld/golgi)
        B = images.shape[0]
        crop = images.shape[-1]
        det_terms, prompt_terms = [], []
        det_parts = {k: [] for k in ("cls", "giou", "dfl")}
        prompt_parts = {k: [] for k in ("mask", "dice")}
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
            dloss = yolo_loss(det_sub, e_boxes, e_labels, crop)
            det_terms.append(dloss["loss"])
            for key in det_parts:
                det_parts[key].append(dloss[key])
            if bidx_pos:
                pm_sub = [out_m["prompt"][i]["pred_masks"][b] for b in bidx_pos]
                e_masks_pos = [masks[b][(mask_labels[b].long() == cls_id)] for b in bidx_pos]
                ploss = prompt_mask_loss(pm_sub, e_masks_pos)
                prompt_terms.append(ploss["loss"])
                for key in prompt_parts:
                    prompt_parts[key].append(ploss[key])
        if det_terms:
            det_l = torch.stack(det_terms).mean()
            parts["det_cls"] = GAIN_CLS * torch.stack(det_parts["cls"]).mean()
            parts["det_box"] = GAIN_BOX * torch.stack(det_parts["giou"]).mean()
            parts["det_dfl"] = GAIN_DFL * torch.stack(det_parts["dfl"]).mean()
        if prompt_terms:
            prompt_l = torch.stack(prompt_terms).mean()
            parts["prompt_mask"] = 5.0 * torch.stack(prompt_parts["mask"]).mean()
            parts["prompt_dice"] = 5.0 * torch.stack(prompt_parts["dice"]).mean()
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
            parts["er_cls"] = er["cls"]
            parts["er_mask"] = 5.0 * er["mask"]
            parts["er_dice"] = 5.0 * er["dice"]
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
            dloss = yolo_loss(det_sub, e_boxes, e_labels, crop)
            det_l = dloss["loss"]
            parts["det_cls"] = GAIN_CLS * dloss["cls"]
            parts["det_box"] = GAIN_BOX * dloss["giou"]
            parts["det_dfl"] = GAIN_DFL * dloss["dfl"]
            if bidx_pos:
                pm_sub = [out_m["prompt"]["pred_masks"][b] for b in bidx_pos]
                e_masks_pos = []
                for b in bidx_pos:
                    mlab = mask_labels[b].long()
                    msk = masks[b]
                    keep = (mlab == 2) | (mlab == 3)
                    e_masks_pos.append(msk[keep])
                ploss = prompt_mask_loss(pm_sub, e_masks_pos)
                prompt_l = ploss["loss"]
                parts["prompt_mask"] = 5.0 * ploss["mask"]
                parts["prompt_dice"] = 5.0 * ploss["dice"]
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
            parts["er_cls"] = er["cls"]
            parts["er_mask"] = 5.0 * er["mask"]
            parts["er_dice"] = 5.0 * er["dice"]
            loss = loss + er_l
    return loss, seg["loss"], det_l, prompt_l, er_l, parts


@torch.no_grad()
def evaluate(args, model, test_loader, device, cls_weight=None, max_batches: int | None = None):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    totals = dict.fromkeys(METRIC_KEYS, 0.0)
    n = 0
    for bi, batch in enumerate(test_loader):
        if max_batches is not None and bi >= max_batches:
            break
        loss, seg_l, det_l, prompt_l, er_l, parts = compute_losses(args, raw, batch, device, cls_weight)
        if not torch.isfinite(loss):
            continue
        totals["loss"] += float(loss.item())
        totals["seg"] += float(seg_l.item())
        totals["det"] += float(det_l.item())
        totals["prompt"] += float(prompt_l.item())
        totals["er"] += float(er_l.item())
        for key, value in parts.items():
            totals[key] += float(value.item())
        n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


def format_loss_metrics(m):
    return (
        f"loss={m['loss']:.4f} seg={m['seg']:.4f} det={m['det']:.4f} "
        f"prompt={m['prompt']:.4f} er={m['er']:.4f} "
        f"| seg(cls/bce/dice)={m['seg_cls']:.3f}/{m['seg_mask']:.3f}/{m['seg_dice']:.3f} "
        f"det(cls/box/dfl)={m['det_cls']:.3f}/{m['det_box']:.3f}/{m['det_dfl']:.3f} "
        f"prompt(bce/dice)={m['prompt_mask']:.3f}/{m['prompt_dice']:.3f} "
        f"er(cls/bce/dice)={m['er_cls']:.3f}/{m['er_mask']:.3f}/{m['er_dice']:.3f}"
    )


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
    resume_payload = {}
    resume_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resume_val = float(checkpoint.get("val_loss", float("inf")))
        resume_payload = {
            "optimizer": checkpoint.get("optimizer"),
            "scheduler": checkpoint.get("scheduler"),
        }
        best_path = out / f"best_{tag}.pt"
        if rank == 0 and not best_path.exists() and np.isfinite(resume_val):
            torch.save(checkpoint, best_path)
        del checkpoint
        if rank == 0:
            print(f"[resume] loaded {args.resume}; continuing at epoch {start_epoch}", flush=True)
    if use_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if resume_payload.get("optimizer") is not None:
        opt.load_state_dict(resume_payload["optimizer"])
    elif args.resume and rank == 0:
        print("[resume] checkpoint has no optimizer state; AdamW starts fresh", flush=True)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_factor, patience=args.lr_patience,
            threshold=1e-3, threshold_mode="rel", min_lr=args.min_lr,
        )
        if resume_payload.get("scheduler") is not None:
            scheduler.load_state_dict(resume_payload["scheduler"])
        elif np.isfinite(resume_val):
            scheduler.step(resume_val)
    if rank == 0:
        cls_weight = compute_class_weights(ds).to(device)
    else:
        cls_weight = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
    if use_dist:
        dist.broadcast(cls_weight, src=0)
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
    best_val = resume_val
    if args.resume and not np.isfinite(best_val) and rank == 0:
        best_path = out / f"best_{tag}.pt"
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_val = float(best_checkpoint.get("val_loss", best_val))
            del best_checkpoint
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batch_sampler.set_epoch(epoch - 1)
        running = torch.zeros(len(METRIC_KEYS) + 1, device=device, dtype=torch.float64)
        window = torch.zeros_like(running)
        opt.zero_grad(set_to_none=True)
        lr_used = float(opt.param_groups[0]["lr"])
        for step, batch in enumerate(loader):
            loss, seg_l, det_l, prompt_l, er_l, parts = compute_losses(
                args, model, batch, device, cls_weight
            )
            finite = torch.isfinite(loss).to(device=device, dtype=torch.int32)
            if use_dist:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                continue
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad(set_to_none=True)

            vals = torch.stack([
                loss.detach().float(), seg_l.detach().float(), det_l.detach().float(),
                prompt_l.detach().float(), er_l.detach().float(),
                *(parts[k].detach().float() for k in SUBLOSS_KEYS),
            ]).to(torch.float64)
            running[:-1] += vals
            running[-1] += 1
            window[:-1] += vals
            window[-1] += 1
            if (step + 1) % 20 == 0 or (step + 1) == len(loader):
                if use_dist:
                    dist.all_reduce(window, op=dist.ReduceOp.SUM)
                if rank == 0:
                    avg = (window[:-1] / window[-1].clamp_min(1)).cpu().tolist()
                    print(
                        f"[{tag}] epoch {epoch} step {step + 1}/{len(loader)} "
                        + format_loss_metrics(dict(zip(METRIC_KEYS, avg))), flush=True,
                    )
                window.zero_()

        if use_dist:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
        avg = (running[:-1] / running[-1].clamp_min(1)).cpu().tolist()
        mean = dict(zip(METRIC_KEYS, avg))
        val = {}
        if rank == 0 and epoch % args.eval_every == 0:
            val = evaluate(args, model, test_loader, device, cls_weight)
        val_loss_tensor = torch.tensor(
            val.get("val_loss", 0.0) if rank == 0 else 0.0, device=device, dtype=torch.float64
        )
        if use_dist:
            dist.broadcast(val_loss_tensor, src=0)
        if scheduler is not None and epoch % args.eval_every == 0:
            scheduler.step(float(val_loss_tensor.item()))
        lr_next = float(opt.param_groups[0]["lr"])

        if rank == 0:
            msg = f"[{tag}] epoch {epoch} lr={lr_used:.2e}-> {lr_next:.2e} " + format_loss_metrics(mean)
            if val:
                val_mean = {k.removeprefix("val_"): v for k, v in val.items()}
                msg += " | val " + format_loss_metrics(val_mean)
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "dim": args.dim, "lr": lr_used,
                                    "lr_next": lr_next, **mean, **val}) + "\n")
            state = model.module.state_dict() if use_dist else model.state_dict()
            checkpoint_state = {
                "epoch": epoch, "model": state, "mode": args.mode, "dim": args.dim,
                "eos_coef": args.eos_coef, "lr": lr_next, **mean,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(checkpoint_state, out / f"last_{tag}.pt")
            if val and val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save({**checkpoint_state, **val}, out / f"best_{tag}.pt")
                print(f"[{tag}] new best val_loss={best_val:.4f} @ epoch {epoch}", flush=True)
        if rank == 0 and epoch % 5 == 0:
            torch.save(checkpoint_state, out / f"e{epoch}_{tag}.pt")
        if use_dist:
            dist.barrier()

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
