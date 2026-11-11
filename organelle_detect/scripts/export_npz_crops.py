#!/usr/bin/env python3
"""Export CellMap organelle crops to .npz for the seg+det model (parallel).

type1 (instance-centered, jitter-style):
    For each mito/lyso/ld instance (NOT er), take the union of the bounding
    boxes of all same-class instances that overlap its 6-layer z slab; that
    union box, expanded by --scale in xy, is the crop. The window is fitted
    INSIDE the region's label bbox (z slab shifted, xy clipped) so no instance
    is cut by the label boundary; the target organelle stays fully contained.
    Crops with black bands (>1% zero-filled EM voxels) are discarded.
    The crop is resized to the standard backbone input size.

type2 (random):
    Random 256x256x6 windows of raw EM (no resize), sampled STRICTLY INSIDE
    the region's label bounding box so every voxel is covered by all 4 class
    labels (no unlabeled "fake background"). Count is user-set.

Only regions (dataset/crop) that annotate ALL 4 classes are used; regions
with partial class coverage are dropped entirely.

Every crop stores:
  image        float32 [6, S, S]   6-layer raw EM (type1 resized to S, type2 S=256)
  masks        bool    [N, 6, S, S] one mask per organelle instance/cc in the crop
  mask_labels  int64   [N]         merged class id per mask (1 er,2 mito,3 lyso,4 ld)
  boxes        float32 [M, 6]      GT 3D boxes for instance organelles (no er)
  box_labels   int64   [M]         merged class id per box
  meta         json                dataset/crop/origin/kind

Merged classes: 0 bg, 1 er, 2 mito, 3 lyso, 4 ld. Substructures merged.
ER masks are 3D connected components of the semantic ER label (no boxes).
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.cellmap_io import (  # noqa: E402
    UNKNOWN_ID,
    em_group,
    index_to_world,
    label_group,
    open_array,
    read_multiscale_transform,
    world_to_index,
)
from data.datasets import build_instance_index  # noqa: E402

CLASS_NAMES = ("er", "mito", "lyso", "ld")
CLASS_TO_ID = {n: i + 1 for i, n in enumerate(CLASS_NAMES)}
ER_ID = CLASS_TO_ID["er"]
INSTANCE_CLASSES = ("mito", "lyso", "ld")
FULL_INDEX = ROOT / "outputs" / "instance_index_full.json"


def _resample(vol, lab_xf, em_xf, z0, y0, x0, dz, dy, dx):
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    em_coords = np.stack([zz + z0 + 0.5, yy + y0 + 0.5, xx + x0 + 0.5], axis=-1).reshape(-1, 3)
    world = index_to_world(em_coords, em_xf)
    lab_idx = world_to_index(world, lab_xf)
    coords = lab_idx.T.reshape(3, dz, dy, dx)
    return ndi.map_coordinates(vol.astype(np.float32), coords, order=0, mode="constant", cval=0).astype(vol.dtype)


def _resize(img, out, is_mask):
    t = torch.from_numpy(np.asarray(img, np.float32))[None, None]
    mode = "nearest" if is_mask else "trilinear"
    kw = {} if is_mask else {"align_corners": False}
    t = F.interpolate(t, size=out, mode=mode, **kw)
    return t[0, 0].numpy()


def _em_levels(em_path):
    """Available EM scale levels, ordered s0, s1, s2, ..."""
    levels = [p.name for p in em_path.iterdir() if p.is_dir() and p.name.startswith("s")]
    return sorted(levels, key=lambda s: int(s[1:]))


def _read_em_window(em_path, origin_world, dz, dy, dx, s0_scale):
    """Read a 6-layer EM window defined in world coords.

    The output grid has (dz, dy, dx) voxels at the dataset's s0 spacing. Tries
    EM levels s0, s1, ... until the window contains real (non-blank) data —
    some datasets only have EM written at coarser levels. Returns the window
    or None if every level is blank/out-of-bounds.
    """
    zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    grid_world = np.stack(
        [
            origin_world[0] + (zz + 0.5) * s0_scale[0],
            origin_world[1] + (yy + 0.5) * s0_scale[1],
            origin_world[2] + (xx + 0.5) * s0_scale[2],
        ],
        axis=-1,
    ).reshape(-1, 3)
    for level in _em_levels(em_path):
        xf = read_multiscale_transform(em_path, level)
        arr = open_array(em_path, level)
        idx = world_to_index(grid_world, xf)
        lo = np.floor(idx.min(axis=0)).astype(int) - 1
        hi = np.ceil(idx.max(axis=0)).astype(int) + 2
        if np.any(lo < 0) or np.any(hi > np.array(arr.shape)):
            continue  # out of bounds at this level
        slab = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
        if slab.size == 0 or slab.std() < 1e-3:
            continue  # blank (unwritten chunks) at this level
        coords = (idx - lo[None, :]).T.reshape(3, dz, dy, dx)
        img = ndi.map_coordinates(slab, coords, order=1, mode="constant", cval=0)
        return img.astype(np.float32), level
    return None, None


def _extract(ds_name, crop_name, z0, y0, x0, dz, dy, dx, out_size, resize):
    """Pull EM + all organelle masks + instance boxes for one window.

    The window (z0,y0,x0,dz,dy,dx) is in EM s0 index space; the EM read falls
    back to coarser levels when s0 is unwritten. Returns None for blank crops.
    """
    em_path = em_group(ds_name)
    em_xf = read_multiscale_transform(em_path, "s0")
    s0_scale = em_xf.scale
    origin_world = index_to_world(np.array([z0, y0, x0], dtype=np.float64), em_xf)
    image, level = _read_em_window(em_path, origin_world, dz, dy, dx, s0_scale)
    if image is None:
        return None
    # black-band check: >1% exact-zero voxels means the window overlaps
    # unwritten EM chunks (zero fill) — discard rather than train on it
    if float((image == 0).mean()) > 0.01:
        return None

    masks, mask_labels, boxes, box_labels = [], [], [], []
    for cls in CLASS_NAMES:
        g = label_group(ds_name, crop_name, cls)
        if not (g / "s0" / ".zarray").exists():
            continue
        vol = np.asarray(open_array(g, "s0"))
        lab_xf = read_multiscale_transform(g, "s0")
        sampled = _resample(vol, lab_xf, em_xf, z0, y0, x0, dz, dy, dx)
        if cls in INSTANCE_CLASSES:
            for inst_id in np.unique(sampled):
                if inst_id in (0, UNKNOWN_ID):
                    continue
                m = sampled == inst_id
                if m.sum() < 8:
                    continue
                zs, ys, xs = np.where(m)
                boxes.append([zs.min(), ys.min(), xs.min(), zs.max() + 1, ys.max() + 1, xs.max() + 1])
                masks.append(m)
                mask_labels.append(CLASS_TO_ID[cls])
                box_labels.append(CLASS_TO_ID[cls])
        else:
            binary = (sampled > 0) & (sampled != UNKNOWN_ID)
            labeled, nlab = ndi.label(binary)
            for cid in range(1, nlab + 1):
                m = labeled == cid
                if m.sum() < 16:
                    continue
                masks.append(m)
                mask_labels.append(ER_ID)

    if resize:
        out = (image.shape[0], out_size, out_size)
        image = _resize(image, out, False)
        masks = [_resize(m, out, True) > 0.5 for m in masks]
        scale_f = np.array([1.0, out_size / dy, out_size / dx] * 2, np.float32)
        boxes = [np.asarray(b, np.float32) * scale_f for b in boxes]
        S = out_size
    else:
        S = dy
    DZ = image.shape[0]
    image = (image - image.mean()) / (image.std() + 1e-6)
    masks_a = np.stack(masks) if masks else np.zeros((0, DZ, S, S), bool)
    return {
        "image": image.astype(np.float32),
        "masks": masks_a.astype(bool),
        "mask_labels": np.asarray(mask_labels, np.int64),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 6),
        "box_labels": np.asarray(box_labels, np.int64),
        "em_level": level,
    }


def _write_shard(batch, out_dir, kind, idx):
    path = out_dir / f"{kind}_{idx:05d}.npz"
    n = len(batch)
    image = np.stack([b["image"] for b in batch])
    masks = np.empty(n, object)
    mask_labels = np.empty(n, object)
    boxes = np.empty(n, object)
    box_labels = np.empty(n, object)
    for i, b in enumerate(batch):
        masks[i] = b["masks"]
        mask_labels[i] = b["mask_labels"]
        boxes[i] = b["boxes"]
        box_labels[i] = b["box_labels"]
    np.savez_compressed(
        path, image=image, masks=masks, mask_labels=mask_labels,
        boxes=boxes, box_labels=box_labels,
        meta=json.dumps([b["meta"] for b in batch]),
    )
    return path.name


def _work(task):
    """Process one chunk of crop specs, write one npz shard."""
    kind, shard_idx, specs, out_dir, out_size = task
    batch = []
    for spec in specs:
        ds_name, crop_name, z0, y0, x0, dz, dy, dx, resize, meta = spec
        out = _extract(ds_name, crop_name, z0, y0, x0, dz, dy, dx, out_size, resize)
        if out is None:
            continue
        meta = dict(meta)
        meta["em_level"] = out.pop("em_level")
        out["meta"] = meta
        batch.append(out)
    if not batch:
        return None
    return _write_shard(batch, Path(out_dir), kind, shard_idx)


def _label_bbox_em(ds_name, crop_name, em_xf):
    """Intersection (over all 4 classes) of the label-array bounding boxes, in
    EM s0 index space. Returns (lo, hi) (hi exclusive) or None when any class
    has no s0 label array for this crop — such regions only annotate a subset
    of classes, so unlabeled organelles there would become fake background."""
    lo_all = np.full(3, -np.inf)
    hi_all = np.full(3, np.inf)
    for cls in CLASS_NAMES:
        g = label_group(ds_name, crop_name, cls)
        if not (g / "s0" / ".zarray").exists():
            return None
        arr = open_array(g, "s0")
        xf = read_multiscale_transform(g, "s0")
        w0 = index_to_world(np.zeros(3), xf)
        w1 = index_to_world(np.array(arr.shape, dtype=np.float64), xf)
        i0 = world_to_index(w0, em_xf)
        i1 = world_to_index(w1, em_xf)
        lo_all = np.maximum(lo_all, np.minimum(i0, i1))
        hi_all = np.minimum(hi_all, np.maximum(i0, i1))
    if np.any(hi_all <= lo_all):
        return None
    return lo_all, hi_all


def _union_box_type1(rec, inst_by_ds_crop, em_xf, layers, scale=1.0):
    """Union box of same-class instances overlapping the target's 6-layer slab,
    then EXPAND the xy field of view by `scale` about the union centre so the
    organelle sits in context instead of touching the crop edges. z stays at
    `layers` slices. Out-of-bounds windows are discarded later (no padding)."""
    ds_name, crop_name, cls = rec["dataset"], rec["crop"], rec["class"]
    zmin_w = np.array(rec["world_min"])[0]
    zmax_w = np.array(rec["world_max"])[0]
    lo, hi = min(zmin_w, zmax_w), max(zmin_w, zmax_w)
    zc = 0.5 * (lo + hi)
    half = 0.5 * layers * em_xf.scale[0]
    slab_lo, slab_hi = zc - half, zc + half
    union_min = np.array(rec["world_min"], np.float64)
    union_max = np.array(rec["world_max"], np.float64)
    for other in inst_by_ds_crop.get((ds_name, crop_name), []):
        if other["class"] != cls:
            continue
        omin, omax = np.array(other["world_min"]), np.array(other["world_max"])
        o_lo, o_hi = min(omin[0], omax[0]), max(omin[0], omax[0])
        if o_hi < slab_lo or o_lo > slab_hi:
            continue
        union_min = np.minimum(union_min, omin)
        union_max = np.maximum(union_max, omax)
    emin = world_to_index(union_min, em_xf)
    emax = world_to_index(union_max, em_xf)
    # z slab is centred on the TARGET instance (not the union): a chained
    # union can drift far in z, and the 6-layer slab must still contain it
    t0 = world_to_index(np.array(rec["world_min"], np.float64), em_xf)[0]
    t1 = world_to_index(np.array(rec["world_max"], np.float64), em_xf)[0]
    z0 = int(round(0.5 * (min(t0, t1) + max(t0, t1)) - layers / 2))
    # tight union extents in xy
    y_lo, y_hi = min(emin[1], emax[1]), max(emin[1], emax[1])
    x_lo, x_hi = min(emin[2], emax[2]), max(emin[2], emax[2])
    yc, xc = 0.5 * (y_lo + y_hi), 0.5 * (x_lo + x_hi)
    # expand field of view by `scale` about the centre
    dy = max((y_hi - y_lo) * scale, 1.0)
    dx = max((x_hi - x_lo) * scale, 1.0)
    y0 = int(round(yc - dy / 2))
    x0 = int(round(xc - dx / 2))
    return z0, y0, x0, layers, int(round(dy)), int(round(dx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--distance-thr", type=float, default=0.6)
    ap.add_argument("--out-size", type=int, default=256, help="type1 xy resize target")
    ap.add_argument("--type2-xy", type=int, default=256)
    ap.add_argument("--type2-count", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16, help="crops per npz shard")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "outputs" / "npz_crops_full")
    ap.add_argument("--index", type=Path, default=FULL_INDEX)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = build_instance_index(max_per_crop=100000, cache=args.index)
    inst_records = [r for r in records if r["class"] in INSTANCE_CLASSES]
    inst_by_ds_crop: dict[tuple[str, str], list[dict]] = {}
    for r in inst_records:
        inst_by_ds_crop.setdefault((r["dataset"], r["crop"]), []).append(r)

    # keep only regions that annotate ALL 4 classes; compute their label bboxes
    # (intersection of the 4 label arrays) in EM s0 index space
    em_xfs: dict[str, object] = {}
    region_label_boxes: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for ds_name, crop_name in sorted(inst_by_ds_crop):
        if ds_name not in em_xfs:
            em_xfs[ds_name] = read_multiscale_transform(em_group(ds_name), "s0")
        bb = _label_bbox_em(ds_name, crop_name, em_xfs[ds_name])
        if bb is not None:
            region_label_boxes[(ds_name, crop_name)] = bb
    n_drop = len(inst_by_ds_crop) - len(region_label_boxes)
    inst_records = [r for r in inst_records if (r["dataset"], r["crop"]) in region_label_boxes]
    inst_by_ds_crop = {}
    for r in inst_records:
        inst_by_ds_crop.setdefault((r["dataset"], r["crop"]), []).append(r)
    print(
        f"instances (mito/lyso/ld): {len(inst_records)} in {len(region_label_boxes)} fully-labeled regions "
        f"(dropped {n_drop} partially-labeled regions)",
        flush=True,
    )

    tasks = []
    # ---- type 1 specs ----
    # Fit the (union-box x scale) window into the region's label bbox instead
    # of letting it poke out: outside the labels, instances would be cut by the
    # LABEL boundary (visible in EM but annotated as background). z: shift the
    # 6-layer slab to fit (depth fixed). xy: clip the expanded window to the
    # label box — the instance union lives inside the box, so clipping only
    # removes outer context. The target instance's xy extent must stay fully
    # contained, and the z slab must still overlap it; otherwise discard.
    t1_specs = []
    n_outside = 0
    scale_ladder = [args.scale, 1.5, 1.25, 1.0]
    for rec in inst_records:
        ds_name, crop_name = rec["dataset"], rec["crop"]
        em_xf = em_xfs[ds_name]
        lo, hi = region_label_boxes[(ds_name, crop_name)]
        tmin = world_to_index(np.array(rec["world_min"], np.float64), em_xf)
        tmax = world_to_index(np.array(rec["world_max"], np.float64), em_xf)
        t_lo, t_hi = np.minimum(tmin, tmax), np.maximum(tmin, tmax)
        placed = None
        # shrink the FOV expansion until the window fits inside the label box
        for sc in dict.fromkeys(scale_ladder):
            z0, y0, x0, dz, dy, dx = _union_box_type1(rec, inst_by_ds_crop, em_xf, args.layers, sc)
            if hi[0] - lo[0] < dz:
                break  # label box shallower than the slab: no scale can fix
            z0 = int(np.clip(z0, np.ceil(lo[0]), np.floor(hi[0]) - dz))
            y_lo = max(y0, int(np.ceil(lo[1])))
            x_lo = max(x0, int(np.ceil(lo[2])))
            y_hi = min(y0 + dy, int(np.floor(hi[1])))
            x_hi = min(x0 + dx, int(np.floor(hi[2])))
            if y_hi - y_lo < 16 or x_hi - x_lo < 16:
                continue
            z_overlap = z0 < t_hi[0] and t_lo[0] < z0 + dz
            xy_contained = (
                y_lo <= t_lo[1] and t_hi[1] <= y_hi and x_lo <= t_lo[2] and t_hi[2] <= x_hi
            )
            if z_overlap and xy_contained:
                placed = (z0, y_lo, x_lo, dz, y_hi - y_lo, x_hi - x_lo)
                break
        if placed is None:
            n_outside += 1
            continue
        z0, y0, x0, dz, dy, dx = placed
        meta = {"dataset": ds_name, "crop": crop_name, "origin": [z0, y0, x0], "kind": "type1", "seed": rec["class"]}
        t1_specs.append((ds_name, crop_name, z0, y0, x0, dz, dy, dx, True, meta))
    print(f"type1: {len(t1_specs)} specs ({n_outside} discarded: cannot fit in label bbox)", flush=True)
    for i in range(0, len(t1_specs), args.batch):
        tasks.append(("type1", i // args.batch, t1_specs[i : i + args.batch], str(args.out), args.out_size))
    n1 = len(t1_specs)

    # ---- type 2 specs (random, no resize) ----
    # Sample windows STRICTLY INSIDE the region's label bbox: every voxel of a
    # type2 window is then covered by all 4 class labels, so visible organelles
    # can never be unlabeled "fake background". (v4 sampled within ±256 voxels
    # of the instance union -> 82% of windows landed outside the labels.)
    rng = np.random.default_rng(0)
    regions = sorted(region_label_boxes)
    t2_specs = []
    L, XY = args.layers, args.type2_xy
    win = np.array([L, XY, XY])
    attempts = 0
    while len(t2_specs) < args.type2_count and attempts < args.type2_count * 3:
        attempts += 1
        ds_name, crop_name = regions[rng.integers(len(regions))]
        lo, hi = region_label_boxes[(ds_name, crop_name)]
        em = open_array(em_group(ds_name), "s0")
        ez, ey, ex = em.shape
        # origin range so the whole window stays inside the label box AND the EM volume
        o_lo = np.maximum(np.ceil(lo).astype(int), [0, 0, 0])
        o_hi = np.minimum(np.floor(hi).astype(int) - win, [ez - L, ey - XY, ex - XY])
        if np.any(o_hi <= o_lo):
            continue  # label box smaller than the window
        z0 = int(rng.integers(o_lo[0], o_hi[0]))
        y0 = int(rng.integers(o_lo[1], o_hi[1]))
        x0 = int(rng.integers(o_lo[2], o_hi[2]))
        meta = {"dataset": ds_name, "crop": crop_name, "origin": [z0, y0, x0], "kind": "type2"}
        t2_specs.append((ds_name, crop_name, z0, y0, x0, L, XY, XY, False, meta))
    for j in range(0, len(t2_specs), args.batch):
        tasks.append(("type2", j // args.batch, t2_specs[j : j + args.batch], str(args.out), args.out_size))
    n2 = len(t2_specs)

    print(f"type1 specs: {n1}  type2 specs: {n2}  shards: {len(tasks)}  workers: {args.workers}", flush=True)
    done = 0
    with Pool(args.workers) as pool:
        for name in pool.imap_unordered(_work, tasks):
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] last shard: {name}", flush=True)

    # count actually-written crops (blank/out-of-bounds windows were skipped)
    actual = {"type1": 0, "type2": 0}
    for f in args.out.glob("*.npz"):
        kind = f.stem.split("_")[0]
        with np.load(f) as z:
            actual[kind] = actual.get(kind, 0) + int(z["image"].shape[0])
    summary = {"type1": actual.get("type1", 0), "type2": actual.get("type2", 0),
               "type1_specs": n1, "type2_specs": n2,
               "layers": args.layers, "out_size": args.out_size,
               "type2_xy": args.type2_xy, "scale": args.scale, "distance_thr": args.distance_thr}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
