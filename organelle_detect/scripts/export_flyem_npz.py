"""Export flyEM fish2 annotation volumes (lysosome 2500^2x250, mito 1280^2x256)
to .npz crops compatible with train_npz.py (m2f_bin / det_sam, dim 3d).

Crop spec: 6 z-slices (6x30nm=180nm) x 125x125 xy (1000nm @8nm), native res.

Grayscale unification across volumes:
  1. per-volume robust percentile normalization clip(p0.5, p99.8) -> [0,1]
  2. per-crop z-score (same convention as export_npz_crops.py)

Crops:
  type1  one per instance, centered on instance bbox center
  type2  uniform random windows
  aug    4 per base crop: xy-plane rotation U(0,360) + translation
         (xy +/-40px, z +/-8 slices); kept only if the rotated box stays
         fully inside the volume (fallback: pure rotation -> pure shift -> copy)

Split: each volume divided into 5 vertical xy stripes (meta crop = ..._s{k});
split.json marks stripe 2 of each volume as test (2/10 = 20%).

Class ids follow organelle_classes: mito=2, lyso=3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

FLYEM = Path("/path/to/project_data/flyEM")
OUT_DIR = Path("/path/to/JaneliaEM/outputs/npz_crops_flyem_1000nm")
VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/flyem_1000nm")

DZ, XY = 6, 125          # 6 slices (180nm) x 1000nm
N_TYPE2 = 500            # random crops per volume
N_AUG = 4                # augmented crops per base crop
TRANS_XY = 40.0          # +/- px
TRANS_Z = 8              # +/- slices
BATCH = 16               # crops per npz shard
N_STRIPES = 5
TEST_STRIPE = 2
SEED = 42

MITO_ID, LYSO_ID = 2, 3  # organelle_classes convention

VOLUMES = [
    dict(
        dataset="fish2_2500", cls="lyso", cls_id=LYSO_ID,
        em=FLYEM / "2500x2500x250/raw.ome.tif",
        lab=FLYEM / "2500x2500x250/annotation/lysosome_instances_v0924.zarr",
    ),
    dict(
        dataset="fish2_1280", cls="mito", cls_id=MITO_ID,
        em=FLYEM / "1280x1280x256/raw.ome.tif",
        lab=FLYEM / "1280x1280x256/annotation/mitochondria_0917",
    ),
]


def load_volume(spec):
    em = tifffile.imread(str(spec["em"])).astype(np.float32)
    lab = np.asarray(zarr.open(str(spec["lab"]), mode="r"))
    assert em.shape == lab.shape, (em.shape, lab.shape)
    # per-volume robust gray unification -> [0,1]
    sub = em[::4, ::4, ::4]
    lo, hi = np.percentile(sub, [0.5, 99.8])
    em = np.clip((em - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    print(f"[{spec['dataset']}] shape={em.shape} gray p0.5={lo:.1f} p99.8={hi:.1f} -> [0,1]",
          flush=True)
    return em, lab.astype(np.uint32)


def stripe_of(cx: float, width: int) -> int:
    return min(N_STRIPES - 1, int(cx / (width / N_STRIPES)))


def masks_boxes_from_label(lab_crop: np.ndarray, cls_id: int, min_vox: int = 8):
    """Per-instance bool masks + tight 3D boxes from an instance-id crop."""
    masks, boxes = [], []
    for iid in np.unique(lab_crop):
        if iid == 0:
            continue
        m = lab_crop == iid
        if m.sum() < min_vox:
            continue
        zs, ys, xs = np.where(m)
        masks.append(m)
        boxes.append([zs.min(), ys.min(), xs.min(), zs.max() + 1, ys.max() + 1, xs.max() + 1])
    return masks, boxes


def finalize(em_crop, masks, boxes, cls_id, meta):
    """Per-crop z-score + array packing (export_npz_crops convention)."""
    img = (em_crop - em_crop.mean()) / (em_crop.std() + 1e-6)
    n = len(masks)
    return {
        "image": img.astype(np.float32),
        "masks": np.stack(masks).astype(bool) if n else np.zeros((0, DZ, XY, XY), bool),
        "mask_labels": np.full(n, cls_id, np.int64),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 6),
        "box_labels": np.full(len(boxes), cls_id, np.int64),
        "meta": meta,
    }


def extract_base(em, lab, z0, y0, x0, cls_id, meta):
    em_crop = em[z0:z0 + DZ, y0:y0 + XY, x0:x0 + XY]
    lab_crop = lab[z0:z0 + DZ, y0:y0 + XY, x0:x0 + XY]
    masks, boxes = masks_boxes_from_label(lab_crop, cls_id)
    return finalize(em_crop, masks, boxes, cls_id, meta)


def extract_aug(em, lab, cx, cy, cz, angle, cls_id, meta):
    """Rotate the crop box in the xy plane around (cz,cy,cx); volume already
    validated to contain the rotated box. Zero-pad of the extraction region is
    safe (never sampled inside the valid rotated box)."""
    half = XY / 2.0
    reg = int(np.ceil(XY * np.sqrt(2) / 2)) + 4  # 181
    z0, z1 = int(round(cz)) - DZ // 2, int(round(cz)) + DZ // 2
    icy, icx = int(round(cy)), int(round(cx))
    Z, Y, X = em.shape
    em_reg = np.zeros((DZ, 2 * reg, 2 * reg), np.float32)
    lb_reg = np.zeros((DZ, 2 * reg, 2 * reg), np.uint32)
    sy0, sy1 = max(0, icy - reg), min(Y, icy + reg)
    sx0, sx1 = max(0, icx - reg), min(X, icx + reg)
    py0 = sy0 - (icy - reg)
    px0 = sx0 - (icx - reg)
    em_reg[:, py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = em[z0:z1, sy0:sy1, sx0:sx1]
    lb_reg[:, py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = lab[z0:z1, sy0:sy1, sx0:sx1]
    rot_em = np.stack([ndi.rotate(em_reg[z], angle, reshape=False, order=1,
                                  mode="constant", cval=0) for z in range(DZ)])
    rot_lb = np.stack([ndi.rotate(lb_reg[z], angle, reshape=False, order=0,
                                  mode="constant", cval=0) for z in range(DZ)])
    c0, c1 = reg - XY // 2, reg - XY // 2 + XY
    em_crop = rot_em[:, c0:c1, c0:c1]
    lb_crop = rot_lb[:, c0:c1, c0:c1]
    masks, boxes = masks_boxes_from_label(lb_crop, cls_id)
    return finalize(em_crop, masks, boxes, cls_id, meta)


def aug_valid(cx, cy, cz, angle, shape):
    """Rotated xy corners + z range inside the volume (2px margin)."""
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)
    Z, Y, X = shape
    if not (0 <= cz - DZ / 2 and cz + DZ / 2 <= Z):
        return False
    for dy in (-XY / 2, XY / 2):
        for dx in (-XY / 2, XY / 2):
            x = cx + c * dx - s * dy
            y = cy + s * dx + c * dy
            if not (2 <= x < X - 2 and 2 <= y < Y - 2):
                return False
    return True


def write_shards(items, kind, shard_start):
    paths = []
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        path = OUT_DIR / f"{kind}_{shard_start:05d}.npz"
        shard_start += 1
        masks = np.empty(len(batch), object)
        mask_labels = np.empty(len(batch), object)
        boxes = np.empty(len(batch), object)
        box_labels = np.empty(len(batch), object)
        for j, b in enumerate(batch):
            masks[j] = b["masks"]
            mask_labels[j] = b["mask_labels"]
            boxes[j] = b["boxes"]
            box_labels[j] = b["box_labels"]
        np.savez_compressed(path, image=np.stack([b["image"] for b in batch]),
                            masks=masks, mask_labels=mask_labels,
                            boxes=boxes, box_labels=box_labels,
                            meta=json.dumps([b["meta"] for b in batch]))
        paths.append(path.name)
    return shard_start, paths


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    all_items = {"type1": [], "type2": []}
    stats = {}
    for spec in VOLUMES:
        em, lab = load_volume(spec)
        Z, Y, X = em.shape
        cls_id, ds, cls = spec["cls_id"], spec["dataset"], spec["cls"]

        # ---- type1: instance-centered ----
        base = []
        objs = ndi.find_objects(lab)
        n_inst = 0
        for iid, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            n_inst += 1
            cz = 0.5 * (sl[0].start + sl[0].stop)
            cy = 0.5 * (sl[1].start + sl[1].stop)
            cx = 0.5 * (sl[2].start + sl[2].stop)
            z0 = int(np.clip(round(cz - DZ / 2), 0, Z - DZ))
            y0 = int(np.clip(round(cy - XY / 2), 0, Y - XY))
            x0 = int(np.clip(round(cx - XY / 2), 0, X - XY))
            base.append((z0, y0, x0, "type1"))
        # ---- type2: uniform random ----
        for _ in range(N_TYPE2):
            z0 = int(rng.integers(0, Z - DZ))
            y0 = int(rng.integers(0, Y - XY))
            x0 = int(rng.integers(0, X - XY))
            base.append((z0, y0, x0, "type2"))
        print(f"[{ds}] instances={n_inst} type2={N_TYPE2} base total={len(base)}", flush=True)

        n_aug_ok = 0
        for z0, y0, x0, kind in base:
            cx, cy, cz = x0 + XY / 2, y0 + XY / 2, z0 + DZ / 2
            st = stripe_of(cx, X)
            meta = dict(dataset=ds, crop=f"{cls}_s{st}", origin=[z0, y0, x0], kind=kind)
            all_items[kind].append(extract_base(em, lab, z0, y0, x0, cls_id, meta))
            # ---- aug: rotate & translate, keep if in bounds ----
            for _ in range(N_AUG):
                placed = None
                for _try in range(100):
                    ang = float(rng.uniform(0, 360))
                    ncx = cx + float(rng.uniform(-TRANS_XY, TRANS_XY))
                    ncy = cy + float(rng.uniform(-TRANS_XY, TRANS_XY))
                    ncz = cz + float(rng.integers(-TRANS_Z, TRANS_Z + 1))
                    if aug_valid(ncx, ncy, ncz, ang, em.shape):
                        placed = (ncx, ncy, ncz, ang)
                        break
                if placed is None:  # fallback: pure rotation around base center
                    for _try in range(60):
                        ang = float(rng.uniform(0, 360))
                        if aug_valid(cx, cy, cz, ang, em.shape):
                            placed = (cx, cy, cz, ang)
                            break
                if placed is None:  # fallback: shift only
                    for _try in range(60):
                        ncx = cx + float(rng.uniform(-TRANS_XY, TRANS_XY))
                        ncy = cy + float(rng.uniform(-TRANS_XY, TRANS_XY))
                        ncz = cz + float(rng.integers(-TRANS_Z, TRANS_Z + 1))
                        if aug_valid(ncx, ncy, ncz, 0.0, em.shape):
                            placed = (ncx, ncy, ncz, 0.0)
                            break
                if placed is None:
                    placed = (cx, cy, cz, 0.0)
                ncx, ncy, ncz, ang = placed
                st = stripe_of(ncx, X)
                meta = dict(dataset=ds, crop=f"{cls}_s{st}",
                            origin=[z0, y0, x0], kind=f"{kind}_aug",
                            angle=round(ang, 1),
                            trans=[round(ncx - cx, 1), round(ncy - cy, 1), round(ncz - cz, 1)])
                all_items[kind].append(extract_aug(em, lab, ncx, ncy, ncz, ang, cls_id, meta))
                n_aug_ok += 1
        stats[ds] = dict(instances=n_inst, base=len(base), aug=n_aug_ok)
        print(f"[{ds}] done base={len(base)} aug={n_aug_ok} ({time.time()-t0:.0f}s)", flush=True)
        del em, lab

    # ---- write shards ----
    shard = 0
    for kind in ("type1", "type2"):
        rng.shuffle(all_items[kind])
        shard, names = write_shards(all_items[kind], kind, shard)
        print(f"[write] {kind}: {len(all_items[kind])} crops in {len(names)} shards", flush=True)

    # ---- split.json: stripe 2 of each volume as test (2/10 = 20%) ----
    test_regions = [f"{OUT_DIR.name}/{v['dataset']}/{v['cls']}_s{TEST_STRIPE}" for v in VOLUMES]
    (OUT_DIR / "split.json").write_text(json.dumps(
        {"test_regions": test_regions, "test_frac": 0.2, "scheme": "5 xy-stripes/volume"}, indent=2))
    summary = dict(stats=stats, dz=DZ, xy=XY, xy_nm=XY * 8, z_nm=DZ * 30,
                   n_type1=len(all_items["type1"]), n_type2=len(all_items["type2"]),
                   test_regions=test_regions,
                   norm="per-volume clip(p0.5,p99.8)->[0,1] then per-crop z-score")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # ---- prep check viz ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    picks = []
    for kind in ("type1", "type2"):
        its = all_items[kind]
        step = max(1, len(its) // 6)
        picks += [(kind, its[i]) for i in range(0, len(its), step)[:6]]
    for ax, (kind, it) in zip(axes.ravel(), picks):
        z = DZ // 2
        ax.imshow(it["image"][z], cmap="gray")
        if it["masks"].shape[0]:
            m = it["masks"][:, z].any(0)
            ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.5)
        m = it["meta"]
        ax.set_title(f"{kind} {m['dataset']}/{m['crop']} {m['kind']}\n"
                     f"a={m.get('angle', 0)} n={it['masks'].shape[0]}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("flyem 1000nm crops prep check (mid-z)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "prep_check.png", dpi=110)
    print(f"[viz] {VIZ_DIR/'prep_check.png'}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
