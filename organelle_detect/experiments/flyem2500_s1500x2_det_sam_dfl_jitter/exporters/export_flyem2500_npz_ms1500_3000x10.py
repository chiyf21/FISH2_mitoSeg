"""Export flyEM fish2 2500^2x250 (mito + lyso) to multi-scale .npz crops for
train_npz.py (det_sam, dim 3d).

Two physical scales, same tensor shape 10 x 188 x 188:
  s1500  native 8 nm/px:  188 px  = 1504 nm FOV
  s3000  2x xy downsample: 376 native px -> 188 px @16 nm = 3008 nm FOV
z is never resampled (10 slices x 30 nm).

Sampling per scale (2x the 1500x1500x10 export):
  type1  one per instance (mito and lyso), centered on instance bbox center,
         + N_AUG_TYPE1 augmented copies
  type2  N_TYPE2 uniform random windows + N_AUG_TYPE2 augmented copies
  aug    xy-plane rotation U(0,360) + translation (xy +/-60 px at 1500 scale,
         scaled with FOV; z +/-8 slices); kept only if the rotated box stays
         fully inside the volume (fallback: pure rotation -> pure shift -> copy)
Rotation is done at native resolution, then the crop is downsampled.

Normalization: per-volume clip(p0.5, p99.8) -> [0,1], then per-crop z-score.
Split: 5 vertical xy stripes by crop center; stripe 2 is test for both scales.
Class ids follow organelle_classes: mito=2, lyso=3.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

FLYEM = Path("/path/to/project_data/flyEM")
OUT_DIR = Path("/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_ms1500_3000x10")
VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/flyem2500_ms1500_3000x10")

DZ, XY = 10, 188
SCALES = [
    dict(name="s1500", ds=1),
    dict(name="s3000", ds=2),
]
N_TYPE2 = 1000
N_AUG_TYPE1 = 17         # 1 base + 17 aug = 2x (1 + 8)
N_AUG_TYPE2 = 8
TRANS_XY = 60.0          # +/- output px; native shift = TRANS_XY * ds
TRANS_Z = 8
BATCH = 16
N_STRIPES = 5
TEST_STRIPE = 2
SEED = 42
N_PROC = 48

MITO_ID, LYSO_ID = 2, 3
DATASET = "fish2_2500"
EM_PATH = FLYEM / "2500x2500x250/raw.ome.tif"
LABELS = [
    (MITO_ID, FLYEM / "2500x2500x250/annotation/mitochondria_0922.zarr"),
    (LYSO_ID, FLYEM / "2500x2500x250/annotation/lysosome_instances_v0924.zarr"),
]

# fork-pool worker globals (copy-on-write, set before Pool creation)
_EM = None
_LABS = None  # list of (cls_id, uint16 volume)


def load_inputs():
    em = tifffile.imread(str(EM_PATH)).astype(np.float32)
    sub = em[::4, ::4, ::4]
    lo, hi = np.percentile(sub, [0.5, 99.8])
    em = np.clip((em - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    labs = []
    for cid, p in LABELS:
        lab = np.asarray(zarr.open(str(p), mode="r"))
        assert lab.shape == em.shape, (p, lab.shape, em.shape)
        labs.append((cid, lab))
    print(f"[{DATASET}] shape={em.shape} gray p0.5={lo:.1f} p99.8={hi:.1f} -> [0,1]", flush=True)
    return em, labs


def stripe_of(cx: float, width: int) -> int:
    return min(N_STRIPES - 1, int(cx / (width / N_STRIPES)))


def downsample(em_c: np.ndarray, lab_cs: list[np.ndarray], ds: int):
    if ds == 1:
        return em_c, lab_cs
    em_d = em_c.reshape(DZ, XY, ds, XY, ds).mean(axis=(2, 4))
    lab_d = [l[:, ds // 2::ds, ds // 2::ds] for l in lab_cs]
    return em_d, lab_d


def masks_boxes(lab_crop: np.ndarray, cls_id: int, min_vox: int = 8):
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


def finalize(em_crop, lab_crops, meta):
    img = (em_crop - em_crop.mean()) / (em_crop.std() + 1e-6)
    masks, boxes, labels = [], [], []
    for (cid, _), lc in zip(_LABS, lab_crops):
        m, b = masks_boxes(lc, cid)
        masks += m
        boxes += b
        labels += [cid] * len(m)
    n = len(masks)
    return {
        "image": img.astype(np.float32),
        "masks": np.stack(masks).astype(bool) if n else np.zeros((0, DZ, XY, XY), bool),
        "mask_labels": np.asarray(labels, np.int64),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 6),
        "box_labels": np.asarray(labels, np.int64),
        "meta": meta,
    }


def extract_base(z0, y0, x0, ds, meta):
    n = XY * ds
    em_c = _EM[z0:z0 + DZ, y0:y0 + n, x0:x0 + n]
    lab_c = [lab[z0:z0 + DZ, y0:y0 + n, x0:x0 + n] for _, lab in _LABS]
    em_c, lab_c = downsample(em_c, lab_c, ds)
    return finalize(em_c, lab_c, meta)


def extract_aug(cx, cy, cz, angle, ds, meta):
    """Rotate the native crop box in xy around (cz,cy,cx), then downsample.
    The volume is already validated to contain the rotated box, so zero-padding
    of the extraction region is never sampled."""
    n = XY * ds
    reg = int(np.ceil(n * np.sqrt(2) / 2)) + 4
    z0, z1 = int(round(cz)) - DZ // 2, int(round(cz)) + DZ // 2
    icy, icx = int(round(cy)), int(round(cx))
    Z, Y, X = _EM.shape
    sy0, sy1 = max(0, icy - reg), min(Y, icy + reg)
    sx0, sx1 = max(0, icx - reg), min(X, icx + reg)
    py0, px0 = sy0 - (icy - reg), sx0 - (icx - reg)
    c0, c1 = reg - n // 2, reg - n // 2 + n

    def rot(vol, order, dtype):
        r = np.zeros((DZ, 2 * reg, 2 * reg), dtype)
        r[:, py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = vol[z0:z1, sy0:sy1, sx0:sx1]
        out = np.stack([ndi.rotate(r[z], angle, reshape=False, order=order, mode="constant", cval=0)
                        for z in range(DZ)])
        return out[:, c0:c1, c0:c1]

    em_c = rot(_EM, 1, np.float32)
    lab_c = [rot(lab, 0, lab.dtype) for _, lab in _LABS]
    em_c, lab_c = downsample(em_c, lab_c, ds)
    return finalize(em_c, lab_c, meta)


def aug_valid(cx, cy, cz, angle, shape, n):
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)
    Z, Y, X = shape
    if not (0 <= cz - DZ / 2 and cz + DZ / 2 <= Z):
        return False
    for dy in (-n / 2, n / 2):
        for dx in (-n / 2, n / 2):
            x = cx + c * dx - s * dy
            y = cy + s * dx + c * dy
            if not (2 <= x < X - 2 and 2 <= y < Y - 2):
                return False
    return True


def plan_aug(rng, cx, cy, cz, shape, n, txy):
    placed = None
    for _ in range(100):
        ang = float(rng.uniform(0, 360))
        ncx = cx + float(rng.uniform(-txy, txy))
        ncy = cy + float(rng.uniform(-txy, txy))
        ncz = cz + float(rng.integers(-TRANS_Z, TRANS_Z + 1))
        if aug_valid(ncx, ncy, ncz, ang, shape, n):
            return ncx, ncy, ncz, ang
    for _ in range(60):
        ang = float(rng.uniform(0, 360))
        if aug_valid(cx, cy, cz, ang, shape, n):
            return cx, cy, cz, ang
    for _ in range(60):
        ncx = cx + float(rng.uniform(-txy, txy))
        ncy = cy + float(rng.uniform(-txy, txy))
        ncz = cz + float(rng.integers(-TRANS_Z, TRANS_Z + 1))
        if aug_valid(ncx, ncy, ncz, 0.0, shape, n):
            return ncx, ncy, ncz, 0.0
    return placed or (cx, cy, cz, 0.0)


def _do_task(task):
    if task[0] == "base":
        _, z0, y0, x0, ds, meta = task
        return extract_base(z0, y0, x0, ds, meta)
    _, cx, cy, cz, ang, ds, meta = task
    return extract_aug(cx, cy, cz, ang, ds, meta)


def write_shards(items, kind, shard_start):
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        path = OUT_DIR / f"{kind}_{shard_start:05d}.npz"
        shard_start += 1
        obj = {k: np.empty(len(batch), object) for k in ("masks", "mask_labels", "boxes", "box_labels")}
        for j, b in enumerate(batch):
            for k in obj:
                obj[k][j] = b[k]
        np.savez(path, image=np.stack([b["image"] for b in batch]), **obj,
                 meta=json.dumps([b["meta"] for b in batch]))
    return shard_start


def main():
    global _EM, _LABS
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    _EM, _LABS = load_inputs()
    Z, Y, X = _EM.shape

    centers = []
    for cid, lab in _LABS:
        for sl in ndi.find_objects(lab):
            if sl is None:
                continue
            centers.append((0.5 * (sl[0].start + sl[0].stop),
                            0.5 * (sl[1].start + sl[1].stop),
                            0.5 * (sl[2].start + sl[2].stop), cid))
    n_mito = sum(c[3] == MITO_ID for c in centers)
    print(f"[{DATASET}] instances mito={n_mito} lyso={len(centers) - n_mito}", flush=True)

    ctx = mp.get_context("fork")
    shard = {"type1": 0, "type2": 0}
    stats, viz_items = {}, []
    for sc in SCALES:
        name, ds = sc["name"], sc["ds"]
        n = XY * ds
        base = []
        for cz, cy, cx, _cid in centers:
            z0 = int(np.clip(round(cz - DZ / 2), 0, Z - DZ))
            y0 = int(np.clip(round(cy - n / 2), 0, Y - n))
            x0 = int(np.clip(round(cx - n / 2), 0, X - n))
            base.append((z0, y0, x0, "type1"))
        for _ in range(N_TYPE2):
            base.append((int(rng.integers(0, Z - DZ)), int(rng.integers(0, Y - n)),
                         int(rng.integers(0, X - n)), "type2"))

        tasks = []
        for z0, y0, x0, kind in base:
            cx, cy, cz = x0 + n / 2, y0 + n / 2, z0 + DZ / 2
            common = dict(dataset=DATASET, scale=name, ds=ds, xy_nm_per_px=8 * ds)
            meta = dict(common, crop=f"{name}_s{stripe_of(cx, X)}", origin=[z0, y0, x0], kind=kind)
            tasks.append(("base", z0, y0, x0, ds, meta))
            for _ in range(N_AUG_TYPE1 if kind == "type1" else N_AUG_TYPE2):
                ncx, ncy, ncz, ang = plan_aug(rng, cx, cy, cz, _EM.shape, n, TRANS_XY * ds)
                meta = dict(common, crop=f"{name}_s{stripe_of(ncx, X)}", origin=[z0, y0, x0],
                            kind=f"{kind}_aug", angle=round(ang, 1),
                            trans=[round(ncx - cx, 1), round(ncy - cy, 1), round(ncz - cz, 1)])
                tasks.append(("aug", ncx, ncy, ncz, ang, ds, meta))
        print(f"[{name}] base={len(base)} tasks={len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

        with ctx.Pool(N_PROC) as pool:
            results = pool.map(_do_task, tasks, chunksize=8)
        by_kind = {"type1": [], "type2": []}
        for it in results:
            by_kind["type1" if it["meta"]["kind"].startswith("type1") else "type2"].append(it)
        n_inst = {"mito": 0, "lyso": 0}
        for it in results:
            n_inst["mito"] += int((it["mask_labels"] == MITO_ID).sum())
            n_inst["lyso"] += int((it["mask_labels"] == LYSO_ID).sum())
        for kind in ("type1", "type2"):
            rng.shuffle(by_kind[kind])
            shard[kind] = write_shards(by_kind[kind], kind, shard[kind])
            viz_items += [(name, kind, it) for it in by_kind[kind][:3]]
        stats[name] = dict(ds=ds, fov_nm=XY * 8 * ds, n_type1=len(by_kind["type1"]),
                           n_type2=len(by_kind["type2"]), mask_rows=n_inst)
        print(f"[{name}] written {stats[name]} ({time.time()-t0:.0f}s)", flush=True)
        del results, by_kind

    test_regions = [f"{OUT_DIR.name}/{DATASET}/{sc['name']}_s{TEST_STRIPE}" for sc in SCALES]
    (OUT_DIR / "split.json").write_text(json.dumps(
        {"test_regions": test_regions, "test_frac": 0.2, "scheme": "5 xy-stripes by crop center, per scale"},
        indent=2))
    (OUT_DIR / "class_supervision.json").write_text(json.dumps(
        {"comment": "fish2_2500 has both mito (mitochondria_0922) and lyso (lysosome_instances_v0924)",
         DATASET: [MITO_ID, LYSO_ID]}, indent=2))
    (OUT_DIR / "cross_neg.json").write_text(json.dumps({"enabled": False}))
    summary = dict(stats=stats, dz=DZ, xy=XY, z_nm=DZ * 30, test_regions=test_regions,
                   norm="per-volume clip(p0.5,p99.8)->[0,1], per-crop z-score; s3000 = 2x2 mean / center-pixel labels")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    colors = {MITO_ID: "autumn", LYSO_ID: "winter"}
    for ax, (name, kind, it) in zip(axes.ravel(), viz_items):
        z = DZ // 2
        ax.imshow(it["image"][z], cmap="gray")
        for cid, cmap in colors.items():
            sel = it["mask_labels"] == cid
            if sel.any():
                m = it["masks"][sel][:, z].any(0)
                ax.imshow(np.ma.masked_where(~m, m), cmap=cmap, alpha=0.45)
        ax.set_title(f"{name} {it['meta']['kind']} a={it['meta'].get('angle', 0)}\n"
                     f"mito={int((it['mask_labels'] == MITO_ID).sum())} "
                     f"lyso={int((it['mask_labels'] == LYSO_ID).sum())}", fontsize=8)
        ax.axis("off")
    fig.suptitle("flyem2500 multi-scale crops (top: 1500nm @8nm, bottom: 3000nm @16nm), mid-z; red=mito blue=lyso")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "prep_check.png", dpi=110)
    print(f"[viz] {VIZ_DIR / 'prep_check.png'}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
