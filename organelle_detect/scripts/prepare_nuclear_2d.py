"""Prepare 2D nucleus segmentation patches from flyEM nuclear_anno slices.

Per annotated slice (2500x2500 uint8 EM @8nm + uint16 instance labels):
  1. base crops : 50 patches of 800x800 (7x7 grid, stride auto-computed for
     overlap, +1 random, shuffled)
  2. aug crops  : 200 patches - each base box rotated (U(0,360) deg) and
     translated (U(-T,+T) px); kept only if the rotated box stays inside the
     image (no out-of-bound sampling)

Output (memmap-friendly):
  outputs/nuclear2d_patches/images.npy  (N,800,800) uint8
  outputs/nuclear2d_patches/masks.npy   (N,800,800) uint8  (binary 0/1)
  outputs/nuclear2d_patches/meta.json   per-patch provenance
  outputs/nuclear2d_patches/split.json  train/test indices (0.8/0.2, seed 42)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

ANNO_ROOT = Path("/path/to/project_data/flyEM/nuclear_anno")
OUT_DIR = Path("/path/to/JaneliaEM/outputs/nuclear2d_patches")
VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/nuclear_unet2d")

PATCH = 800          # patch size (px)
N_BASE = 50          # base crops per slice
N_AUG_PER_BASE = 4   # aug crops per base box -> 200 per slice
TRANS_RANGE = 250.0  # translation range (+/- px)
SEED = 42
MARGIN = 2           # keep rotated corners this far inside the image


def find_slices(root: Path):
    """Match slice_XX tifs with output_slice_XX label dirs."""
    tifs = {}
    for f in root.glob("slice_*.tif"):
        m = re.match(r"slice_(\d+)_", f.name)
        if m:
            tifs[int(m.group(1))] = f
    labels = {}
    for d in root.glob("output_slice_*/curation_labels.zarr"):
        m = re.match(r"output_slice_(\d+)_", d.parent.name)
        if m:
            labels[int(m.group(1))] = d
    common = sorted(set(tifs) & set(labels))
    print(f"[slices] {len(tifs)} tifs, {len(labels)} label dirs, {len(common)} paired: {common}")
    return [(i, tifs[i], labels[i]) for i in common]


def base_positions(img_size: int, patch: int, n_want: int, rng: np.random.Generator):
    """Grid positions covering the image with auto-computed overlap stride.

    n_side = ceil(sqrt(n_want)) would give 64 (8x8) for 50; use floor grid
    7x7=49 with stride=(img-patch)/(n_side-1) then +1 random -> 50, shuffled.
    """
    n_side = int(np.floor(np.sqrt(n_want)))  # 7
    lo, hi = 0, img_size - patch
    stride = (img_size - patch) / (n_side - 1)
    pos1d = np.linspace(lo, hi, n_side).round().astype(int)
    pos = [(int(y), int(x)) for y in pos1d for x in pos1d]  # 49
    while len(pos) < n_want:
        pos.append((int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))))
    rng.shuffle(pos)
    overlap = patch - stride
    print(f"[base] grid {n_side}x{n_side} stride={stride:.1f}px overlap={overlap:.1f}px "
          f"({100*overlap/patch:.0f}%) + {n_want-len(pos1d)**2} random -> {len(pos)}")
    return pos[:n_want]


def rotated_corners_ok(cx: float, cy: float, angle_deg: float, half: float,
                       img_size: int, margin: float) -> bool:
    """Check the 4 corners of the rotated sampling box stay inside the image.

    (corner set is identical for +angle/-angle so direction convention-free)
    """
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    ok = True
    for dy in (-half, half):
        for dx in (-half, half):
            rx = c * dx - s * dy
            ry = s * dx + c * dy
            x, y = cx + rx, cy + ry
            if not (margin <= x < img_size - margin and margin <= y < img_size - margin):
                ok = False
    return ok


def extract_rotated_patch(em: np.ndarray, mask: np.ndarray, cx: float, cy: float,
                          angle_deg: float, patch: int):
    """Rotate around (cx,cy) by angle, crop patch x patch. Zero-pad region is
    safe: validity check guarantees sampled box is inside the image."""
    img_size = em.shape[0]
    reg_half = int(np.ceil(patch * np.sqrt(2) / 2)) + 8  # 574 -> margin for interp
    icx, icy = int(round(cx)), int(round(cy))
    y0, y1 = icy - reg_half, icy + reg_half
    x0, x1 = icx - reg_half, icx + reg_half
    # extract with zero padding where region exceeds image
    py0, py1 = max(0, -y0), max(0, y1 - img_size)
    px0, px1 = max(0, -x0), max(0, x1 - img_size)
    reg_em = np.zeros((2 * reg_half, 2 * reg_half), dtype=em.dtype)
    reg_mk = np.zeros((2 * reg_half, 2 * reg_half), dtype=mask.dtype)
    sy0, sy1 = max(0, y0), min(img_size, y1)
    sx0, sx1 = max(0, x0), min(img_size, x1)
    reg_em[py0:2 * reg_half - py1, px0:2 * reg_half - px1] = em[sy0:sy1, sx0:sx1]
    reg_mk[py0:2 * reg_half - py1, px0:2 * reg_half - px1] = mask[sy0:sy1, sx0:sx1]
    # rotate (same transform on both); order 1 for EM, order 0 (nn) for mask
    rot_em = ndimage.rotate(reg_em, angle_deg, reshape=False, order=1, mode="constant", cval=0)
    rot_mk = ndimage.rotate(reg_mk, angle_deg, reshape=False, order=0, mode="constant", cval=0)
    h = patch // 2
    c0, c1 = reg_half - h, reg_half + h
    return rot_em[c0:c1, c0:c1], rot_mk[c0:c1, c0:c1]


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    slices = find_slices(ANNO_ROOT)
    n_per = N_BASE * (1 + N_AUG_PER_BASE)  # 250
    n_total = n_per * len(slices)
    print(f"[plan] {len(slices)} slices x {n_per} patches = {n_total}")

    img_mm = np.lib.format.open_memmap(OUT_DIR / "images.npy", mode="w+",
                                       dtype=np.uint8, shape=(n_total, PATCH, PATCH))
    msk_mm = np.lib.format.open_memmap(OUT_DIR / "masks.npy", mode="w+",
                                       dtype=np.uint8, shape=(n_total, PATCH, PATCH))
    meta = []
    idx = 0
    for sid, tif_p, zarr_p in slices:
        em = tifffile.imread(str(tif_p)).squeeze()
        lab = zarr.open(str(zarr_p), mode="r")[:].squeeze()
        assert em.shape == (2500, 2500) and lab.shape == (2500, 2500), (em.shape, lab.shape)
        mask = (lab > 0).astype(np.uint8)
        n_inst = len(np.unique(lab)) - 1
        fg = float(mask.mean())

        positions = base_positions(em.shape[0], PATCH, N_BASE, rng)
        n_base_written = n_aug_written = 0
        for bi, (y, x) in enumerate(positions):
            cx, cy = x + PATCH / 2, y + PATCH / 2
            # base crop (axis-aligned)
            img_mm[idx] = em[y:y + PATCH, x:x + PATCH]
            msk_mm[idx] = mask[y:y + PATCH, x:x + PATCH]
            meta.append(dict(slice=sid, kind="base", cx=cx, cy=cy, angle=0.0))
            idx += 1
            n_base_written += 1
            # aug crops: rotate & translate the base box
            for _ in range(N_AUG_PER_BASE):
                got = False
                for _try in range(100):
                    ang = float(rng.uniform(0.0, 360.0))
                    dx = float(rng.uniform(-TRANS_RANGE, TRANS_RANGE))
                    dy = float(rng.uniform(-TRANS_RANGE, TRANS_RANGE))
                    ncx, ncy = cx + dx, cy + dy
                    if rotated_corners_ok(ncx, ncy, ang, PATCH / 2, em.shape[0], MARGIN):
                        got = True
                        break
                if not got:  # fall back: pure rotation around base center
                    for _try in range(100):
                        ang = float(rng.uniform(0.0, 360.0))
                        ncx, ncy = cx, cy
                        if rotated_corners_ok(ncx, ncy, ang, PATCH / 2, em.shape[0], MARGIN):
                            got = True
                            break
                assert got, f"no valid aug for slice {sid} base {bi}"
                p_em, p_mk = extract_rotated_patch(em, mask, ncx, ncy, ang, PATCH)
                img_mm[idx] = p_em
                msk_mm[idx] = p_mk
                meta.append(dict(slice=sid, kind="aug", cx=ncx, cy=ncy, angle=ang))
                idx += 1
                n_aug_written += 1
        print(f"[slice {sid:02d}] instances={n_inst:3d} fg={fg:.3f} "
              f"base={n_base_written} aug={n_aug_written} ({time.time()-t0:.0f}s)")

    assert idx == n_total, (idx, n_total)
    img_mm.flush()
    msk_mm.flush()

    # split 0.8 / 0.2
    perm = rng.permutation(n_total).tolist()
    n_train = int(round(0.8 * n_total))
    split = dict(train=sorted(perm[:n_train]), test=sorted(perm[n_train:]),
                 seed=SEED, n_total=n_total)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta))
    (OUT_DIR / "split.json").write_text(json.dumps(split))
    print(f"[done] {n_total} patches -> {OUT_DIR} "
          f"(train={len(split['train'])} test={len(split['test'])}) in {time.time()-t0:.0f}s")

    # quick sanity viz
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    show = [0, 1, 2, 250, 251, 252, 1000, 1001, 2000, 2001, 3000, 3001]
    for ax, i in zip(axes.ravel(), show):
        ax.imshow(img_mm[i], cmap="gray")
        ax.imshow(np.ma.masked_where(msk_mm[i] == 0, msk_mm[i]), cmap="autumn", alpha=0.45)
        m = meta[i]
        ax.set_title(f"#{i} s{m['slice']:02d} {m['kind']} a={m['angle']:.0f}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("prep check: EM + nucleus mask overlay")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "prep_check.png", dpi=110)
    print(f"[viz] {VIZ_DIR/'prep_check.png'}")


if __name__ == "__main__":
    sys.exit(main())
