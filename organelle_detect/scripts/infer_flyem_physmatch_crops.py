#!/usr/bin/env python3
"""Crop flyEM windows at v6 training physical FOV, infer with m2f_bin, visualize.

1) Measure type2 crop FOV in nm from CellMap s0 scales × (6,256,256).
2) On Fish2 raw.ome.tif (8×8×30 nm XYZ → 30×8×8 ZYX), crop windows matching
   those physical sizes, then resize to model input 6×256×256 (type1-style).
3) Run m2f_bin_mixed final ckpt and save a montage.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.cellmap_io import em_group, read_multiscale_transform  # noqa: E402
from scripts.viz_val import CLASS_COLORS, CLASS_NAMES, draw_masks, load_model, m2f_pred  # noqa: E402

NPZ_V6 = ROOT.parent / "outputs" / "npz_crops_v6"
CKPT = ROOT.parent / "outputs" / "train_m2f_bin_mixed" / "last_m2f_bin.pt"
FLY_TIF = Path("/path/to/project_data/flyEM/raw.ome.tif")
# Fish2 documented voxel size (XYZ nm) → ZYX
FLY_ZYX_NM = (30.0, 8.0, 8.0)
OUT_DIR = ROOT.parent / "outputs" / "logs" / "flyem_physmatch"
MODEL_Z, MODEL_XY = 6, 256
N_CROPS_PER_FOV = 4
SEED = 0


def v6_fov_report() -> dict:
    scales: dict[str, tuple[float, float, float]] = {}
    crop_by_scale: Counter = Counter()
    for f in sorted(NPZ_V6.glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            for m in json.loads(str(z["meta"])):
                ds = m["dataset"]
                if ds not in scales:
                    scales[ds] = tuple(
                        float(x) for x in read_multiscale_transform(em_group(ds), "s0").scale
                    )
                crop_by_scale[scales[ds]] += 1
    tot = sum(crop_by_scale.values())
    rows = []
    for sc, n in crop_by_scale.most_common():
        sz, sy, sx = sc
        fov = (MODEL_Z * sz, MODEL_XY * sy, MODEL_XY * sx)
        rows.append(
            {
                "voxel_zyx_nm": list(sc),
                "n_crops": n,
                "frac": n / tot,
                "type2_fov_zyx_nm": list(fov),
            }
        )
    w_fov = [
        sum(r["type2_fov_zyx_nm"][i] * r["n_crops"] for r in rows) / tot for i in range(3)
    ]
    return {
        "layers": MODEL_Z,
        "out_size": MODEL_XY,
        "note": "type2 FOV = native (no xy resize); type1 same tensor shape but FOV varies with union-box before resize",
        "by_scale": rows,
        "weighted_mean_fov_zyx_nm": w_fov,
        "mode_fov_zyx_nm": rows[0]["type2_fov_zyx_nm"],
    }


def fly_voxels_for_fov(fov_zyx_nm: tuple[float, float, float]) -> tuple[int, int, int]:
    """Map physical FOV to Fish2 voxel counts (at least 1³)."""
    cz = max(1, int(round(fov_zyx_nm[0] / FLY_ZYX_NM[0])))
    cy = max(1, int(round(fov_zyx_nm[1] / FLY_ZYX_NM[1])))
    cx = max(1, int(round(fov_zyx_nm[2] / FLY_ZYX_NM[2])))
    return cz, cy, cx


def resize_to_model(crop: np.ndarray) -> np.ndarray:
    """crop [Dz,Dy,Dx] → [6,256,256] float32 (trilinear-ish via zoom)."""
    dz, dy, dx = crop.shape
    factors = (MODEL_Z / dz, MODEL_XY / dy, MODEL_XY / dx)
    out = zoom(crop.astype(np.float32), factors, order=1)
    # zoom can be off by 1 due to rounding
    out = out[:MODEL_Z, :MODEL_XY, :MODEL_XY]
    if out.shape != (MODEL_Z, MODEL_XY, MODEL_XY):
        pad = [(0, MODEL_Z - out.shape[0]), (0, MODEL_XY - out.shape[1]), (0, MODEL_XY - out.shape[2])]
        out = np.pad(out, pad, mode="edge")[:MODEL_Z, :MODEL_XY, :MODEL_XY]
    return out


def normalize(crop: np.ndarray) -> np.ndarray:
    x = crop.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def sample_origins(vol_shape, cz, cy, cx, n, rng) -> list[tuple[int, int, int]]:
    Z, Y, X = vol_shape
    if Z < cz or Y < cy or X < cx:
        raise ValueError(f"volume {vol_shape} too small for crop {(cz, cy, cx)}")
    origins = []
    for _ in range(n * 40):
        z0 = int(rng.integers(0, Z - cz + 1))
        y0 = int(rng.integers(0, Y - cy + 1))
        x0 = int(rng.integers(0, X - cx + 1))
        origins.append((z0, y0, x0))
        if len(origins) >= n * 8:
            break
    # keep diverse + non-blank
    return origins


def pick_nonblank(vol, cz, cy, cx, candidates, n) -> list[tuple[int, int, int, np.ndarray]]:
    picked = []
    for z0, y0, x0 in candidates:
        crop = vol[z0 : z0 + cz, y0 : y0 + cy, x0 : x0 + cx]
        if float(crop.std()) < 5.0:
            continue
        picked.append((z0, y0, x0, crop))
        if len(picked) >= n:
            break
    return picked


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = v6_fov_report()
    (OUT_DIR / "v6_crop_fov_nm.json").write_text(json.dumps(report, indent=2))
    print("=== v6 type2 crop FOV (nm) ===", flush=True)
    for r in report["by_scale"]:
        fov = r["type2_fov_zyx_nm"]
        print(
            f"  voxel {r['voxel_zyx_nm']}: FOV Z/Y/X = {fov[0]:.1f}/{fov[1]:.1f}/{fov[2]:.1f} nm "
            f"({r['frac']*100:.1f}% crops)",
            flush=True,
        )
    print(f"  weighted mean FOV nm: {report['weighted_mean_fov_zyx_nm']}", flush=True)

    # Representative FOVs to crop on flyEM
    fov_targets = [
        ("mode_2nm", tuple(report["mode_fov_zyx_nm"])),  # 12×512×512
        ("cellmap_4nm", (24.0, 1024.0, 1024.0)),
        ("cellmap_8nm", (48.0, 2048.0, 2048.0)),
        ("weighted_mean", tuple(report["weighted_mean_fov_zyx_nm"])),
    ]

    print(f"\nLoading flyEM {FLY_TIF} ...", flush=True)
    vol = np.asarray(tifffile.imread(str(FLY_TIF)))  # ZYX
    print(f"flyEM shape ZYX={vol.shape} voxel_nm_zyx={FLY_ZYX_NM}", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading ckpt {CKPT} on {device}", flush=True)
    model = load_model("m2f_bin", CKPT)
    ep = torch.load(CKPT, map_location="cpu").get("epoch", "?")

    rng = np.random.default_rng(SEED)
    all_rows = []  # for viz: (title, img6, preds)
    crop_meta = []

    for name, fov in fov_targets:
        cz, cy, cx = fly_voxels_for_fov(fov)
        actual_fov = (cz * FLY_ZYX_NM[0], cy * FLY_ZYX_NM[1], cx * FLY_ZYX_NM[2])
        print(
            f"\n[{name}] target FOV nm={tuple(round(x,1) for x in fov)} "
            f"→ fly voxels ZYX=({cz},{cy},{cx}) actual FOV nm={tuple(round(x,1) for x in actual_fov)}",
            flush=True,
        )
        cands = sample_origins(vol.shape, cz, cy, cx, N_CROPS_PER_FOV, rng)
        rng.shuffle(cands)
        picked = pick_nonblank(vol, cz, cy, cx, cands, N_CROPS_PER_FOV)
        if len(picked) < N_CROPS_PER_FOV:
            # fallback denser grid
            Z, Y, X = vol.shape
            grid = [
                (z, y, x)
                for z in range(0, Z - cz + 1, max(1, (Z - cz) // 4 or 1))
                for y in range(0, Y - cy + 1, max(1, (Y - cy) // 4 or 1))
                for x in range(0, X - cx + 1, max(1, (X - cx) // 4 or 1))
            ]
            rng.shuffle(grid)
            picked = pick_nonblank(vol, cz, cy, cx, grid, N_CROPS_PER_FOV)
        print(f"  picked {len(picked)} non-blank crops", flush=True)

        for i, (z0, y0, x0, raw) in enumerate(picked):
            resized = resize_to_model(raw)
            normed = normalize(resized)
            img_t = torch.from_numpy(normed)[None]  # [1,6,S,S] expected by m2f_pred via [1,6,S,S] then [None]→ wait
            # viz_val.m2f_pred expects img [1,6,S,S] OR [6,S,S]? It does img[None] so wants [C?,6,S,S]
            # dataset returns image [1,6,S,S]. Here pass [1,6,S,S]
            img_in = torch.from_numpy(normed)[None]  # [1,6,256,256]
            preds = m2f_pred(model, img_in)
            title = f"{name}#{i} origin=({z0},{y0},{x0}) fly={cz}x{cy}x{cx}"
            all_rows.append((title, normed, preds, actual_fov))
            crop_meta.append(
                {
                    "fov_name": name,
                    "target_fov_zyx_nm": list(fov),
                    "fly_voxels_zyx": [cz, cy, cx],
                    "actual_fov_zyx_nm": list(actual_fov),
                    "origin_zyx": [z0, y0, x0],
                    "n_pred": len(preds),
                    "pred_classes": sorted({CLASS_NAMES[c] for c, _, _ in preds}),
                }
            )

    (OUT_DIR / "crop_meta.json").write_text(json.dumps(crop_meta, indent=2))

    # Visualize: rows × (raw | pred)
    n = len(all_rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n))
    if n == 1:
        axes = np.array([axes])
    for r, (title, img, preds, fov) in enumerate(all_rows):
        # best z by pred mass else mid
        if preds:
            tot = np.stack([m.sum(axis=(1, 2)) for _, m, _ in preds]).sum(0)
            z = int(np.argmax(tot)) if tot.sum() > 0 else MODEL_Z // 2
        else:
            z = MODEL_Z // 2
        img2d = img[z]
        axes[r, 0].imshow(img2d, cmap="gray")
        axes[r, 0].set_title(f"{title}\nFOV≈{fov[0]:.0f}×{fov[1]:.0f}×{fov[2]:.0f}nm →6×256", fontsize=8)
        axes[r, 0].axis("off")
        draw_masks(axes[r, 1], img2d, preds, z)
        axes[r, 1].set_title(f"m2f_bin e{ep} pred ({len(preds)} objs) z={z}", fontsize=8)
        axes[r, 1].axis("off")
    # legend
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS[c], lw=6, label=CLASS_NAMES[c]) for c in sorted(CLASS_NAMES)
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(
        f"flyEM phys-matched crops | Fish2 {FLY_ZYX_NM} nm/vox ZYX | ckpt {CKPT.name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    out_png = OUT_DIR / "viz_flyem_physmatch_m2f_bin.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nsaved {out_png}", flush=True)
    print(f"report {OUT_DIR / 'v6_crop_fov_nm.json'}", flush=True)
    print(f"meta   {OUT_DIR / 'crop_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
