#!/usr/bin/env python3
"""Link per-slice 2D instance masks along z into 3D instances, then filter by 3D size.

Input: a (Z, Y, X) volume where every z-slice holds 2D instance ids (each mask a connected
component in its slice; ids may repeat between slices), plus the class of each voxel, either
  --inst INST --sem SEM           one 2D instance volume + a semantic label volume, or
  --class-inst mito=PATH ...      one 2D instance volume per class.
Two masks of the same class in adjacent slices are connected when they share at least
--min-overlap pixels (z-direct overlap; with --z-diag, contacts through the 8 in-plane
neighbours of the next slice count as well). Connected masks are merged with sparse connected components (union over the
whole stack at once), and 3D instances smaller than the per-class threshold are removed.

Volumes may be .zarr, .tif/.tiff or .npy. Outputs (in --out):
  {prefix}_{cls}_instance3d.zarr   uint32, 0 = background, consecutive ids per class
  {prefix}_instance3d.zarr         all classes in one label volume (ids offset per class)
  {prefix}_instances3d.json        id, class, voxels, z range, number of 2D masks
  {prefix}_summary.json            counts and timings
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CLASS_IDS = {"mito": 2, "lyso": 3}
CHUNKS = (10, 256, 256)
THREADS = 16


def load(path: Path) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p, mmap_mode="r")
    if p.endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(p)
    import zarr
    return np.asarray(zarr.open(p, mode="r"))


def save_zarr(path: Path, arr: np.ndarray, attrs: dict | None = None):
    import zarr
    from numcodecs import Blosc
    z = zarr.open_array(str(path), mode="w", shape=arr.shape, chunks=CHUNKS, dtype=arr.dtype,
                        compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE))
    for z0 in range(0, arr.shape[0], CHUNKS[0]):
        z[z0:z0 + CHUNKS[0]] = arr[z0:z0 + CHUNKS[0]]
    if attrs:
        z.attrs.update(attrs)


def globalize(inst2d: np.ndarray, keep: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Per-slice ids -> globally unique ids (slice offset); voxels outside `keep` -> 0."""
    Z = inst2d.shape[0]
    out = np.empty(inst2d.shape, np.int64)

    def one(z):
        s = np.asarray(inst2d[z]).astype(np.int64)
        if keep is not None:
            s[~keep[z]] = 0
        out[z] = s
        return int(s.max())

    with ThreadPoolExecutor(THREADS) as ex:
        mx = list(ex.map(one, range(Z)))
    off = np.concatenate([[0], np.cumsum(mx)[:-1]])

    def shift(z):
        s = out[z]
        np.add(s, off[z], out=s, where=s > 0)

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(shift, range(Z)))
    return out, int(sum(mx))


def z_pairs(g: np.ndarray, z: int, diag: bool, min_overlap: int) -> np.ndarray:
    """Connected (id_z, id_z+1) pairs between slice z and z+1 -> [K, 2] (ids < 2^32).
    Contact count = overlapping pixels (summed over the 9 in-plane offsets when diag)."""
    a, b = g[z], g[z + 1]
    shifts = [(0, 0)] + ([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)] if diag else [])
    Y, X = a.shape
    keys = []
    for dy, dx in shifts:
        ya, yb = slice(max(0, -dy), Y - max(0, dy)), slice(max(0, dy), Y - max(0, -dy))
        xa, xb = slice(max(0, -dx), X - max(0, dx)), slice(max(0, dx), X - max(0, -dx))
        aa, bb = a[ya, xa], b[yb, xb]
        m = (aa > 0) & (bb > 0)
        if m.any():
            keys.append((aa[m] << 32) | bb[m])
    if not keys:
        return np.zeros((0, 2), np.int64)
    u, c = np.unique(np.concatenate(keys), return_counts=True)
    u = u[c >= min_overlap]
    return np.stack([u >> 32, u & 0xFFFFFFFF], 1)


def link_class(inst2d, keep, min_vox, diag, min_overlap, name):
    t0 = time.time()
    g, n = globalize(inst2d, keep)
    t_glob = time.time() - t0
    Z = g.shape[0]
    with ThreadPoolExecutor(THREADS) as ex:
        pairs = list(ex.map(lambda z: z_pairs(g, z, diag, min_overlap), range(Z - 1)))
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
    t_pairs = time.time() - t0 - t_glob
    area = np.bincount(g.ravel(), minlength=n + 1)
    present = area > 0
    present[0] = False
    adj = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n + 1, n + 1))
    _, comp = connected_components(adj, directed=False)
    # components of the 2D masks that exist; size = sum of member areas
    ids2d = np.flatnonzero(present)
    vox = np.bincount(comp[ids2d], weights=area[ids2d])
    nmask = np.bincount(comp[ids2d])
    roots = np.unique(comp[ids2d])
    kept = roots[vox[roots] >= min_vox]
    lut_comp = np.zeros(comp.max() + 1, np.uint32)
    lut_comp[kept] = np.arange(1, len(kept) + 1, dtype=np.uint32)
    lut = lut_comp[comp]
    lut[0] = 0
    out = np.empty(g.shape, np.uint32)

    def write(z):
        out[z] = lut[g[z]]

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(write, range(Z)))
    del g
    t_total = time.time() - t0
    zr = ndi.find_objects(out)
    table = []
    for k, root in enumerate(kept, start=1):
        sl = zr[k - 1]
        table.append(dict(id=k, cls=name, voxels=int(vox[root]), n_masks2d=int(nmask[root]),
                          z0=int(sl[0].start), z1=int(sl[0].stop)))
    stats = dict(n_masks2d=int(present.sum()), n_links=int(len(pairs)), n_instances3d=int(len(roots)),
                 n_kept=int(len(kept)), min_voxels=int(min_vox),
                 seconds=dict(globalize=round(t_glob, 1), pairs=round(t_pairs, 1), total=round(t_total, 1)))
    print(f"[{name}] 2D masks={stats['n_masks2d']} links={stats['n_links']} -> 3D instances={len(roots)} "
          f"-> kept (>= {min_vox} vox)={len(kept)}  ({t_total:.1f}s)", flush=True)
    return out, table, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inst", type=Path, help="2D instance volume (all classes)")
    ap.add_argument("--sem", type=Path, help="semantic label volume giving each voxel's class id")
    ap.add_argument("--class-inst", nargs="+", default=[], metavar="CLS=PATH",
                    help="per-class 2D instance volumes, e.g. mito=a.zarr lyso=b.zarr")
    ap.add_argument("--min-mito", type=int, default=5000, help="min 3D voxels of a mito instance")
    ap.add_argument("--min-lyso", type=int, default=2000, help="min 3D voxels of a lyso instance")
    ap.add_argument("--min-overlap", type=int, default=1,
                    help="pixels two masks must share between adjacent slices to be connected")
    ap.add_argument("--z-diag", action="store_true",
                    help="also connect masks touching diagonally between slices (26-connectivity)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="linked")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    min_vox = {"mito": args.min_mito, "lyso": args.min_lyso}
    t0 = time.time()

    jobs = []
    if args.class_inst:
        for spec in args.class_inst:
            name, p = spec.split("=", 1)
            jobs.append((name, load(Path(p)), None))
    else:
        if args.inst is None or args.sem is None:
            ap.error("give --inst and --sem, or --class-inst")
        inst, sem = load(args.inst), load(args.sem)
        for name, cid in CLASS_IDS.items():
            jobs.append((name, inst, np.asarray(sem) == cid))
    t_load = time.time() - t0

    combined, tables, stats, offset = None, [], {}, 0
    for name, inst2d, keep in jobs:
        out, table, st = link_class(inst2d, keep, min_vox.get(name, 0), args.z_diag, args.min_overlap, name)
        save_zarr(args.out / f"{args.prefix}_{name}_instance3d.zarr", out,
                  dict(cls=name, min_voxels=min_vox.get(name, 0), z_diag=args.z_diag, min_overlap=args.min_overlap))
        if combined is None:
            combined = np.zeros(out.shape, np.uint32)
        np.add(combined, out + np.uint32(offset), out=combined, where=out > 0)
        for t in table:
            tables.append(dict(t, id=t["id"] + offset, class_local_id=t["id"]))
        offset += len(table)
        stats[name] = st
        del out
    save_zarr(args.out / f"{args.prefix}_instance3d.zarr", combined)
    (args.out / f"{args.prefix}_instances3d.json").write_text(json.dumps(tables, indent=1))
    summary = dict(inputs=dict(inst=str(args.inst), sem=str(args.sem), class_inst=args.class_inst),
                   params=dict(min_mito=args.min_mito, min_lyso=args.min_lyso, min_overlap=args.min_overlap,
                               z_diag=args.z_diag),
                   classes=stats, seconds_load=round(t_load, 1), seconds_total=round(time.time() - t0, 1))
    (args.out / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
