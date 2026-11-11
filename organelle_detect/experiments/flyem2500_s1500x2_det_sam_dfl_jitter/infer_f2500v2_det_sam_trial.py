#!/usr/bin/env python3
"""flyEM det_sam inference: sliding windows + edge-truncation trial windows + mask-based instance stitching.

Per-window logic is unchanged from infer_flyem_nucleus_filtered.infer_det_sam:
  per-volume clip(p0.5,p99.8)->[0,1], per-crop z-score; per class: top-12 boxes,
  keep score > 0.3, prompt-decoder mask, voxels with logit > 0.

Pipeline:
  0. nucleus mask (2D UNet); windows lying completely inside the nucleus are skipped.
  1. normal windows: crop 10x188x188, stride --stride-z / --stride-xy.
  2. mask stitching (pass 1): fragments are added in priority order (untruncated first,
     then by score). A fragment joins the existing instance (any class) with the highest
     mask IoU computed inside the fragment's window, if that IoU >= MERGE_IOU; otherwise it
     starts a new instance. A fragment joins at most one instance, and only paints voxels
     that are still free (no overwriting of other instances). Class = argmax of
     score*voxels votes over the instance's fragments.
  3. trials: an instance is complete if one of its fragments is not truncated on any axis
     along which the instance fits a window. Incomplete instances get a window centered
     on their bbox; the trial fragment with the highest mask IoU (>= MATCH_IOU) to the
     instance is taken, and the move is repeated up to MAX_TRIALS times while it is
     still truncated.
  4. final stitching (pass 2) of normal + trial fragments, complete trials first; each
     instance is then split into 3D connected components.
  5. nucleus exclusion (instance dropped if >= 10% of voxels in nucleus).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model_segdet3d import OrganelleSegDet3D, det_cfg_from_state  # noqa: E402

FLYEM = Path("/path/to/project_data/flyEM")
TAG = "f2500v2"
TIF = FLYEM / "2500x2500x250_v2/raw.ome_v2.tif"
NUC = ROOT.parent / "outputs/infer_flyem1500x10_det_sam_v3/f2500v2_nucleus.npy"
CKPT = ROOT.parent / "outputs/train_det_sam_flyem2500_s1500x2_slice2d_dfl/best_det_sam.pt"
CKPT_NUC = ROOT.parent / "outputs/train_nuclear_unet2d/best.pt"
OUT_DIR = Path("/path/to/project_data/JaneliaEM_infer/f2500v2_s1500x2_slice2d_trial")
VIZ_DIR = ROOT.parent / "outputs/logs/infer_f2500v2_s1500x2_slice2d_trial"
CLASSES = {"mito": 2, "lyso": 3}
CLS_LIST = list(CLASSES.values())
CLS_IDX = {c: i for i, c in enumerate(CLS_LIST)}
CLASS_COLORS = {2: (1.0, 0.25, 0.25), 3: (0.35, 1.0, 0.35)}

NUC_WIN, NUC_STRIDE, NUC_THR = 800, 400, 0.5
CZ, CXY = 10, 188
SZ, SXY = 3, 94
TOPK, SCORE_THR, MASK_LOGIT_THR = 12, 0.3, 0.0
BATCH = 16
WORKER_THREADS = 2
MIN_FRAG_VOX = 32
FIT_MARGIN = (1, 4, 4)  # an axis is fittable if extent <= crop - 2*margin
MAX_TRIALS = 4
TRIAL_QUANT = (3, 24, 24)  # trial windows with the same quantized origin are deduplicated
MATCH_IOU = 0.3
MERGE_IOU = 0.3
MERGE_CONT = 0.5  # or: intersection / smaller of (fragment, instance-in-region) >= this
NEW_FREE_FRAC = 0.5  # an unmerged fragment starts a new instance only if this fraction of it is unclaimed
STITCH_ORDER = "raster"
MERGE_REGION = "box"  # instance voxels compared against the fragment: inside its box or its whole window
FRAG_MIN = 0.1  # the overlap must also be at least this fraction of the fragment
CONTACT_THR = 1.5
NUC_OVERLAP_THR = 0.10
MIN_VOX_INST = 32
CROP = np.array([CZ, CXY, CXY])


def iter_windows(Z, Y, X, cz, cy, cx, sz, sy, sx):
    def starts(L, c, s):
        v = list(range(0, max(L - c, 0) + 1, s))
        if not v or v[-1] != L - c:
            v.append(max(L - c, 0))
        return v
    for z0 in starts(Z, cz, sz):
        for y0 in starts(Y, cy, sy):
            for x0 in starts(X, cx, sx):
                yield z0, y0, x0


def is_window_in_nucleus(w, nuc, cz, cy, cx, skip_thr=1.0):
    """Check if window w=(z,y,x) falls completely (or >= skip_thr) inside the nucleus mask."""
    z, y, x = int(w[0]), int(w[1]), int(w[2])
    if skip_thr >= 1.0:
        if (nuc[z, y, x] == 1 and
            nuc[z + cz - 1, y + cy - 1, x + cx - 1] == 1 and
            nuc[z, y + cy - 1, x] == 1 and
            nuc[z + cz - 1, y, x + cx - 1] == 1):
            return bool(np.all(nuc[z:z + cz, y:y + cy, x:x + cx] == 1))
        return False
    return bool(nuc[z:z + cz, y:y + cy, x:x + cx].mean() >= skip_thr)


@torch.no_grad()
def segment_nuclei(vol_u8: np.ndarray, ckpt_path: Path, dev: str = "cuda:0", tag: str = "vol",
                   out_path: Path | None = None) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_nuclear_unet2d import UNet2D
    m = UNet2D(base=32).to(dev).eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    Z, Y, X = vol_u8.shape
    prob = np.zeros((Z, Y, X), np.float32)
    wins = [(y0, x0) for _, y0, x0 in iter_windows(1, Y, X, 1, NUC_WIN, NUC_WIN, 1, NUC_STRIDE, NUC_STRIDE)]
    t0 = time.time()
    print(f"[nuc {tag}] segmenting nuclei with UNet2D (best ep {ck.get('epoch')}, dice {ck.get('dice', 0):.4f}) on {dev}...", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for z in range(Z):
            img = vol_u8[z].astype(np.float32) / 255.0
            p2 = np.zeros((Y, X), np.float32)
            c2 = np.zeros((Y, X), np.float32)
            for i in range(0, len(wins), BATCH):
                chunk = wins[i:i + BATCH]
                t = torch.from_numpy(np.stack([img[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] for y0, x0 in chunk]))[:, None]
                lg = m(t.to(dev))
                pr = torch.sigmoid(lg.float()).cpu().numpy()[:, 0]
                for (y0, x0), p in zip(chunk, pr):
                    p2[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] += p
                    c2[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] += 1
            prob[z] = p2 / np.maximum(c2, 1)
            if (z + 1) % 50 == 0 or z == Z - 1:
                print(f"  [nuc {tag}] slice {z+1}/{Z} ({time.time() - t0:.1f}s)", flush=True)
    mask = (prob > NUC_THR).astype(np.uint8)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mask)
        print(f"[nuc {tag}] saved mask to {out_path} (fg={mask.mean():.4f})", flush=True)
    return mask


# ---------------- GPU workers ----------------
_W: dict = {}


def _init_worker(gpus, ckpt, vol_path, workers_per_gpu):
    idx = mp.current_process()._identity[0] - 1
    dev = f"cuda:{gpus[(idx // workers_per_gpu) % len(gpus)]}"
    # many workers share the CPU; torch's default (one thread per core each) oversubscribes it
    torch.set_num_threads(WORKER_THREADS)
    torch.cuda.set_device(dev)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ck["model"].items()}
    m = OrganelleSegDet3D(mode="det_sam", dim="3d", **det_cfg_from_state(sd))
    m.load_state_dict(sd)
    _W.update(model=m.to(dev).eval(), dev=dev, vol=np.load(vol_path, mmap_mode="r"))


@torch.no_grad()
def _infer_chunk(task):
    """task = (ids, origins) -> fragments (one per kept box mask) of every class."""
    ids, origins = task
    model, dev, vol = _W["model"], _W["dev"], _W["vol"]
    L = np.array(vol.shape)
    crops = np.stack([vol[z0:z0 + CZ, y0:y0 + CXY, x0:x0 + CXY] for z0, y0, x0 in origins]).astype(np.float32)
    crops = (crops - crops.mean(axis=(1, 2, 3), keepdims=True)) / (crops.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
    t = torch.from_numpy(crops[:, None]).to(dev)
    frags = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = model.backbone(t)
        p3, p4, p5 = model.neck(feats)
        py = [p3, p4, p5]
        for cid in CLASSES.values():
            h = model.instance_ids.index(cid)
            d = model.det_heads[h](py)
            boxes, probs = model._top_boxes_binary(d, topk=TOPK)
            keep = [torch.where(pb[:, 0] > SCORE_THR)[0] for pb in probs]
            boxes_f = [bb[k] for bb, k in zip(boxes, keep)]
            cls_probs = [torch.ones(bb.shape[0], 1, device=bb.device, dtype=bb.dtype) for bb in boxes_f]
            pm_list = model.prompt_decoders[h](p3, boxes_f, cls_probs)["pred_masks"]
            for b, origin in enumerate(origins):
                n = min(boxes_f[b].shape[0], pm_list[b].shape[0])
                if n == 0:
                    continue
                pm = F.interpolate(pm_list[b][:n].float().unsqueeze(1), size=(CZ, CXY, CXY), mode="nearest")
                masks = (pm.squeeze(1) > MASK_LOGIT_THR).cpu().numpy()
                scores = probs[b][keep[b], 0][:n].float().cpu().numpy()
                o = np.array(origin)
                for k in range(n):
                    m = masks[k]
                    if m.sum() < MIN_FRAG_VOX:
                        continue
                    nz = np.nonzero(m)
                    lo = np.array([a.min() for a in nz])
                    hi = np.array([a.max() + 1 for a in nz])
                    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                    t_lo = (lo == 0) & (o > 0)
                    t_hi = (hi == CROP) & (o + CROP < L)
                    frags.append(dict(wid=ids[b], cid=cid, score=float(scores[k]), origin=o, lo=lo, hi=hi,
                                      shape=sub.shape, bits=np.packbits(sub), trunc=np.concatenate([t_lo, t_hi])))
    return frags


def run_windows(pool, ids, origins, label):
    tasks = [(ids[i:i + BATCH], origins[i:i + BATCH]) for i in range(0, len(origins), BATCH)]
    out, t0 = [], time.time()
    for j, fr in enumerate(pool.imap_unordered(_infer_chunk, tasks)):
        out += fr
        if j % 200 == 0 or j == len(tasks) - 1:
            print(f"  [{label}] {min((j + 1) * BATCH, len(origins))}/{len(origins)} windows, "
                  f"{len(out)} fragments ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------- fragments ----------------
def gbox(f):
    return np.concatenate([f["origin"] + f["lo"], f["origin"] + f["hi"]])


def frag_mask(f):
    return np.unpackbits(f["bits"], count=int(np.prod(f["shape"]))).reshape(f["shape"]).astype(bool)


def trunc_axes(f):
    return f["trunc"][:3] | f["trunc"][3:]


def fit_axes(box):
    return (box[3:] - box[:3]) <= CROP - 2 * np.array(FIT_MARGIN)


def win_slice(o):
    return tuple(slice(int(a), int(a) + int(c)) for a, c in zip(o, CROP))


def box_origin(box, L):
    ctr = (box[:3] + box[3:]) / 2
    return np.clip(np.round(ctr - CROP / 2).astype(int), 0, L - CROP)


# ---------------- mask stitching ----------------
class Stitcher:
    """Global instance label built from fragments; merge decided by mask IoU inside the fragment's window."""

    def __init__(self, shape, nuc):
        self.label = np.zeros(shape, np.uint32)
        self.score = np.zeros(shape, np.float16)
        self.nuc = nuc
        self.vote = [np.zeros(len(CLS_LIST))]
        self.vox = [0]
        self.nuc_vox = [0]
        self.frags = [[]]
        self.box = [None]

    def _new(self):
        self.vote.append(np.zeros(len(CLS_LIST)))
        self.vox.append(0)
        self.nuc_vox.append(0)
        self.frags.append([])
        self.box.append(None)
        return len(self.vote) - 1

    def add(self, f):
        o = f["origin"]
        gb = gbox(f)
        bsl = tuple(slice(int(a), int(b)) for a, b in zip(gb[:3], gb[3:]))
        m = frag_mask(f)
        nf = int(m.sum())
        sub = self.label[bsl]
        ids, inter = np.unique(sub[m], return_counts=True)
        best, best_rank = 0, 0.0
        reg = None
        for i, c in zip(ids.tolist(), inter.tolist()):
            if i == 0:
                continue
            if MERGE_REGION == "box":
                iw = int(np.count_nonzero(sub == i))
            else:
                if reg is None:
                    reg = self.label[win_slice(o)]
                iw = int(np.count_nonzero(reg == i))
            iou = c / max(nf + iw - c, 1)
            cont = c / max(min(nf, iw), 1)
            if c < FRAG_MIN * nf:
                continue
            if (iou >= MERGE_IOU or cont >= MERGE_CONT) and iou + cont > best_rank:
                best, best_rank = i, iou + cont
        free = m & (sub == 0)
        if not best and int(free.sum()) < NEW_FREE_FRAC * nf:
            f["inst"] = 0
            return 0
        iid = best if best else self._new()
        sub[free] = iid
        s = self.score[bsl]
        s[free] = np.float16(f["score"])
        nfree = int(free.sum())
        self.vox[iid] += nfree
        self.nuc_vox[iid] += int(np.count_nonzero(self.nuc[bsl][free]))
        self.vote[iid][CLS_IDX[f["cid"]]] += f["score"] * nf
        self.frags[iid].append(f)
        b = self.box[iid]
        self.box[iid] = gb.copy() if b is None else np.concatenate([np.minimum(b[:3], gb[:3]), np.maximum(b[3:], gb[3:])])
        f["inst"] = iid
        return iid

    def cls_of(self):
        return np.array([0] + [CLS_LIST[int(np.argmax(v))] for v in self.vote[1:]], np.int64)


def priority(f):
    # complete trial fragments, then untruncated normal fragments, then truncated ones; by score within
    tier = 0 if f.get("complete") else (1 if not f["trunc"].any() else 2)
    return (tier, -f["score"])


def raster(f):
    # window by window along z, then y, x, so instances grow from one window into the next
    return (int(f["origin"][0]), int(f["origin"][1]), int(f["origin"][2]), -f["score"])


def stitch(frags, shape, nuc, label="stitch"):
    st = Stitcher(shape, nuc)
    t0 = time.time()
    order = sorted(frags, key=raster if STITCH_ORDER == "raster" else priority)
    for k, f in enumerate(order):
        st.add(f)
        if k % 20000 == 0:
            print(f"  [{label}] {k}/{len(order)} fragments, {len(st.vote) - 1} instances ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"  [{label}] {len(order)} fragments -> {len(st.vote) - 1} instances ({time.time() - t0:.0f}s)", flush=True)
    return st


def is_complete(st, iid):
    fa = fit_axes(st.box[iid])
    return any(not (trunc_axes(f) & fa).any() for f in st.frags[iid])


# ---------------- truncation trials ----------------
def run_trials(pool, st, L, nuc=None, skip_thr=1.0):
    """Move windows onto instances that are truncated in every window that saw them."""
    pending = {}
    n_complete0 = n_nuc = n_small = n_unfit = 0
    for iid in range(1, len(st.vote)):
        if st.vox[iid] < MIN_VOX_INST:
            n_small += 1
            continue
        if st.nuc_vox[iid] / max(st.vox[iid], 1) >= NUC_OVERLAP_THR:
            n_nuc += 1
            continue
        if is_complete(st, iid):
            n_complete0 += 1
            continue
        if not fit_axes(st.box[iid]).any():
            n_unfit += 1
            continue
        pending[iid] = st.box[iid].copy()
    print(f"[trial] instances={len(st.vote) - 1} complete_in_normal={n_complete0} in_nucleus={n_nuc} "
          f"small={n_small} cannot_fit={n_unfit} -> need trial={len(pending)}", flush=True)
    trial_frags, stats = [], []
    first_origin = {iid: st.frags[iid][0]["origin"] for iid in pending}
    for r in range(MAX_TRIALS):
        plan: dict = {}
        n_nucwin = 0
        for iid, bb in pending.items():
            o = box_origin(bb, L)
            if nuc is not None and is_window_in_nucleus(tuple(o.tolist()), nuc, CZ, CXY, CXY, skip_thr=skip_thr):
                n_nucwin += 1
                continue
            key = tuple((o // np.array(TRIAL_QUANT)).tolist())
            plan.setdefault(key, (o, []))[1].append(iid)
        items = list(plan.values())
        print(f"[trial round {r + 1}] instances={len(pending)} windows={len(items)} skipped_nucleus={n_nucwin}",
              flush=True)
        if not items:
            stats.append(dict(round=r + 1, instances=len(pending), windows=0))
            break
        frags = run_windows(pool, list(range(len(items))), [tuple(o.tolist()) for o, _ in items], f"trial r{r + 1}")
        by_w: dict[int, list] = {}
        for f in frags:
            by_w.setdefault(f["wid"], []).append(f)
        nxt, n_ok, n_fail, n_stuck = {}, 0, 0, 0
        for wid, (o, iids) in enumerate(items):
            cand = by_w.get(wid, [])
            if not cand:
                n_fail += len(iids)
                continue
            reg = st.label[win_slice(o)]
            masks = [frag_mask(f) for f in cand]
            for iid in iids:
                iw = int(np.count_nonzero(reg == iid))
                best, best_rank = None, 0.0
                for f, m in zip(cand, masks):
                    lo, hi = f["lo"], f["hi"]
                    inter = int(np.count_nonzero(reg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][m] == iid))
                    if inter == 0:
                        continue
                    nf = int(m.sum())
                    iou = inter / max(nf + iw - inter, 1)
                    cont = inter / max(min(nf, iw), 1)
                    if (iou >= MATCH_IOU or cont >= MERGE_CONT) and iou + cont > best_rank:
                        best, best_rank = f, iou + cont
                if best is None:
                    n_fail += 1
                    continue
                f = best
                f["trial"] = r + 1
                f["seed_origin"] = first_origin[iid]
                if not any(f is g for g in trial_frags):
                    trial_frags.append(f)
                gb = gbox(f)
                bb = np.concatenate([np.minimum(pending[iid][:3], gb[:3]), np.maximum(pending[iid][3:], gb[3:])])
                fa = fit_axes(bb)
                if not (trunc_axes(f) & fa).any():
                    f["complete"] = True
                    n_ok += 1
                elif fa.any() and r < MAX_TRIALS - 1:
                    nxt[iid] = bb
                else:
                    n_stuck += 1
        stats.append(dict(round=r + 1, instances=len(pending), windows=len(items), complete=n_ok,
                          still_truncated=len(nxt), gave_up=n_stuck, no_match=n_fail))
        print(f"[trial round {r + 1}] complete={n_ok} still_truncated={len(nxt)} gave_up={n_stuck} "
              f"no_match={n_fail}", flush=True)
        pending = nxt
        if not pending:
            break
    return trial_frags, stats, dict(complete_in_normal=n_complete0, in_nucleus=n_nuc, small=n_small,
                                    cannot_fit=n_unfit)


def merge_by_contact(label, cls_inst, thr):
    """Merge same-class instances whose shared face is large relative to the smaller one:
    contact_voxels / min(size)^(2/3) >= thr. Pieces of one object cut between windows share
    a whole cross-section; distinct touching organelles share a small patch."""
    n = len(cls_inst)
    if thr <= 0 or n < 2:
        return np.arange(n)
    size = np.bincount(label.ravel(), minlength=n).astype(np.int64)

    def pairs(z0, z1):
        # face pairs whose lower voxel lies in slices [z0, z1)
        keys = []
        for ax in range(3):
            if ax == 0:
                x, y = label[z0:min(z1, label.shape[0] - 1)], label[z0 + 1:min(z1, label.shape[0] - 1) + 1]
            else:
                a = label[z0:z1]
                sa = [slice(None)] * 3
                sb = [slice(None)] * 3
                sa[ax], sb[ax] = slice(None, -1), slice(1, None)
                x, y = a[tuple(sa)], a[tuple(sb)]
            k = (x != y) & (x > 0) & (y > 0)
            xs, ys = x[k].astype(np.int64), y[k].astype(np.int64)
            keys.append(np.minimum(xs, ys) * n + np.maximum(xs, ys))
        return np.concatenate(keys)

    from concurrent.futures import ThreadPoolExecutor
    step = 8
    with ThreadPoolExecutor(16) as ex:
        keys = np.concatenate(list(ex.map(lambda z: pairs(z, z + step), range(0, label.shape[0], step))))
    u, c = np.unique(keys, return_counts=True)
    contact = dict(zip(u.tolist(), c.tolist()))
    root = np.arange(n)

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for kk, c in sorted(contact.items(), key=lambda t: -t[1]):
        a, b = divmod(kk, n)
        if cls_inst[a] != cls_inst[b] or c < thr * min(size[a], size[b]) ** (2 / 3):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def split_components(label, cls_of_inst):
    """Split every instance into 26-connected components; drop pieces < MIN_VOX_INST."""
    st26 = np.ones((3, 3, 3), bool)
    new_cls = [0]
    parent = [0]
    nxt = 1
    out = np.zeros_like(label)
    for iid, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        m = label[sl] == iid
        cc, n = ndi.label(m, structure=st26)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        o = out[sl]
        for k in np.argsort(-sizes):
            if sizes[k] < MIN_VOX_INST:
                break
            o[cc == k + 1] = nxt
            new_cls.append(int(cls_of_inst[iid]))
            parent.append(iid)
            nxt += 1
    return out, np.array(new_cls, np.int64), np.array(parent, np.int64)


SLICE_ST = np.zeros((3, 3, 3), bool)
SLICE_ST[1] = ndi.generate_binary_structure(2, 1)  # in-plane connectivity only: fills holes slice by slice


def fill_holes(label):
    """Fill enclosed holes of every instance (3D and per z-slice). Background hole voxels are
    given to the instance; hole voxels already owned by another instance are left to it and
    recorded as an overlap (outer id written to the returned `overlap` volume) for manual review."""
    overlap = np.zeros_like(label)
    pairs = []
    n_filled = 0
    sizes = np.bincount(label.ravel())
    for iid, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        sub = label[sl]
        m = sub == iid
        f = ndi.binary_fill_holes(m) | ndi.binary_fill_holes(m, structure=SLICE_ST)
        hole = f & ~m
        if not hole.any():
            continue
        bg = hole & (sub == 0)
        sub[bg] = iid
        n_filled += int(bg.sum())
        occ = hole & (sub != iid)
        if not occ.any():
            continue
        overlap[sl][occ] = iid
        ids, cnt = np.unique(sub[occ], return_counts=True)
        for j, c in zip(ids.tolist(), cnt.tolist()):
            pairs.append(dict(outer=iid, inner=int(j), voxels=int(c), inner_frac=round(c / max(int(sizes[j]), 1), 3)))
    return overlap, pairs, n_filled


def viz_review(vol, label, overlap, pairs, out_png, n=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    top = sorted(pairs, key=lambda p: -p["voxels"])[:n]
    if not top:
        return
    sls = ndi.find_objects(label)
    cols = 4
    rows = (len(top) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), top):
        sl = sls[p["outer"] - 1]
        zs = np.nonzero((overlap[sl] == p["outer"]).any(axis=(1, 2)))[0]
        z = sl[0].start + int(zs[len(zs) // 2])
        pad = 20
        y0, y1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, label.shape[1])
        x0, x1 = max(sl[2].start - pad, 0), min(sl[2].stop + pad, label.shape[2])
        lz = label[z, y0:y1, x0:x1]
        ax.imshow(vol[z, y0:y1, x0:x1], cmap="gray")
        rgba = np.zeros((*lz.shape, 4))
        rgba[find_boundaries(lz == p["outer"])] = (1, 0, 0, 1)
        rgba[find_boundaries(lz == p["inner"])] = (0, 1, 1, 1)
        rgba[overlap[z, y0:y1, x0:x1] == p["outer"]] += (1, 1, 0, 0.35)
        ax.imshow(np.clip(rgba, 0, 1))
        ax.set_title(f"outer {p['outer']} (red) / inner {p['inner']} (cyan)\n"
                     f"z={z} overlap={p['voxels']} vox, {p['inner_frac']:.0%} of inner", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"[viz] {out_png}", flush=True)


def instance_table(label, score, cls_of, parent, nfrag_parent, complete_parent, nuc):
    out = []
    nuc_b = nuc.astype(bool)
    for iid, sl in enumerate(ndi.find_objects(label), start=1):
        if sl is None:
            continue
        m = label[sl] == iid
        vox = int(m.sum())
        ov = float((m & nuc_b[sl]).sum()) / vox
        p = int(parent[iid])
        out.append(dict(
            id=iid, cls=int(cls_of[iid]), voxels=vox,
            box=[int(sl[0].start), int(sl[1].start), int(sl[2].start),
                 int(sl[0].stop), int(sl[1].stop), int(sl[2].stop)],
            score=float(score[sl][m].max()), n_fragments=int(nfrag_parent[p]),
            complete_trial=bool(complete_parent[p]), stitch_id=p,
            nuc_overlap=round(ov, 3), kept=bool(ov < NUC_OVERLAP_THR),
        ))
    return out


# ---------------- viz ----------------
def viz(vol, label, cls_of, table, complete, out_png, roi_origin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = vol.shape[0]
    z = Z // 2
    kept = np.zeros(len(cls_of) + 1, bool)
    for t in table:
        kept[t["id"]] = t["kept"]
    rng = np.random.default_rng(0)
    pal = rng.uniform(0.2, 1.0, (len(cls_of) + 1, 3))
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 4)
    ax0 = fig.add_subplot(gs[:2, :2])
    ax1 = fig.add_subplot(gs[:2, 2:])
    em = vol[z]
    ax0.imshow(em, cmap="gray")
    sem = np.zeros((*em.shape, 4))
    lz = label[z]
    for cid, col in CLASS_COLORS.items():
        sem[(cls_of[lz] == cid) & (lz > 0) & kept[lz]] = (*col, 0.6)
    ax0.imshow(sem)
    ax0.set_title(f"{TAG} z={z}: kept instances by class (red mito, green lyso)", fontsize=10)
    ax1.imshow(em, cmap="gray")
    inst = np.zeros((*em.shape, 4))
    vis = (lz > 0) & kept[lz]
    inst[vis, :3] = pal[lz[vis]]
    inst[vis, 3] = 0.65
    ax1.imshow(inst)
    ax1.set_title(f"stitched instances (random colors), z={z}", fontsize=10)
    ex = sorted([f for f in complete if f["trial"] >= 2], key=lambda f: -f["bits"].size)[:4]
    ex += sorted([f for f in complete if f["trial"] == 1], key=lambda f: -f["bits"].size)[:4 - len(ex)]
    for k, f in enumerate(ex[:4]):
        ax = fig.add_subplot(gs[2, k])
        so, fo = f["seed_origin"], f["origin"]
        zc = int(fo[0] + (f["lo"][0] + f["hi"][0]) // 2)
        y0 = max(0, int(min(so[1], fo[1])) - 20)
        x0 = max(0, int(min(so[2], fo[2])) - 20)
        y1 = min(vol.shape[1], int(max(so[1], fo[1])) + CXY + 20)
        x1 = min(vol.shape[2], int(max(so[2], fo[2])) + CXY + 20)
        ax.imshow(vol[zc, y0:y1, x0:x1], cmap="gray")
        m = frag_mask(f)
        zl = zc - int(fo[0]) - int(f["lo"][0])
        if 0 <= zl < m.shape[0]:
            mm = np.zeros((y1 - y0, x1 - x0), bool)
            ys, xs = int(fo[1] + f["lo"][1]) - y0, int(fo[2] + f["lo"][2]) - x0
            mm[ys:ys + m.shape[1], xs:xs + m.shape[2]] = m[zl]
            col = CLASS_COLORS[f["cid"]]
            ov = np.zeros((*mm.shape, 4))
            ov[mm] = (*col, 0.55)
            ax.imshow(ov)
        for o, c, lab in ((so, "yellow", "normal (truncated)"), (fo, "cyan", f"trial {f['trial']} (complete)")):
            ax.add_patch(plt.Rectangle((int(o[2]) - x0, int(o[1]) - y0), CXY, CXY, fill=False, ec=c, lw=1.5,
                                       label=lab))
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"cls={f['cid']} trial window, z={zc + roi_origin[0]}", fontsize=9)
        ax.axis("off")
    ax0.axis("off")
    ax1.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"[viz] {out_png}", flush=True)


def save_zarr(path, arr, chunks=(10, 256, 256)):
    import zarr
    z = zarr.open(str(path), mode="w", shape=arr.shape, chunks=chunks, dtype=arr.dtype)
    z[:] = arr


def main():
    global OUT_DIR, VIZ_DIR, TAG, TIF, NUC, SZ, SXY, MERGE_IOU, MERGE_CONT, NEW_FREE_FRAC, STITCH_ORDER, MERGE_REGION, FRAG_MIN, CONTACT_THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--ckpt-nuc", type=Path, default=CKPT_NUC)
    ap.add_argument("--tif", type=Path, default=TIF)
    ap.add_argument("--nuc", type=Path, default=NUC)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--recompute-nuc", action="store_true", help="force recomputing nucleus segmentation with UNet")
    ap.add_argument("--nuc-skip-thr", type=float, default=1.0,
                    help="window mean nucleus fraction >= this threshold to skip (1.0 = completely inside nucleus)")
    ap.add_argument("--stride-z", type=int, default=SZ)
    ap.add_argument("--stride-xy", type=int, default=SXY)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    ap.add_argument("--roi", type=int, nargs=6, default=None, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                    help="debug: run on a sub-volume")
    ap.add_argument("--gpus", default=os.environ.get("INFER_GPUS", "0,1,2,3,4,5,6,7"))
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--merge-iou", type=float, default=MERGE_IOU)
    ap.add_argument("--merge-cont", type=float, default=MERGE_CONT)
    ap.add_argument("--new-free-frac", type=float, default=NEW_FREE_FRAC)
    ap.add_argument("--order", choices=["raster", "priority"], default=STITCH_ORDER)
    ap.add_argument("--merge-region", choices=["box", "window"], default=MERGE_REGION)
    ap.add_argument("--frag-min", type=float, default=FRAG_MIN)
    ap.add_argument("--contact-thr", type=float, default=CONTACT_THR, help="0 disables the contact merge")
    ap.add_argument("--from-frags", type=Path, default=None,
                    help="re-stitch saved fragments.pkl (no GPU inference)")
    args = ap.parse_args()
    OUT_DIR, VIZ_DIR = args.out_dir, args.viz_dir
    TAG, TIF, NUC = args.tag, args.tif, args.nuc
    SZ, SXY = args.stride_z, args.stride_xy
    MERGE_IOU, MERGE_CONT, NEW_FREE_FRAC = args.merge_iou, args.merge_cont, args.new_free_frac
    STITCH_ORDER, MERGE_REGION, FRAG_MIN = args.order, args.merge_region, args.frag_min
    CONTACT_THR = args.contact_thr
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    t0 = time.time()

    vol = tifffile.imread(str(TIF)).astype(np.uint8)
    t_nuc_0 = time.time()
    if not args.recompute_nuc and NUC.exists():
        nuc = np.load(NUC)
        print(f"[nuc] loaded precomputed mask from {NUC} (fg={nuc.mean():.4f})", flush=True)
    else:
        nuc = segment_nuclei(vol, args.ckpt_nuc, dev=f"cuda:{gpus[0]}", tag=TAG, out_path=NUC)
    t_nuc = time.time() - t_nuc_0

    sub = vol[::4, ::4, ::4]
    lo, hi = np.percentile(sub, [0.5, 99.8])
    roi_origin = (0, 0, 0)
    if args.roi:
        z0, z1, y0, y1, x0, x1 = args.roi
        vol, nuc = vol[z0:z1, y0:y1, x0:x1], nuc[z0:z1, y0:y1, x0:x1]
        roi_origin = (z0, y0, x0)
    vol01 = np.clip((vol.astype(np.float32) - np.float32(lo)) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
    vol_path = Path("/dev/shm") / f"{TAG}_vol01_{os.getpid()}.npy"
    np.save(vol_path, vol01)
    L = np.array(vol01.shape)
    del vol01
    print(f"[{TAG}] shape={tuple(L)} roi={args.roi} gray p0.5={lo:.1f} p99.8={hi:.1f} ckpt={args.ckpt} "
          f"(loaded in {time.time() - t0:.0f}s)", flush=True)

    all_wins = list(iter_windows(*L, CZ, CXY, CXY, SZ, SXY, SXY))
    wins = [w for w in all_wins if not is_window_in_nucleus(w, nuc, CZ, CXY, CXY, skip_thr=args.nuc_skip_thr)]
    n_nuc_skipped = len(all_wins) - len(wins)
    print(f"[windows] crop={CZ}x{CXY}x{CXY} stride={SZ}/{SXY}/{SXY} total={len(all_wins)} "
          f"skipped_in_nucleus={n_nuc_skipped} ({100 * n_nuc_skipped / len(all_wins):.1f}%) "
          f"infer_windows={len(wins)} gpus={gpus}", flush=True)

    if args.from_frags:
        vol_path.unlink(missing_ok=True)
        with open(args.from_frags, "rb") as fh:
            saved = pickle.load(fh)
        normal, trial_frags = saved["normal"], saved["trial"]
        trial_stats, trial_pre = saved["trial_stats"], saved["trial_pre"]
        for f in normal + trial_frags:
            f.pop("inst", None)
        t_normal = t_trial = t_stitch1 = 0.0
        print(f"[frags] loaded {len(normal)} normal + {len(trial_frags)} trial fragments from {args.from_frags}",
              flush=True)
    else:
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(len(gpus) * args.workers_per_gpu, initializer=_init_worker,
                          initargs=(gpus, str(args.ckpt), str(vol_path), args.workers_per_gpu)) as pool:
                t_normal_0 = time.time()
                normal = run_windows(pool, list(range(len(wins))), wins, "normal")
                t_normal = time.time() - t_normal_0
                n_trunc = sum(bool(f["trunc"].any()) for f in normal)
                print(f"[normal] fragments={len(normal)} touching interior face={n_trunc} ({t_normal:.0f}s)",
                      flush=True)
                t_s1 = time.time()
                st1 = stitch(normal, tuple(L), nuc, "stitch pass1")
                t_stitch1 = time.time() - t_s1
                t_trial_0 = time.time()
                trial_frags, trial_stats, trial_pre = run_trials(pool, st1, L, nuc=nuc, skip_thr=args.nuc_skip_thr)
                t_trial = time.time() - t_trial_0
                del st1
        finally:
            vol_path.unlink(missing_ok=True)
        with open(OUT_DIR / "fragments.pkl", "wb") as fh:
            pickle.dump(dict(normal=normal, trial=trial_frags, trial_stats=trial_stats, trial_pre=trial_pre), fh)

    complete = [f for f in trial_frags if f.get("complete")]
    t_s2 = time.time()
    st = stitch(normal + trial_frags, tuple(L), nuc, "stitch pass2")
    cls_inst = st.cls_of()
    nfrag_inst = np.array([0] + [len(fs) for fs in st.frags[1:]], np.int64)
    complete_inst = np.array([False] + [any(f.get("complete") for f in fs) for fs in st.frags[1:]], bool)
    n_stitched = len(st.vote) - 1
    t_c = time.time()
    root = merge_by_contact(st.label, cls_inst, CONTACT_THR)
    if (root != np.arange(len(root))).any():
        st.label = root.astype(np.uint32)[st.label]
        nfrag_inst = np.bincount(root, weights=nfrag_inst, minlength=len(root)).astype(np.int64)
        complete_inst = np.bincount(root, weights=complete_inst, minlength=len(root)) > 0
    n_contact = len(np.unique(root)) - 1
    print(f"[contact] {n_stitched} -> {n_contact} instances (thr={CONTACT_THR}, {time.time() - t_c:.0f}s)",
          flush=True)
    label, cls_of, parent = split_components(st.label, cls_inst)
    score = st.score
    del st
    t_stitch = time.time() - t_s2 + t_stitch1
    print(f"[split] stitched instances={n_stitched} -> connected-component instances={len(cls_of) - 1}", flush=True)
    t_f = time.time()
    overlap, review_pairs, n_filled = fill_holes(label)
    print(f"[fill] filled {n_filled} background hole voxels; {len(review_pairs)} overlapping pairs "
          f"({len({p['outer'] for p in review_pairs})} outer instances) flagged ({time.time() - t_f:.0f}s)", flush=True)
    table = instance_table(label, score, cls_of, parent, nfrag_inst, complete_inst, nuc)
    by_id = {t["id"]: t for t in table}
    for t in table:
        t["review"], t["contains"], t["inside"] = False, [], []
    for p in review_pairs:
        for k, role, other in ((p["outer"], "contains", p["inner"]), (p["inner"], "inside", p["outer"])):
            if k in by_id:
                by_id[k]["review"] = True
                by_id[k][role].append(other)
    for p in review_pairs:
        p["outer_cls"] = int(cls_of[p["outer"]])
        p["inner_cls"] = int(cls_of[p["inner"]])
        p["outer_box"] = by_id[p["outer"]]["box"]
    sem = cls_of.astype(np.uint8)[label]
    kept_ids = np.zeros(len(cls_of), bool)
    for t in table:
        kept_ids[t["id"]] = t["kept"]
    label_kept = np.where(kept_ids[label], label, 0).astype(np.uint32)

    summary = dict(
        tag=TAG, ckpt=str(args.ckpt), ckpt_nuc=str(args.ckpt_nuc), roi=args.roi,
        gpus=gpus, n_gpus=len(gpus), workers_per_gpu=args.workers_per_gpu,
        crop=[CZ, CXY, CXY], stride=[SZ, SXY, SXY],
        n_windows_raw=len(all_wins),
        n_windows_nuc_skipped=n_nuc_skipped,
        n_windows=len(wins),
        n_normal_fragments=len(normal),
        trial_selection=trial_pre, trial_rounds=trial_stats,
        n_trial_windows=int(sum(s.get("windows", 0) for s in trial_stats)),
        n_trial_fragments=len(trial_frags), n_complete_trials=len(complete),
        n_stitched_instances=n_stitched,
        n_fill_voxels=n_filled, n_review_pairs=len(review_pairs),
        n_review_instances=sum(t["review"] for t in table),
        instances={name: dict(total=sum(t["cls"] == cid for t in table),
                              kept=sum(t["cls"] == cid and t["kept"] for t in table),
                              multi_fragment=sum(t["cls"] == cid and t["n_fragments"] > 1 for t in table),
                              with_complete_trial=sum(t["cls"] == cid and t["complete_trial"] for t in table))
                   for name, cid in CLASSES.items()},
        params=dict(score_thr=SCORE_THR, topk=TOPK, mask_logit_thr=MASK_LOGIT_THR, max_trials=MAX_TRIALS,
                    fit_margin=FIT_MARGIN, merge_iou=MERGE_IOU, merge_cont=MERGE_CONT, new_free_frac=NEW_FREE_FRAC, order=STITCH_ORDER, merge_region=MERGE_REGION, frag_min=FRAG_MIN, contact_thr=CONTACT_THR,
                    match_iou=MATCH_IOU,
                    nuc_overlap_thr=NUC_OVERLAP_THR, min_vox_inst=MIN_VOX_INST, nuc_skip_thr=args.nuc_skip_thr),
        timings=dict(
            seconds_nuc=round(t_nuc),
            seconds_normal=round(t_normal),
            seconds_trials=round(t_trial),
            seconds_stitch=round(t_stitch),
            seconds_total=round(time.time() - t0),
        ),
    )
    pre = OUT_DIR / f"{TAG}_det_sam"
    (OUT_DIR / f"{TAG}_det_sam_instances.json").write_text(json.dumps(table, indent=1))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    save_zarr(f"{pre}_instance_label.zarr", label)
    save_zarr(f"{pre}_instance_label_kept.zarr", label_kept)
    save_zarr(f"{pre}_label.zarr", sem)
    save_zarr(f"{pre}_score.zarr", score)
    save_zarr(f"{pre}_fill_overlap.zarr", overlap)
    (OUT_DIR / f"{TAG}_det_sam_review.json").write_text(json.dumps(review_pairs, indent=1))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[save] outputs written ({time.time() - t0:.0f}s since start)", flush=True)
    viz(vol, label, cls_of, table, complete, VIZ_DIR / f"{TAG}_viz.png", roi_origin)
    viz_review(vol, label, overlap, review_pairs, VIZ_DIR / f"{TAG}_review_overlaps.png")
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
