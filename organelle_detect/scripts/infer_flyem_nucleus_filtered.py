#!/usr/bin/env python3
"""flyEM inference pipeline: nucleus UNet masking + m2f_bin/det_sam (v2 ckpts).

Per volume:
  A. nucleus segmentation: 2D UNet (train_nuclear_unet2d/best.pt), per-z-slice,
     800x800 windows, stride 400, averaged probs, thr 0.5 -> nucleus mask (ZYX)
  B. 3D sliding-window inference (crop 6x125x125 = 1000nm, stride 3/62/62):
       - m2f_bin: per-voxel best-score semantic labels (seg head)
       - det_sam: per-class det boxes + prompt masks painted by score
     input norm: per-volume clip(p0.5,p99.8)->[0,1] then per-crop z-score
     (matches npz_crops_flyem_1000nm training)
  C. nucleus-overlap exclusion: instance dropped if >=10% of its voxels
     overlap the nucleus mask
  D. instance tables (json) + visualization

Volumes / classes:
  f1280   (1280x1280x256)        -> lyso
  f2500   (2500x2500x250)        -> lyso
  f2500v2 (2500x2500x250_v2)     -> lyso + mito
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.model_segdet3d import OrganelleSegDet3D, det_cfg_from_state  # noqa: E402
from train_nuclear_unet2d import UNet2D  # noqa: E402

FLYEM = Path("/path/to/project_data/flyEM")
OUT_DIR = Path("/path/to/JaneliaEM/outputs/infer_flyem1000")
VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/infer_flyem1000")
CLASS_SUP_PATH = Path("/path/to/JaneliaEM/outputs/npz_crops_flyem_1000nm/class_supervision.json")
VOXEL_MUTEX = False
MIN_VOX_INST = 32  # drop tiny instances after mutex

TAG_TO_DATASET = {"f1280": "fish2_1280", "f2500": "fish2_2500", "f2500v2": "fish2_2500"}

CKPT_NUC = Path("/path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt")
CKPT_BIN = Path("/path/to/JaneliaEM/outputs/train_m2f_bin_flyem1000_v2/last_m2f_bin.pt")
CKPT_DET = Path("/path/to/JaneliaEM/outputs/train_det_sam_flyem1000_v2/last_det_sam.pt")

# class ids (organelle_classes): mito=2, lyso=3
VOLUMES = [
    dict(tag="f1280", tif=FLYEM / "1280x1280x256/raw.ome.tif", classes={"mito": 2}),
    dict(tag="f2500", tif=FLYEM / "2500x2500x250/raw.ome.tif", classes={"lyso": 3}),
    dict(tag="f2500v2", tif=FLYEM / "2500x2500x250_v2/raw.ome_v2.tif", classes={"lyso": 3, "mito": 2}),
]

NUC_WIN, NUC_STRIDE, NUC_THR = 800, 400, 0.5
CZ, CXY = 6, 125
SZ, SXY = 3, 94  # 25% xy overlap like previous infer (192/256)
SCORE_THR_BIN, SCORE_THR_DET = 0.5, 0.3
NUC_OVERLAP_THR = 0.10
BATCH = int(os.environ.get("INFER_BATCH", "16"))
DEV = os.environ.get("INFER_DEV", "cuda")
DET_MODE = "det_sam"
DET_TAG = "det_sam"
SKIP_M2F_BIN = False

CLASS_COLORS = {2: (1.0, 0.25, 0.25), 3: (0.35, 1.0, 0.35)}  # mito red, lyso green


def load_class_supervision() -> dict:
    if not CLASS_SUP_PATH.exists():
        return {}
    d = json.loads(CLASS_SUP_PATH.read_text())
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "comment"}


def filter_classes(spec: dict, class_sup: dict) -> dict:
    ds = TAG_TO_DATASET.get(spec["tag"], spec["tag"])
    allowed = set(class_sup.get(ds, list(spec["classes"].values())))
    return {n: i for n, i in spec["classes"].items() if i in allowed}


def merge_voxel_scores(score_by_c: dict[int, np.ndarray], thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel argmax over class score maps."""
    cids = sorted(score_by_c)
    if not cids:
        z = next(iter(score_by_c.values()), None)
        sh = z.shape if z is not None else (0, 0, 0)
        return np.zeros(sh, np.uint8), np.zeros(sh, np.float16)
    best = np.zeros_like(score_by_c[cids[0]], np.float32)
    label = np.zeros_like(best, np.uint8)
    for cid in cids:
        sc = score_by_c[cid].astype(np.float32)
        upd = sc > best
        label[upd] = cid
        best[upd] = sc[upd]
    label[best < thr] = 0
    return label, best.astype(np.float16)


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


# ---------------- stage A: nucleus ----------------
@torch.no_grad()
def segment_nuclei(vol_u8: np.ndarray, nuc_model, tag: str) -> np.ndarray:
    Z, Y, X = vol_u8.shape
    prob = np.zeros((Z, Y, X), np.float32)
    cnt = np.zeros((Z, Y, X), np.float32)
    wins = [(y0, x0) for _, y0, x0 in iter_windows(1, Y, X, 1, NUC_WIN, NUC_WIN, 1, NUC_STRIDE, NUC_STRIDE)]
    t0 = time.time()
    for z in range(Z):
        img = vol_u8[z].astype(np.float32) / 255.0
        p2 = np.zeros((Y, X), np.float32)
        c2 = np.zeros((Y, X), np.float32)
        for i in range(0, len(wins), BATCH):
            chunk = wins[i:i + BATCH]
            t = torch.from_numpy(np.stack([img[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] for y0, x0 in chunk]))[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = nuc_model(t.to(DEV))
            pr = torch.sigmoid(lg.float()).cpu().numpy()[:, 0]
            for (y0, x0), p in zip(chunk, pr):
                p2[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] += p
                c2[y0:y0 + NUC_WIN, x0:x0 + NUC_WIN] += 1
        prob[z] = p2 / np.maximum(c2, 1)
        if z % 50 == 0:
            print(f"  [nuc {tag}] z={z}/{Z} ({time.time()-t0:.0f}s)", flush=True)
    mask = (prob > NUC_THR).astype(np.uint8)
    np.save(OUT_DIR / f"{tag}_nucleus.npy", mask)
    print(f"  [nuc {tag}] done fg={mask.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return mask


# ---------------- stage B: 3D models ----------------
def load_3d_model(ckpt_path, mode):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in ckpt["model"].items()}
    m = OrganelleSegDet3D(mode=mode, dim="3d", **det_cfg_from_state(sd))
    m.load_state_dict(sd)
    return m.to(DEV).eval(), ckpt.get("epoch", "?")


@torch.no_grad()
def infer_m2f_bin(model, vol01: np.ndarray, cls_ids: dict, tag: str):
    """Per-voxel best-score semantic label map (only requested classes)."""
    Z, Y, X = vol01.shape
    score = np.zeros((Z, Y, X), np.float16)
    label = np.zeros((Z, Y, X), np.uint8)
    score_by_c = {cid: np.zeros((Z, Y, X), np.float16) for cid in cls_ids.values()} if VOXEL_MUTEX else None
    wins = list(iter_windows(Z, Y, X, CZ, CXY, CXY, SZ, SXY, SXY))
    t0 = time.time()
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        crops = np.stack([vol01[z0:z0 + CZ, y0:y0 + CXY, x0:x0 + CXY] for z0, y0, x0 in chunk])
        crops = (crops - crops.mean(axis=(1, 2, 3), keepdims=True)) / (crops.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        t = torch.from_numpy(crops[:, None]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(t)["seg"]
        logits = out["pred_logits"].float()
        masks = out["pred_masks"].float().sigmoid()
        masks = F.interpolate(masks, size=(CZ, CXY, CXY), mode="nearest")  # 128->125
        qpc = int(out["queries_per_class"])
        fg = logits.softmax(-1)[..., 1]  # [B,Q]
        for b, (z0, y0, x0) in enumerate(chunk):
            for q in torch.where(fg[b] > SCORE_THR_BIN)[0]:
                cid = int(q) // qpc + 1
                cname = next((n for n, i in cls_ids.items() if i == cid), None)
                if cname is None:
                    continue
                sc = float(fg[b, q])
                m = (masks[b, q] > 0.5).cpu().numpy()
                sl = (slice(z0, z0 + CZ), slice(y0, y0 + CXY), slice(x0, x0 + CXY))
                if score_by_c is not None:
                    reg_s = score_by_c[cid][sl]
                    upd = m & (sc > reg_s)
                    reg_s[upd] = np.float16(sc)
                else:
                    reg_s = score[sl]
                    reg_l = label[sl]
                    upd = m & (sc > reg_s)
                    reg_l[upd] = cid
                    reg_s[upd] = np.float16(sc)
        if (i // BATCH) % 200 == 0:
            print(f"  [m2f_bin {tag}] {i}/{len(wins)} ({time.time()-t0:.0f}s)", flush=True)
    if score_by_c is not None:
        label, score = merge_voxel_scores(score_by_c, SCORE_THR_BIN)
    return label, score


@torch.no_grad()
def infer_det_sam(model, vol01: np.ndarray, cls_ids: dict, tag: str):
    """Det boxes + prompt masks painted per voxel by score (requested classes).

    Bypasses model.forward: full forward runs the prompt decoder on top-20
    boxes of ALL 4 instance heads (80 sequential per-box decoder runs per
    sample) plus seg/er heads we don't need -> ~120x slower than m2f_bin.
    Here: backbone+neck once, then only the requested heads with topk=12.
    """
    Z, Y, X = vol01.shape
    score = np.zeros((Z, Y, X), np.float16)
    label = np.zeros((Z, Y, X), np.uint8)
    score_by_c = {cid: np.zeros((Z, Y, X), np.float16) for cid in cls_ids.values()} if VOXEL_MUTEX else None
    boxes_all = []  # (cid, box6 global, score)
    wins = list(iter_windows(Z, Y, X, CZ, CXY, CXY, SZ, SXY, SXY))
    t0 = time.time()
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        crops = np.stack([vol01[z0:z0 + CZ, y0:y0 + CXY, x0:x0 + CXY] for z0, y0, x0 in chunk])
        crops = (crops - crops.mean(axis=(1, 2, 3), keepdims=True)) / (crops.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        t = torch.from_numpy(crops[:, None]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.backbone(t)
            p3, p4, p5 = model.neck(feats)
            py = [p3, p4, p5]
            head_outs = []
            for cname, cid in cls_ids.items():
                if cid not in model.instance_ids:
                    continue
                head_idx = model.instance_ids.index(cid)
                d = model.det_heads[head_idx](py)
                boxes, probs = model._top_boxes_binary(d, topk=12)
                # only decode masks for boxes passing the score threshold
                # (~1.5/sample instead of 12 -> prompt decoder nearly free)
                keep = [torch.where(pb[:, 0] > SCORE_THR_DET)[0] for pb in probs]
                boxes_f = [bb[idx] for bb, idx in zip(boxes, keep)]
                cls_probs = [torch.ones(bb.shape[0], 1, device=bb.device, dtype=bb.dtype)
                             for bb in boxes_f]
                pm_list = model.prompt_decoders[head_idx](p3, boxes_f, cls_probs)["pred_masks"]
                head_outs.append((cid, boxes_f, probs, keep, pm_list))
        for cid, boxes_f, probs, keep, pm_list in head_outs:
            for b, (z0, y0, x0) in enumerate(chunk):
                kb = boxes_f[b]  # already score-filtered, aligned with pm
                n = min(kb.shape[0], pm_list[b].shape[0])
                if n == 0:
                    continue
                pm, pb = pm_list[b][:n], probs[b][keep[b], 0][:n]
                # vectorized: one interpolate + one host transfer per sample/class
                pm_v = pm.float().unsqueeze(1)  # [Nv,1,D,H,W]
                pm_v = F.interpolate(pm_v, size=(CZ, CXY, CXY), mode="nearest").squeeze(1) > 0
                sc_v = pb.float().view(-1, 1, 1, 1)
                sv = (pm_v.float() * sc_v).amax(0).cpu().numpy()  # [D,H,W] best score
                sl = (slice(z0, z0 + CZ), slice(y0, y0 + CXY), slice(x0, x0 + CXY))
                if score_by_c is not None:
                    reg_s = score_by_c[cid][sl]
                    upd = sv > reg_s
                    reg_s[upd] = sv[upd].astype(np.float16)
                else:
                    reg_s = score[sl]
                    reg_l = label[sl]
                    upd = sv > reg_s
                    reg_l[upd] = cid
                    reg_s[upd] = sv[upd].astype(np.float16)
                off = np.array([z0, y0, x0, z0, y0, x0])
                for k in range(n):
                    gb = kb[k].cpu().numpy() + off
                    boxes_all.append((cid, [round(v, 1) for v in gb.tolist()],
                                     round(float(pb[k]), 3)))
        if (i // BATCH) % 200 == 0:
            print(f"  [det_sam {tag}] {i}/{len(wins)} ({time.time()-t0:.0f}s)", flush=True)
    if score_by_c is not None:
        label, score = merge_voxel_scores(score_by_c, SCORE_THR_DET)
    return label, score, boxes_all


@torch.no_grad()
def infer_det_sam_mc(model, vol01: np.ndarray, cls_ids: dict, tag: str):
    """Multiclass det (mito/lyso) + single class-conditioned prompt decoder."""
    Z, Y, X = vol01.shape
    score = np.zeros((Z, Y, X), np.float16)
    label = np.zeros((Z, Y, X), np.uint8)
    score_by_c = {cid: np.zeros((Z, Y, X), np.float16) for cid in cls_ids.values()} if VOXEL_MUTEX else None
    boxes_all = []
    wins = list(iter_windows(Z, Y, X, CZ, CXY, CXY, SZ, SXY, SXY))
    t0 = time.time()
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        crops = np.stack([vol01[z0:z0 + CZ, y0:y0 + CXY, x0:x0 + CXY] for z0, y0, x0 in chunk])
        crops = (crops - crops.mean(axis=(1, 2, 3), keepdims=True)) / (crops.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        t = torch.from_numpy(crops[:, None]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.backbone(t)
            p3, p4, p5 = model.neck(feats)
            py = [p3, p4, p5]
            d = model.det_head(py)
            boxes, probs = model._top_boxes_mc(d, topk=12)
        p3f = p3.float()
        for b, (z0, y0, x0) in enumerate(chunk):
            kb, pb = boxes[b], probs[b]
            if kb.shape[0] == 0:
                continue
            fg_sc = pb.max(dim=-1).values
            keep = fg_sc > SCORE_THR_DET
            if not keep.any():
                continue
            kb, pb = kb[keep], pb[keep]
            fg_sc = fg_sc[keep]
            cids = torch.where(pb[:, 0] >= pb[:, 1], 2, 3)
            cls_probs = pb.float()
            with torch.autocast("cuda", enabled=False):
                pm = model.prompt_decoder(p3f[b : b + 1], [kb.float()], [cls_probs])["pred_masks"][0]
            n = min(kb.shape[0], pm.shape[0])
            pm_v = pm[:n].float().unsqueeze(1)
            pm_v = F.interpolate(pm_v, size=(CZ, CXY, CXY), mode="nearest").squeeze(1) > 0
            sc_v = fg_sc[:n].view(-1, 1, 1, 1)
            for k in range(n):
                cid = int(cids[k])
                if cid not in cls_ids.values():
                    continue
                sv = (pm_v[k].float() * sc_v[k]).cpu().numpy()
                sl = (slice(z0, z0 + CZ), slice(y0, y0 + CXY), slice(x0, x0 + CXY))
                if score_by_c is not None:
                    reg_s = score_by_c[cid][sl]
                    upd = sv > reg_s
                    reg_s[upd] = sv[upd].astype(np.float16)
                else:
                    reg_s = score[sl]
                    reg_l = label[sl]
                    upd = sv > reg_s
                    reg_l[upd] = cid
                    reg_s[upd] = sv.astype(np.float16)
                off = np.array([z0, y0, x0, z0, y0, x0])
                gb = kb[k].cpu().numpy() + off
                boxes_all.append((cid, [round(v, 1) for v in gb.tolist()], round(float(fg_sc[k]), 3)))
        if (i // BATCH) % 200 == 0:
            print(f"  [det_sam_mc {tag}] {i}/{len(wins)} ({time.time()-t0:.0f}s)", flush=True)
    if score_by_c is not None:
        label, score = merge_voxel_scores(score_by_c, SCORE_THR_DET)
    return label, score, boxes_all


# ---------------- stage C: instances + nucleus exclusion ----------------
def instances_from_label(label: np.ndarray, score: np.ndarray, cid: int, nuc: np.ndarray):
    """Connected components of one class -> instance table with exclusion flag."""
    binm = label == cid
    cc, n = ndi.label(binm)
    objs = ndi.find_objects(cc)
    nuc_b = nuc.astype(bool)
    out = []
    for iid, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = cc[sl] == iid
        vox = int(m.sum())
        if vox < 8:
            continue
        ov = float((m & nuc_b[sl]).sum()) / vox
        out.append(dict(
            id=iid, cls=cid, voxels=vox,
            box=[int(sl[0].start), int(sl[1].start), int(sl[2].start),
                 int(sl[0].stop), int(sl[1].stop), int(sl[2].stop)],
            score=float(score[sl][m].max()), nuc_overlap=round(ov, 3),
            kept=bool(ov < NUC_OVERLAP_THR),
        ))
    return out, cc


# ---------------- main ----------------
def main():
    global OUT_DIR, VIZ_DIR, VOXEL_MUTEX, CKPT_BIN, CKPT_DET
    global CZ, CXY, SZ, SXY, DET_MODE, DET_TAG, SKIP_M2F_BIN
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutex-voxels", action="store_true", help="per-voxel argmax across classes")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--viz-dir", type=Path, default=None)
    ap.add_argument("--ckpt-bin", type=Path, default=None)
    ap.add_argument("--ckpt-det", type=Path, default=None)
    ap.add_argument("--det-mode", choices=("det_sam", "det_sam_mc"), default="det_sam")
    ap.add_argument("--det-only", action="store_true", help="skip m2f_bin volume inference")
    ap.add_argument("--cz", type=int, default=None, help="crop depth (default 6 for 1000nm, use 10 for 1500x10)")
    ap.add_argument("--cxy", type=int, default=None, help="crop xy (default 125 @1000nm, 188 @1500nm)")
    ap.add_argument("--stride-z", type=int, default=None)
    ap.add_argument("--stride-xy", type=int, default=None)
    args = ap.parse_args()
    VOXEL_MUTEX = args.mutex_voxels
    DET_MODE = args.det_mode
    DET_TAG = args.det_mode
    SKIP_M2F_BIN = args.det_only
    if args.cz is not None:
        CZ = args.cz
    if args.cxy is not None:
        CXY = args.cxy
    if args.stride_z is not None:
        SZ = args.stride_z
    if args.stride_xy is not None:
        SXY = args.stride_xy
    if args.out_dir:
        OUT_DIR = args.out_dir
    if args.viz_dir:
        VIZ_DIR = args.viz_dir
    if args.ckpt_bin:
        CKPT_BIN = args.ckpt_bin
    if args.ckpt_det:
        CKPT_DET = args.ckpt_det
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    class_sup = load_class_supervision()
    nuc_model = UNet2D(base=32).to(DEV).eval()
    nuc_ck = torch.load(CKPT_NUC, map_location="cpu", weights_only=False)
    nuc_model.load_state_dict(nuc_ck["model"])
    print(f"[nuc] ckpt epoch={nuc_ck.get('epoch')} dice={nuc_ck.get('dice'):.4f}", flush=True)
    m_bin = ep_bin = None
    if not SKIP_M2F_BIN:
        m_bin, ep_bin = load_3d_model(CKPT_BIN, "m2f_bin")
    m_det, ep_det = load_3d_model(CKPT_DET, DET_MODE)
    print(
        f"[3d] crop={CZ}x{CXY}x{CXY} stride={SZ}/{SXY} "
        f"m2f_bin={f'e{ep_bin}' if m_bin else 'skip'}  {DET_TAG}=e{ep_det} mutex={VOXEL_MUTEX}",
        flush=True,
    )

    summary = {}
    for spec in VOLUMES:
        tag = spec["tag"]
        cls_ids = filter_classes(spec, class_sup) if class_sup else spec["classes"]
        if not cls_ids:
            print(f"[skip] {tag}: no supervised classes", flush=True)
            continue
        print(f"===== {tag} {spec['tif']} classes={list(cls_ids)} mutex={VOXEL_MUTEX}", flush=True)
        vol = tifffile.imread(str(spec["tif"])).astype(np.uint8)
        # stage A
        nuc_path = OUT_DIR / f"{tag}_nucleus.npy"
        nuc = np.load(nuc_path) if nuc_path.exists() else segment_nuclei(vol, nuc_model, tag)
        # per-volume gray unification -> [0,1]
        sub = vol[::4, ::4, ::4]
        lo, hi = np.percentile(sub, [0.5, 99.8])
        vol01 = np.clip((vol.astype(np.float32) - np.float32(lo)) / max(float(hi - lo), 1e-6),
                        0, 1).astype(np.float32)
        print(f"[{tag}] gray p0.5={lo:.1f} p99.8={hi:.1f}", flush=True)
        # stage B + C per model
        res = {}
        model_names = [] if SKIP_M2F_BIN else ["m2f_bin"]
        model_names.append(DET_TAG)
        for mname in model_names:
            lab_p = OUT_DIR / f"{tag}_{mname}_label.npy"
            sc_p = OUT_DIR / f"{tag}_{mname}_score.npy"
            is_det = mname in ("det_sam", "det_sam_mc")
            if lab_p.exists():
                label, score = np.load(lab_p), np.load(sc_p)
                boxes = json.loads((OUT_DIR / f"{tag}_{mname}_boxes.json").read_text()) if is_det else []
            else:
                if mname == "m2f_bin":
                    label, score = infer_m2f_bin(m_bin, vol01, cls_ids, tag)
                    boxes = []
                elif mname == "det_sam_mc":
                    label, score, boxes = infer_det_sam_mc(m_det, vol01, cls_ids, tag)
                else:
                    label, score, boxes = infer_det_sam(m_det, vol01, cls_ids, tag)
                np.save(lab_p, label)
                np.save(sc_p, score)
                if is_det:
                    (OUT_DIR / f"{tag}_{mname}_boxes.json").write_text(json.dumps(boxes))
            inst = {}
            for cname, cid in cls_ids.items():
                table, cc = instances_from_label(label, score, cid, nuc)
                inst[cname] = table
                kept = [t for t in table if t["kept"]]
                print(f"[{tag}/{mname}/{cname}] instances={len(table)} kept={len(kept)} "
                      f"excluded_by_nucleus={len(table)-len(kept)}", flush=True)
            (OUT_DIR / f"{tag}_{mname}_instances.json").write_text(json.dumps(inst, indent=1))
            res[mname] = (label, inst)
        summary[tag] = {
            m: {c: dict(total=len(inst[c]), kept=sum(t["kept"] for t in inst[c]))
                for c in cls_ids} for m, (_, inst) in res.items()
        }
        # stage D viz
        viz_volume(spec, vol, nuc, res)
        del vol, vol01, nuc, res
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def viz_volume(spec, vol, nuc, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = spec["tag"]
    Z, Y, X = vol.shape
    z = Z // 2
    nrows = 1 + len(res)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.6 * nrows))
    axes = np.atleast_2d(axes)
    em = vol[z]
    axes[0, 0].imshow(em, cmap="gray"); axes[0, 0].set_title(f"{tag} EM z={z}", fontsize=9)
    axes[0, 1].imshow(em, cmap="gray")
    axes[0, 1].imshow(np.ma.masked_where(nuc[z] == 0, nuc[z]), cmap="cool", alpha=0.5)
    axes[0, 1].set_title(f"nucleus mask (fg={nuc.mean():.3f})", fontsize=9)
    axes[0, 2].imshow(vol[:, Y // 2, :], cmap="gray", aspect=30 / 8)
    axes[0, 2].set_title("xz profile", fontsize=9)
    for r, (mname, (label, inst)) in enumerate(res.items(), start=1):
        axes[r, 0].imshow(em, cmap="gray")
        kept_rgb = np.zeros((*em.shape, 4), float)
        for cname, cid in spec["classes"].items():
            m = label[z] == cid
            kept_rgb[m] = (*CLASS_COLORS[cid], 0.75)
        axes[r, 0].imshow(kept_rgb)
        axes[r, 0].set_title(f"{mname} raw pred z={z}", fontsize=9)
        # kept vs excluded overlay via instance boxes
        for c, show_kept in ((1, True), (2, False)):
            axes[r, c].imshow(em, cmap="gray")
            for cname, table in inst.items():
                cid = spec["classes"][cname]
                for t in table:
                    if t["kept"] != show_kept:
                        continue
                    z0, y0, x0, z1, y1, x1 = t["box"]
                    if not (z0 <= z < z1):
                        continue
                    col = CLASS_COLORS[cid] if show_kept else (0.2, 0.6, 1.0)
                    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                         edgecolor=col, lw=1.2)
                    axes[r, c].add_patch(rect)
            n_show = sum(1 for cname, table in inst.items() for t in table if t["kept"] == show_kept)
            ttl = "kept (nucleus-free)" if show_kept else "excluded (nucleus overlap)"
            axes[r, c].set_title(f"{mname} {ttl}: {n_show}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{tag}: nucleus-filtered organelle inference", fontsize=12)
    fig.tight_layout()
    out = VIZ_DIR / f"{tag}_viz.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[viz] {out}", flush=True)


if __name__ == "__main__":
    main()
