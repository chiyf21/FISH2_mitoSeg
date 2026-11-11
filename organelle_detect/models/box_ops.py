"""3D detection utilities: boxes are (z1,y1,x1,z2,y2,x2) in crop voxel coords."""

from __future__ import annotations

import math

import torch


def box_iou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N,6], b: [M,6] -> [N,M] IoU."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a = a[:, None, :]
    b = b[None, :, :]
    z1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x1 = torch.maximum(a[..., 2], b[..., 2])
    z2 = torch.minimum(a[..., 3], b[..., 3])
    y2 = torch.minimum(a[..., 4], b[..., 4])
    x2 = torch.minimum(a[..., 5], b[..., 5])
    inter = (z2 - z1).clamp(min=0) * (y2 - y1).clamp(min=0) * (x2 - x1).clamp(min=0)
    vol_a = (a[..., 3] - a[..., 0]).clamp(min=0) * (a[..., 4] - a[..., 1]).clamp(min=0) * (a[..., 5] - a[..., 2]).clamp(min=0)
    vol_b = (b[..., 3] - b[..., 0]).clamp(min=0) * (b[..., 4] - b[..., 1]).clamp(min=0) * (b[..., 5] - b[..., 2]).clamp(min=0)
    union = vol_a + vol_b - inter
    return inter / union.clamp(min=1e-6)


def box_giou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Paired GIoU, a and b [N,6]."""
    z1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x1 = torch.maximum(a[:, 2], b[:, 2])
    z2 = torch.minimum(a[:, 3], b[:, 3])
    y2 = torch.minimum(a[:, 4], b[:, 4])
    x2 = torch.minimum(a[:, 5], b[:, 5])
    inter = (z2 - z1).clamp(min=0) * (y2 - y1).clamp(min=0) * (x2 - x1).clamp(min=0)
    vol_a = (a[:, 3] - a[:, 0]).clamp(min=0) * (a[:, 4] - a[:, 1]).clamp(min=0) * (a[:, 5] - a[:, 2]).clamp(min=0)
    vol_b = (b[:, 3] - b[:, 0]).clamp(min=0) * (b[:, 4] - b[:, 1]).clamp(min=0) * (b[:, 5] - b[:, 2]).clamp(min=0)
    union = vol_a + vol_b - inter
    iou = inter / union.clamp(min=1e-6)
    cz1 = torch.minimum(a[:, 0], b[:, 0])
    cy1 = torch.minimum(a[:, 1], b[:, 1])
    cx1 = torch.minimum(a[:, 2], b[:, 2])
    cz2 = torch.maximum(a[:, 3], b[:, 3])
    cy2 = torch.maximum(a[:, 4], b[:, 4])
    cx2 = torch.maximum(a[:, 5], b[:, 5])
    hull = (cz2 - cz1).clamp(min=0) * (cy2 - cy1).clamp(min=0) * (cx2 - cx1).clamp(min=0)
    return iou - (hull - union) / hull.clamp(min=1e-6)


def dist2bbox3d(distance: torch.Tensor, anchor_zyx: torch.Tensor) -> torch.Tensor:
    """distance: [B, 6, A] (z1,y1,x1,z2,y2,x2 distances), anchors [3, A] -> boxes [B,A,6]."""
    z, y, x = anchor_zyx
    z1 = z - distance[:, 0]
    y1 = y - distance[:, 1]
    x1 = x - distance[:, 2]
    z2 = z + distance[:, 3]
    y2 = y + distance[:, 4]
    x2 = x + distance[:, 5]
    return torch.stack([z1, y1, x1, z2, y2, x2], dim=-1)


def bbox2dist3d(anchor_zyx: torch.Tensor, boxes: torch.Tensor, stride6: torch.Tensor, reg_max: int):
    """Inverse of dist2bbox3d in stride units for DFL targets.

    anchor_zyx [N,3], boxes [N,6], stride6 [N,6] -> distances [N,6] clamped to [0, reg_max-1-0.01].
    """
    d = torch.cat([anchor_zyx - boxes[:, :3], boxes[:, 3:] - anchor_zyx], dim=1) / stride6
    return d.clamp(0, reg_max - 1 - 0.01)


def bbox_ciou_2d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU of (y1,x1,y2,x2) boxes, broadcast over leading dims (Ultralytics bbox_iou CIoU=True)."""
    ay1, ax1, ay2, ax2 = a.unbind(-1)
    by1, bx1, by2, bx2 = b.unbind(-1)
    ah, aw = ay2 - ay1, (ax2 - ax1).clamp(min=eps)
    bh, bw = by2 - by1, (bx2 - bx1).clamp(min=eps)
    inter = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0) * (
        torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)
    ).clamp(min=0)
    union = ah * aw + bh * bw - inter + eps
    iou = inter / union
    ch = torch.maximum(ay2, by2) - torch.minimum(ay1, by1)
    cw = torch.maximum(ax2, bx2) - torch.minimum(ax1, bx1)
    c2 = ch.pow(2) + cw.pow(2) + eps
    rho2 = ((by1 + by2 - ay1 - ay2).pow(2) + (bx1 + bx2 - ax1 - ax2).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(bw / bh.clamp(min=eps)) - torch.atan(aw / ah.clamp(min=eps))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def slice_boxes_from_masks(masks: torch.Tensor, labels: torch.Tensor, min_pixels: int = 4):
    """Per-z-slice 2D GT boxes from instance masks.

    masks [N,D,H,W] bool, labels [N] -> boxes [M,6] as (z, y1, x1, z+1, y2, x2) with
    exclusive max edges, labels [M], inst [M] (source instance index). Slices where
    an instance has fewer than min_pixels voxels are skipped.
    """
    n, d, h, w = masks.shape
    dev = masks.device
    if n == 0:
        return masks.new_zeros((0, 6), dtype=torch.float32), labels.new_zeros((0,)), labels.new_zeros((0,))
    rows = masks.any(-1)  # [N,D,H]
    cols = masks.any(-2)  # [N,D,W]
    area = masks.flatten(2).sum(-1)  # [N,D]
    ys = torch.arange(h, device=dev)
    xs = torch.arange(w, device=dev)
    y1 = torch.where(rows, ys, h).amin(-1)
    y2 = torch.where(rows, ys, -1).amax(-1) + 1
    x1 = torch.where(cols, xs, w).amin(-1)
    x2 = torch.where(cols, xs, -1).amax(-1) + 1
    ni, zi = (area >= min_pixels).nonzero(as_tuple=True)
    zf = zi.float()
    boxes = torch.stack([zf, y1[ni, zi].float(), x1[ni, zi].float(), zf + 1,
                         y2[ni, zi].float(), x2[ni, zi].float()], dim=1)
    return boxes, labels[ni], ni


def link_slice_boxes(boxes: torch.Tensor, scores: torch.Tensor, payload: torch.Tensor | None = None,
                     link_iou: float = 0.3, max_gap: int = 1):
    """Chain per-slice 2D detections into 3D instances (greedy IoU tracking along z).

    boxes [K,6] thin (z, y1, x1, z+1, y2, x2), scores [K], payload [K,C] optional.
    A detection on slice z extends the best-overlapping track last seen on z-1..z-1-max_gap
    if its 2D IoU >= link_iou. Returns 3D union boxes [T,6], score [T] (max over the
    track), payload of the max-score member [T,C], and the member index lists.
    """
    if boxes.shape[0] == 0:
        c = payload.shape[1] if payload is not None else 0
        return boxes.new_zeros((0, 6)), scores.new_zeros((0,)), scores.new_zeros((0, c)), []
    z = boxes[:, 0].round().long().tolist()
    yx = boxes[:, [1, 2, 4, 5]]
    by_z: dict[int, list[int]] = {}
    for i in scores.argsort(descending=True).tolist():
        by_z.setdefault(z[i], []).append(i)
    tracks: list[list[int]] = []
    last_z: list[int] = []
    for zz in sorted(by_z):
        cand = by_z[zz]
        open_t = [t for t in range(len(tracks)) if 1 <= zz - last_z[t] <= 1 + max_gap]
        taken_c, taken_t = set(), set()
        if open_t:
            a = yx[cand][:, None]
            b = yx[[tracks[t][-1] for t in open_t]][None]
            inter = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0) * (
                torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])
            ).clamp(min=0)
            area = lambda t: (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
            iou = inter / (area(a) + area(b) - inter).clamp(min=1e-6)  # [C,T]
            for flat in iou.flatten().argsort(descending=True).tolist():
                ci, ti = divmod(flat, len(open_t))
                if iou[ci, ti] < link_iou:
                    break
                if ci in taken_c or ti in taken_t:
                    continue
                taken_c.add(ci)
                taken_t.add(ti)
                tracks[open_t[ti]].append(cand[ci])
                last_z[open_t[ti]] = zz
        for ci, i in enumerate(cand):
            if ci not in taken_c:
                tracks.append([i])
                last_z.append(zz)
    out_b, out_s, out_p = [], [], []
    for t in tracks:
        tb = boxes[t]
        out_b.append(torch.cat([tb[:, :3].amin(0), tb[:, 3:].amax(0)]))
        best = t[int(scores[t].argmax())]
        out_s.append(scores[best])
        if payload is not None:
            out_p.append(payload[best])
    out_p = torch.stack(out_p) if payload is not None else scores.new_zeros((len(tracks), 0))
    return torch.stack(out_b), torch.stack(out_s), out_p, tracks


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Greedy 3D NMS; boxes [N,6], scores [N] -> kept indices sorted by score desc."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    iou = box_iou_3d(boxes[order].float(), boxes[order].float())
    suppressed = torch.zeros(order.shape[0], dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(order.shape[0]):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_thr
    return order[torch.as_tensor(keep, dtype=torch.long, device=boxes.device)]


def make_anchors_3d(feats: list[torch.Tensor], strides: tuple, grid_cell=0.5):
    """Return anchors [3, A] and per-axis stride [A, 3] in input-crop voxels.

    strides may be per-level scalars or per-level (z, y, x) tuples. Always float32:
    bf16 cannot represent half-voxel anchor centers above 128.
    """
    anchor_points, stride_tensor = [], []
    dtype, device = torch.float32, feats[0].device
    for i, feat in enumerate(feats):
        _, _, d, h, w = feat.shape
        s = strides[i]
        if isinstance(s, (tuple, list)):
            sz_s, sy_s, sx_s = float(s[0]), float(s[1]), float(s[2])
        else:
            sz_s = sy_s = sx_s = float(s)
        sz = (torch.arange(d, device=device, dtype=dtype) + grid_cell) * sz_s
        sy = (torch.arange(h, device=device, dtype=dtype) + grid_cell) * sy_s
        sx = (torch.arange(w, device=device, dtype=dtype) + grid_cell) * sx_s
        zz, yy, xx = torch.meshgrid(sz, sy, sx, indexing="ij")
        anchor_points.append(torch.stack([zz, yy, xx], dim=0).reshape(3, -1))
        st = torch.tensor([sz_s, sy_s, sx_s], device=device, dtype=dtype)
        stride_tensor.append(st[None, :].expand(d * h * w, 3))
    return torch.cat(anchor_points, dim=1), torch.cat(stride_tensor, dim=0)


# ---- 2D counterparts (boxes are (y1,x1,y2,x2) in crop voxels) ----


def box_iou_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N,4], b: [M,4] -> [N,M] IoU."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a = a[:, None, :]
    b = b[None, :, :]
    y1 = torch.maximum(a[..., 0], b[..., 0])
    x1 = torch.maximum(a[..., 1], b[..., 1])
    y2 = torch.minimum(a[..., 2], b[..., 2])
    x2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (y2 - y1).clamp(min=0) * (x2 - x1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def box_giou_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Paired GIoU, a and b [N,4]."""
    y1 = torch.maximum(a[:, 0], b[:, 0])
    x1 = torch.maximum(a[:, 1], b[:, 1])
    y2 = torch.minimum(a[:, 2], b[:, 2])
    x2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (y2 - y1).clamp(min=0) * (x2 - x1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=1e-6)
    cy1 = torch.minimum(a[:, 0], b[:, 0])
    cx1 = torch.minimum(a[:, 1], b[:, 1])
    cy2 = torch.maximum(a[:, 2], b[:, 2])
    cx2 = torch.maximum(a[:, 3], b[:, 3])
    hull = (cy2 - cy1).clamp(min=0) * (cx2 - cx1).clamp(min=0)
    return iou - (hull - union) / hull.clamp(min=1e-6)


def dist2bbox2d(distance: torch.Tensor, anchor_yx: torch.Tensor) -> torch.Tensor:
    """distance: [B, 4, A] (y1,x1,y2,x2 distances), anchors [2, A] -> boxes [B,A,4]."""
    y, x = anchor_yx
    y1 = y - distance[:, 0]
    x1 = x - distance[:, 1]
    y2 = y + distance[:, 2]
    x2 = x + distance[:, 3]
    return torch.stack([y1, x1, y2, x2], dim=-1)


def make_anchors_2d(feats: list[torch.Tensor], strides: tuple, grid_cell=0.5):
    """Return anchors [2, A] and per-axis stride [A, 2] in input-crop voxels."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, feat in enumerate(feats):
        _, _, h, w = feat.shape
        s = strides[i]
        sy_s = sx_s = float(s) if not isinstance(s, (tuple, list)) else float(s[-1])
        if isinstance(s, (tuple, list)) and len(s) >= 2:
            sy_s, sx_s = float(s[-2]), float(s[-1])
        sy = (torch.arange(h, device=device, dtype=dtype) + grid_cell) * sy_s
        sx = (torch.arange(w, device=device, dtype=dtype) + grid_cell) * sx_s
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack([yy, xx], dim=0).reshape(2, -1))
        st = torch.tensor([sy_s, sx_s], device=device, dtype=dtype)
        stride_tensor.append(st[None, :].expand(h * w, 2))
    return torch.cat(anchor_points, dim=1), torch.cat(stride_tensor, dim=0)
