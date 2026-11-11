"""Coordinate-safe 2D duplicate grouping and overlapping-video 3D assembly."""
from collections import defaultdict
import numpy as np


def starts(n, window, stride):
    if not 0 < stride <= window:
        raise ValueError('Require 0 < stride <= window')
    if n <= window:
        return [0]
    return sorted(set(range(0, n-window+1, stride)) | {n-window})


class UnionFind:
    def __init__(self, n): self.p=list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self, a, b):
        a,b=self.find(a),self.find(b)
        if a != b:self.p[max(a,b)]=min(a,b)


def mask_iou(a, b):
    """Full-mask IoU in global coordinates; NOT IoU just within crop overlap."""
    ax,ay=a['origin_xy'];bx,by=b['origin_xy'];am=a['mask'];bm=b['mask']
    lx=max(ax,bx);ly=max(ay,by);hx=min(ax+am.shape[1],bx+bm.shape[1]);hy=min(ay+am.shape[0],by+bm.shape[0])
    inter=0
    if hx>lx and hy>ly:
        inter=int((am[ly-ay:hy-ay,lx-ax:hx-ax] & bm[ly-by:hy-by,lx-bx:hx-bx]).sum())
    union=int(am.sum())+int(bm.sum())-inter
    return inter/union if union else 0.


def partial_overlap(a,b):
    """Evidence for crop-truncated fragments; evaluated only in shared image FOV."""
    if not (a.get('touches_crop_edge') or b.get('touches_crop_edge')):return 0.,0
    ax,ay=a['origin_xy'];bx,by=b['origin_xy'];am=a['mask'];bm=b['mask']
    # Identical fields of view are competing segmentations, not partial fragments.
    if (ax,ay,am.shape)==(bx,by,bm.shape):return 0.,0
    lx=max(ax,bx);ly=max(ay,by);hx=min(ax+am.shape[1],bx+bm.shape[1]);hy=min(ay+am.shape[0],by+bm.shape[0])
    if hx-lx<16 or hy-ly<16:return 0.,0
    aa=am[ly-ay:hy-ay,lx-ax:hx-ax];bb=bm[ly-by:hy-by,lx-bx:hx-bx]
    inter=int((aa&bb).sum());union=int((aa|bb).sum())
    return inter/max(1,union),inter


def assemble(tracks, threshold=.8, min_shared_layers=2):
    """Same-slice connected components select highest SAM IoU-head score.

    Video identity links slices. Track fragments are joined only when >=2
    shared Z layers independently have full-mask IoU > threshold (configurable).
    A single-layer fragment requires its one shared layer. No morphology edits.
    """
    by_z=defaultdict(list)
    for ti,track in enumerate(tracks):
        for frame in track['frames']:
            if frame['mask'].any():by_z[frame['z']].append((ti,frame))
    uf=UnionFind(len(tracks));support=defaultdict(set);partial_support=defaultdict(set);layer_groups={};edges=[];partial_edges=[]
    for z,frames in sorted(by_z.items()):
        local=UnionFind(len(frames))
        for i,(ta,a) in enumerate(frames):
            for j in range(i):
                tb,b=frames[j]
                if ta==tb:continue
                iou=mask_iou(a,b)
                if iou>threshold:
                    local.union(i,j);key=tuple(sorted((ta,tb)));support[key].add(z)
                    edges.append({'z':int(z),'track_a':ta,'track_b':tb,'mask_iou':iou})
                else:
                    overlap,inter=partial_overlap(a,b)
                    if overlap>threshold and inter>=16:
                        partial_support[tuple(sorted((ta,tb)))].add(z)
                        partial_edges.append({'z':int(z),'track_a':ta,'track_b':tb,'shared_fov_iou':overlap,'shared_foreground_pixels':inter})
        groups=defaultdict(list)
        for i,pair in enumerate(frames):groups[local.find(i)].append(pair)
        layer_groups[z]=list(groups.values())
    for (a,b),zz in support.items():
        na=sum(f['mask'].any() for f in tracks[a]['frames']);nb=sum(f['mask'].any() for f in tracks[b]['frames'])
        if len(zz)>=min(min_shared_layers,na,nb):uf.union(a,b)
    # Partial fragments require two independent shared Z slices. Keep BOTH masks
    # so their nonoverlapping spatial support survives the global union.
    for (a,b),zz in partial_support.items():
        if len(zz)>=min_shared_layers:uf.union(a,b)
    objects=defaultdict(list);ambiguous=[]
    for z,groups in layer_groups.items():
        for members in groups:
            winner=max(members,key=lambda v:(v[1]['score'],v[1]['detection_score'],-v[0]))
            ids=sorted({uf.find(ti) for ti,_ in members})
            # Same-slice duplicate still appears once, even if 3D support is weak.
            if len(ids)>1:ambiguous.append({'z':int(z),'roots':ids,'selected_track':winner[0]})
            objects[uf.find(winner[0])].append({**winner[1],'track_index':winner[0],'duplicate_count':len(members)})
    return dict(objects),{'duplicate_edges':edges,'partial_fragment_edges':partial_edges,'weak_3d_links':ambiguous,'threshold':threshold,'min_shared_layers':min_shared_layers}


def paste(frame, bbox_xyz):
    x0,y0,z0,x1,y1,z1=bbox_xyz;out=np.zeros((y1-y0,x1-x0),bool)
    x,y=frame['origin_xy'];m=frame['mask'];lx=max(x0,x);ly=max(y0,y);hx=min(x1,x+m.shape[1]);hy=min(y1,y+m.shape[0])
    if hx>lx and hy>ly:out[ly-y0:hy-y0,lx-x0:hx-x0]|=m[ly-y:hy-y,lx-x:hx-x]
    return out
