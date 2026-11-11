"""Pure rules for forward seeding; candidates all see the SAME propagated snapshot."""
import math
import numpy as np

def global_slice_nms(detections, threshold=.6):
    """Greedy same-Z NMS across every cached tile/window, in native XY.

    Retain the original highest slice-score box. Stable ties use detection ID.
    Only already retained boxes suppress later boxes; no transitive merging.
    """
    byz={}
    for d in detections:byz.setdefault(d['z'],[]).append(d)
    kept=[];audit=[]
    for z,rows in sorted(byz.items()):
        selected=[]
        for d in sorted(rows,key=lambda d:(-d['slice_score'],d['global_detection_id'])):
            b=np.asarray(d['box_xyxy'],dtype=float)
            if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:raise ValueError('invalid box geometry')
            suppressor=None
            for k in selected:
                a=np.asarray(k['box_xyxy'],dtype=float)
                inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
                union=float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
                overlap=inter/union
                if overlap>threshold:suppressor=(k['global_detection_id'],overlap);break
            if suppressor is None:selected.append(d);kept.append(d)
            audit.append(dict(detection_id=d['global_detection_id'],z=z,nms_suppressed=suppressor is not None,nms_suppressor=suppressor[0] if suppressor else None,nms_iou=suppressor[1] if suppressor else None))
    return kept,audit
def mask_box_iom(mask,origin_xy,box):
    area=int(mask.sum())
    if not area:return 0.
    x,y=origin_xy;x0,y0,x1,y1=[int(math.ceil(float(v)-.5)) for v in box];lx=max(x0,x);ly=max(y0,y);hx=min(x1,x+mask.shape[1]);hy=min(y1,y+mask.shape[0])
    inter=int(mask[ly-y:hy-y,lx-x:hx-x].sum()) if hx>lx and hy>ly else 0
    return inter/area

def mask_box_overlap(mask, origin_xy, box):
    """Both directional coverages on the same native pixel-centre grid.

    Box denominator includes its entire extent, including outside crop/source.
    """
    x,y=origin_xy;x0,y0,x1,y1=[int(math.ceil(float(v)-.5)) for v in box]
    lx,ly=max(x0,x),max(y0,y);hx,hy=min(x1,x+mask.shape[1]),min(y1,y+mask.shape[0])
    inter=int(mask[ly-y:hy-y,lx-x:hx-x].sum()) if hx>lx and hy>ly else 0
    mask_area=int(mask.sum());box_area=max(0,x1-x0)*max(0,y1-y0)
    iom=inter/mask_area if mask_area else 0.;opm=inter/box_area if box_area else 0.
    return dict(iom=iom,opm=opm,overlap_max=max(iom,opm),intersection=inter,mask_area=mask_area,box_area=box_area)

def box_foreground_fraction(plane, box):
    """Count only the requested box ROI; avoid rescanning a full nuclear plane."""
    x0,y0,x1,y1=[int(math.ceil(float(v)-.5)) for v in box]
    area=max(0,x1-x0)*max(0,y1-y0)
    lx,ly=max(0,x0),max(0,y0);hx,hy=min(plane.shape[1],x1),min(plane.shape[0],y1)
    inter=int(plane[ly:hy,lx:hx].sum()) if hx>lx and hy>ly else 0
    return inter/area if area else 0.
def filter_boxes(boxes,propagated,origin_xy,threshold=.8):
    result=[]
    for b in boxes:
        options=[(mask_box_iom(m,origin_xy,b['box_xyxy']),-int(tid)) for tid,m in propagated]
        best=max(options,default=(0.,0));result.append(dict(detection_id=b['global_detection_id'],suppressed=bool(best[0]>threshold),iom=best[0],covered_by_track=-best[1] if best[0]>threshold else None))
    return result
