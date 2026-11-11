"""Exact seed IoM against the entire source instance slice, on the augmented grid."""
from common import *
from functools import lru_cache
import math,numpy as np
from scipy import ndimage as ndi
from source_data import labels,affine

BOUNDS=ROOT/'source_cache/mito_full_xyz_bounds.json'

def prepare_bounds():
    if not BOUNDS.exists():
        bounds={}
        for iid,box in enumerate(ndi.find_objects(labels(2)),1):
            if box is not None:bounds[str(iid)]=[[s.start,s.stop] for s in box]
        save(BOUNDS,bounds)

@lru_cache(1)
def bounds():return json.loads(BOUNDS.read_text())

@lru_cache(1)
def source_labels():return labels(2)

@lru_cache(256)
def complete_mask(iid,z,cx,cy,cz,angle):
    plan={'center_xyz':[cx,cy,cz],'angle':angle};_,matrix,offset=affine(plan)
    data=source_labels();bb=bounds()[str(iid)];assert bb[0][0]<=z<bb[0][1]
    # Pad tight input bounds so nearest-neighbor edge support matches a full
    # source-slice transform, including fractional coordinates near mask edges.
    y0=max(0,bb[1][0]-2);y1=min(data.shape[1],bb[1][1]+2)
    x0=max(0,bb[2][0]-2);x1=min(data.shape[2],bb[2][1]+2)
    binary=np.asarray(data[z,y0:y1,x0:x1]==iid)
    ys,xs=np.nonzero(binary)
    if not len(ys):return np.zeros((1,1),bool),np.array([0,0])
    corners=np.array([[ys.min()+y0-.5,xs.min()+x0-.5],[ys.min()+y0-.5,xs.max()+x0+.5],[ys.max()+y0+.5,xs.min()+x0-.5],[ys.max()+y0+.5,xs.max()+x0+.5]])
    transformed=(matrix.T@(corners-offset).T).T
    lo=np.floor(transformed.min(0)).astype(int)-3;hi=np.ceil(transformed.max(0)).astype(int)+4
    full=ndi.affine_transform(binary,matrix,offset=offset+matrix@lo-np.array([y0,x0]),output_shape=tuple(hi-lo),order=0,mode='constant',cval=0,prefilter=False)
    return full,lo

def measure(rec):
    cx,cy,cz=rec['plan']['center_xyz'];z=rec['video_z_bounds'][0]+rec['seed_frame']
    mask,origin=complete_mask(rec['source_instance_id'],z,cx,cy,cz,rec['plan']['angle'])
    area=int(mask.sum());assert area>0,rec['sample_id']
    x0,y0,x1,y1=rec['seed_box_xyxy']
    yy0=max(0,int(math.ceil(y0-.5))-int(origin[0]));yy1=min(mask.shape[0],int(math.ceil(y1-.5))-int(origin[0]))
    xx0=max(0,int(math.ceil(x0-.5))-int(origin[1]));xx1=min(mask.shape[1],int(math.ceil(x1-.5))-int(origin[1]))
    inter=int(mask[yy0:yy1,xx0:xx1].sum()) if yy1>yy0 and xx1>xx0 else 0
    return {'iom':inter/area,'seed_gt_pixels_full':area,'seed_gt_pixels_in_box':inter,'seed_gt_pixels_outside_box':area-inter,'iom_reference':'complete source GT instance slice on identical augmented pixel grid'}
