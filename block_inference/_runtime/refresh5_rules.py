"""Pure helpers for exact mask-derived prompts and 26-neighbour assembly."""
import numpy as np
from scipy import ndimage as ndi

def tight_prompt(mask,origin_xy):
    yy,xx=np.nonzero(mask)
    if not len(xx):return None
    return [origin_xy[0]+int(xx.min()),origin_xy[1]+int(yy.min()),origin_xy[0]+int(xx.max())+1,origin_xy[1]+int(yy.max())+1]

def assemble26(binary,min_voxels=1000):
    # Global 26-CC is equivalent to first splitting each XY slice with 8-CC,
    # then connecting those components through adjacent Z with 26-neighbours.
    labels,count=ndi.label(binary,structure=np.ones((3,3,3),bool))
    sizes=np.bincount(labels.ravel());objects=ndi.find_objects(labels)
    keep=[]
    for cid,sl in enumerate(objects,1):
        if sl is not None and sizes[cid]>=min_voxels and sl[0].stop-sl[0].start>=2:
            keep.append((cid,sl,int(sizes[cid])))
    return labels,keep,count
