"""Keep the 26-connected union component most supported by original reference mask."""
import numpy as np
from scipy import ndimage as ndi

def retain_connected_body(mask,origin,seed,seed_xy,seed_z,original=None,original_origin=None):
    cc,n=ndi.label(mask,structure=np.ones((3,3,3),bool))
    sizes=np.bincount(cc.reshape(-1),minlength=n+1);sizes[0]=0
    support=np.zeros(n+1,np.int64);j=int(seed_z-origin[2]);x=int(seed_xy[0]-origin[0]);y=int(seed_xy[1]-origin[1])
    if 0<=j<len(mask):
        yy,xx=np.nonzero(seed);xx=xx+x;yy=yy+y;ok=(xx>=0)&(xx<mask.shape[2])&(yy>=0)&(yy<mask.shape[1])
        support=np.bincount(cc[j,yy[ok],xx[ok]],minlength=n+1);support[0]=0
    original_votes=np.zeros(n+1,np.int64)
    if original is not None:
        x,y,z=np.asarray(original_origin)-np.asarray(origin)
        original_votes=np.bincount(cc[z:z+len(original),y:y+original.shape[1],x:x+original.shape[2]][original],minlength=n+1);original_votes[0]=0
    winner=max(range(1,n+1),key=lambda c:(int(support[c]),int(original_votes[c]),int(sizes[c]),-c)) if n else 0
    out=(cc==winner) if winner else np.zeros_like(mask)
    audit=dict(connectivity=26,components_before=int(n),components_after=int(bool(winner)),selected_component=int(winner),selection='maximum overlap with original reference seed, then overlap with original 3D mask, then volume, then label',seed_z=int(seed_z),seed_support_voxels=int(support[winner]) if winner else 0,kept_voxels=int(out.sum()),removed_components=max(0,int(n)-int(bool(winner))),removed_voxels=int(mask.sum()-out.sum()),uses_gt=False)
    assert ndi.label(out,structure=np.ones((3,3,3),bool))[1]<=1
    assert not np.any(out&~mask)
    return out,audit
