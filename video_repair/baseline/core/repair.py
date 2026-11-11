"""Shared native-grid black-plane repair. Padding is never treated as missing EM."""
import numpy as np


def interpolate_stack(a, valid=None):
    a=np.asarray(a)
    valid=np.ones(a.shape,bool) if valid is None else np.asarray(valid,bool)
    if a.ndim!=3 or valid.shape!=a.shape:raise ValueError('Expected ZYX image and matching validity')
    # Classify only real source pixels; do not classify entirely padded planes.
    bad=np.array([v.any() and np.all(im[v]<=1) for im,v in zip(a,valid)])
    good=np.array([i for i,v in enumerate(valid) if v.any() and not bad[i]])
    if bad.any() and not len(good):raise ValueError('No valid bracketing source plane')
    out=a.copy();audit=[]
    for z in np.flatnonzero(bad):
        lo=good[good<z];hi=good[good>z];l=int(lo[-1]) if len(lo) else int(hi[0]);h=int(hi[0]) if len(hi) else l
        w=(z-l)/(h-l) if h!=l else 0.
        available=valid[z]&valid[l]&valid[h]
        if not np.array_equal(available,valid[z]):raise ValueError('Interpolation source coverage differs')
        values=(1-w)*a[l].astype(np.float32)+w*a[h].astype(np.float32)
        if np.issubdtype(a.dtype,np.integer):values=np.rint(values)
        out[z,available]=values[available].astype(a.dtype)
        audit.append({'z':int(z),'lower_z':l,'upper_z':h,'upper_weight':float(w),'method':'linear' if l!=h else 'boundary_nearest'})
    return out,audit
