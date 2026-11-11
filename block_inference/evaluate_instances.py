"""Optional explicit GT evaluation, restricted to the configured detection ROI."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from common import config,save,roi_for
from block_raw import BlockRaw

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--gt',required=True,help='Native integer ZYX instance labels,0 background');a=p.parse_args();d=config(a.config);raw=BlockRaw(d['raw']);roi=roi_for(raw,d)
    pred=np.load(Path(d['output'])/'video/assembly/labels.npy',mmap_mode='r');gt=np.load(a.gt,mmap_mode='r')
    if gt.shape!=pred.shape or not np.issubdtype(gt.dtype,np.integer):raise ValueError('GT must be integer instance labels with same native ZYX shape as raw')
    pa=Counter();ga=Counter();overlap=Counter();x0,y0,z0,x1,y1,z1=roi
    for z in range(z0,z1):
        for y in range(y0,y1,128):
            b=pred[z,y:min(y+128,y1),x0:x1];g=gt[z,y:min(y+128,y1),x0:x1]
            if g.min()<0 or g.max()>2**32-1:raise ValueError('GT instance IDs must fit uint32')
            ids,n=np.unique(b,return_counts=True);pa.update({int(i):int(v) for i,v in zip(ids,n) if i})
            ids,n=np.unique(g,return_counts=True);ga.update({int(i):int(v) for i,v in zip(ids,n) if i})
            valid=(b>0)&(g>0);keys=(b[valid].astype(np.uint64)<<32)|g[valid].astype(np.uint64);ids,n=np.unique(keys,return_counts=True);overlap.update({(int(k)>>32,int(k)&0xffffffff):int(v) for k,v in zip(ids,n)})
    rows=[]
    for pid,area in sorted(pa.items()):
        votes=[(v,gid) for (pp,gid),v in overlap.items() if pp==pid];inter,gid=max(votes,default=(0,None));iou=inter/(area+ga[gid]-inter) if gid else None
        rows.append(dict(instance_id=pid,gt_id=gid,pred_voxels_in_roi=area,gt_voxels_in_roi=ga[gid] if gid else None,intersection=inter,iou=iou))
    save(Path(d['output'])/'evaluation.json',dict(roi=roi,gt=str(Path(a.gt).resolve()),matching='maximum GT voxel vote restricted to detection ROI',instances=rows,unmatched_predictions=sum(r['gt_id'] is None for r in rows)))

if __name__=='__main__':main()
