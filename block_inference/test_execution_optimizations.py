import numpy as np
import tempfile,shutil
from pathlib import Path
from common import ROOT
from common import save
from sequential_rules import mask_box_overlap
from execution_optimizations import SnapshotIndex,overlap
from assemble_instances import assemble

def main():
    rng=np.random.default_rng(19);snapshot=[]
    for i in range(70):
        shape=tuple(map(int,rng.integers(1,150,2)));mask=rng.random(shape)>.7;xy=list(map(int,rng.integers(-100,600,2)))
        if i%13==0:mask[:]=0
        snapshot.append((i,mask,dict(origin_xy=xy)))
    index=SnapshotIndex(snapshot)
    for j in range(500):
        x,y=rng.uniform(-160,700,2);w,h=rng.uniform(0,300,2);box=[x,y,x+w,y+h]
        original=[(mask_box_overlap(m,f['origin_xy'],box)['overlap_max'],-i) for i,m,f in snapshot]
        baseline=max(original,default=(0.,0));candidates=[]
        for k,(i,m,f) in index.query(box):
            a=overlap(m,f['origin_xy'],box,index.areas[k]);b=mask_box_overlap(m,f['origin_xy'],box);assert a==b;candidates.append((a['overlap_max'],-i))
        fast=max(candidates,default=(0.,0));assert fast[0]==baseline[0]
        if baseline[0]>0:assert fast==baseline
    print('PASS exact spatial filtering, both coverage fractions, mask-area cache, zero/tie/native negative-coordinate cases')
    root=Path(tempfile.mkdtemp(prefix='sparse_26cc_test_'));a=root/'a';(a/'tracks').mkdir(parents=True);(a/'frames').mkdir();frames=[]
    for z in [1,2,5,6]:
        np.savez_compressed(a/'frames'/f'z{z:03d}_track00001.npz',mask=np.ones((30,40),bool),origin_xy=[580,140],score=.8);frames.append(dict(z=z))
    save(a/'tracks/track_00001.json',dict(track_id=1,frames=frames));b=root/'b';shutil.copytree(a,b)
    ra=assemble(a,(8,300,1200));rb=assemble(b,(8,300,1200),optimized=True)
    assert ra==rb and np.array_equal(np.load(a/'assembly/labels.npy'),np.load(b/'assembly/labels.npy'))
    print('PASS sparse zero-plane/tile writes preserve labels, IDs and Z gaps exactly')

if __name__=='__main__':main()
