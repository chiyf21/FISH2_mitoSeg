"""Meaningful CPU regressions for bounded I/O, NMS and out-of-core 26CC."""
import tempfile,json
from pathlib import Path
import numpy as np
from common import save
from block_raw import BlockRaw
from filter_boxes import nms
from sequential_rules import global_slice_nms
from repair import interpolate_stack
from assemble_instances import assemble
from scipy import ndimage as ndi

def main():
    root=Path(tempfile.mkdtemp(prefix='em_bundle_test_'))
    rng=np.random.default_rng(7);raw=rng.integers(2,256,(7,40,50),dtype=np.uint8);raw[0]=0;raw[2]=0;raw[3]=1;np.save(root/'raw.npy',raw)
    reader=BlockRaw(root/'raw.npy',cache_mib=.1);expected,_=interpolate_stack(raw)
    for z in range(7):
        a,v=reader.slice_crop(z,0,0,50);assert np.array_equal(a[:40],expected[z]);assert not a[40:].any();assert v.sum()==2000
    a,v=reader.slice_crop(2,-10,-5,60);assert not a[:5].any() and not a[:,:10].any();assert np.array_equal(a[5:45,10:60],expected[2])
    print('PASS exact native black-plane interpolation, boundary nearest, padding and bounded cache')
    rows=[]
    for i in range(180):
        x,y=rng.uniform(0,100,2);w,h=rng.uniform(5,35,2);rows.append(dict(z=i%3,global_detection_id=i,box_xyxy=[x,y,x+w,y+h],slice_score=float(rng.integers(1,10)/10)))
    a,_=nms(rows,.4);b,_=global_slice_nms(rows,.4);assert [r['global_detection_id'] for r in a]==[r['global_detection_id'] for r in b]
    print('PASS vectorized NMS exactly matches reference greedy NMS with score ties')
    run=root/'assembly';(run/'tracks').mkdir(parents=True);(run/'frames').mkdir();m=np.zeros((6,70,70),bool);m[0:2,0:25,0:25]=True;m[2:4,25:50,25:50]=True;m[5,0:35,0:35]=True;m[0:2,60:62,60:62]=True
    for z in range(6):np.savez_compressed(run/'frames'/f'z{z:03d}_track00001.npz',mask=m[z],origin_xy=[0,0],score=.8)
    save(run/'tracks/track_00001.json',dict(track_id=1,frames=[dict(z=z) for z in range(6)]));rows=assemble(run,m.shape)
    result=np.load(run/'assembly/labels.npy');ref,n=ndi.label(m,np.ones((3,3,3),bool));expected=np.zeros_like(m)
    for cid,sl in enumerate(ndi.find_objects(ref),1):
        if (ref==cid).sum()>=1000 and sl[0].stop-sl[0].start>=2:expected|=ref==cid
    assert np.array_equal(result>0,expected) and len(rows)==1 and rows[0]['voxels']==2500
    print('PASS streamed26CC equals full3D26CC, including diagonal contact, size and single-Z filtering')
    print('ALL CPU REGRESSIONS PASSED')

if __name__=='__main__':main()
