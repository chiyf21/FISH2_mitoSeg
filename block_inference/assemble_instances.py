"""Out-of-core exact26-connected components using only adjacent Z planes."""
from pathlib import Path
from array import array
import json,time,os
import numpy as np
from scipy import ndimage as ndi
from common import save

def assemble(run,shape_zyx,min_voxels=1000,min_z=2,optimized=False):
    run=Path(run);out=run/'assembly';out.mkdir(exist_ok=True);started=time.time();byz={}
    # Only complete, referenced frames enter assembly; ignore orphan files
    # left by an interrupted seed before its cache-completion marker.
    accepted=set(json.loads((run/'accepted_tracks.json').read_text())) if (run/'accepted_tracks.json').exists() else None
    for p in (run/'tracks').glob('track_*.json'):
        t=json.loads(p.read_text())
        if accepted is not None and t['track_id'] not in accepted:continue
        for f in t['frames']:byz.setdefault(f['z'],[]).append((run/'frames'/f'z{f["z"]:03d}_track{t["track_id"]:05d}.npz'))
    parents=array('q',[0]);stats=[None]
    def find(v):
        while parents[v]!=v:parents[v]=parents[parents[v]];v=parents[v]
        return v
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    provisional=np.lib.format.open_memmap(out/'provisional.npy',mode='w+',dtype=np.uint32,shape=tuple(shape_zyx));previous=None;nonempty_planes=[]
    for z in range(shape_zyx[0]):
        if optimized and z not in byz:
            previous=None
            save(out/'progress.json',dict(phase='plane_union',z_done=z+1,total_z=shape_zyx[0],provisional_components=len(parents)-1));continue
        plane=np.zeros(shape_zyx[1:],bool)
        for path in byz.get(z,[]):
            with np.load(path) as f:
                m=f['mask'];x,y=map(int,f['origin_xy']);h,w=m.shape;plane[y:y+h,x:x+w]|=m
        current,n=ndi.label(plane,structure=np.ones((3,3),bool));offset=len(parents)-1
        if offset+n>=2**32:raise RuntimeError('More than uint32 provisional IDs; split input into smaller blocks')
        sizes=np.bincount(current.ravel());objects=ndi.find_objects(current)
        for local,sl in enumerate(objects,1):
            gid=offset+local;parents.append(gid);stats.append([int(sizes[local]),sl[1].start,sl[0].start,z,sl[1].stop,sl[0].stop,z+1])
        current=current.astype(np.uint32);current[current>0]+=offset
        if previous is not None and current.any() and previous.any():
            h,w=current.shape
            for dy in [-1,0,1]:
                ya,yb=max(0,-dy),min(h,h-dy)
                for dx in [-1,0,1]:
                    xa,xb=max(0,-dx),min(w,w-dx)
                    for y in range(ya,yb,128):
                        end=min(y+128,yb);a=current[y:end,xa:xb];b=previous[y+dy:end+dy,xa+dx:xb+dx];valid=(a>0)&(b>0)
                        if valid.any():
                            packed=(a[valid].astype(np.uint64)<<32)|b[valid].astype(np.uint64)
                            for key in np.unique(packed):union(int(key)>>32,int(key)&0xffffffff)
        if optimized:
            if n:
                nonempty_planes.append(z)
                for y in range(0,shape_zyx[1],128):
                    for x in range(0,shape_zyx[2],512):
                        tile=current[y:y+128,x:x+512]
                        if tile.any():provisional[z,y:y+128,x:x+512]=tile
            previous=current if n else None
        else:provisional[z]=current;previous=current
        save(out/'progress.json',dict(phase='plane_union',z_done=z+1,total_z=shape_zyx[0],provisional_components=len(parents)-1))
    provisional.flush();merged={}
    for cid in range(1,len(parents)):
        root=find(cid);v=stats[cid]
        if root not in merged:merged[root]=list(v)
        else:
            t=merged[root];t[0]+=v[0]
            for j in range(1,4):t[j]=min(t[j],v[j])
            for j in range(4,7):t[j]=max(t[j],v[j])
    keep={k:v for k,v in sorted(merged.items()) if v[0]>=min_voxels and v[6]-v[3]>=min_z};root_to_final={k:i+1 for i,k in enumerate(keep)}
    lut=np.zeros(len(parents),np.uint32)
    for i in range(1,len(parents)):lut[i]=root_to_final.get(find(i),0)
    labels=np.lib.format.open_memmap(out/'labels.npy',mode='w+',dtype=np.uint32,shape=tuple(shape_zyx))
    for z in nonempty_planes if optimized else range(shape_zyx[0]):
        for y in range(0,shape_zyx[1],128):
            if optimized:
                for x in range(0,shape_zyx[2],512):
                    tile=provisional[z,y:y+128,x:x+512]
                    if tile.any():labels[z,y:y+128,x:x+512]=lut[tile]
            else:labels[z,y:y+128]=lut[provisional[z,y:y+128]]
        save(out/'progress.json',dict(phase='final_labels',z_done=z+1,total_z=shape_zyx[0],instances=len(keep)))
    labels.flush();save(out/'progress.json',dict(phase='complete',z_done=shape_zyx[0],total_z=shape_zyx[0],instances=len(keep)))
    rows=[dict(instance_id=root_to_final[k],voxels=v[0],bbox_xyz=v[1:],z_layers=v[6]-v[3]) for k,v in keep.items()]
    save(out/'instances.json',rows);save(out/'completed.json',dict(instances=len(rows),raw_components=len(merged),shape_zyx=list(shape_zyx),min_voxels=min_voxels,min_z_layers=min_z,seconds=time.time()-started,limited_run=False))
    # This is our own temporary label volume, removed only after final output
    # flush and completion marker. No inputs or other experiments are touched.
    del provisional;os.remove(out/'provisional.npy')
    return rows
