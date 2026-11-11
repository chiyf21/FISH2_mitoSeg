"""AdjacentZ detection connectivity: IoU>0.5 OR intersection/min(area)>0.8."""
from collections import defaultdict
import numpy as np
class UnionFind:
    def __init__(self,n):self.p=list(range(n));self.size=[1]*n
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.p[b]=a;self.size[a]+=self.size[b]
def box_metrics(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));aa=(a[2]-a[0])*(a[3]-a[1]);ab=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter),inter/min(aa,ab)
def group_detections(detections,iou_threshold=.5,iom_threshold=.8):
    byz=defaultdict(list)
    for i,r in enumerate(detections):
        b=np.asarray(r['box_xyxy'],np.float64)
        if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:raise ValueError(f'Invalid box{i}: {b}')
        byz[int(r['z'])].append(i)
    uf=UnionFind(len(detections));edges=[]
    for z in sorted(byz):
        if z+1 not in byz:continue
        left=byz[z];right=byz[z+1];b=np.asarray([detections[i]['box_xyxy'] for i in right],np.float64);ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
        for start in range(0,len(left),256):
            ids=left[start:start+256];a=np.asarray([detections[i]['box_xyxy'] for i in ids],np.float64);aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1);iou=inter/(aa[:,None]+ab[None,:]-inter);iom=inter/np.minimum(aa[:,None],ab[None,:]);rr,cc=np.nonzero((iou>iou_threshold)|(iom>iom_threshold))
            for x,y in zip(rr,cc):
                ia,ib=ids[x],right[y];uf.union(ia,ib);edges.append((ia,ib,float(iou[x,y]),float(iom[x,y])))
    components=defaultdict(list)
    for i in range(len(detections)):components[uf.find(i)].append(i)
    groups=[]
    for gid,members in enumerate(sorted(components.values(),key=min),1):
        def rank(i):
            d=detections[i];b=d['box_xyxy'];return ((b[2]-b[0])*(b[3]-b[1]),float(d.get('slice_score',0)),-int(d['z']),-i)
        seed=max(members,key=rank);zs=sorted({int(detections[i]['z']) for i in members});groups.append(dict(index=gid,group_id=gid,member_ids=members,member_count=len(members),z_values=zs,z_range=[zs[0],zs[-1]],seed_id=seed,seed=dict(detections[seed]),source='adjacentZ box graph; no GT'))
    return groups,edges
