"""Vectorized global same-Z greedy NMS, keeping original high-score boxes."""
import numpy as np

def nms(rows,threshold=.4):
    byz={};kept=[];audit=[]
    for b in rows:byz.setdefault(b['z'],[]).append(b)
    for z,items in sorted(byz.items()):
        items=sorted(items,key=lambda b:(-b['slice_score'],b['global_detection_id']));boxes=np.asarray([b['box_xyxy'] for b in items],float);area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);remaining=np.arange(len(items));status={}
        while len(remaining):
            i=int(remaining[0]);kept.append(items[i]);remaining=remaining[1:]
            inter=np.maximum(0,np.minimum(boxes[remaining,2:],boxes[i,2:])-np.maximum(boxes[remaining,:2],boxes[i,:2])).prod(1)
            overlap=inter/(area[i]+area[remaining]-inter);bad=overlap>threshold
            for j,v in zip(remaining[bad],overlap[bad]):status[int(j)]=(items[i]['global_detection_id'],float(v))
            remaining=remaining[~bad]
        for i,b in enumerate(items):audit.append(dict(detection_id=b['global_detection_id'],z=z,nms_suppressed=i in status,suppressor=status.get(i,[None,None])[0],iou=status.get(i,[None,None])[1]))
    return kept,audit
