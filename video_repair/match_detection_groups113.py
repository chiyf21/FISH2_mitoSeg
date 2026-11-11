"""Select one existing graph group per target GT instance; seed never changed."""
import json,time
from pathlib import Path
import numpy as np
from fast_runtime import ROOT
from infer_block import save
REF=ROOT.parent/'sam_video_hardcases_20261102';RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103';TRAIN=ROOT.parent/'detector_video_iom06_20261031'
def main():
    t=time.perf_counter();cohort=json.loads((REF/'cohort.json').read_text());detections=json.loads((RUN/'detections.json').read_text())['detections'];groups=json.loads((RUN/'groups.json').read_text());labels=np.load(TRAIN/'source_cache/labels_2.npy',mmap_mode='r');byid={g['index']:g for g in groups};owner=np.zeros(len(detections),np.int32)
    for g in groups:owner[g['member_ids']]=g['index']
    boxes=np.asarray([d['box_xyxy'] for d in detections]);zs=np.asarray([d['z'] for d in detections]);scores=np.asarray([d['slice_score'] for d in detections]);matches=[];selected={}
    for rec in cohort:
        x0,y0,z0,x1,y1,z1=rec['gt_bbox'];z,y,x=np.nonzero(labels[z0:z1,y0:y1,x0:x1]==rec['gt_id']);z+=z0;y+=y0;x+=x0;best=np.full(len(x),-np.inf);ids=np.zeros(len(x),np.int32);local={int(q):np.flatnonzero(z==q) for q in np.unique(z)}
        candidates=np.flatnonzero((zs>=z0)&(zs<z1)&(boxes[:,0]<x1)&(boxes[:,2]>x0)&(boxes[:,1]<y1)&(boxes[:,3]>y0))
        for i in candidates:
            at=local.get(int(zs[i]))
            if at is None:continue
            b=boxes[i];inside=(x[at]+.5>=b[0])&(x[at]+.5<b[2])&(y[at]+.5>=b[1])&(y[at]+.5<b[3]);at=at[inside];better=scores[i]>best[at];at=at[better];best[at]=scores[i];ids[at]=owner[i]
        count=np.bincount(ids,minlength=len(groups)+1);count[0]=0;winner=int(count.argmax()) if count.max()>0 else None;order=np.flatnonzero(count);order=sorted(order,key=lambda i:(-count[i],i))
        row=dict(**rec,matched_group_id=winner,gt_voxels=len(x),votes=int(count[winner]) if winner else 0,coverage=float(count[winner]/len(x)) if winner and len(x) else 0,covered_by_any_group=int(np.count_nonzero(ids)),top_groups=[dict(group_id=int(i),votes=int(count[i])) for i in order[:5]],match_rule='Each GT voxel votes to highest detector-slice-score covering box, then its graph group; max votes; ties smaller group ID; no zero-overlap forced match')
        matches.append(row)
        if winner:selected.setdefault(winner,[]).append(rec['gt_id'])
    jobs=[]
    for gid,gtids in selected.items():
        g=dict(byid[gid]);g['matched_gt_ids']=gtids;jobs.append(g)
    jobs.sort(key=lambda g:(0 if 3154 in g['matched_gt_ids'] else 1,-g['member_count'],g['index']))
    save(RUN/'target_matches.json',matches);save(RUN/'selected_groups.json',jobs)
    summary=dict(target_instances=len(cohort),matched_instances=sum(r['matched_group_id'] is not None for r in matches),unmatched_instances=sum(r['matched_group_id'] is None for r in matches),all_graph_groups=len(groups),selected_unique_groups=len(jobs),shared_groups=sum(len(ids)>1 for ids in selected.values()),selection_uses_target_gt=True,gt_not_used_for_grouping_prompt_or_video=True,seconds=time.perf_counter()-t)
    save(RUN/'target_selection_completed.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
