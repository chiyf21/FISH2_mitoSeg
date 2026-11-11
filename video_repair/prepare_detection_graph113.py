import json,time
from pathlib import Path
import numpy as np
from detection_graph import group_detections
from fast_runtime import ROOT
from infer_block import save
REF=ROOT.parent/'sam_video_hardcases_20261102';RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103'
def main():
    RUN.mkdir(exist_ok=True);t=time.perf_counter();records=json.loads((REF/'cohort.json').read_text());assert len(records)==113;unique={};shas=set();tags=set();files=[];missing=[]
    for r in records:
        path=REF/f'cases/instance_{r["index"]:03d}/video_repaired_overlap/detections.json'
        if not path.exists():missing.append(r['index']);continue
        data=json.loads(path.read_text());shas.add(data['detector_sha256']);tags.add(data['source_tag']);files.append(str(path))
        for row in data['detections']:
            key=(int(row['z']),*map(float,row['box_xyxy']))
            if key not in unique:unique[key]={**row,'source_case_indices':[r['index']]}
            else:
                prior=unique[key];cases=prior['source_case_indices']+[r['index']]
                if row['slice_score']>prior['slice_score']:prior={**row}
                prior['source_case_indices']=sorted(set(cases));unique[key]=prior
    if missing:raise RuntimeError(f'Detection caches missing for{missing}; do not silently infer a partial113 cohort')
    assert len(shas)==1 and len(tags)==1,'Cannot mix detector weights/raw data versions'
    detections=[unique[k] for k in sorted(unique)]
    for i,d in enumerate(detections):d['global_detection_id']=i
    save(RUN/'detections.json',dict(detections=detections,detector_sha256=next(iter(shas)),source_tag=next(iter(tags)),source_files=files,exact_duplicate_only=True))
    groups,edges=group_detections(detections);save(RUN/'groups.json',groups);np.savez_compressed(RUN/'graph_edges.npz',edges=np.asarray(edges,np.float64).reshape(-1,4))
    save(RUN/'inference_config.json',dict(iou_threshold_exclusive=.5,iom_threshold_exclusive=.8,iom_denominator='min(box_area_a,box_area_b)',adjacency='actualZ+1 only',transitive_many_to_many=True,seed='largest-area original detector box; ties score then lowerZ then ID',prompt_padding=0,score_threshold=.5,score_stop_count=1,crop='max(100,3*prompt longest side), adaptive >100 margin',propagate='both directions to first score<0.5 or true source boundary',detection_scope='all cached113 evaluation blocks; not claiming full2500x2500x250 detector coverage',gt_used_for_grouping_or_seed=False,group_id_identity_preserved=True))
    summary=dict(source_cases=len(records),detections=len(detections),groups=len(groups),edges=len(edges),max_group_boxes=max(g['member_count'] for g in groups),max_group_z_span=max(g['z_range'][1]-g['z_range'][0]+1 for g in groups),single_layer_groups=sum(len(g['z_values'])==1 for g in groups),seconds=time.perf_counter()-t)
    save(RUN/'graph_completed.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
