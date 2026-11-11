"""Three direction-limited routes; choose one nonempty whole mask per Z."""
import json,time
import numpy as np
from repair_video113 import ROOT,RUN,existing
from repair_three_prompt import choose_prompts,shared_geometry,MultiPromptEngine
from infer_edge_shift import rerun_track_adaptive
from infer_block import save
from repair_quality import inspect

def choose_slice_winners(routes,threshold=.4):
    winners={}
    for role in ('maxclean','head','tail'):
        route=next(r for r in routes if r['role']==role)
        for f in route['frames']:
            if f['score']<threshold or not f['mask'].any():continue
            if f['z'] not in winners or f['score']>winners[f['z']]['score']:winners[f['z']]=f
    return [winners[z] for z in sorted(winners)]

def compact(frames,xy,fallback_z):
    occupied=[]
    for f in frames:
        yy,xx=np.nonzero(f['mask'])
        if len(xx):occupied.append((f['z'],int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1))
    if not occupied:return np.zeros((1,1,1),bool),[max(0,xy[0]),max(0,xy[1]),fallback_z]
    z0=min(p[0] for p in occupied);z1=max(p[0] for p in occupied)+1;x0=min(p[1] for p in occupied);y0=min(p[2] for p in occupied);x1=max(p[3] for p in occupied);y1=max(p[4] for p in occupied)
    mask=np.zeros((z1-z0,y1-y0,x1-x0),bool)
    for f in frames:
        if z0<=f['z']<z1:mask[f['z']-z0]=f['mask'][y0:y1,x0:x1]
    return mask,[xy[0]+x0,xy[1]+y0,z0]

def frame_meta(f):return {k:v for k,v in f.items() if k not in ('mask','probability')}

def run_case(engine,rec):
    start=time.perf_counter();d=RUN/f'cases/instance_{rec["index"]:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'completed.json').exists():return
    old,origin=existing(rec)
    if old is None:save(d/'completed.json',dict(index=rec['index'],status='skipped_no_existing_mask',repair_actions=[],limited_run=False));return
    np.save(d/'existing_mask.npy',old);save(d/'existing_origin.json',origin);save(d/'input_quality.json',inspect(old,origin,block_boundary_z=[0,249],straight_threshold=50,end_cap_threshold=500))
    primary,anchors,selection=choose_prompts(old,origin);save(d/'reference_selection.json',selection);save(d/'prompt_frames.json',anchors)
    if primary is None:save(d/'completed.json',dict(index=rec['index'],status='skipped_no_clean_reference',repair_actions=[],limited_run=False));return
    primary['directional_routes']=True;geometry,side=shared_geometry(anchors,int(engine.raw.size[2]));t=time.perf_counter();track,history=rerun_track_adaptive(engine,primary,geometry,rec['block'],initial_side=side);inference=time.perf_counter()-t
    t=time.perf_counter();selected=choose_slice_winners(track['routes']);xy=track['geometry'][:2];final,fo=compact(selected,xy,primary['z']);merge_seconds=time.perf_counter()-t
    np.save(d/'mask.npy',final);save(d/'mask_origin.json',fo)
    choices=[dict(z=f['z'],score=f['score'],source_route=f['source_route'],area=int(f['mask'].sum())) for f in selected];save(d/'slice_winners.json',choices)
    actions=[]
    for i,r in enumerate(track['routes']):
        ad=d/f'action_{i:03d}';ad.mkdir(exist_ok=True);m,o=compact(r['frames'],xy,r['seed_z']);np.save(ad/'selected_mask.npy',m);save(ad/'origin.json',o)
        tr={k:v for k,v in track.items() if k not in ('frames','routes','evaluated_scores','stop_events')};tr.update(frames=[frame_meta(f) for f in r['frames']],evaluated_scores=r['evaluated_scores'],stop_events=r['stop_events'],direction_end_reasons=r['direction_end_reasons'],role=r['role'],directions=r['directions'],conditioning_prompts=[r['prompt']],conditioning_frame_count=1,crop_history=history);save(ad/'track.json',tr)
        actions.append(dict(kind='route_'+r['role'],branch=r['role'],bidirectional=r['role']=='maxclean',directions=r['directions'],seed_z=r['seed_z'],target_z=r['seed_z'],prompt=r['prompt'],prompts=[r["prompt"]],crop_geometry=track['geometry'],initial_crop_side=history[0]['crop_side'],initial_crop_min_side=100,final_side=track['crop_side'],crop_history=history,stop_events=r['stop_events'],both_direction_end_reasons=r['direction_end_reasons'],status='predicted' if m.any() else 'empty_prediction',same_video_memory=False,independent_route_memories=True))
    detail={k:v for k,v in track.items() if k not in ('frames','routes')};detail.update(frames=[frame_meta(f) for f in selected],selected_frames=choices,crop_history=history,routes=[{**{k:v for k,v in r.items() if k!='frames'},'frames':[frame_meta(f) for f in r['frames']]} for r in track['routes']]);save(d/'track.json',detail)
    save(d/'completed.json',dict(index=rec['index'],status='repaired' if final.any() else 'empty_prediction',repair_actions=actions,detector_prompt=primary,reference_selection=selection,crop_geometry=track['geometry'],final_side=track['crop_side'],initial_crop_min_side=100,pipeline='three_directional_routes',prompt_count=len(anchors),score_threshold=.4,score_stop_count=1,same_video_memory=False,independent_route_memories=True,merge_rule=track['merge_rule'],no_pixelwise_or=True,no_final_component_pruning=True,inference_seconds=inference,assembly_seconds=merge_seconds,elapsed_seconds=time.perf_counter()-start,nucleus_excluded=True,whole_prediction_replaces_original=True,limited_run=False))
