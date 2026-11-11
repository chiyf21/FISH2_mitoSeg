"""Independent single-prompt routes; endpoint 2D components >500; preserve old mask."""
import time,json,os
import numpy as np
from scipy import ndimage as ndi
from repair_video113 import ROOT,RUN,existing,maxarea_prompt
from repair_three_prompt import MultiPromptEngine,shared_geometry
from repair_directional_routes import compact,frame_meta
from infer_edge_shift import rerun_track_adaptive
from infer_block import save
from repair_quality import inspect

def tight_prompt(plane,xy,z,role):
    yy,xx=np.nonzero(plane);assert len(xx)
    x,y=map(int,xy);b=[x+int(xx.min()),y+int(yy.min()),x+int(xx.max())+1,y+int(yy.max())+1]
    return dict(z=int(z),tight_bbox_xyxy=b,box_xyxy=list(b),roles=[role],padding_native_pixels=0,detection_id=-1,slice_score=None,prompt_source='corresponding_seed_mask_exact_tight_bbox',seed_mask_area=int(plane.sum()))

def plan(mask,origin,depth):
    specs=[];audit=[];occupied=np.flatnonzero(mask.any((1,2)))
    if not len(occupied):return [],[]
    main,selection=maxarea_prompt(mask,origin,True,straight_threshold=50)
    if main is not None:
        p=tight_prompt(mask[main['z']-origin[2]],origin[:2],main['z'],'maxclean')
        specs.append(dict(role='maxclean',prompt=p,directions=['forward','backward'],selection=selection))
    for side,j,direction in [('head',int(occupied[0]),'backward'),('tail',int(occupied[-1]),'forward')]:
        z=origin[2]+j;cc,n=ndi.label(mask[j],structure=np.ones((3,3),bool))
        for k in range(1,n+1):
            plane=cc==k;area=int(plane.sum());boundary=z in (0,depth-1);eligible=area>500 and not boundary
            entry=dict(side=side,z=z,component_id=k,component_area=area,threshold=500,comparison='>',source_boundary=boundary,eligible=eligible)
            if eligible:
                role=f'{side}_cc{k}';p=tight_prompt(plane,origin[:2],z,role)
                specs.append(dict(role=role,prompt=p,directions=[direction],endpoint_component=entry))
                entry['prompt']=p
            audit.append(entry)
    return specs,audit

def union_volumes(volumes):
    vols=[(m,o) for m,o in volumes if m.any()]
    if not vols:return np.zeros((1,1,1),bool),list(volumes[0][1])
    lo=np.min([o for m,o in vols],axis=0).astype(int)
    hi=np.max([[o[0]+m.shape[2],o[1]+m.shape[1],o[2]+m.shape[0]] for m,o in vols],axis=0).astype(int)
    out=np.zeros(tuple((hi-lo)[::-1]),bool)
    for m,o in vols:
        x,y,z=np.asarray(o)-lo;out[z:z+m.shape[0],y:y+m.shape[1],x:x+m.shape[2]]|=m
    return out,lo.tolist()

def run_case(engine,rec):
    start=time.perf_counter();d=RUN/f'cases/instance_{rec["index"]:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'completed.json').exists():return
    old,origin=existing(rec)
    if old is None:save(d/'completed.json',dict(index=rec['index'],status='skipped_no_existing_mask',repair_actions=[],limited_run=False));return
    np.save(d/'existing_mask.npy',old);save(d/'existing_origin.json',origin)
    save(d/'input_quality.json',inspect(old,origin,block_boundary_z=[0,int(engine.raw.size[2])-1],straight_threshold=50,end_cap_threshold=500))
    specs,audit=plan(old,origin,int(engine.raw.size[2]));save(d/'component_endpoint_audit.json',audit);save(d/'planned_routes.json',specs)
    actions=[];volumes=[(old,origin)];all_scores=[];stops=[];frames=[];elapsed=0.;root_primary=None;root_history=[]
    for i,spec in enumerate(specs):
        p=spec['prompt'];role=spec['role'];ad=d/f'action_{i:03d}';ad.mkdir(exist_ok=True)
        # Save the actual 2D seed used to define the box; never inferred from GT.
        j=p['z']-origin[2]
        if role=='maxclean':seed=old[j]
        else:
            cc,n=ndi.label(old[j],structure=np.ones((3,3),bool));seed=cc==spec['endpoint_component']['component_id']
        np.save(ad/'seed_mask.npy',seed);save(ad/'seed_origin_xy.json',origin[:2])
        assert tight_prompt(seed,origin[:2],p['z'],role)['box_xyxy']==p['box_xyxy']
        det={**p,'anchors':[p],'all_prompt_boxes':[p['box_xyxy']],'directional_routes':True,'route_specs':[(role,spec['directions'])]}
        geometry,side=shared_geometry([p],int(engine.raw.size[2]));t=time.perf_counter()
        tr,hist=rerun_track_adaptive(engine,det,geometry,rec['block'],initial_side=side);dt=time.perf_counter()-t;elapsed+=dt
        route=tr['routes'][0];m,o=compact(route['frames'],tr['geometry'][:2],p['z']);np.save(ad/'selected_mask.npy',m);save(ad/'origin.json',o);volumes.append((m,o))
        detail={k:v for k,v in tr.items() if k not in ('frames','routes')};detail.update(frames=[frame_meta(f) for f in route['frames']],crop_history=hist,role=role,directions=spec['directions'],evaluated_scores=route['evaluated_scores'],prompt_box_audit=dict(passed=True,padding_each_side=0,tight_bbox_xyxy=p['tight_bbox_xyxy'],actual_prompt_xyxy=p['box_xyxy']))
        save(ad/'track.json',detail)
        actions.append(dict(kind='component_route_'+role,branch=role,bidirectional=role=='maxclean',directions=spec['directions'],seed_z=p['z'],target_z=p['z'],prompt=p,prompts=[p],crop_geometry=tr['geometry'],initial_crop_side=hist[0]['crop_side'],initial_crop_min_side=100,final_side=tr['crop_side'],crop_history=hist,stop_events=route['stop_events'],both_direction_end_reasons=route['direction_end_reasons'],status='predicted' if m.any() else 'empty_prediction',same_video_memory=False,independent_route_memories=True,endpoint_component=spec.get('endpoint_component'),inference_seconds=dt))
        all_scores.extend(route['evaluated_scores']);stops.extend(route['stop_events']);frames.extend([frame_meta(f) for f in route['frames']])
        if root_primary is None:root_primary=p;root_history=hist
        save(d/'repair_actions.json',actions)
    t=time.perf_counter();final,fo=union_volumes(volumes);assembly=time.perf_counter()-t
    # Preserve original mask exactly: every original positive remains positive.
    x,y,z=np.asarray(origin)-np.asarray(fo);assert np.all(final[z:z+old.shape[0],y:y+old.shape[1],x:x+old.shape[2]][old])
    if os.environ.get('REPAIR_KEEP3D_COMPONENT')=='1':
        from union_connectivity import retain_connected_body
        main=next((s for s in specs if s['role']=='maxclean'),None)
        seed_z=main['prompt']['z'] if main else origin[2]+int(old.sum((1,2)).argmax())
        final,component_audit=retain_connected_body(final,fo,old[seed_z-origin[2]],origin[:2],seed_z,old,origin)
        save(d/'connectivity_audit.json',component_audit)
    np.save(d/'mask.npy',final);save(d/'mask_origin.json',fo)
    save(d/'track.json',dict(frames=frames,evaluated_scores=all_scores,stop_events=stops,crop_history=root_history,merge_rule='pixelwise union: original mask + maxclean route + all eligible endpoint-component routes',independent_route_memories=True,same_video_memory=False))
    save(d/'completed.json',dict(index=rec['index'],status='component_ends_union' if actions else 'original_preserved',repair_actions=actions,detector_prompt=root_primary,reference_selection=dict(mode='exact_seed_tight_bbox_no_padding',largest_mask_area_px=(root_primary or {}).get('seed_mask_area')),initial_crop_min_side=100,pipeline='component_ends_union',score_threshold=float(getattr(engine,'score_threshold',.4)),score_stop_count=int(getattr(engine,'score_stop_count',1)),original_mask_preserved=True,endpoint_component_threshold=500,endpoint_connectivity=8,endpoint_audit=audit,same_video_memory=False,inference_seconds=elapsed,assembly_seconds=assembly,elapsed_seconds=time.perf_counter()-start,limited_run=False))
