"""Three box-conditioned frames in one SAM2 Video memory, score cutoff0.4."""
import json,math,time,os
import numpy as np
import torch
from scipy import ndimage as ndi
from repair_video113 import ROOT,RUN,existing,maxarea_prompt,ScoreStoppedEngine,frame_box_iou
from infer_edge_shift import rerun_track_adaptive
from infer_block import save
from repair_quality import inspect

def choose_prompts(mask,origin):
    primary,selection=maxarea_prompt(mask,origin,True,straight_threshold=50)
    if primary is None:return None,[],selection
    occupied=np.flatnonzero(mask.any((1,2)));x,y,z0=origin;byz={}
    for role,z in [('maxclean',primary['z']),('head',z0+int(occupied[0])),('tail',z0+int(occupied[-1]))]:
        if z not in byz:
            yy,xx=np.nonzero(mask[z-z0]);tight=[x+int(xx.min()),y+int(yy.min()),x+int(xx.max())+1,y+int(yy.max())+1]
            byz[z]=dict(z=z,box_xyxy=[tight[0]-10,tight[1]-10,tight[2]+10,tight[3]+10],tight_bbox_xyxy=tight,roles=[],detection_id=-1,slice_score=None)
        byz[z]['roles'].append(role)
    anchors=[byz[z] for z in sorted(byz)]
    primary={**primary,'anchors':anchors,'all_prompt_boxes':[a['box_xyxy'] for a in anchors]}
    return primary,anchors,selection

def shared_geometry(anchors,depth=250):
    boxes=[a['box_xyxy'] for a in anchors];x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes);x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
    longest=max(max(b[2]-b[0],b[3]-b[1]) for b in boxes)
    side=int(math.ceil(max(100,3*longest,x1-x0,y1-y0)))
    return [math.floor((x0+x1-side)/2),math.floor((y0+y1-side)/2),0,depth],side

class MultiPromptEngine(ScoreStoppedEngine):
    @torch.inference_mode()
    def track(self,encoded,geometry,side,det):
        x,y,z0,n=geometry;m=self.model;threshold=float(getattr(self,'score_threshold',.4));anchors={a['z']-z0:a for a in det['anchors']};all_z=sorted(a['z'] for a in det['anchors'])
        if m.max_cond_frames_in_attn!=-1 and m.max_cond_frames_in_attn<len(anchors):m.max_cond_frames_in_attn=len(anchors)
        def decode(j,point,is_cond,state,reverse):
            features={k:[a.cuda(non_blocking=True) for a in encoded[j][k]] for k in ('backbone_fpn','vision_pos_enc')}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _,feat,pos,sizes=m._prepare_backbone_features(features)
                cur,outs,_,_=m._track_step(j,is_cond,feat,pos,sizes,point,None,state,n,reverse,None)
                _,_,ious,lo,hi,ptr,obj=outs;cur.update(pred_masks=lo,obj_ptr=ptr)
                m._encode_memory_in_output(feat,sizes,point,True,hi,obj,cur)
            score=float(ious.float().reshape(-1)[0]);assert math.isfinite(score)
            prob=torch.nn.functional.interpolate(hi.float(),size=(side,side),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
            prob[self.nuc_crop(z0+j,x,y,side)]=0
            valid=np.zeros((side,side),bool);h,w=self.nuc.shape[1:];lx=max(0,-x);ly=max(0,-y);hx=min(side,w-x);hy=min(side,h-y)
            if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
            prob[~valid]=0;binary=prob>=.5
            frame=dict(z=z0+j,origin_xy=[x,y],mask=binary,probability=prob,score=score,object_probability=float(obj.float().sigmoid().reshape(-1)[0]),detection_score=None,seed_z=det['z'],detection_id=-1,is_prompt_frame=j in anchors,available_conditioning_z=all_z)
            kept={k:cur[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
            return frame,kept
        cond={};prompt_outputs={};evaluated=[];best={};stops=[]
        # Prepare all actual box prompts BEFORE any propagation: every tracked
        # frame may attend to head, middle and tail conditioning memories.
        for j,a in anchors.items():
            box=np.asarray(a['box_xyxy'])-np.array([x,y,x,y]);point=dict(point_coords=torch.tensor(box,dtype=torch.float32,device='cuda').reshape(1,2,2)/side*1024,point_labels=torch.tensor([[2,3]],dtype=torch.int32,device='cuda'))
            frame,kept=decode(j,point,True,dict(cond_frame_outputs={},non_cond_frame_outputs={}),False)
            cond[j]=kept;prompt_outputs[j]=frame;evaluated.append(dict(z=z0+j,score=frame['score'],direction='prompt'))
            if frame['score']>=threshold:best[j]=frame
        seed=det['z']-z0
        if det.get('directional_routes'):
            routes=[];all_frames=[];all_stops=[]
            specs=det.get('route_specs') or [('maxclean',('forward','backward')),('head',('backward',)),('tail',('forward',))]
            for role,directions in specs:
                anchor=next(a for a in det['anchors'] if role in a['roles']);s=anchor['z']-z0;route_frames={};route_scores=[];route_stops=[]
                for direction in directions:
                    reverse=direction=='backward'
                    begin=s-1 if reverse and role=='maxclean' else s
                    order=range(begin,-1,-1) if reverse else range(begin,n)
                    # Each route sees ONLY its own prompted frame. Image features
                    # and crop geometry may be reused; temporal memories may not.
                    state=dict(cond_frame_outputs={s:cond[s]},non_cond_frame_outputs={})
                    for j in order:
                        if j==s:base=prompt_outputs[s];kept=cond[s]
                        else:base,kept=decode(j,None,False,state,reverse)
                        frame={**base,'source_route':role,'seed_z':anchor['z'],'is_prompt_frame':j==s,'available_conditioning_z':[anchor['z']]};route_scores.append(dict(z=z0+j,score=frame['score'],direction=direction,route=role,conditioning_z=[anchor['z']]))
                        if frame['score']<threshold:
                            route_stops.append(dict(direction=direction,route=role,low_z=[z0+j],threshold=threshold,required_low_frames=1,excluded_low_frames=True));break
                        if j not in route_frames or frame['score']>route_frames[j]['score']:route_frames[j]=frame
                        if j!=s:state['non_cond_frame_outputs'][j]=kept
                fs=[route_frames[j] for j in sorted(route_frames)];ended={q['direction'] for q in route_stops}
                route=dict(role=role,seed_z=anchor['z'],prompt=anchor,directions=list(directions),conditioning_z=[anchor['z']],conditioning_frame_count=1,frames=fs,evaluated_scores=route_scores,stop_events=route_stops,direction_end_reasons={direction:'low_score' if direction in ended else ('source_volume_boundary' if direction in directions else 'not_requested') for direction in ('forward','backward')})
                routes.append(route);all_frames.extend(fs);evaluated.extend(route_scores);all_stops.extend(route_stops)
            lo=min(r['z'] for r in evaluated);hi=max(r['z'] for r in evaluated)+1
            reasons={q:next((r['direction_end_reasons'][q] for r in reversed(routes) if q in r['directions']),'not_requested') for q in ('forward','backward')}
            return dict(geometry=[x,y,lo,hi-lo],source_video_geometry=list(geometry),seed_detection=det,frames=all_frames,routes=routes,stop_events=all_stops,evaluated_scores=evaluated,crop_side=side,encoded_z_batches=encoded.batches,conditioning_prompts=det['anchors'],conditioning_frame_count=1,independent_route_memories=True,max_cond_frames_in_attn=m.max_cond_frames_in_attn,direction_end_reasons=reasons,same_video_memory=False,merge_rule='union of nonempty Z slices, highest-score entire mask per Z, ties maxclean/head/tail')
        for direction,reverse,order in [('forward',False,range(seed,n)),('backward',True,range(seed-1,-1,-1))]:
            state=dict(cond_frame_outputs=dict(cond),non_cond_frame_outputs={})
            for j in order:
                if j in prompt_outputs:frame=prompt_outputs[j];kept=cond[j]
                else:frame,kept=decode(j,None,False,state,reverse)
                evaluated.append(dict(z=z0+j,score=frame['score'],direction=direction,conditioning_z=all_z))
                if frame['score']<threshold:
                    stops.append(dict(direction=direction,low_z=[z0+j],threshold=threshold,required_low_frames=1,excluded_low_frames=True));break
                if j not in best or frame['score']>best[j]['score']:best[j]=frame
                if j not in cond:state['non_cond_frame_outputs'][j]=kept
        frames=[best[j] for j in sorted(best)];lo=min(r['z'] for r in evaluated);hi=max(r['z'] for r in evaluated)+1
        ended={s['direction'] for s in stops}
        return dict(geometry=[x,y,lo,hi-lo],source_video_geometry=list(geometry),seed_detection=det,frames=frames,stop_events=stops,evaluated_scores=evaluated,crop_side=side,encoded_z_batches=encoded.batches,conditioning_prompts=det['anchors'],conditioning_frame_count=len(cond),max_cond_frames_in_attn=m.max_cond_frames_in_attn,direction_start_z=dict(forward=det['z'],backward=det['z']-1),direction_end_reasons={d:'low_score' if d in ended else 'source_volume_boundary' for d in ('forward','backward')},same_video_memory=True,overlap_policy='highest score between two propagation directions; prompted frames cached once')

def select_prompt_supported_component(binary,origin,anchors):
    cc,n=ndi.label(binary,structure=np.ones((3,3,3),bool));support={};votes=[]
    for a in anchors:
        j=a['z']-origin[2]
        if not 0<=j<len(binary):continue
        parts,k=ndi.label(binary[j],structure=np.ones((3,3),bool));options=[]
        for label in range(1,k+1):
            p=parts==label;iou=frame_box_iou(p,origin[:2],a['box_xyxy']);component=int(cc[j][p][0]);options.append((iou,component))
        if not options:continue
        iou,component=max(options)
        if iou<=0:continue
        s=support.setdefault(component,dict(prompt_frames=0,total_iou=0.));s['prompt_frames']+=1;s['total_iou']+=iou;votes.append(dict(z=a['z'],component=component,iou=iou))
    if not support:return np.zeros_like(binary),dict(reason='no_retained_prompt_component',votes=votes)
    winner=max(support,key=lambda c:(support[c]['prompt_frames'],support[c]['total_iou'],-c))
    return cc==winner,dict(component=winner,support=support,votes=votes,all_components=int(n),rule='most prompt-frame support, then summed prompt-box IoU; no GT')

def run_case(engine,rec):
    start=time.perf_counter();d=RUN/f'cases/instance_{rec["index"]:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'completed.json').exists():return
    old,origin=existing(rec)
    if old is None:save(d/'completed.json',dict(index=rec['index'],status='skipped_no_existing_mask',repair_actions=[],limited_run=False));return
    np.save(d/'existing_mask.npy',old);save(d/'existing_origin.json',origin);save(d/'input_quality.json',inspect(old,origin,block_boundary_z=[0,249],straight_threshold=50,end_cap_threshold=500))
    primary,anchors,selection=choose_prompts(old,origin);save(d/'reference_selection.json',selection);save(d/'prompt_frames.json',anchors)
    if primary is None:save(d/'completed.json',dict(index=rec['index'],status='skipped_no_clean_reference',repair_actions=[],limited_run=False));return
    geometry,side=shared_geometry(anchors,int(engine.raw.size[2]));t=time.perf_counter();track,history=rerun_track_adaptive(engine,primary,geometry,rec['block'],initial_side=side);seconds=time.perf_counter()-t
    x,y,z,n=track['geometry'];side=track['crop_side'];binary=np.zeros((n,side,side),bool)
    for f in track['frames']:binary[f['z']-z]=f['mask']
    t=time.perf_counter();chosen,component=select_prompt_supported_component(binary,[x,y,z],anchors);assembly=time.perf_counter()-t
    zz,yy,xx=np.nonzero(chosen)
    if len(zz):final=chosen[zz.min():zz.max()+1,yy.min():yy.max()+1,xx.min():xx.max()+1];fo=[x+int(xx.min()),y+int(yy.min()),z+int(zz.min())]
    else:final=np.zeros((1,1,1),bool);fo=[max(0,x),max(0,y),z]
    np.save(d/'mask.npy',final);save(d/'mask_origin.json',fo);save(d/'component_selection.json',component)
    detail={k:v for k,v in track.items() if k!='frames'};detail['frames']=[{k:v for k,v in f.items() if k not in ('mask','probability')} for f in track['frames']];detail['crop_history']=history;save(d/'track.json',detail)
    a=dict(kind='three_prompt_same_video',branch='simple',bidirectional=True,seed_z=primary['z'],target_z=primary['z'],prompt=primary,prompts=anchors,crop_geometry=track['geometry'],initial_crop_side=history[0]['crop_side'],initial_crop_min_side=100,final_side=side,crop_history=history,stop_events=track['stop_events'],both_direction_end_reasons=track['direction_end_reasons'],status='repaired' if final.any() else 'empty_prediction',inference_seconds=seconds)
    ad=d/'action_000';ad.mkdir(exist_ok=True)
    for name,source in [('selected_mask.npy','mask.npy'),('origin.json','mask_origin.json'),('track.json','track.json')]:
        if not (ad/name).exists():(ad/name).symlink_to(d/source)
    save(d/'completed.json',dict(index=rec['index'],status=a['status'],repair_actions=[a],detector_prompt=primary,reference_selection=selection,initial_crop_min_side=100,pipeline='simple_three_prompt',prompt_count=len(anchors),score_threshold=.4,score_stop_count=1,same_video_memory=True,inference_seconds=seconds,assembly_seconds=assembly,elapsed_seconds=time.perf_counter()-start,nucleus_excluded=True,whole_prediction_replaces_original=True,limited_run=False))
