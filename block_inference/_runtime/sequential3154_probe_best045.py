"""Minimum-five forward probe, best-mask tight prompt, bidirectional Video .45."""
import json,time,traceback,math
from types import SimpleNamespace
import numpy as np
import torch
import sequential3154_lookahead10 as prep
from sequential_rules import mask_box_overlap
from refresh5_rules import tight_prompt
from infer_block import save

engine=prep.engine
engine.RUN=engine.ROOT/'sequential3154_probe5_best_bidir045_cc26_20261103'
engine.GALLERY=engine.TRAIN/'dashboard/sam_video_hardcases_20261102/sequential3154_probe5_best_bidir045_cc26'
engine.DETECTIONS_PATH=engine.RUN/'detections_filtered.json'
engine.SCORE_THRESHOLD=.45

def progress(z,tid,phase,frame_z,started,extra=None):
    save(engine.RUN/'phase_progress.json',dict(z=z,track_id=tid,phase=phase,frame_z=frame_z,seconds=time.time()-started,**(extra or {})))

def infer_track(e,b,tid,started):
    roi=engine.ROI;origin_z=b['z'];record=SimpleNamespace(id=tid,seed=b,frames=[],history=[],active=False,stop=None)
    probe=engine.Tracker(e,b,tid,score_threshold=-math.inf);first_low=None;tested=[];best=None;probe_reason=None
    for q in range(origin_z,roi[5]):
        progress(origin_z,tid,'forward_probe',q,started)
        out=probe.advance(q)
        if out is None:probe_reason=probe.stop;break
        mask,f=out;box=tight_prompt(mask,f['origin_xy']);tested.append(dict(**f,tight_bbox=box,mask_area=int(mask.sum())))
        if f['score']<.45 and first_low is None:first_low=q
        if box is not None and f['score']>=.45 and (best is None or f['score']>best['score']):best=dict(z=q,score=f['score'],box_xyxy=box,mask_area=int(mask.sum()))
        if first_low is not None and q>=origin_z+5:
            probe_reason=dict(reason='first_low_after_minimum5',first_low_z=first_low,last_probe_z=q);break
    if probe_reason is None:probe_reason=dict(reason='requested_roi_z_boundary',z=roi[5]-1)
    probe.state={}
    save(engine.RUN/'tracks'/f'probe_{tid:05d}.json',dict(initial_detection=b,frames=tested,first_low_z=first_low,stop=probe_reason,best=best))
    if best is None:
        record.stop=dict(reason='probe_no_nonempty_mask_score_ge045',probe_stop=probe_reason);return record,{}
    anchor={**b,'z':best['z'],'box_xyxy':best['box_xyxy']};record.seed=anchor
    outputs={};direction_stops={};prompts=[dict(z=origin_z,box_xyxy=b['box_xyxy'],source='detector',role='probe_only'),dict(z=best['z'],box_xyxy=best['box_xyxy'],source='best_probe_mask_tight_bbox',padding=0,score=best['score'],role='bidirectional_anchor')]
    for direction,reverse in [('forward',False),('backward',True)]:
        inner=engine.Tracker(e,anchor,tid,reverse=reverse);order=range(anchor['z'],roi[2]-1,-1) if reverse else range(anchor['z'],roi[5]);segment=0
        for q in order:
            progress(origin_z,tid,direction,q,started,dict(anchor_z=anchor['z']))
            out=inner.advance(q)
            if out is None:direction_stops[direction]=inner.stop;break
            mask,f=out;offset=abs(q-anchor['z'])
            if offset>0 and offset%5==0 and (q>roi[2] if reverse else q<roi[5]-1):
                box=tight_prompt(mask,f['origin_xy'])
                if box is None:direction_stops[direction]=dict(reason='empty_mask_at_refresh',z=q);break
                record.history.extend([dict(direction=direction,**h) for h in inner.history]);inner.state={};segment+=1
                reseed={**anchor,'z':q,'box_xyxy':box};prompts.append(dict(z=q,box_xyxy=box,source='predicted_mask_tight_bbox',padding=0,direction=direction,segment=segment))
                inner=engine.Tracker(e,reseed,tid,reverse=reverse);out=inner.advance(q)
                if out is None:direction_stops[direction]=dict(reason='refresh_prompt_low_score',z=q,detail=inner.stop);break
                mask,f=out
            f=dict(f,direction=direction,segment=segment,conditioning_z=inner.seed['z'],prompt_box_xyxy=inner.seed['box_xyxy'],is_prompt_frame=q==inner.seed['z'])
            if q not in outputs or f['score']>outputs[q][1]['score']:outputs[q]=(mask,f)
        if direction not in direction_stops:direction_stops[direction]=dict(reason='requested_roi_z_boundary',z=roi[2] if reverse else roi[5]-1)
        record.history.extend([dict(direction=direction,**h) for h in inner.history]);inner.state={}
    record.frames=[outputs[q][1] for q in sorted(outputs)];record.stop=dict(reason='bidirectional_finished',direction_stops=direction_stops)
    record.propagation_mode='bidirectional_score_stop';record.direction_stops=direction_stops
    record.initial_detection=b;record.probe_best=best
    save(engine.RUN/'tracks'/f'prompts_{tid:05d}.json',prompts)
    save(engine.RUN/'tracks'/f'bidirectional_{tid:05d}.json',dict(initial_detection=b,anchor=anchor,direction_stops=direction_stops,probe_best=best))
    return record,outputs

def save_record(t):
    save(engine.RUN/'tracks'/f'track_{t.id:05d}.json',dict(track_id=t.id,seed=t.seed,frames=t.frames,crop_history=t.history,stop=t.stop,active=False,propagation_mode=getattr(t,'propagation_mode','probe_failed'),direction_stops=getattr(t,'direction_stops',{}),initial_detection=getattr(t,'initial_detection',t.seed),probe_best=getattr(t,'probe_best',None)))

def main():
    import fcntl
    r,g=engine.RUN,engine.GALLERY;r.mkdir(parents=True,exist_ok=True);g.mkdir(parents=True,exist_ok=True);lock=open(r/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (r/'completed.json').exists():return
    for name in ['tracks','frames']:(r/name).mkdir(exist_ok=True)
    prior=g.parent/'sequential3154_iom08_score05'
    for name in ['raw','gt']:
        if not (g/name).exists():(g/name).symlink_to(prior/name,target_is_directory=True)
    prep.prepare()
    html=prep.HTML.replace('score&lt;0.5','score&lt;0.45').replace('逐帧补种追踪：3154所在块＋全局检测框','试推最高分mask → tight bbox双向Video 0.45')
    html=html.replace('已有实例先续推一层。','原框向后试推至首次低分，至少5帧；最高分非空mask的tight bbox作独立双向prompt；score0.45，每5层更新prompt。')
    (g/'index.html').write_text(html)
    config=dict(roi=engine.ROI,score_threshold=.45,probe_min_forward_steps=5,probe_end='max(first low-score forward offset,5), limited by ROI boundary; no10frame cap',anchor='highest score >=.45 nonempty probe mask tight bbox, zero padding',propagation='independent forward/backward to score<.45 or ROI boundary, refresh tight bbox every5steps',global_nms_iou=.4,nucleus_box_fraction_threshold=.6,overlap_reject='max(mask coverage,box coverage)>.5',iou_scope='detection_roi',iou_bbox_xyz=engine.ROI,assembly='same-Z union then26CC,>=1000voxels,>=2Z')
    save(r/'config.json',config);save(g/'config.json',config)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25);started=time.time();assets=engine.ROOT/'baseline/assets'
    e=engine.create_local(engine.REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif',engine.REF/'instance_centered_20261102/nucleus.npy',assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True)
    byz={};roi=engine.ROI
    for b in json.loads(engine.DETECTIONS_PATH.read_text())['detections']:
        x0,y0,x1,y1=b['box_xyxy']
        if roi[0]<=(x0+x1)/2<roi[3] and roi[1]<=(y0+y1)/2<roi[4] and roi[2]<=b['z']<roi[5]:byz.setdefault(b['z'],[]).append(b)
    buckets={};tracks=[];all_decisions={};engine.SUPPRESSED=0
    for z in range(roi[2],roi[5]):
        snapshot=list(buckets.get(z,[]));decisions=[]
        for b in sorted(byz.get(z,[]),key=lambda d:d['global_detection_id']):
            best=dict(iom=0.,opm=0.,overlap_max=0.);winner=None
            for tid,mask,f in snapshot:
                v=mask_box_overlap(mask,f['origin_xy'],b['box_xyxy'])
                if v['overlap_max']>best['overlap_max']:best=v;winner=tid
            rejected=best['overlap_max']>.5;d=dict(detection_id=b['global_detection_id'],suppressed=rejected,covered_by_track=winner if rejected else None,**best)
            if rejected:engine.SUPPRESSED+=1
            else:
                t,outputs=infer_track(e,b,len(tracks)+1,started);tracks.append(t);d['track_id']=t.id;d['seed_stop']=t.stop
                for q,(mask,f) in outputs.items():
                    buckets.setdefault(q,[]).append((t.id,mask,f));np.savez_compressed(r/'frames'/f'z{q:03d}_track{t.id:05d}.npz',mask=mask,origin_xy=f['origin_xy'],score=f['score'])
                save_record(t)
            decisions.append(d)
        all_decisions[z]=decisions;engine.publish(z,buckets.get(z,[]),tracks,decisions,started)
    # Later anchors can predict earlier layers. Refresh all map layers, without
    # retroactively changing historical box suppression decisions.
    for z in range(roi[2],roi[5]):engine.publish(z,buckets.get(z,[]),tracks,all_decisions[z],started,z==roi[5]-1)
    for t in tracks:save_record(t)
    save(r/'inference_completed.json',dict(limited_run=False,layers=44,tracks=len(tracks),seconds=time.time()-started));e.fast_encoder.io.close()
    from assemble_refresh5 import main as assemble
    assemble(r,g)

if __name__=='__main__':
    try:main()
    except Exception:
        save(engine.RUN/'error.json',dict(trace=traceback.format_exc(),time=time.time()));raise
