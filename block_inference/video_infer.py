"""Arbitrary block: min5 probe -> highest-score tight bbox -> bidirectional .45."""
import argparse,json,time,sys
from pathlib import Path
from common import ROOT,config,gpu,guard,save,signature,roi_for
from block_raw import BlockRaw
from filter_boxes import nms
from sequential_rules import box_foreground_fraction,mask_box_overlap
import numpy as np

def run(d):
    out=guard(Path(d['output'])/'video',d,'video')
    if (out/'completed.json').exists():return
    gpu(d)
    import torch
    import sequential3154_probe_best045 as method
    raw=BlockRaw(d['raw'],d.get('raw_cache_mib',512));roi=roi_for(raw,d)
    nucpath=Path(d.get('nucleus') or Path(d['output'])/'nucleus/nucleus.npy');nuc=np.load(nucpath,mmap_mode='r')
    if nuc.shape!=raw.a.shape or nuc.dtype!=bool:raise ValueError('nucleus mask must be bool .npy with same full ZYX shape as raw')
    detpath=Path(d.get('detections') or Path(d['output'])/'detection/detections.json');source=json.loads(detpath.read_text())
    if source.get('source_tag') and source['source_tag']!=raw.source_tag:raise ValueError('Detection cache belongs to another raw source')
    candidates=[];audit=[]
    for b in source['detections']:
        if not 0<=b['z']<raw.size[2]:raise ValueError('Detection Z outside source')
        ratio=box_foreground_fraction(nuc[b['z']],b['box_xyxy']);audit.append(dict(detection_id=b['global_detection_id'],nucleus_box_fraction=ratio,nucleus_suppressed=ratio>.6))
        if ratio<=.6:candidates.append(b)
    filtered,nms_audit=nms(candidates,.4);save(out/'filter_audit.json',dict(nucleus=audit,nms=nms_audit));save(out/'detections_filtered.json',dict(detections=filtered,roi=roi))
    method.engine.RUN=out;method.engine.GALLERY=out/'map_unused';method.engine.ROI=roi;method.engine.SCORE_THRESHOLD=.45
    for name in ['frames','tracks']:(out/name).mkdir(exist_ok=True)
    e=method.engine.create_local(d['raw'],nucpath,d['models']['video'],d['models']['sam2_base'],d['models']['sam2_code'],True)
    e.raw=raw;e.fast_encoder.io.raw=raw
    settings=d.get('execution_optimizations',True)
    if settings:
        from execution_optimizations import install,SnapshotIndex,overlap
        save(out/'execution_optimizations.json',install(e,settings))
    manager=None
    if d.get('advanced_optimizations'):
        from advanced_execution import install as install_advanced
        manager=install_advanced(e,method,d['advanced_optimizations'])
    save(out/'models.json',dict(video=signature(d['models']['video']),sam2_base=signature(d['models']['sam2_base']),nucleus_mask=signature(nucpath,False),detector_file=signature(detpath)))
    byz={}
    for b in filtered:
        x0,y0,x1,y1=b['box_xyxy']
        if roi[0]<=(x0+x1)/2<roi[3] and roi[1]<=(y0+y1)/2<roi[4] and roi[2]<=b['z']<roi[5]:byz.setdefault(b['z'],[]).append(b)
    buckets={};accepted=[];rejected=0;started=time.time();seed_timings=[]
    def infer_one(b):
        tid=b['global_detection_id']+1;marker=out/'tracks'/f'cached_{tid:05d}.json';trackfile=out/'tracks'/f'track_{tid:05d}.json'
        if marker.exists():
            t=json.loads(trackfile.read_text());outputs={};seed_seconds=json.loads(marker.read_text()).get('inference_seconds');reused=True
            for f in t['frames']:
                with np.load(out/'frames'/f'z{f["z"]:03d}_track{tid:05d}.npz') as a:outputs[f['z']]=(a['mask'].copy(),f)
        else:
            seed_started=time.perf_counter();record,outputs=method.infer_track(e,b,tid,started);seed_seconds=time.perf_counter()-seed_started;reused=False
            for q,(mask,f) in outputs.items():
                target=out/'frames'/f'z{q:03d}_track{tid:05d}.npz';tmp=target.with_suffix('.partial.npz');np.savez_compressed(tmp,mask=mask,origin_xy=f['origin_xy'],score=f['score']);tmp.replace(target)
            method.save_record(record);save(marker,dict(seed_id=tid,complete=True,inference_seconds=seed_seconds))
        return tid,outputs,seed_seconds,reused
    for z in range(roi[2],roi[5]):
        snapshot=list(buckets.get(z,[]));decisions=[];index=SnapshotIndex(snapshot) if settings else None;jobs=[]
        for b in sorted(byz.get(z,[]),key=lambda b:b['global_detection_id']):
            best=dict(iom=0.,opm=0.,overlap_max=0.);winner=None
            matches=index.query(b['box_xyxy']) if index else enumerate(snapshot)
            for i,(tid,mask,f) in matches:
                v=overlap(mask,f['origin_xy'],b['box_xyxy'],index.areas[i]) if index else mask_box_overlap(mask,f['origin_xy'],b['box_xyxy'])
                if v['overlap_max']>best['overlap_max']:best=v;winner=tid
            reject=best['overlap_max']>.5;decision=dict(detection_id=b['global_detection_id'],suppressed=reject,covered_by_track=winner if reject else None,**best)
            if reject:rejected+=1
            else:
                decision['track_id']=b['global_detection_id']+1;jobs.append(b)
            decisions.append(decision)
        # Same-frame new seeds never suppress one another. Compute them in
        # parallel, but commit outputs in original detection order for ties.
        if manager and manager.workers>1 and jobs:
            from concurrent.futures import ThreadPoolExecutor
            manager.parallel_width=min(manager.workers,len(jobs))
            with ThreadPoolExecutor(max_workers=manager.parallel_width) as pool:results=list(pool.map(infer_one,jobs))
        else:results=[infer_one(b) for b in jobs]
        for tid,outputs,seed_seconds,reused in results:
            seed_timings.append(dict(track_id=tid,inference_seconds=seed_seconds,reused=reused,retained_frames=len(outputs),has_mask=any(mask.any() for mask,_ in outputs.values())))
            accepted.append(tid)
            for q,(mask,f) in outputs.items():buckets.setdefault(q,[]).append((tid,mask,f))
        save(out/'decisions'/f'z{z:05d}.json',decisions);save(out/'accepted_tracks.json',accepted);save(out/'seed_timings.json',seed_timings)
        # A Z layer is never visited again. Release masks older than current Z;
        # backward results are still on disk for final 3D assembly.
        for q in list(buckets):
            if q<=z:del buckets[q]
        save(out/'progress.json',dict(z=z,layers_done=z-roi[2]+1,total_layers=roi[5]-roi[2],seed_attempts=len(accepted),boxes_suppressed=rejected,seconds=time.time()-started));print('video Z',z,'seed attempts',len(accepted),flush=True)
    if manager:save(out/'advanced_execution_stats.json',manager.close())
    e.fast_encoder.io.close();save(out/'black_plane_interpolation.json',list(raw.repair_log.values()))
    from execution_optimizations import stats
    save(out/'execution_stats.json',stats(e))
    save(out/'inference_completed.json',dict(limited_run=False,roi=roi,seed_attempts=len(accepted),seconds=time.time()-started))
    from assemble_instances import assemble
    rows=assemble(out,raw.a.shape,1000,2,optimized=bool(settings))
    save(out/'completed.json',dict(instances=len(rows),roi=roi,origin_xyz=d.get('origin_xyz',[0,0,0]),labels=str(out/'assembly/labels.npy'),instances_file=str(out/'assembly/instances.json'),limited_run=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);run(config(p.parse_args().config))
