"""113-only true global tracking length smoke; isolated0.4, explicit censoring."""
import argparse,json,time,os,sys,types,traceback,gc,fcntl
from pathlib import Path
import numpy as np
from fast_runtime import ROOT
from global_runtime import create_global
from mito_repair import finalize_case,read
from infer_block import save
import torch
class SampleBudgetExceeded(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=6);p.add_argument('--seconds-per-case',type=int,default=180);a=p.parse_args()
    run=ROOT/'smoke_tracking_score04_113_20261103';run.mkdir(exist_ok=True);lock=(run/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(run/'process.json',dict(pid=os.getpid(),time=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),threshold=.4,production_policy_unchanged=True))
    plan=ROOT.parent/'sam_video_hardcases_20261102/repair20k_prefill200_plan_20261103/flagged_manifest.json';pool=read(plan);rng=np.random.default_rng(11304)
    chosen=[];used=set()
    for lo,hi in [(1,10),(10,15),(15,21),(21,35),(35,61),(61,10000)][:a.samples]:
        candidates=[r for r in pool if lo<=r['original_z_span']<hi and r['index'] not in used]
        r=candidates[int(rng.integers(len(candidates)))];chosen.append(r);used.add(r['index'])
    save(run/'manifest.json',chosen);save(run/'inference_policy.json',dict(score_threshold=.4,score_stop_count=1,strict_less_than=True,smoke_only=True,case_wall_budget_seconds=a.seconds_per_case,no_repair_z_cap=True))
    start=time.perf_counter();engine=create_global('http://127.0.0.1:18085',ROOT/'io_access.token','/path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt','/path/to/JaneliaEM/organelle_detect/scripts/train_nuclear_unet2d.py',ROOT/'nucleus_global_cache',True);engine.score_threshold=.4;engine.score_stop_count=1
    import repair_component_ends as pipeline
    pipeline.RUN=run;pipeline.existing=lambda rec:(np.load(rec['mask_path'],mmap_mode='r'),rec['origin_xyz']);os.environ['REPAIR_KEEP3D_COMPONENT']='0';current={};results=[]
    original_step=engine.model._track_step;original_rerun=pipeline.rerun_track_adaptive;original_track=engine.track
    def progress(force=False):
        if force or time.monotonic()-current.get('last_update',0)>4:
            current['last_update']=time.monotonic();save(run/'progress.json',dict(done=len(results),total=len(chosen),current_index=current.get('index'),original_z_span=current.get('original_span'),route=current.get('role'),attempt=current.get('attempt'),distinct_z_evaluated=len(current.get('z',set())),total_model_steps=current.get('steps',0),last_z=current.get('last_z'),last_score=current.get('last_score'),elapsed_current_seconds=time.monotonic()-current.get('started',time.monotonic()),case_budget_seconds=a.seconds_per_case,threshold=.4,completed_cases=sum(r['status']=='naturally_completed' for r in results),censored_cases=sum(r['status']=='time_budget_censored' for r in results)))
    def step(self,*args,**kw):
        if time.monotonic()>current['deadline']:raise SampleBudgetExceeded('Smoke wall-time budget, not a model stop')
        output=original_step(*args,**kw);z=int(args[0]);score=float(output[1][2].float().reshape(-1)[0]);current['z'].add(z);current['steps']+=1;current['last_z']=z;current['last_score']=score;progress();return output
    def rerun(e,det,*args,**kw):
        current['role']='/'.join(det.get('roles',[]));return original_rerun(e,det,*args,**kw)
    def track(self,*args,**kw):
        current['attempt']=current.get('attempt',0)+1;return original_track(*args,**kw)
    engine.model._track_step=types.MethodType(step,engine.model);engine.track=types.MethodType(track,engine);pipeline.rerun_track_adaptive=rerun
    def summarize():
        completed=[r for r in results if r['status']=='naturally_completed'];censored=[r for r in results if r['status']=='time_budget_censored'];errors=[r for r in results if r['status']=='error']
        summary=dict(requested=len(chosen),processed=len(results),completed=len(completed),censored=len(censored),errors=len(errors),score_threshold=.4,mean_original_z_span=float(np.mean([r['original_z_span'] for r in results])) if results else None,mean_tracked_z_completed=float(np.mean([r['accepted_distinct_evaluated_z'] for r in completed])) if completed else None,mean_nonempty_prediction_z_completed=float(np.mean([r['video_nonempty_z'] for r in completed])) if completed else None,mean_distinct_z_actually_evaluated_all=float(np.mean([r['observed_distinct_z'] for r in results])) if results else None,median_distinct_z_actually_evaluated_all=float(np.median([r['observed_distinct_z'] for r in results])) if results else None,censored_counts_are_lower_bounds_for_actual_compute_not_final_mask_z=True,completed_only_mean_is_selection_biased_if_censored=bool(censored),elapsed_seconds=time.perf_counter()-start,results=results,limited_run=bool(censored or errors or len(results)!=len(chosen)),no_full_batch_launched=True,production_threshold_unchanged=.5)
        save(run/'RESULTS.json',summary);return summary
    for r in chosen:
        current.clear();t=time.monotonic();current.update(index=r['index'],original_span=r['original_z_span'],started=t,deadline=t+a.seconds_per_case,z=set(),steps=0);progress(True)
        base=dict(index=r['index'],record_id=r['record_id'],body_id=r['body_id'],original_z_span=r['original_z_span'],origin_xyz=r['origin_xyz'])
        try:
            pipeline.run_case(engine,dict(**r,gt_id=r['index'],block=[0,0,0,*map(int,engine.raw.size)]))
            d=run/f'cases/instance_{r["index"]:03d}';meta=read(d/'completed.json');evaluated=set();nonempty=set();reasons={}
            for i,act in enumerate(meta['repair_actions']):
                ad=d/f'action_{i:03d}';tr=read(ad/'track.json');evaluated.update(int(f['z']) for f in tr['evaluated_scores']);m=np.load(ad/'selected_mask.npy',mmap_mode='r');o=read(ad/'origin.json');nonempty.update(o[2]+int(j) for j in np.flatnonzero(m.any((1,2))))
                for why in tr['direction_end_reasons'].values():reasons[why]=reasons.get(why,0)+1
            base.update(status='naturally_completed',accepted_distinct_evaluated_z=len(evaluated),video_nonempty_z=len(nonempty),stop_reasons=reasons,limited_run=False)
        except SampleBudgetExceeded:base.update(status='time_budget_censored',unfinished_route=current.get('role'),unfinished_attempt=current.get('attempt'),limited_run=True)
        except Exception as exc:base.update(status='error',error=repr(exc),traceback=traceback.format_exc(),limited_run=True)
        base.update(observed_distinct_z=len(current['z']),total_model_steps=current['steps'],observed_z_range=[min(current['z']),max(current['z'])] if current['z'] else None,last_z=current.get('last_z'),last_score=current.get('last_score'),wall_seconds=time.monotonic()-t)
        results.append(base);save(run/f'case_result_{r["index"]:06d}.json',base);summarize();print(json.dumps(base),flush=True);gc.collect();torch.cuda.empty_cache()
    engine.fast_encoder.io.close();engine.nuc.pool.shutdown(wait=True);engine.raw.pool.shutdown(wait=True);summary=summarize();save(run/'smoke_finished.json',dict(processed=len(results),completed=summary['completed'],censored=summary['censored'],limited_run=summary['limited_run']));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
