"""Bounded timing probe, NOT a completed repair. Hash raw SAM outputs before abort."""
import argparse,json,time,types,hashlib,os,sys
from pathlib import Path
from fast_runtime import ROOT
from global_runtime import create_global
from infer_block import save
from policy import load_policy
import numpy as np
class ProfileComplete(Exception):pass
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','optimized'],required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args();tag='score'+str(load_policy()['score_threshold']).replace('.','');out=ROOT/('smoke_global_profile_'+tag+'_20261103')/a.mode;out.mkdir(parents=True,exist_ok=True);rows=json.loads((ROOT/'global_smoke_inputs/manifest.json').read_text());row=rows[0];start=time.perf_counter()
    engine=create_global('http://127.0.0.1:18085',ROOT/'io_access.token','/path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt','/path/to/JaneliaEM/organelle_detect/scripts/train_nuclear_unet2d.py',ROOT/'nucleus_global_cache',a.mode=='optimized');setup=time.perf_counter()-start
    import repair_component_ends as pipeline
    pipeline.RUN=out;pipeline.existing=lambda r:(np.load(ROOT/'global_smoke_inputs'/r['mask_path'],mmap_mode='r'),r['origin_xyz']);os.environ['REPAIR_KEEP3D_COMPONENT']='0';original=engine.model._track_step;frames=[];t0=time.perf_counter()
    def step(self,*args,**kwargs):
        result=original(*args,**kwargs);outs=result[1];hi=outs[4].detach().contiguous().cpu();score=float(outs[2].float().reshape(-1)[0]);blob=hi.view(__import__('torch').uint8).numpy().tobytes()
        frames.append(dict(z=int(args[0]),reverse=bool(args[9]),is_prompt=bool(args[1]),score=score,logits_sha256=hashlib.sha256(blob).hexdigest(),logits_shape=list(hi.shape),dtype=str(hi.dtype)))
        save(out/'prefix_progress.json',dict(steps=len(frames),target=a.steps,z=int(args[0]),score=score,elapsed_seconds=time.perf_counter()-t0,limited_run=True))
        if len(frames)>=a.steps:raise ProfileComplete()
        return result
    engine.model._track_step=types.MethodType(step,engine.model);natural=False
    try:pipeline.run_case(engine,dict(**row,gt_id=row['index'],block=[0,0,0,*map(int,engine.raw.size)]));natural=True
    except ProfileComplete:pass
    elapsed=time.perf_counter()-t0
    if a.mode=='optimized':engine.fast_encoder.io.close()
    engine.nuc.pool.shutdown(wait=True);engine.raw.pool.shutdown(wait=True)
    report=dict(mode=a.mode,index=row['index'],original_z_span=row['original_z_span'],steps=len(frames),setup_seconds=setup,wall_seconds=elapsed,seconds_per_step=elapsed/max(1,len(frames)),frames=frames,raw_stats=dict(engine.raw.stats),nucleus_stats=dict(engine.nuc.stats),fast_stats=dict(engine.fast_stats) if a.mode=='optimized' else {},naturally_completed=natural,limited_run=not natural,repair_result_usable=natural)
    save(out/'RESULTS.json',report)
    if a.mode=='optimized':
        base=json.loads((out.parent/'baseline/RESULTS.json').read_text());assert base['frames']==frames,'SAM prefix output differs'
        save(out.parent/'COMPARISON.json',dict(prefix_logits_and_scores_pixel_exact=True,steps=len(frames),baseline_wall_seconds=base['wall_seconds'],optimized_wall_seconds=elapsed,speedup=base['wall_seconds']/elapsed,limited_run=not natural,not_full_instance_completion=True))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}),flush=True)
if __name__=='__main__':main()
