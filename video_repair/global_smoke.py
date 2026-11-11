"""True global-EM smoke: baseline vs optimized; never truncate propagation Z."""
import argparse,time,json,os,sys,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT
from global_runtime import create_global
from mito_repair import finalize_case,read
from infer_block import save
from policy import load_policy
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--nuclear-code',required=True);p.add_argument('--nuclear-weight',required=True);p.add_argument('--mode',choices=['baseline','optimized'],required=True);p.add_argument('--limit',type=int,default=4);a=p.parse_args()
    tag='score'+str(load_policy()['score_threshold']).replace('.','');manifest=ROOT/'global_smoke_inputs/manifest.json';rows=read(manifest)[:a.limit];out=ROOT/('smoke_global_'+tag+'_20261103')/a.mode;out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    engine=create_global('http://127.0.0.1:18085',ROOT/'io_access.token',a.nuclear_weight,a.nuclear_code,ROOT/'nucleus_global_cache',a.mode=='optimized')
    import repair_component_ends as pipeline
    pipeline.RUN=out;pipeline.existing=lambda r:(np.load(manifest.parent/r['mask_path'],mmap_mode='r'),r['origin_xyz']);os.environ['REPAIR_KEEP3D_COMPONENT']='0';results=[];post=ThreadPoolExecutor(max_workers=2);pending=[]
    def finish(row,t0,infer):
        t=time.perf_counter();d=out/f'cases/instance_{row["index"]:03d}';m,o,audit=finalize_case(d);np.save(d/'connected_mask.npy',m);save(d/'connected_origin.json',o)
        identical=None
        if a.mode=='optimized':
            ref=out.parent/'baseline/cases'/d.name
            identical=o==read(ref/'connected_origin.json') and np.array_equal(m,np.load(ref/'connected_mask.npy',mmap_mode='r'))
            if not identical:raise AssertionError('Global baseline/optimized pixel mismatch: '+d.name)
        meta=read(d/'completed.json');result=dict(index=row['index'],source_z_span=row['original_z_span'],routes=len(meta['repair_actions']),inference_wall_seconds=infer,postprocess_seconds=time.perf_counter()-t,end_to_end_seconds=time.perf_counter()-t0,exact_match=identical,encoded_batches=sum(len(read(d/f'action_{i:03d}/track.json')['encoded_z_batches']) for i in range(len(meta['repair_actions']))))
        save(d/'smoke_result.json',result);return result
    for row in rows:
        t=time.perf_counter();save(out/'progress.json',dict(mode=a.mode,current_index=row['index'],done=len(results),total=len(rows),source_z_span=row['original_z_span'],started=time.time()))
        pipeline.run_case(engine,dict(**row,gt_id=row['index'],block=[0,0,0,*map(int,engine.raw.size)]));dt=time.perf_counter()-t
        if a.mode=='baseline':results.append(finish(row,t,dt))
        else:pending.append(post.submit(finish,row,t,dt))
        for f in pending:
            if f.done():f.result()
    results.extend(f.result() for f in pending);post.shutdown(wait=True)
    if a.mode=='optimized':engine.fast_encoder.io.close()
    engine.nuc.pool.shutdown(wait=True)
    stats=dict(engine.fast_stats) if a.mode=='optimized' else {};report=dict(mode=a.mode,cases=results,wall_seconds=time.perf_counter()-started,raw_stats=dict(engine.raw.stats),nucleus_stats=dict(engine.nuc.stats),fast_stats=stats,all_pixel_exact=all(r['exact_match'] for r in results) if a.mode=='optimized' else None,limited_run=False)
    save(out/'RESULTS.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
