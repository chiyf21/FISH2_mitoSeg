"""One persistent SAM model per GPU; dynamic jobs; CPU26CC overlaps next GPU job."""
import argparse,json,os,time,sys,urllib.request,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT,create_local
from mito_repair import finalize_case,read
from infer_block import save
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--token',required=True);p.add_argument('--worker',required=True);p.add_argument('--out',required=True);p.add_argument('--raw',required=True);p.add_argument('--nucleus',required=True);p.add_argument('--reference',required=True);p.add_argument('--baseline',action='store_true');a=p.parse_args()
    def rpc(path,**data):
        req=urllib.request.Request(a.queue+path,data=json.dumps(dict(token=a.token,worker=a.worker,**data)).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assets=ROOT/'baseline/assets';start=time.perf_counter();engine=create_local(a.raw,a.nucleus,assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',not a.baseline)
    import repair_component_ends as pipeline
    pipeline.RUN=out;pipeline.existing=lambda r:(np.load(r['mask_path'],mmap_mode='r'),r['origin_xyz'])
    os.environ['REPAIR_KEEP3D_COMPONENT']='0'
    post=ThreadPoolExecutor(max_workers=2);pending=[];case_times=[]
    def finish(row,t0,inference_wall):
        t=time.perf_counter();d=out/f'cases/instance_{row["index"]:03d}';final,o,audit=finalize_case(d)
        gold=Path(a.reference)/'cases'/d.name;expected=np.load(gold/'mask.npy',mmap_mode='r');eo=read(gold/'mask_origin.json')
        same=o==eo and np.array_equal(final,expected)
        np.save(d/'connected_mask.npy',final);save(d/'connectivity_audit.json',audit)
        result=dict(index=row['index'],exact_match=same,inference_wall_seconds=inference_wall,postprocess_seconds=time.perf_counter()-t,end_to_end_seconds=time.perf_counter()-t0)
        save(d/'smoke_result.json',result)
        if not same:raise AssertionError(f'Pixel difference for {d.name}')
        rpc('/complete',index=row['index'],result=result);return result
    while True:
        reply=rpc('/claim');row=reply['job']
        if row is None:break
        try:
            t=time.perf_counter();pipeline.run_case(engine,dict(**row,gt_id=row['index'],block=[0,0,0,*map(int,engine.raw.size)]));dt=time.perf_counter()-t
            if a.baseline:case_times.append(finish(row,t,dt))
            else:pending.append(post.submit(finish,row,t,dt))
            # Surface asynchronous failures promptly, not only at process exit.
            for f in pending:
                if f.done():f.result()
        except Exception as e:
            rpc('/fail',index=row['index'],result=dict(error=repr(e),trace=traceback.format_exc()));raise
    case_times.extend(f.result() for f in pending);post.shutdown(wait=True)
    if not a.baseline:engine.fast_encoder.io.close()
    stats=dict(engine.fast_stats) if not a.baseline else {}
    save(out/f'worker_{a.worker}.json',dict(worker=a.worker,baseline=a.baseline,cases=case_times,wall_seconds=time.perf_counter()-start,stats=stats,feature_cache_bytes=engine.fast_encoder.cache.bytes if not a.baseline else None,limited_run=False))
if __name__=='__main__':main()
