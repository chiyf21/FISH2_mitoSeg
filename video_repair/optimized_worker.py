"""Reusable local/global optimized worker, independent GPU memories and async26CC."""
import argparse,json,os,time,urllib.request,traceback,socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT,create_local
from mito_repair import finalize_case,read
from infer_block import save
from repair_component_ends import plan
from repair_three_prompt import shared_geometry
import numpy as np
from scipy import ndimage as ndi

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',default='http://127.0.0.1:18084');p.add_argument('--token-file',required=True);p.add_argument('--worker',required=True);p.add_argument('--out',required=True);p.add_argument('--source',choices=['local','global'],required=True);p.add_argument('--raw');p.add_argument('--nucleus');p.add_argument('--raw-url',default='http://127.0.0.1:18085');p.add_argument('--raw-token-file',default=str(ROOT/'io_access.token'));p.add_argument('--nuclear-code');p.add_argument('--nuclear-weight');p.add_argument('--nuclear-cache',default=str(ROOT/'nucleus_global_cache'));a=p.parse_args()
    token_data=read(a.token_file);token=token_data['token']
    def rpc(path,**data):
        req=urllib.request.Request(a.queue+path,data=json.dumps(dict(token=token,worker=a.worker,**data)).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assets=ROOT/'baseline/assets';t0=time.perf_counter()
    if a.source=='global':
        from global_runtime import create_global
        engine=create_global(a.raw_url,a.raw_token_file,a.nuclear_weight,a.nuclear_code,a.nuclear_cache,True)
    else:engine=create_local(a.raw,a.nucleus,assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True)
    import repair_component_ends as pipeline
    pipeline.RUN=out;pipeline.existing=lambda r:(np.load(r['prepared_mask_path'],mmap_mode='r'),r['origin_xyz']);os.environ['REPAIR_KEEP3D_COMPONENT']='0';post=ThreadPoolExecutor(max_workers=2);pre=ThreadPoolExecutor(max_workers=1);pending=[]
    policy_path=out/'inference_policy.json'
    if policy_path.exists() and read(policy_path)!=engine.inference_policy:raise RuntimeError('Output directory belongs to a different stopping policy; choose a new output path')
    save(policy_path,engine.inference_policy)
    def prepare(row):
        source=Path(row.get('mask_path_by_host',{}).get(socket.gethostname(),row['mask_path']))
        if not source.exists():
            source=Path(row.get('original_mask_path112','')) if str(ROOT).startswith('/mnt/data21T_3') else Path(row['mask_path'])
        m=np.array(np.load(source,mmap_mode='r'),dtype=bool);added=0
        if row.get('prefill_lt200',False):
            for j in range(len(m)):
                h=ndi.binary_fill_holes(m[j])&~m[j];cc,n=ndi.label(h);sizes=np.bincount(cc.ravel());keep=np.flatnonzero((sizes>0)&(sizes<200));keep=keep[keep!=0];fill=np.isin(cc,keep);added+=int(fill.sum());m[j]|=fill
        if 'expected_prepared_voxels' in row:assert int(m.sum())==row['expected_prepared_voxels'],'Input/prefill differs from verified plan'
        d=out/f'inputs/{row["index"]:06d}';d.mkdir(parents=True,exist_ok=True);np.save(d/'prepared.npy',m);row={**row,'prepared_mask_path':str(d/'prepared.npy')};save(d/'prefill.json',dict(added_voxels=added,threshold_exclusive=200,source=str(source)))
        specs,_=plan(m,row['origin_xyz'],int(engine.raw.size[2]))
        if specs:
            g,side=shared_geometry([specs[0]['prompt']],int(engine.raw.size[2]));engine.raw.slice_crop(specs[0]['prompt']['z'],g[0],g[1],side)
        return row
    def finish(row,begin,gpu_wall):
        d=out/f'cases/instance_{row["index"]:03d}';t=time.perf_counter();final,o,audit=finalize_case(d)
        if not (d/'union_mask.npy').exists():(d/'mask.npy').rename(d/'union_mask.npy')
        tmp=d/'final.partial.npy';np.save(tmp,final);tmp.replace(d/'mask.npy');save(d/'connectivity_audit.json',audit)
        meta=read(d/'completed.json');meta.update(final_connectivity=26,component_selection=audit,original_mask_preserved='before26CC',optimized=True);save(d/'completed.json',meta)
        result=dict(index=row['index'],host=socket.gethostname(),mask=str(d/'mask.npy'),origin_xyz=o,inference_wall_seconds=gpu_wall,postprocess_seconds=time.perf_counter()-t,wall_seconds=time.perf_counter()-begin,limited_run=False)
        save(d/'finalized.json',result);rpc('/complete',index=row['index'],result=result);return result
    current=rpc('/claim')['job'];next_prepared=pre.submit(prepare,current) if current else None
    while next_prepared:
        try:row=next_prepared.result()
        except Exception as exc:rpc('/fail',index=current['index'],result=dict(error=repr(exc),phase='input_prepare'));raise
        next_job=rpc('/claim')['job'];next_prepared=pre.submit(prepare,next_job) if next_job else None;current=next_job
        try:
            t=time.perf_counter();pipeline.run_case(engine,dict(**row,gt_id=row['index'],block=[0,0,0,*map(int,engine.raw.size)]));pending.append(post.submit(finish,row,t,time.perf_counter()-t))
            for f in pending:
                if f.done():f.result()
        except Exception as exc:rpc('/fail',index=row['index'],result=dict(error=repr(exc),trace=traceback.format_exc()));raise
    results=[f.result() for f in pending];post.shutdown(wait=True);pre.shutdown(wait=True);engine.fast_encoder.io.close()
    if a.source=='global':engine.nuc.pool.shutdown(wait=True);engine.raw.pool.shutdown(wait=True)
    save(out/f'worker_{a.worker}.json',dict(worker=a.worker,host=socket.gethostname(),cases=results,wall_seconds=time.perf_counter()-t0,stats=dict(engine.fast_stats),limited_run=False))
if __name__=='__main__':main()
