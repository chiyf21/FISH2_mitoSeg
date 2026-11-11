"""Baseline/optimized stratified smoke on four idle GPUs, exact saved-reference check."""
import os,json,time,subprocess,sys,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent.resolve();REF=ROOT.parent/'sam_video_hardcases_20261102';RUN=ROOT/'smoke_local_20261103'
def main():
    from policy import load_policy
    if load_policy()['score_threshold']!=.4:raise RuntimeError('Historical0.4 golden smoke: current policy differs; use a fresh0.5 baseline/new output, do not overwrite historical comparisons')
    RUN.mkdir(exist_ok=True);lock=(RUN/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True);mem={int(r.split(',')[0]):int(r.split(',')[1]) for r in stats.splitlines()};assert all(mem[i]<500 for i in (2,3,4,5)),mem
    assets=ROOT/'baseline/assets'
    if not assets.exists():assets.symlink_to(ROOT.parent/'mito_video_repair_v1/assets',target_is_directory=True)
    source=REF/'simple113_component500_union_tight_score04_20261103';gold=REF/'simple113_component500_union_tight_cc26_20261103';records=[]
    for d in source.glob('cases/instance_*'):
        if not (d/'existing_mask.npy').exists():continue
        idx=int(d.name.split('_')[-1]);meta=json.loads((d/'completed.json').read_text());m=np.load(d/'existing_mask.npy',mmap_mode='r');o=json.loads((d/'existing_origin.json').read_text())
        records.append(dict(index=idx,mask_path=str(d/'existing_mask.npy'),origin_xyz=o,cost=sum(a['initial_crop_side']**2 for a in meta['repair_actions'])*len(m),source_z=len(m),route_count=len(meta['repair_actions'])))
    records.sort(key=lambda r:r['cost']);chosen=[records[int(i)] for i in np.linspace(0,len(records)-1,24,dtype=int)]
    # Spatial order is independent of task identity. Dynamic queue preserves it.
    chosen.sort(key=lambda r:(r['origin_xyz'][2]//32,r['origin_xyz'][1]//512,r['origin_xyz'][0]//512))
    phase_results={}
    for phase,rows in [('baseline',chosen[::2]),('optimized',chosen)]:
        out=RUN/phase;out.mkdir(exist_ok=True);manifest=out/'manifest.json';manifest.write_text(json.dumps(rows));log=(out/'queue.log').open('a')
        server=subprocess.Popen([sys.executable,'-u',str(ROOT/'queue_service.py'),'--manifest',str(manifest),'--out',str(out),'--port','18084'],stdout=log,stderr=log)
        try:
            deadline=time.time()+20
            while not (out/'queue_access.json').exists():
                if server.poll() is not None:raise RuntimeError('Queue failed')
                if time.time()>deadline:raise RuntimeError('Queue startup timeout')
                time.sleep(.2)
            token=json.loads((out/'queue_access.json').read_text())['token'];jobs=[];start=time.time()
            for gpu in (2,3,4,5):
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');f=(out/f'gpu{gpu}.log').open('a')
                cmd=[sys.executable,'-u',str(ROOT/'smoke_worker.py'),'--queue','http://127.0.0.1:18084','--token',token,'--worker',str(gpu),'--out',str(out),'--raw',str(REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif'),'--nucleus',str(REF/'instance_centered_20261102/nucleus.npy'),'--reference',str(gold)]
                if phase=='baseline':cmd+=['--baseline']
                jobs.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=f));f.close()
            (RUN/'progress.json').write_text(json.dumps(dict(phase=phase,pids=[p.pid for p in jobs],samples=len(rows),start=start)))
            codes=[p.wait() for p in jobs]
            if any(codes):raise RuntimeError(f'{phase} failed: {codes}')
            reports=[json.loads(p.read_text()) for p in out.glob('worker_*.json')];cases=[c for r in reports for c in r['cases']];assert len(cases)==len(rows) and all(c['exact_match'] for c in cases)
            phase_results[phase]=dict(samples=len(cases),wall_seconds=time.time()-start,cases=cases,workers=reports)
        finally:server.terminate();server.wait(timeout=10);log.close()
    b={c['index']:c for c in phase_results['baseline']['cases']};f={c['index']:c for c in phase_results['optimized']['cases']};matched=[b[k]['inference_wall_seconds']/f[k]['inference_wall_seconds'] for k in b]
    result=dict(phases=phase_results,all_final_masks_pixel_exact=True,matched_median_inference_speedup=float(np.median(matched)),limited_run=False)
    (RUN/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(all_masks_exact=True,median_speedup=result['matched_median_inference_speedup'],baseline_wall=phase_results['baseline']['wall_seconds'],optimized_wall=phase_results['optimized']['wall_seconds'])),flush=True)
if __name__=='__main__':main()
