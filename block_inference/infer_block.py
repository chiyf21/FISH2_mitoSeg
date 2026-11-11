"""Unified nucleus -> detection -> Video -> 3D instance inference entry."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--check',action='store_true',help='Validate paths/assets and print input/ROI without running models');a=p.parse_args()
    from common import ROOT,config,guard,save,roi_for
    from block_raw import BlockRaw
    d=config(a.config);raw=BlockRaw(d['raw'],d.get('raw_cache_mib',512));roi=roi_for(raw,d)
    for key,path in d['models'].items():
        if not Path(path).exists():raise FileNotFoundError(f'{key}: {path}')
    if a.check:print(json.dumps(dict(shape_zyx=list(raw.a.shape),roi=roi,models=d['models'],output=d['output']),indent=2));return
    import fcntl
    out=guard(d['output'],d,'pipeline');lock=open(out/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
    stages=[]
    if not d.get('nucleus'):stages.append('nucleus_infer.py')
    if not d.get('detections'):stages.append('detection_infer.py')
    stages.append('video_infer.py')
    timings=[]
    for script in stages:
        save(out/'state.json',dict(stage=script,status='running',seconds=time.time()-started))
        stage_dir={'nucleus_infer.py':'nucleus','detection_infer.py':'detection','video_infer.py':'video'}[script]
        reused=(out/stage_dir/'completed.json').exists();stage_started=time.perf_counter()
        subprocess.run([sys.executable,'-u',str(ROOT/script),'--config',str(Path(a.config).resolve())],check=True)
        timings.append(dict(stage=stage_dir,wall_seconds=time.perf_counter()-stage_started,reused_completed_output=reused));save(out/'stage_timings.json',timings)
    result=json.loads((out/'video/completed.json').read_text());result.update(seconds=time.time()-started,config=str(Path(a.config).resolve()),stage_timings=timings)
    save(out/'completed.json',result);save(out/'state.json',dict(stage='all',status='complete'));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
