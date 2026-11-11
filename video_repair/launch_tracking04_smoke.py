import subprocess,sys,json,os,time
from pathlib import Path
ROOT=Path(__file__).parent;RUN=ROOT/'smoke_tracking_score04_113_20261103';RUN.mkdir(exist_ok=True)
if (RUN/'process.json').exists():
    p=json.loads((RUN/'process.json').read_text());proc=Path(f'/proc/{p["pid"]}')
    if proc.exists() and (proc/'cmdline').read_bytes():raise RuntimeError('Same smoke already running')
gpu='6';stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True);rows={r.split(',')[0].strip():r.split(',') for r in stats.splitlines()};uuid=rows[gpu][1].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert uuid not in apps and int(rows[gpu][2])<1000,'GPU6 acquired by another compute job'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (RUN/'smoke.log').open('a') as f:
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tracking_length_smoke04.py'),'--samples','6','--seconds-per-case','180'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
(RUN/'launched.json').write_text(json.dumps(dict(pid=p.pid,time=time.time(),gpu=gpu,threshold=.4,production_unchanged=True)));print((RUN/'launched.json').read_text())
