import json,time,subprocess,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;ref=ROOT.parent/'sam_video_hardcases_20261102';source=ref/'simple113_component500_union_tight_score04_20261103';gold=ref/'simple113_component500_union_tight_cc26_20261103';out=ROOT/'smoke_production_worker_20261103';out.mkdir(exist_ok=True);rows=[]
from policy import load_policy
if load_policy()['score_threshold']!=.4:raise RuntimeError('This historical golden reference uses0.4; do not compare it as a0.5 run')
for idx in (3,63):
    d=source/f'cases/instance_{idx:03d}';rows.append(dict(index=idx,mask_path=str(d/'existing_mask.npy'),origin_xyz=json.loads((d/'existing_origin.json').read_text())))
(out/'manifest.json').write_text(json.dumps(rows));service=subprocess.Popen([sys.executable,str(ROOT/'queue_service.py'),'--manifest',str(out/'manifest.json'),'--out',str(out),'--port','18086'],stdout=subprocess.DEVNULL)
try:
    deadline=time.time()+10
    while not (out/'queue_access.json').exists():
        if time.time()>deadline:raise RuntimeError('Queue startup failed')
        time.sleep(.1)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,str(ROOT/'optimized_worker.py'),'--queue','http://127.0.0.1:18086','--token-file',str(out/'queue_access.json'),'--worker','production-smoke','--out',str(out/'results'),'--source','local','--raw',str(ref.parent.parent/'flyEM/2500x2500x250/raw.ome.tif'),'--nucleus',str(ref/'instance_centered_20261102/nucleus.npy')],env=env,check=True)
    for row in rows:
        d=out/'results/cases'/f'instance_{row["index"]:03d}';g=gold/'cases'/d.name
        assert json.loads((d/'mask_origin.json').read_text())==json.loads((g/'mask_origin.json').read_text()) and np.array_equal(np.load(d/'mask.npy'),np.load(g/'mask.npy'))
    (out/'VERIFIED.json').write_text(json.dumps(dict(reusable_worker_exact=True,instances=[3,63],next_instance_prefetch=True,async_26cc=True,limited_run=False)));print('PASS:reusable optimized worker, next-instance prefetch and async26CC output exact')
finally:service.terminate();service.wait(timeout=10)
