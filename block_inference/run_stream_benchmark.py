import json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
B=P/'validation/advanced_20261104_013608/full'
base=json.loads((B/'B0_current.json').read_text())
opts=dict(replay_batch_encoder=True,replay_batch_size=4,replay_mode='streams',image_streams=2,prompt_initialization_reuse=True,prefix_cache=True,multi_track_decoder=True,track_workers=4,decode_batch_size=4,batch_wait_ms=1,decode_mode='streams',decode_streams=4)
name='T_streams';out=B/name;cfg=B/(name+'.json');base.update(output=str(out),advanced_optimizations=opts);cfg.write_text(json.dumps(base,indent=2))
if out.exists():raise ValueError(f'Fresh output required: {out}')
with (B/(name+'.log')).open('w') as log:subprocess.run([sys.executable,'-u',str(P/'infer_block.py'),'--config',str(cfg)],cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
from run_advanced_benchmark import compare,evaluate
r=json.loads((out/'completed.json').read_text());r['comparison']=compare(B/'B0_current',out);r['gt']=evaluate(out/'video/assembly/labels.npy',__import__('numpy').load(B.parent/'manual_gt.npy'));(B/(name+'_result.json')).write_text(json.dumps(r,indent=2))
