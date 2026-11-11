"""Reproduce all113 CPU selections and one real CUDA instance from the standalone CLI."""
import argparse,subprocess,sys,os,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference-experiment',required=True);p.add_argument('--gpu',default='2');a=p.parse_args();root=Path(__file__).parent.resolve();ref=Path(a.reference_experiment).resolve();validation=root/'validation';validation.mkdir(exist_ok=True)
source=ref/'simple113_component500_union_tight_score04_20261103';expected=ref/'simple113_component500_union_tight_cc26_20261103';start=time.time()
def cmd(*args,env=None):subprocess.run([sys.executable,str(root/'mito_repair.py'),*map(str,args)],env=env,check=True)
cmd('finalize','--source-run',source,'--output',validation/'cpu113','--expected-run',expected)
idx=3;d=source/f'cases/instance_{idx:03d}';origin=json.loads((d/'existing_origin.json').read_text());manifest=validation/'instance.json';manifest.write_text(json.dumps([dict(index=idx,mask_path=str(d/'existing_mask.npy'),origin_xyz=origin)]))
stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True);mem={r.split(',')[0].strip():int(r.split(',')[1]) for r in stats.splitlines()};assert mem[a.gpu]<500,mem
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
raw=ref.parent.parent/'flyEM/2500x2500x250/raw.ome.tif'
cmd('run','--raw',raw,'--nucleus',ref/'instance_centered_20261102/nucleus.npy','--manifest',manifest,'--output',validation/'gpu_one','--panel',env=env)
actual=validation/f'gpu_one/cases/instance_{idx:03d}';gold=expected/f'cases/instance_{idx:03d}'
x=np.load(actual/'mask.npy');y=np.load(gold/'mask.npy');xo=json.loads((actual/'mask_origin.json').read_text());yo=json.loads((gold/'mask_origin.json').read_text());assert xo==yo and np.array_equal(x,y),'Standalone GPU inference differs from frozen reference'
manifest_panel=json.loads((validation/'gpu_one/panel/manifest.json').read_text());assert len(manifest_panel)==1
for f in manifest_panel[0]['files']:assert (validation/'gpu_one/panel'/f).stat().st_size>0
result=dict(cpu113_pixel_exact=True,cuda_instance=idx,cuda_pixel_exact=True,actual_cuda_device=a.gpu,panel_images=len(manifest_panel[0]['files']),tests=['component threshold and tight bbox rules','26 connectivity and seed selection','113 cached cases exact output','one uncached standalone SAM inference exact output','optional panel image references'],wall_seconds=time.time()-start,limited_run=False)
(root/'VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
