import json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P/'validation/advanced_20261104_013608/full'
from run_advanced_benchmark import compare,evaluate
import numpy as np
for name,src in [('R_baseline','B0_current'),('R_prompt','S_cache_only'),('R_streams','T_streams')]:
 c=json.loads((B/(src+'.json')).read_text());o=B/name;c['output']=str(o)
 if name=='R_prompt':c['advanced_optimizations']['prefix_cache']=False
 if o.exists():raise ValueError(f'Fresh output required: {o}')
 f=B/(name+'.json');f.write_text(json.dumps(c,indent=2))
 with (B/(name+'.log')).open('w') as log:subprocess.run([sys.executable,'-u',str(P/'infer_block.py'),'--config',str(f)],cwd=P,stdout=log,stderr=subprocess.STDOUT,check=True)
 r=json.loads((o/'completed.json').read_text());r['comparison']=compare(B/'B0_current',o);(B/(name+'_result.json')).write_text(json.dumps(r,indent=2));print(name,r['seconds'],r['comparison']['strict_exact'],flush=True)
