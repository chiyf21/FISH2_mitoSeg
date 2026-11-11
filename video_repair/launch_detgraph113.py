import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).parent;RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103'
if (RUN/'launched.json').exists():raise RuntimeError('Run already launched; inspect existing controller')
with (RUN/'controller.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'detgraph_controller.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
print(json.dumps(dict(controller_pid=p.pid,run=str(RUN))))
