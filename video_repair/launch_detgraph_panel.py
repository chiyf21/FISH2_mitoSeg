import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).parent;RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103';marker=RUN/'publisher_process.json'
if marker.exists():
    proc=Path(f'/proc/{json.loads(marker.read_text())["pid"]}')
    if proc.exists() and (proc/'cmdline').read_bytes():raise RuntimeError('Publisher already alive')
with (RUN/'publisher.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'render_detgraph113.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
marker.write_text(json.dumps(dict(pid=p.pid)));print(marker.read_text())
