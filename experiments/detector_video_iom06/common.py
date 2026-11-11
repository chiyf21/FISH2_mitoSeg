import os,sys,json,hashlib,time
from pathlib import Path
BASE=Path('/path/to/project_data')
DET=BASE/'JaneliaEM_experiments/detsam_dfljitter_v1_hardneg_z000_079_20261021'
ROOT=BASE/'JaneliaEM_experiments/detector_video_iom06_20261031'
SAM=BASE/'EM/point-MAE/runs/detsam_sam2base_cohort5k_20261031'
PYTHON=SAM/'venv/bin/python'
CODE=DET/'code/organelle_detect'
sys.path.insert(0,str(CODE))
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.partial');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def append(p,data):
 with Path(p).open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(2**20),b''):h.update(a)
 return h.hexdigest()
def stamp():
 from datetime import datetime,timezone,timedelta
 return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
