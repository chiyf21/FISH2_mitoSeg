"""Portable public CLI utilities. No historical dataset is implicitly opened."""
import sys,json,hashlib,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'_runtime'))
import fast_runtime
from infer_block import save,digest
import numpy as np

def config(path):
    p=Path(path).resolve();d=json.loads(p.read_text());base=p.parent
    for k in ['raw','output','nucleus','detections']:
        if d.get(k):d[k]=str((base/Path(d[k])).resolve())
    for k,v in list(d.get('models',{}).items()):d['models'][k]=str((ROOT/Path(v)).resolve())
    return d

def identity(d,stage):
    raw=Path(d['raw']);st=raw.stat();models={}
    for k,v in d.get('models',{}).items():
        p=Path(v)
        if p.is_file():s=p.stat();models[k]=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    external={}
    for k in ['nucleus','detections']:
        if d.get(k):p=Path(d[k]);s=p.stat();external[k]=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    return dict(stage=stage,raw=dict(path=str(raw),size=st.st_size,mtime_ns=st.st_mtime_ns),models=models,external=external,config=d)

def guard(out,d,stage):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);v=identity(d,stage);p=out/'identity.json'
    if p.exists() and json.loads(p.read_text())!=v:raise ValueError(f'Output belongs to different inputs/config: {out}; choose a new output')
    save(p,v);return out

def roi_for(raw,d):
    roi=list(d.get('bbox_xyz') or [0,0,0,*map(int,raw.size)])
    if len(roi)!=6 or any(type(v)!=int for v in roi):raise ValueError('bbox_xyz must be6 native XYZ integers [x0,y0,z0,x1,y1,z1]')
    if any(roi[i]<0 or roi[i]>=roi[i+3] or roi[i+3]>raw.size[i] for i in range(3)):raise ValueError('ROI outside raw dimensions')
    return roi

def gpu(d):
    os.environ['CUDA_VISIBLE_DEVICES']=str(d.get('gpu',0));os.environ.setdefault('OMP_NUM_THREADS','4')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch GPU is required')
    torch.set_num_threads(int(d.get('cpu_threads',4)))
    torch.cuda.set_per_process_memory_fraction(float(d.get('gpu_memory_fraction',.4)))

def signature(path,hash_content=True):
    p=Path(path);s=p.stat();v=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    if hash_content:v['sha256']=digest(p)
    return v
