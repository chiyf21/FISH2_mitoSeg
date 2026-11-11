"""Global native data adapter and exact fixed800/stride400 nuclear probability cache."""
import sys,io,json,time,urllib.request,threading,fcntl,hashlib,types,gzip,os
from pathlib import Path
from collections import Counter,OrderedDict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT,optimize
from policy import apply_policy
from mito_repair import configure_sam
from infer_block import digest
from stitch_core import starts
import numpy as np
import torch

class GlobalRaw:
    def __init__(self,url,token_file):
        self.url=url.rstrip('/');self.headers={'Authorization':'Bearer '+Path(token_file).read_text().strip()};info=json.load(urllib.request.urlopen(urllib.request.Request(self.url,headers=self.headers),timeout=30));self.size=np.asarray(info['bounds'][3:]);self.source_tag=hashlib.sha256(info['source'].encode()).hexdigest();self.stats=Counter();self.lock=threading.RLock();self.pending=OrderedDict();self.pool=ThreadPoolExecutor(max_workers=4)
    def _read(self,key):
        z,x,y,side=key;t=time.perf_counter();req=urllib.request.Request(self.url+'/crop',data=json.dumps(dict(z=z,x=x,y=y,side=side)).encode(),headers={**self.headers,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as r:raw=np.load(io.BytesIO(r.read()))
        with self.lock:self.stats['rpc_seconds']+=time.perf_counter()-t;self.stats['rpc_crop_requests']+=1
        assert raw.shape==(side,side) and raw.dtype==np.uint8;return raw
    def prefetch(self,z,x,y,side):
        key=(z,x,y,side)
        with self.lock:
            # Small exact-coordinate2D cache, shared by SAM and nucleus in worker.
            value=self.pending.get(key)
            if value is not None:self.stats['crop_hits']+=1;self.pending.move_to_end(key)
            else:
                value=self.pool.submit(self._read,key);self.pending[key]=value
                while len(self.pending)>96:self.pending.popitem(last=False)
        return value
    def slice_crop(self,z,x,y,side):
        value=self.prefetch(z,x,y,side).result()
        valid=np.zeros_like(value,bool);lx=max(0,-x);ly=max(0,-y);hx=min(side,int(self.size[0])-x);hy=min(side,int(self.size[1])-y)
        if 0<=z<self.size[2] and hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
        return value,valid

class NuclearCache:
    def __init__(self,raw,weight,code,cache):
        sys.path.insert(0,str(Path(code).parent));from train_nuclear_unet2d import UNet2D
        self.raw=raw;self.cache=Path(cache)/digest(weight);self.cache.mkdir(parents=True,exist_ok=True);self.stats=Counter();self.model=UNet2D(base=32).cuda().eval();ck=torch.load(weight,map_location='cpu',weights_only=False);self.model.load_state_dict(ck['model']);self.pool=ThreadPoolExecutor(max_workers=4)
        self.xs=starts(int(raw.size[0]),800,400);self.ys=starts(int(raw.size[1]),800,400)
        self.shape=(int(raw.size[2]),int(raw.size[1]),int(raw.size[0]));self.memory=OrderedDict();self.prefetch_enabled=False;self.compress_cache=os.environ.get('MITO_NUCLEAR_LOSSLESS_ARCHIVE')=='1'
    @torch.inference_mode()
    def probability(self,z,xi,yi):
        index=yi*len(self.xs)+xi;group=index//8;d=self.cache/f'z{z:05d}';d.mkdir(exist_ok=True);file=d/f'window_{index:07d}.npy'
        key=(z,index)
        if key in self.memory:self.stats['nucleus_memory_hits']+=1;self.memory.move_to_end(key);return self.memory[key]
        with (d/f'group_{group:06d}.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            compressed=file.with_suffix('.npy.gz')
            if file.exists() or compressed.exists():self.stats['nucleus_disk_hits']+=1
            else:
                t=time.perf_counter();indices=list(range(group*8,min((group+1)*8,len(self.xs)*len(self.ys))))
                # Preserve the original full-volume row-major groups of8, even
                # when only one tile is requested. No altered conv batch shape.
                futures=[self.pool.submit(self.raw.slice_crop,z,self.xs[k%len(self.xs)],self.ys[k//len(self.xs)],800) for k in indices]
                aa=np.stack([f.result()[0] for f in futures]).astype(np.float32)/255
                with torch.autocast('cuda',dtype=torch.bfloat16):prob=self.model(torch.from_numpy(aa[:,None]).cuda()).float().sigmoid().cpu().numpy()[:,0]
                for k,pp in zip(indices,prob):
                    target=d/(f'window_{k:07d}.npy.gz' if self.compress_cache else f'window_{k:07d}.npy');tmp=target.with_suffix('.partial.gz' if self.compress_cache else '.partial.npy')
                    # Lossless compression: no float quantization or thresholds change.
                    if self.compress_cache:
                        with gzip.open(tmp,'wb',compresslevel=1) as stream:np.save(stream,pp)
                    else:np.save(tmp,pp)
                    tmp.replace(target)
                self.stats['nucleus_compute_seconds']+=time.perf_counter()-t;self.stats['nucleus_batches']+=1
            if file.exists():pp=np.load(file,mmap_mode='r')
            else:
                with gzip.open(compressed,'rb') as stream:pp=np.load(stream)
            self.memory[key]=pp
            while len(self.memory)>24:self.memory.popitem(last=False)
            return pp
    def prefetch_crop(self,z,x,y,side):
        groups=set()
        for yi,y0 in enumerate(self.ys):
            if y0>=y+side or y0+800<=y:continue
            for xi,x0 in enumerate(self.xs):
                if x0<x+side and x0+800>x:groups.add((yi*len(self.xs)+xi)//8)
        for ordinal,group in enumerate(sorted(groups)):
            if ordinal>=8:break #Bound speculative reads to64 tiles.
            for k in range(group*8,min((group+1)*8,len(self.xs)*len(self.ys))):
                f=self.cache/f'z{z:05d}'/f'window_{k:07d}.npy'
                if not f.exists() and not f.with_suffix('.npy.gz').exists():self.raw.prefetch(z,self.xs[k%len(self.xs)],self.ys[k//len(self.xs)],800)
    def crop(self,z,x,y,side):
        t=time.perf_counter();prob=np.zeros((side,side),np.float32);count=np.zeros_like(prob)
        if self.prefetch_enabled:self.prefetch_crop(z,x,y,side)
        xx=[(i,q) for i,q in enumerate(self.xs) if q<x+side and q+800>x];yy=[(i,q) for i,q in enumerate(self.ys) if q<y+side and q+800>y]
        for yi,y0 in yy:
            for xi,x0 in xx:
                p=self.probability(z,xi,yi);lx=max(x,x0,0);ly=max(y,y0,0);hx=min(x+side,x0+800,int(self.raw.size[0]));hy=min(y+side,y0+800,int(self.raw.size[1]))
                if hx>lx and hy>ly:prob[ly-y:hy-y,lx-x:hx-x]+=p[ly-y0:hy-y0,lx-x0:hx-x0];count[ly-y:hy-y,lx-x:hx-x]+=1
        self.stats['nucleus_crop_seconds']+=time.perf_counter()-t
        return prob/np.maximum(count,1)>.5

def create_global(url,token_file,nuclear_weight,nuclear_code,nuclear_cache,fast=True,nucleus_proxy_url=None):
    from repair_three_prompt import MultiPromptEngine
    assets=ROOT/'baseline/assets';configure_sam(assets/'sam2');from sam2.build_sam import build_sam2
    torch.set_num_threads(3)
    class Engine(MultiPromptEngine):
        def __init__(self):
            self.raw=GlobalRaw(url,token_file)
            if nucleus_proxy_url:
                from nucleus_proxy import NuclearProxy
                self.nuc=NuclearProxy(nucleus_proxy_url,token_file,self.raw.size)
            else:self.nuc=NuclearCache(self.raw,nuclear_weight,nuclear_code,nuclear_cache)
            apply_policy(self)
            self.model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(assets/'models/sam2.1_hiera_base_plus.pt'),device='cuda',apply_postprocessing=False,hydra_overrides_extra=['++model.multimask_output_in_sam=false','++model.multimask_output_for_tracking=false'])
            ck=torch.load(assets/'models/video_sam_best.pt',map_location='cpu',weights_only=False);self.model.load_state_dict(ck['model'],strict=True);self.model.eval()
        def nuc_crop(self,z,x,y,side):return self.nuc.crop(z,x,y,side)
    engine=Engine()
    if fast:engine.nuc.prefetch_enabled=True;return optimize(engine)
    engine.nuc.compress_cache=False
    return engine
