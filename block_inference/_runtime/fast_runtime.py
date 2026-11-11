"""Scheduling/cache optimizations only: exact original8+2 batch shapes and decoder."""
import sys,time,types,threading
from pathlib import Path
from collections import OrderedDict,Counter
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'baseline/core'),str(ROOT/'baseline')]
import numpy as np
import torch
from PIL import Image
from mito_repair import runtime
from policy import apply_policy

class ByteLRU:
    def __init__(self,limit):self.limit=limit;self.items=OrderedDict();self.bytes=0;self.hits=0;self.misses=0
    def get(self,key):
        if key not in self.items:self.misses+=1;return None
        self.hits+=1;self.items.move_to_end(key);return self.items[key][0]
    def put(self,key,value,size):
        if key in self.items:self.bytes-=self.items.pop(key)[1]
        self.items[key]=(value,size);self.bytes+=size
        while self.bytes>self.limit and self.items:
            k,(v,n)=self.items.popitem(last=False);self.bytes-=n

class FastIO:
    def __init__(self,raw,stats):
        self.raw=raw;self.stats=stats;self.pool=ThreadPoolExecutor(max_workers=4);self.lock=threading.RLock();self.pending=OrderedDict()
    def _rgb(self,key):
        x,y,z,side=key;t=time.perf_counter();im=self.raw.slice_crop(z,x,y,side)[0]
        rgb=np.asarray(Image.fromarray(im).convert('RGB').resize((1024,1024))).copy()
        with self.lock:self.stats['raw_resize_cpu_seconds']+=time.perf_counter()-t;self.stats['prepared_frames']+=1
        return torch.from_numpy(rgb).permute(2,0,1).float()/255.
    def prefetch(self,geometry,side):
        x,y,z,n=geometry
        with self.lock:
            for j in range(n):
                key=(x,y,z+j,side)
                if key in self.pending:self.pending.move_to_end(key);self.stats['prepared_frame_hits']+=1
                else:self.pending[key]=self.pool.submit(self._rgb,key)
            # Bound speculative CPU memory. Running futures may complete normally.
            while len(self.pending)>48:self.pending.popitem(last=False)
    def frames(self,geometry,side):
        self.prefetch(geometry,side);x,y,z,n=geometry
        with self.lock:fs=[self.pending[(x,y,z+j,side)] for j in range(n)]
        t=time.perf_counter();out=[f.result() for f in fs];self.stats['input_wait_seconds']+=time.perf_counter()-t
        return out
    def close(self):self.pool.shutdown(wait=True)

class FastEncoder:
    def __init__(self,engine,stats,gpu_gib=2):
        self.engine=engine;self.stats=stats;self.cache=ByteLRU(int(gpu_gib*2**30));self.io=FastIO(engine.raw,stats)
        self.synchronize_for_timing=True;self.reuse_normalization=False;self.mean=None;self.std=None;self.cuda_events=[]
    @torch.inference_mode()
    def encode(self,geometry,side):
        x,y,z,n=geometry;result=[]
        # Batch start, length and position are part of the cache key: do not
        # silently reuse a feature computed under a different CUDA batch shape.
        for start in range(0,n,8):
            count=min(8,n-start);key=(x,y,z+start,count,side)
            cached=self.cache.get(key)
            if cached is not None:self.stats['encoded_frame_cache_hits']+=count;result.extend(cached);continue
            aa=self.io.frames((x,y,z+start,count),side);t=time.perf_counter()
            xx=torch.stack(aa).pin_memory().cuda(non_blocking=True)
            mean=self.mean if self.reuse_normalization else torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
            std=self.std if self.reuse_normalization else torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
            xx.sub_(mean).div_(std)
            if not self.synchronize_for_timing:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):features=self.engine.model.forward_image(xx)
            frames=[{k:[a[j:j+1] for a in features[k]] for k in ('backbone_fpn','vision_pos_enc')} for j in range(count)]
            if self.synchronize_for_timing:torch.cuda.synchronize();self.stats['encoder_seconds']+=time.perf_counter()-t
            else:end.record();self.cuda_events.append((begin,end));self.stats['encoder_enqueue_cpu_seconds']+=time.perf_counter()-t
            self.stats['encoder_batches']+=1;self.stats['encoded_frames']+=count
            size=sum(a.numel()*a.element_size() for k in ('backbone_fpn','vision_pos_enc') for a in features[k])
            self.cache.put(key,frames,size);result.extend(frames)
        return result

class FastClip:
    def __init__(self,encoder,geometry,side):self.encoder=encoder;self.geometry=geometry;self.side=side;self.cache=OrderedDict();self.last=None;self.batches=[]
    def __getitem__(self,j):
        x,y,z,n=self.geometry
        if j not in self.cache:
            reverse=self.last is not None and j<self.last
            lo=max(0,j-9) if reverse else j;hi=j+1 if reverse else min(n,j+10)
            frames=self.encoder.encode((x,y,z+lo,hi-lo),self.side)
            for k,f in enumerate(frames):self.cache[lo+k]=f
            self.batches.append(dict(z_start=z+lo,z_end_exclusive=z+hi,direction='backward' if reverse else 'forward'))
            while len(self.cache)>20:self.cache.popitem(last=False)
            # CPU-only speculation: it cannot alter SAM memory or scoring.
            next_lo=max(0,lo-10) if reverse else hi;next_hi=lo if reverse else min(n,hi+10)
            if next_hi>next_lo:self.encoder.io.prefetch((x,y,z+next_lo,next_hi-next_lo),self.side)
        self.last=j;return self.cache[j]

def optimize(engine,gpu_gib=2):
    stats=Counter();encoder=FastEncoder(engine,stats,gpu_gib)
    engine.encode_clip=types.MethodType(lambda self,geometry,side:FastClip(encoder,geometry,side),engine)
    original=engine.track
    def track(self,*args,**kwargs):
        t=time.perf_counter();result=original(*args,**kwargs);stats['track_wall_seconds']+=time.perf_counter()-t;return result
    engine.track=types.MethodType(track,engine)
    engine.fast_stats=stats;engine.fast_encoder=encoder
    return engine

def create_local(raw,nucleus,weights,base,code,fast=True):
    engine=apply_policy(runtime(raw,nucleus,weights,base,code))
    return optimize(engine) if fast else engine
