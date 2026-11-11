"""Execution-only controls; unchanged model arithmetic, batch shapes and rules."""
import math
from collections import defaultdict
import numpy as np

def install(engine,settings):
    import torch
    opts={} if settings is True else dict(settings or {})
    engine.keep_memory_on_gpu=opts.get('gpu_temporal_memory',True)
    engine.cpu_prefetch_steps=int(opts.get('cpu_prefetch_steps',1))
    encoder=engine.fast_encoder
    encoder.synchronize_for_timing=not opts.get('defer_encoder_timing_sync',True)
    encoder.reuse_normalization=True
    encoder.mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
    encoder.std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
    encoder.cache.limit=int(float(opts.get('encoder_cache_gib',2))*2**30)
    return dict(gpu_temporal_memory=engine.keep_memory_on_gpu,cpu_prefetch_steps=engine.cpu_prefetch_steps,defer_encoder_timing_sync=not encoder.synchronize_for_timing,encoder_cache_gib=encoder.cache.limit/2**30,spatial_mask_index=True,arithmetic_batch_shapes_unchanged=True)

def overlap(mask,origin,box,area):
    x,y=origin;x0,y0,x1,y1=[int(math.ceil(float(v)-.5)) for v in box];lx,ly=max(x0,x),max(y0,y);hx,hy=min(x1,x+mask.shape[1]),min(y1,y+mask.shape[0]);inter=int(mask[ly-y:hy-y,lx-x:hx-x].sum()) if hx>lx and hy>ly else 0
    box_area=max(0,x1-x0)*max(0,y1-y0);iom=inter/area if area else 0.;opm=inter/box_area if box_area else 0.
    return dict(iom=iom,opm=opm,overlap_max=max(iom,opm),intersection=inter,mask_area=area,box_area=box_area)

class SnapshotIndex:
    """Conservative spatial grid; restores original order for exact tie breaks."""
    def __init__(self,snapshot,cell=128):
        self.snapshot=snapshot;self.cell=cell;self.bins=defaultdict(list);self.areas=[]
        for i,(_,mask,f) in enumerate(snapshot):
            area=int(mask.sum());self.areas.append(area)
            if not area:continue
            x,y=f['origin_xy'];h,w=mask.shape
            for gx in range(x//cell,(x+w-1)//cell+1):
                for gy in range(y//cell,(y+h-1)//cell+1):self.bins[gx,gy].append(i)
    def query(self,box):
        x0,y0,x1,y1=[int(math.ceil(float(v)-.5)) for v in box];cell=self.cell
        if x1<=x0 or y1<=y0:return []
        if ((x1-1)//cell-x0//cell+1)*((y1-1)//cell-y0//cell+1)>max(16,len(self.bins)*2):ids=range(len(self.snapshot))
        else:
            ids=set()
            for gx in range(x0//cell,(x1-1)//cell+1):
                for gy in range(y0//cell,(y1-1)//cell+1):ids.update(self.bins.get((gx,gy),[]))
            ids=sorted(ids)
        return [(i,self.snapshot[i]) for i in ids if self.areas[i]]

def stats(engine):
    import torch
    encoder=engine.fast_encoder
    if encoder.cuda_events:
        torch.cuda.synchronize();gpu_seconds=sum(begin.elapsed_time(end)/1000 for begin,end in encoder.cuda_events)
    else:gpu_seconds=None
    return dict(counters=dict(engine.fast_stats),encoder_cuda_seconds=gpu_seconds,feature_cache_hits=encoder.cache.hits,feature_cache_misses=encoder.cache.misses,feature_cache_bytes=encoder.cache.bytes,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
