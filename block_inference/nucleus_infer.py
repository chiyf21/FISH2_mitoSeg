"""Whole-raw tiled nuclear UNet inference; full-plane probability averaging."""
import argparse,json,time,gc
from pathlib import Path
from common import ROOT,config,gpu,guard,save,signature
from block_raw import BlockRaw
import numpy as np

def run(d):
    import torch
    from nuclear_model import UNet2D
    from stitch_core import starts
    out=guard(Path(d['output'])/'nucleus',d,'nucleus')
    if (out/'completed.json').exists():return
    gpu(d);raw=BlockRaw(d['raw'],d.get('raw_cache_mib',512));target=out/'nucleus.npy';shape=raw.a.shape
    resume=json.loads((out/'progress.json').read_text())['z_done'] if (out/'progress.json').exists() else 0
    mask=np.lib.format.open_memmap(target,mode='r+' if resume else 'w+',dtype=bool,shape=shape)
    weight=d['models']['nucleus'];m=UNet2D(base=32).cuda().eval();ck=torch.load(weight,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck
    window=800;stride=400;windows=[(x,y) for y in starts(shape[1],window,stride) for x in starts(shape[2],window,stride)];started=time.time()
    with torch.inference_mode():
        for z in range(resume,shape[0]):
            prob=np.zeros(shape[1:],np.float32);count=np.zeros(shape[1:],np.uint16)
            for i in range(0,len(windows),8):
                batch=windows[i:i+8];aa=np.stack([raw.slice_crop(z,x,y,window)[0] for x,y in batch]).astype(np.float32)/255
                with torch.autocast('cuda',dtype=torch.bfloat16):pp=m(torch.from_numpy(aa[:,None]).cuda()).float().sigmoid().cpu().numpy()[:,0]
                for (x,y),p in zip(batch,pp):
                    h,w=min(window,shape[1]-y),min(window,shape[2]-x);prob[y:y+h,x:x+w]+=p[:h,:w];count[y:y+h,x:x+w]+=1
            mask[z]=prob/np.maximum(count,1)>.5;mask.flush();save(out/'progress.json',dict(z_done=z+1,total=shape[0],seconds=time.time()-started));print('nucleus',z+1,'/',shape[0],flush=True)
    save(out/'completed.json',dict(model=signature(weight),threshold=.5,window=800,stride=400,batch_size=8,fusion='mean probabilities',shape_zyx=list(shape),raw_source_tag=raw.source_tag,mask=str(target),seconds=time.time()-started,limited_run=False))
    save(out/'black_plane_interpolation.json',list(raw.repair_log.values()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);run(config(p.parse_args().config))
