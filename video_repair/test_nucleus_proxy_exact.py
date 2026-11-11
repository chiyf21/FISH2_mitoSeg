import json,gzip
import numpy as np
from fast_runtime import ROOT
from nucleus_proxy import NuclearProxy
from stitch_core import starts
from repair_three_prompt import shared_geometry
from infer_block import save
run=ROOT/'smoke_tracking_score04_113_20261103';specs=json.loads((run/'cases/instance_797/planned_routes.json').read_text());p=specs[0]['prompt'];geometry,side=shared_geometry([p],10254);x,y=geometry[:2];z=p['z'];mask=NuclearProxy('http://127.0.0.1:18088',ROOT/'io_access.token',np.array([204800,114688,10254])).crop(z,x,y,side)
xs=starts(204800,800,400);ys=starts(114688,800,400);prob=np.zeros((side,side),np.float32);count=np.zeros_like(prob);cache=ROOT/'nucleus_global_cache/3f773d9076927a58ee68bf0a5cc1b729268104c7ccdc7435aeef50a3c7b67496'/f'z{z:05d}'
for yi,y0 in enumerate(ys):
    if y0>=y+side or y0+800<=y:continue
    for xi,x0 in enumerate(xs):
        if x0>=x+side or x0+800<=x:continue
        f=cache/f'window_{yi*len(xs)+xi:07d}.npy'
        if f.exists():pp=np.load(f,mmap_mode='r')
        else:
            with gzip.open(f.with_suffix('.npy.gz'),'rb') as stream:pp=np.load(stream)
        lx=max(x,x0);ly=max(y,y0);hx=min(x+side,x0+800);hy=min(y+side,y0+800);prob[ly-y:hy-y,lx-x:hx-x]+=pp[ly-y0:hy-y0,lx-x0:hx-x0];count[ly-y:hy-y,lx-x:hx-x]+=1
expected=prob/np.maximum(count,1)>.5;assert np.array_equal(mask,expected),'Remote nuclear mask differs from original113 probability cache'
save(ROOT/'NUCLEUS_PROXY_EXACT.json',dict(exact_same_as_113_cached_nucleus=True,seed_z=z,crop_xy=[x,y],side=side,changed_pixels=int(np.count_nonzero(mask!=expected)),limited_run=False));print('PASS:112 preprocessing return mask exactly equals113 original nuclear probability fusion')
