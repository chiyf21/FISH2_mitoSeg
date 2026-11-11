"""Minimal runtime adapter: repair does not require or load a detector."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
TRAIN=Path(__file__).resolve().parent
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(f'.{os.getpid()}.partial.json');t.write_text(json.dumps(x,indent=2));t.replace(p)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
class Engines:
    @torch.inference_mode()
    def encode_clip(self,geometry,side):
        x,y,z,n=geometry;frames=[self.raw.slice_crop(z+j,x,y,side)[0] for j in range(n)];encoded=[]
        for start in range(0,n,8):
            aa=[]
            for im in frames[start:start+8]:
                rgb=np.asarray(Image.fromarray(im).convert('RGB').resize((1024,1024))).copy()
                aa.append(torch.from_numpy(rgb).permute(2,0,1).float()/255.)
            xx=torch.stack(aa).cuda();xx.sub_(torch.tensor([.485,.456,.406],device='cuda')[:,None,None]).div_(torch.tensor([.229,.224,.225],device='cuda')[:,None,None])
            with torch.autocast('cuda',dtype=torch.bfloat16):features=self.model.forward_image(xx)
            for j in range(len(aa)):encoded.append({k:[a[j:j+1].cpu() for a in features[k]] for k in ('backbone_fpn','vision_pos_enc')})
        return encoded

