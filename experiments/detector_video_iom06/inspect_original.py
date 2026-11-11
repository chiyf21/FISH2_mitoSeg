from common import *
import torch,numpy as np
from models.model_segdet3d import det_cfg_from_state
original=Path('/path/to/JaneliaEM/outputs/train_det_sam_flyem2500_s1500x2_slice2d_dfl_rffjit')
out=[]
for name in ('best_det_sam.pt','last_det_sam.pt','e30_det_sam.pt'):
 p=original/name;c=torch.load(p,map_location='cpu',weights_only=False);s={k.removeprefix('module.'):v for k,v in c['model'].items()}
 out.append({'file':str(p),'epoch':c.get('epoch'),'keys':list(c),'inferred_architecture':det_cfg_from_state(s),'sha256':sha(p),'other':{k:v for k,v in c.items() if k in ('score','best_score','val_loss','mode','dim','prompt_jitter','args','config') and isinstance(v,(dict,list,str,float,int,type(None)))}})
 del c,s
save(ROOT/'original30_audit.json',out);print(json.dumps(out))
p=next(iter(sorted((DET/'data/npz_crops_flyem2500_s1500x2').glob('type1_*.npz'))))
with np.load(p,allow_pickle=True) as z:print('FIRST_META',str(z['meta'])[:1500])
