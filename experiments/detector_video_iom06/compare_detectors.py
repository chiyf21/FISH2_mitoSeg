from common import *
import numpy as np, torch, torch.distributed as dist
from train_detonly_ddp import DetectorOnly,dataset,evaluate,negative_fp
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(rank);device=torch.device('cuda',rank);torch.set_num_threads(3);dist.init_process_group('nccl')
model=DetectorOnly(mode='det_sam',dim='3d',det_box='slice2d',prompt_pe='rff',prompt_jitter=.1).to(device).eval()
old=dataset('test','old');neg=dataset('test','neg');results=[]
for name,path in [('original30_best',ROOT/'original30_best.pt'),('detonly_best',ROOT/'detector_detonly_ft10/best_det_sam.pt'),('detonly_last',ROOT/'detector_detonly_ft10/last_det_sam.pt')]:
    ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);epoch=ck['epoch'];del ck
    row={'name':name,'checkpoint':str(path),'epoch':epoch,'old_validation':evaluate(model,old,device,rank,world,22),'negative_validation':evaluate(model,neg,device,rank,world,22),'negative_detections':negative_fp(model,neg,device,rank,world,22)}
    results.append(row)
    if rank==0:save(ROOT/'detector_comparison.json',{'definition':'Same FP32 detector-only forward; same complete validation datasets; fixed 200 negative probe; sample-weighted loss; per-rank batch22','results':results});print(json.dumps(row),flush=True)
dist.destroy_process_group()
