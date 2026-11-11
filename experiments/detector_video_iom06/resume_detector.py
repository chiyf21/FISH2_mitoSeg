"""Resume the unchanged historical detector, with an exclusive process lock."""
from common import *
import fcntl,shutil,torch
ROOT.mkdir(parents=True,exist_ok=True)
lock=open(ROOT/'detector_resume.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
records=[]
for name in ('last_det_sam.pt','e5_det_sam.pt','best_det_sam.pt'):
 p=DET/'training'/name;c=torch.load(p,map_location='cpu',weights_only=False)
 records.append({'path':str(p),'epoch':c['epoch'],'optimizer_updates':c['optimizer_updates'],'samples_seen':c['samples_seen'],'sha256':sha(p),'has_optimizer':bool(c['optimizer']),'rng_states':all(k in c for k in ('torch_rng','cuda_rng','numpy_rng','python_rng'))})
 if c['epoch']==5:
  target=ROOT/'epoch5_backup'/name;target.parent.mkdir(exist_ok=True)
  if not target.exists():shutil.copy2(p,target)
 del c
assert all(x['has_optimizer'] and x['rng_states'] for x in records)
save(ROOT/'resume_checkpoints.json',records)
save(ROOT/'detector_process.json',{'pid':os.getpid(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':stamp(),'entry':str(DET/'train_hardneg.py'),'resume':str(DET/'training/last_det_sam.pt'),'target_epochs':30})
sys.path.insert(0,str(DET))
import train_hardneg
train_hardneg.train()
