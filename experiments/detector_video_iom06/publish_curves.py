from common import *
import csv,fcntl,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT.mkdir(exist_ok=True);OUT=ROOT/'dashboard';OUT.mkdir(exist_ok=True)
def rows(p):
 result=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:result.append(json.loads(line))
   except json.JSONDecodeError:continue
 return result
def curve(ax,data,x,key,label):
 pts=[]
 for row in data:
  value=row
  for part in key.split('.'):value=value.get(part) if isinstance(value,dict) else None
  if isinstance(value,(float,int)) and x in row:pts.append((row[x],value))
 if pts:ax.plot([p[0] for p in pts],[p[1] for p in pts],label=label,lw=1.5)
def publish():
 old=rows(Path('/path/to/JaneliaEM/outputs/train_det_sam_flyem2500_s1500x2_slice2d_dfl_rffjit/train_log_det_sam.jsonl'))
 det=rows(ROOT/'detector_detonly_ft10/metrics.jsonl');steps=rows(ROOT/'detector_detonly_ft10/steps.jsonl');video=rows(ROOT/'video_fast_fullgt_ft10/metrics.jsonl');vs=rows(ROOT/'video_fast_fullgt_ft10/steps.jsonl')
 state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else {'stage':'detector_finetune10','status':'running'}
 payload={'updated':stamp(),'state':state,'original30':old,'detector10_epochs':det,'detector10_steps':steps,'video10_epochs':video,'video10_steps':vs,'last_detector_step':steps[-1] if steps else None}
 save(OUT/'curves.json',payload)
 fig,ax=plt.subplots(2,3,figsize=(18,10));a=ax.flat
 for k,label in [('loss','train'),('val_loss','original full validation')]:curve(a[0],old,'epoch',k,label)
 a[0].set_title('Original DFL + RFF + jitter 0.1: completed30')
 for k,label in [('train.loss','train (old full + hardneg)'),('positive_validation.val_loss','old full validation'),('negative_validation.val_loss','hardneg validation')]:curve(a[1],det,'epoch',k,label)
 a[1].set_title('Detector ONLY:10 new fine-tuning epochs | GPUs5,6,7')
 for k in ('loss','cls','giou','dfl'):curve(a[2],steps,'optimizer_update',k,k)
 a[2].set_title('Detector: optimizer-update loss (every10)')
 for k,label in [('train.loss','train'),('val.loss','val')]:curve(a[3],video,'epoch',k,label)
 a[3].set_title('SAM2 Video FAST:10 epochs, LR1e-5, GPUs5/6/7')
 for k in ('loss','dice','iou_head','object'):curve(a[4],vs,'optimizer_update',k,k)
 a[4].set_title('SAM2 Video: optimizer-update loss')
 for k,label in [('train.mask_iou','train'),('val.mask_iou','val')]:curve(a[5],video,'epoch',k,label)
 a[5].set_title('SAM2 Video: mask IoU (eligible samples)')
 for i,aa in enumerate(a):
  aa.grid(alpha=.25);aa.set_xlabel('optimizer update' if i in (2,4) else 'epoch');aa.set_ylabel('IoU' if i==5 else 'loss')
  if aa.lines:aa.legend(fontsize=8)
  else:aa.text(.5,.5,'Waiting for this stage',ha='center',va='center',transform=aa.transAxes,color='gray')
 fig.suptitle('Frozen detector -> per-layer IoM>=0.6 -> Video SAM2 | '+stamp(),fontsize=13);fig.tight_layout();tmp=OUT/'loss_curves.partial.png';fig.savefig(tmp,dpi=130);tmp.replace(OUT/'loss_curves.png');plt.close(fig)
 for name,rr in [('original30',old),('detector10_steps',steps),('detector10_epochs',det),('video10_steps',vs),('video10_epochs',video)]:
  if rr:
   with (OUT/f'{name}.jsonl').open('w') as f:
    for r in rr:f.write(json.dumps(r)+'\n')
 return payload

if __name__=='__main__':
 lock=open(ROOT/'publisher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:publish()
  except Exception as e:append(ROOT/'publisher_errors.jsonl',{'time':stamp(),'error':repr(e)})
  time.sleep(20)
