"""Sequential, checkpoint-aware detector fine-tune -> preview -> filter -> Video fine-tune."""
from common import *
import fcntl,subprocess,shutil,torch
lock=open(ROOT/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def state(stage,status,message='',**extra):save(ROOT/'state.json',{'stage':stage,'status':status,'message':message,'time':stamp(),**extra})
def living(script):
 found=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   args=p.read_bytes().split(b'\0');match=any(Path(x.decode(errors='replace')).name==script for x in args if x)
   if match and (p.parent/'cwd').resolve()==ROOT:found.append(int(p.parent.name))
  except (OSError,PermissionError):pass
 return found
def gpu(need):
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);c=[]
 for line in text.splitlines():
  i,free,util=map(int,line.split(','))
  if i in range(4) and free>=need:c.append((util,i,free))
 return min(c)[1] if c else None
def done(p,epochs=None):
 if not p.exists():return False
 d=json.loads(p.read_text())
 return d.get('epochs')==epochs and d.get('limited_run') is False if epochs is not None else True
def execute(stage,script,args,marker,epochs=None,need=None):
 attempts=0
 while not done(marker,epochs):
  live=living(script)
  if live:state(stage,'running','当前阶段正在运行；通过完成校验后会自动进入下一阶段。',worker_pids=live);time.sleep(30);continue
  if attempts>=3:raise RuntimeError(f'{stage} exited without completion three times; inspect its append-only log')
  if script=='launch_video_fast.py':
   usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
   free={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()}
   if not all(free[g]>=24000 for g in (5,6,7)):
    state(stage,'waiting_resources','等待GPU5/6/7剩余显存，不停止其他任务。');time.sleep(30);continue
  device=gpu(need) if need else None
  if need and device is None:state(stage,'waiting_resources','等待足够的空闲显存，不停止其他任务。');time.sleep(30);continue
  attempts+=1;logpath=ROOT/f'{stage}_controller.log';state(stage,'running','正在启动当前阶段。',gpu=device)
  env={**os.environ}
  if device is not None:env['CUDA_VISIBLE_DEVICES']=str(device)
  with logpath.open('a') as f:
   append(ROOT/'events.jsonl',{'time':stamp(),'stage':stage,'event':'launch','attempt':attempts,'gpu':device,'args':args})
   proc=subprocess.Popen([str(PYTHON),'-u',str(ROOT/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   save(ROOT/f'{stage}_process.json',{'pid':proc.pid,'physical_gpu':device,'time':stamp()});code=proc.wait()
  append(ROOT/'events.jsonl',{'time':stamp(),'stage':stage,'event':'exit','code':code})
  if not done(marker,epochs):time.sleep(60)
 state(stage,'completed')

def main():
 assert json.loads((ROOT/'pipeline_preflight_verified.json').read_text())['passed']
 assert json.loads((ROOT/'detector_detonly_ft10/ddp_preflight.json').read_text())['passed']
 execute('detector_detonly_finetune10','launch_detonly.py',[],ROOT/'detector_detonly_ft10/completed.json',epochs=10)
 complete=json.loads((ROOT/'detector_detonly_ft10/completed.json').read_text());assert complete['epochs']==10 and not complete['limited_run'] and complete['objective']=='detector_only'
 best=ROOT/'detector_detonly_ft10/best_det_sam.pt';c=torch.load(best,map_location='cpu',weights_only=False);assert 1<=c['epoch']<=10 and 'model' in c;epoch=c['epoch'];del c
 frozen=ROOT/'detector_frozen.pt'
 if frozen.exists():assert sha(frozen)==sha(best)
 else:
  temporary=frozen.with_suffix('.partial');shutil.copy2(best,temporary);temporary.replace(frozen)
 save(ROOT/'frozen_detector.json',{'source':str(best),'finetune_epoch':epoch,'sha256':sha(frozen),'weights_frozen':True,'original_pretrained_best_epoch':27,'finetuning_completed_epochs':10})
 execute('detector_preview50','visualize_detector50.py',[],ROOT/'dashboard/detector_samples50/completed.json',need=16000)
 execute('prepare_source','source_data.py',[],ROOT/'source_cache/mito_z_bounds.json')
 execute('filter_iom06','filter_samples.py',[],ROOT/'filtered/completed.json',need=16000)
 filtering=json.loads((ROOT/'filtered/completed.json').read_text());assert filtering['full_run'] and filtering['checkpoint_sha256']==sha(frozen)
 # Re-test the actual final detector's eligible samples before starting10 full Video epochs.
 execute('refilter_complete_gt_iom06','refilter_full_gt.py',[],ROOT/'filtered_fullgt/completed.json')
 filtering=json.loads((ROOT/'filtered_fullgt/completed.json').read_text());assert filtering['full_run'] and filtering['denominator']=='complete_source_gt_slice' and filtering['threshold']==.6
 smoke=ROOT/'video_fast_fullgt_preflight/smoke_completed.json'
 if not smoke.exists():execute('video_fast_fullgt_preflight','launch_video_fast.py',['--smoke'],smoke)
 execute('video_fast_fullgt_finetune10','launch_video_fast.py',[],ROOT/'video_fast_fullgt_ft10/completed.json',epochs=10)
 assert sha(frozen)==json.loads((ROOT/'frozen_detector.json').read_text())['sha256']
 state('complete','completed','Detector10 new epochs and Video10 epochs completed; all data filtering and preview50 finished.')
 report='# Detector → Video pipeline completed\n\nOriginal completed30 best(epoch27) → detector-only10 epochs → frozen best → preview50 → FULL SOURCE GT slice IoM≥0.6 → instance-balanced full-Z SAM2 Video10 epochs, LR1e-5, GPUs5/6/7,4seeds/instance/epoch, global batch4. Old clipped-mask runs are retained as invalid historical comparisons.\n\nFrozen detector: '+str(frozen)+'\n\nVideo best: '+str(ROOT/'video_fast_fullgt_ft10/best.pt')+'\n\nVideo last: '+str(ROOT/'video_fast_fullgt_ft10/last.pt')+'\n\nEligible corrected seed pool: '+json.dumps(filtering)+'\n\nFinal Video metrics: '+json.dumps(json.loads((ROOT/'video_fast_fullgt_ft10/metrics.jsonl').read_text().splitlines()[-1]))+'\n\nLoss curves: '+str(ROOT/'dashboard/loss_curves.png')+'\n'
 (ROOT/'RESULTS.md').write_text(report);(ROOT/'dashboard/RESULTS.md').write_text(report);save(ROOT/'completed.json',{'detector_epochs':10,'video_epochs':10,'limited_run':False,'time':stamp()})

if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  state('needs_attention','error',repr(e),traceback=traceback.format_exc());raise
