"""Train mask decoder and temporal memory with differentiable SAM2Base track steps."""
from common import *
import argparse,math,random,numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from source_data import sequence

KEYS=('loss','focal','dice','iou_head','object','mask_iou')
def make_model():
 from sam2.build_sam import build_sam2
 m=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(SAM/'checkpoints/sam2.1_hiera_base_plus.pt'),device='cuda',apply_postprocessing=False,hydra_overrides_extra=['++model.multimask_output_in_sam=false','++model.multimask_output_for_tracking=false'])
 for p in m.parameters():p.requires_grad=False
 prefixes=('sam_mask_decoder.','memory_attention.','memory_encoder.','obj_ptr_proj.','obj_ptr_tpos_proj.')
 for name,p in m.named_parameters():
  if name.startswith(prefixes):p.requires_grad=True
 return m

def load_record(rec):
 images,gt=sequence(rec);side=rec['crop_side'];n=len(images);xx=torch.empty((n,3,1024,1024),dtype=torch.float32);yy=torch.empty((n,1,1024,1024),dtype=torch.float32)
 for j in range(n):
  rgb=np.asarray(Image.fromarray(images[j]).convert('RGB').resize((1024,1024))).copy();xx[j]=torch.from_numpy(rgb).permute(2,0,1).float()/255.
  target=np.asarray(Image.fromarray(gt[j].astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy();yy[j,0]=torch.from_numpy(target)
 xx.sub_(torch.tensor([.485,.456,.406])[:,None,None]).div_(torch.tensor([.229,.224,.225])[:,None,None])
 return xx,yy

def frame_losses(logits,gt,ious,obj):
 logits=logits.float();gt=gt.float();prob=logits.sigmoid();ce=F.binary_cross_entropy_with_logits(logits,gt,reduction='none');pt=prob*gt+(1-prob)*(1-gt);alpha=.25*gt+.75*(1-gt);focal=(ce*(1-pt).square()*alpha).mean()
 dice=1-(2*(prob*gt).sum()+1)/(prob.sum()+gt.sum()+1)
 with torch.no_grad():
  b=logits>0;t=gt>0;inter=(b&t).sum().float();union=(b|t).sum().float();true_iou=torch.where(union>0,inter/union.clamp_min(1),torch.ones_like(union));present=gt.flatten(1).any(1).float().reshape_as(obj)
 iou_loss=F.mse_loss(ious.float().reshape(-1),true_iou.expand_as(ious).reshape(-1));obj_loss=F.binary_cross_entropy_with_logits(obj.float(),present)
 loss=20*focal+dice+iou_loss+obj_loss
 return loss,torch.stack([loss,focal,dice,iou_loss,obj_loss,true_iou]).detach().cpu().numpy()

def detach_tree(x):
 if torch.is_tensor(x):return x.detach()
 if isinstance(x,dict):return {k:detach_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [detach_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(detach_tree(v) for v in x)
 return x

def one_sequence(model,rec,backward=False,scale=1.,chunk_frames=8):
 xx,yy=load_record(rec);n=len(xx);seed=rec['seed_frame'];assert 0<=seed<n;total=np.zeros(len(KEYS));pending=None;chunk_n=0;seed_out=None
 branches=[(False,list(range(seed,n))),(True,list(range(seed-1,-1,-1)))]
 for reverse,order in branches:
  state={'cond_frame_outputs':{} if seed_out is None else {seed:detach_tree(seed_out)},'non_cond_frame_outputs':{}}
  for k,j in enumerate(order):
   image=xx[j:j+1].cuda();target=yy[j:j+1].cuda()
   # Frozen encoder has no parameter gradients; decoder high-resolution projections remain differentiable.
   backbone=model.forward_image(image);_,feat,pos,sizes=model._prepare_backbone_features(backbone)
   point=None
   if j==seed:
    coords=torch.tensor(rec['seed_prompt_xyxy'],dtype=torch.float32,device='cuda').reshape(1,2,2)/rec['crop_side']*1024
    point={'point_coords':coords,'point_labels':torch.tensor([[2,3]],dtype=torch.int32,device='cuda')}
   cur,outs,_,_=model._track_step(j,j==seed,feat,pos,sizes,point,None,state,n,reverse,None)
   low_multi,high_multi,ious,low,high,objptr,obj=outs
   cur.update(pred_masks=low,obj_ptr=objptr)
   model._encode_memory_in_output(feat,sizes,point,True,high,obj,cur)
   loss,vals=frame_losses(high,target,ious,obj);assert torch.isfinite(loss);total+=vals
   if backward:pending=loss*scale/n if pending is None else pending+loss*scale/n
   # Keep only temporal memory outputs, not full-resolution masks or image graphs.
   kept={key:cur[key] for key in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
   state['cond_frame_outputs' if j==seed else 'non_cond_frame_outputs'][j]=kept
   if j==seed:seed_out=kept
   chunk_n+=1
   end=(k==len(order)-1)
   if backward and (chunk_n>=chunk_frames or end):
    pending.backward();pending=None;chunk_n=0;state=detach_tree(state);seed_out=detach_tree(seed_out)
   elif not backward:state=detach_tree(state);seed_out=detach_tree(seed_out)
   # Official tracking accesses <=16 recent object pointers and <=7 memory frames.
   if len(state['non_cond_frame_outputs'])>32:
    oldest=next(iter(state['non_cond_frame_outputs']));del state['non_cond_frame_outputs'][oldest]
   del backbone,feat,pos,cur,outs,low_multi,high_multi,low,high,objptr,obj,loss,image,target
 return total/n,n

def save_ckpt(path,model,opt,epoch,update,best,config):
 state={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'updates':update,'best_val':best,'config':config,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
 temp=path.with_suffix('.partial');torch.save(state,temp);temp.replace(path)

def records(path):
 with path.open() as f:return [json.loads(x) for x in f if x.strip()]

def run(data,out,smoke=False):
 out.mkdir(exist_ok=True);torch.set_num_threads(3);torch.manual_seed(42);np.random.seed(42);random.seed(42)
 m=make_model();parameters=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-4)
 config={'epochs':10,'lr':1e-5,'microbatch_sequences':1,'accumulation':4,'precision':'BF16 autocast, FP32 losses','trainable_prefixes':['sam_mask_decoder','memory_attention','memory_encoder','obj_ptr_proj','obj_ptr_tpos_proj'],'frozen':['image_encoder','sam_prompt_encoder','detector'],'loss':'20*focal(alpha=.25,gamma=2) + dice + IoU-head MSE + object-presence BCE, per-frame mean then per-sequence mean','memory':'prediction-derived memory only; no GT mask/point prompts; full instance Z, bidirectional from detected seed; truncated backprop8 frames, retain32 recent outputs','data':str(data),'limited_run':smoke,'checkpoint':'last+best each epoch, numbered every5','source_checkpoint':str(SAM/'checkpoints/sam2.1_hiera_base_plus.pt')}
 train=records(data/'train.jsonl');val=records(data/'val.jsonl');assert train and val,'No valid training/validation video seeds'
 save(out/'config.json',{**config,'train_samples':len(train),'val_samples':len(val),'train_frames':sum(r['video_z_bounds'][1]-r['video_z_bounds'][0] for r in train),'val_frames':sum(r['video_z_bounds'][1]-r['video_z_bounds'][0] for r in val)})
 epoch0=1;updates=0;best=float('inf')
 if (out/'last.pt').exists() and not smoke:
  c=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert c['config']['lr']==1e-5;m.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);epoch0=c['epoch']+1;updates=c['updates'];best=c['best_val'];torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng']);del c
 for epoch in range(epoch0,(2 if smoke else 11)):
  m.train();m.image_encoder.eval();m.sam_prompt_encoder.eval();opt.zero_grad(set_to_none=True);tt=np.zeros(len(KEYS));window=np.zeros(len(KEYS));wn=0;count=0;start=time.time();order=np.random.default_rng(42+epoch).permutation(len(train))
  if smoke:
   candidates=[i for i,r in enumerate(train) if r['video_z_bounds'][1]-r['video_z_bounds'][0]>=3]
   assert candidates,'Smoke test requires a real multi-frame sequence'
   order=[min(candidates,key=lambda i:train[i]['video_z_bounds'][1]-train[i]['video_z_bounds'][0])]
  for step,idx in enumerate(order):
   group_len=min(4,len(order)-(step//4)*4)
   with torch.autocast('cuda',dtype=torch.bfloat16):vals,n=one_sequence(m,train[int(idx)],True,1/group_len)
   tt+=vals;window+=vals;wn+=1;count+=1
   if (step+1)%4==0 or step+1==len(order):
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
    if smoke:
     groups={prefix:sum(float(p.grad.abs().sum()) for name,p in m.named_parameters() if name.startswith(prefix) and p.grad is not None) for prefix in ('sam_mask_decoder','memory_attention','memory_encoder')}
     assert all(v>0 for v in groups.values()),groups
     assert all(p.grad is None for name,p in m.named_parameters() if name.startswith(('image_encoder','sam_prompt_encoder')))
     save(out/'gradient_check.json',{'trainable_grad_l1':groups,'frozen_encoder_no_grad':True,'video_frames':n,'finite_loss':float(vals[0]),'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'limited_run':True})
    opt.step();opt.zero_grad(set_to_none=True);updates+=1
    if updates%10==0 or step+1==len(order):append(out/'steps.jsonl',{'epoch':epoch,'optimizer_update':updates,'sequence_step':step+1,'sequences_total':len(order),'lr':1e-5,**dict(zip(KEYS,(window/wn).tolist())),'time':time.time()});window[:]=0;wn=0
  m.eval();vv=np.zeros(len(KEYS));vlist=val[:1] if smoke else val
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for rec in vlist:vals,_=one_sequence(m,rec);vv+=vals
  v=vv/len(vlist);t=tt/count;append(out/'metrics.jsonl',{'epoch':epoch,'train':dict(zip(KEYS,t.tolist())),'val':dict(zip(KEYS,v.tolist())),'updates':updates,'seconds':time.time()-start,'limited_run':smoke})
  improved=v[0]<best;best=min(best,float(v[0]))
  if not smoke:
   save_ckpt(out/'last.pt',m,opt,epoch,updates,best,config)
   if improved:save_ckpt(out/'best.pt',m,opt,epoch,updates,best,config)
   if epoch%5==0:save_ckpt(out/f'epoch_{epoch:02d}.pt',m,opt,epoch,updates,best,config)
 if not smoke:save(out/'completed.json',{'epochs':10,'limited_run':False,'updates':updates,'best_val_loss':best})
 else:save(out/'smoke_completed.json',{'completed':True,'limited_run':True,'finite_train_and_val':True,'gradient_check':str(out/'gradient_check.json')})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,default=ROOT/'filtered');a.add_argument('--out',type=Path,default=ROOT/'video_ft10');a.add_argument('--smoke',action='store_true');v=a.parse_args();run(v.data,v.out,v.smoke)
