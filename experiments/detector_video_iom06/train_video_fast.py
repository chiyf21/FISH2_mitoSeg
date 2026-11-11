"""Instance-balanced full-Z SAM2 training on three GPUs with exact batch4."""
from common import *
import argparse,math,random,contextlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
import torch.distributed as dist
from tqdm import tqdm
import train_video as old

OUT=ROOT/'video_fast_fullgt_ft10'
KEYS=old.KEYS

def grouped_records(split):
    groups=defaultdict(list)
    for rec in old.records(ROOT/f'filtered_fullgt/{split}.jsonl'):
        assert rec.get('full_gt_verified') and rec['iom']>=.6
        groups[rec['source_instance_id']].append(rec)
    return groups

def select_records(groups,epoch,validation=False):
    rng=np.random.default_rng(42000 if validation else 42000+epoch)
    chosen=[]
    for iid in sorted(groups):
        pool=groups[iid]
        ids=rng.choice(len(pool),4,replace=len(pool)<4)
        chosen.extend(pool[int(i)] for i in ids)
    if not validation:rng.shuffle(chosen)
    return chosen

def distribute(records,world):
    shards=[[] for _ in range(world)];loads=[0]*world
    for rec in sorted(records,key=lambda r:r['video_z_bounds'][1]-r['video_z_bounds'][0],reverse=True):
        rank=min(range(world),key=lambda r:loads[r]);shards[rank].append(rec);loads[rank]+=rec['video_z_bounds'][1]-rec['video_z_bounds'][0]
    return shards

def one_sequence(model,rec,prepared,backward,scale=1.,encoder_batch=8):
    xx,yy=prepared;n=len(xx);seed=rec['seed_frame'];total=np.zeros(len(KEYS));seed_out=None
    branches=[(False,list(range(seed,n))),(True,list(range(seed-1,-1,-1)))]
    for reverse,order in branches:
        state={'cond_frame_outputs':{} if seed_out is None else {seed:old.detach_tree(seed_out)},'non_cond_frame_outputs':{}}
        # Eight-frame chunks preserve the original truncated gradient horizon.
        for offset in range(0,len(order),8):
            frame_ids=order[offset:offset+8];encoded=[]
            for k in range(0,len(frame_ids),encoder_batch):
                ids=frame_ids[k:k+encoder_batch]
                with torch.no_grad():batch=model.image_encoder(xx[ids].cuda(non_blocking=True))
                for bi in range(len(ids)):
                    encoded.append({'backbone_fpn':[x[bi:bi+1] for x in batch['backbone_fpn']],'vision_pos_enc':[x[bi:bi+1] for x in batch['vision_pos_enc']]})
            pending=None
            for j,features in zip(frame_ids,encoded):
                # Cache frozen encoder outputs only; these decoder projections
                # MUST remain in autograd and must not be precomputed/detached.
                features['backbone_fpn'][0]=model.sam_mask_decoder.conv_s0(features['backbone_fpn'][0])
                features['backbone_fpn'][1]=model.sam_mask_decoder.conv_s1(features['backbone_fpn'][1])
                _,feat,pos,sizes=model._prepare_backbone_features(features)
                point=None
                if j==seed:
                    coords=torch.tensor(rec['seed_prompt_xyxy'],dtype=torch.float32,device='cuda').reshape(1,2,2)/rec['crop_side']*1024
                    point={'point_coords':coords,'point_labels':torch.tensor([[2,3]],dtype=torch.int32,device='cuda')}
                cur,outs,_,_=model._track_step(j,j==seed,feat,pos,sizes,point,None,state,n,reverse,None)
                _,_,ious,low,high,objptr,obj=outs
                cur.update(pred_masks=low,obj_ptr=objptr)
                model._encode_memory_in_output(feat,sizes,point,True,high,obj,cur)
                loss,vals=old.frame_losses(high,yy[j:j+1].cuda(non_blocking=True),ious,obj)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite sequence loss')
                total+=vals
                if backward:pending=loss*scale/n if pending is None else pending+loss*scale/n
                kept={key:cur[key] for key in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
                state['cond_frame_outputs' if j==seed else 'non_cond_frame_outputs'][j]=kept
                if j==seed:seed_out=kept
                if len(state['non_cond_frame_outputs'])>32:del state['non_cond_frame_outputs'][next(iter(state['non_cond_frame_outputs']))]
            if backward:pending.backward()
            state=old.detach_tree(state);seed_out=old.detach_tree(seed_out)
            del encoded,pending,features,feat,pos,cur,outs,low,high,objptr,obj,loss
    return total/n,n

def prefetch(records):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=None
        for i,rec in enumerate(records):
            if future is None:future=executor.submit(old.load_record,rec)
            prepared=future.result()
            future=executor.submit(old.load_record,records[i+1]) if i+1<len(records) else None
            yield rec,prepared

def reduce_gradients(parameters):
    # Each rank accumulates sum(sequence_loss / global_group_size), so SUM
    # across ranks gives the exact gradient of the global batch4 mean.
    used=torch.tensor([int(p.grad is not None) for p in parameters],device='cuda',dtype=torch.int32)
    dist.all_reduce(used,op=dist.ReduceOp.MAX);used=used.tolist()
    for start in range(0,len(parameters),32):
        pp=parameters[start:start+32]
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in pp])
        dist.all_reduce(flat,op=dist.ReduceOp.SUM);cursor=0
        for index,p in enumerate(pp,start):
            grad=flat[cursor:cursor+p.numel()].view_as(p)
            if not used[index]:p.grad=None
            elif p.grad is None:p.grad=grad.clone()
            else:p.grad.copy_(grad)
            cursor+=p.numel()

def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}

def checkpoint(path,model,opt,epoch,updates,best,config,progress,rank,world):
    states=[None]*world;dist.all_gather_object(states,rng_state())
    if rank==0:
        value={'epoch':epoch,'updates':updates,'model':model.state_dict(),'optimizer':opt.state_dict(),'best_val':best,'config':config,'progress':progress,'rng_by_rank':states}
        temp=Path(str(path)+'.partial');torch.save(value,temp);os.replace(temp,path)
    dist.barrier()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==3
    torch.cuda.set_device(rank);torch.set_num_threads(3);dist.init_process_group('nccl',timeout=__import__('datetime').timedelta(hours=2))
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
    out=ROOT/'video_fast_fullgt_preflight' if a.smoke else OUT;out.mkdir(exist_ok=True)
    verified=json.loads((ROOT/'filtered_fullgt/completed.json').read_text());assert verified['denominator']=='complete_source_gt_slice' and verified['threshold']==.6 and verified['audit_regression_passed']
    train=grouped_records('train');val=grouped_records('val')
    model=old.make_model();parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-4)
    config={'epochs':10,'lr':1e-5,'effective_batch_sequences':4,'physical_gpus':[5,6,7],'world_size':3,'synchronization':'sample-weighted manual SUM allreduce once per optimizer update; global4 videos split across3 GPUs','sampler':'4 eligible seed videos per source_instance_id per epoch; change seeds/views deterministically each epoch; replacement only when fewer than4 seeds','val_sampler':'fixed4 eligible seeds per val source instance, seed42000; fullZ; unchanged original split','train_instances':len(train),'val_instances':len(val),'train_samples_per_epoch':4*len(train),'val_samples':4*len(val),'precision':'BF16 autocast, FP32 losses','trainable_prefixes':['sam_mask_decoder','memory_attention','memory_encoder','obj_ptr_proj','obj_ptr_tpos_proj'],'frozen':['image_encoder','sam_prompt_encoder','detector'],'loss':'20*focal(alpha=.25,gamma=2)+dice+IoU-head MSE+object-presence BCE; frame mean then sequence mean','full_instance_z':True,'seed_prompt':'one actual detection box, no GT inputs','bptt_frames':8,'encoder_batch':8,'feature_cache':'temporary frozen image_encoder outputs per8-frame chunk; trainable conv_s0/conv_s1 computed with autograd','prefetch':'one CPU thread per rank','checkpoint':'first update10 -> first_preview.pt; recovery.pt every100 updates and before validation; last/best every epoch; numbered every5; exact group cursor + RNG resume','limited_run':a.smoke,'source_checkpoint':str(SAM/'checkpoints/sam2.1_hiera_base_plus.pt')}
    config.update(data=str(ROOT/'filtered_fullgt'),iom_threshold=.6,iom_denominator='complete source GT target slice on exact augmented grid; never detector188-clipped mask')
    if rank==0:save(out/'config.json',config)
    epoch0=1;updates=0;best=float('inf');cursor0=0;tt0=np.zeros(len(KEYS));count0=0
    candidates=[] if a.smoke else [p for p in [out/'recovery.pt',out/'last.pt',out/'first_preview.pt'] if p.exists()]
    if candidates:
        # Read metadata then restore the latest complete optimizer boundary.
        chosen=max(candidates,key=lambda p:torch.load(p,map_location='cpu',weights_only=False)['updates'])
        ck=torch.load(chosen,map_location='cpu',weights_only=False);assert ck['config']['effective_batch_sequences']==4
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);updates=ck['updates'];best=ck['best_val'];progress=ck['progress'];epoch0=ck['epoch']+int(progress['epoch_complete'])
        if not progress['epoch_complete']:cursor0=progress['group_cursor'];tt0=np.array(progress['train_total']);count0=progress['train_count']
        state=ck['rng_by_rank'][rank];torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);np.random.set_state(state['numpy']);random.setstate(state['python']);del ck
        if rank==0:append(ROOT/'events.jsonl',{'time':stamp(),'event':'video_fast_resume','checkpoint':str(chosen),'updates':updates,'epoch':epoch0,'group_cursor':cursor0})
    # Explicit initial synchronization: no trainable or frozen parameter can
    # differ between ranks, including parameters initialized outside the ckpt.
    for p in model.parameters():dist.broadcast(p.data,src=0)
    for b in model.buffers():dist.broadcast(b,src=0)
    for epoch in range(epoch0,11):
        selected=select_records(train,epoch);vselected=select_records(val,0,True)
        if a.smoke:
            eligible=[r for r in selected if r['video_z_bounds'][1]-r['video_z_bounds'][0]>=3]
            selected=sorted(eligible,key=lambda r:r['video_z_bounds'][1]-r['video_z_bounds'][0])[:4]
            vselected=sorted(vselected,key=lambda r:r['video_z_bounds'][1]-r['video_z_bounds'][0])[:3]
        groups=[selected[k:k+4] for k in range(0,len(selected),4)]
        start_cursor=cursor0 if epoch==epoch0 else 0
        assignments=[distribute(g,world)[rank] for g in groups]
        tasks=[r for group in assignments[start_cursor:] for r in group];prepared=iter(prefetch(tasks))
        total=tt0.copy() if epoch==epoch0 else np.zeros(len(KEYS));count=count0 if epoch==epoch0 else 0;window=np.zeros(len(KEYS));wn=0;t0=time.time()
        model.train();model.image_encoder.eval();model.sam_prompt_encoder.eval()
        bar=tqdm(range(start_cursor,len(groups)),initial=start_cursor,total=len(groups),desc=f'SAM FAST e{epoch}/10',disable=rank!=0)
        for gi in bar:
            opt.zero_grad(set_to_none=True);local=np.zeros(len(KEYS));frames=0
            for rec in assignments[gi]:
                actual,data=next(prepared);assert actual['sample_id']==rec['sample_id']
                with torch.autocast('cuda',dtype=torch.bfloat16):values,n=one_sequence(model,rec,data,True,1/len(groups[gi]))
                local+=values;frames+=n;del data
            # Confirm all frozen parameters stay outside autograd.
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            reduce_gradients(parameters)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
            metrics=torch.tensor([*local,frames],dtype=torch.float64,device='cuda');dist.all_reduce(metrics)
            summed=metrics[:-1].cpu().numpy();total+=summed;count+=len(groups[gi]);window+=summed;wn+=len(groups[gi])
            if a.smoke:
                grads={prefix:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None) for prefix in ('sam_mask_decoder','memory_attention','memory_encoder','sam_mask_decoder.conv_s0','sam_mask_decoder.conv_s1')}
                assert all(v>0 for v in grads.values()),grads
                checks=[None]*world;dist.all_gather_object(checks,{'rank':rank,'grad_l1':grads,'frozen_no_grad':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
                if rank==0:save(out/'gradient_check.json',{'passed':True,'checks':checks,'limited_run':True})
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            progress={'epoch_complete':False,'group_cursor':gi+1,'train_total':total.tolist(),'train_count':count}
            if rank==0 and (updates%10==0 or gi==len(groups)-1 or a.smoke):
                record={'epoch':epoch,'optimizer_update':updates,'sequence_step':count,'sequences_total':len(selected),'lr':1e-5,**dict(zip(KEYS,(window/wn).tolist())),'time':time.time(),'physical_gpus':[5,6,7],'global_batch':len(groups[gi])}
                append(out/'steps.jsonl',record);bar.set_postfix(loss=f'{record["loss"]:.3f}',update=updates);window[:]=0;wn=0
                save(ROOT/'state.json',{'stage':'video_fast_fullgt_finetune10','status':'running','message':f'完整GT IoM≥0.6修正版，第{epoch}/10轮，已处理{count}/{len(selected)}条视频，GPU5/6/7。','time':stamp()})
            if updates==10 and not a.smoke:checkpoint(out/'first_preview.pt',model,opt,epoch,updates,best,config,progress,rank,world)
            if (updates%100==0 or gi==len(groups)-1) and not a.smoke:checkpoint(out/'recovery.pt',model,opt,epoch,updates,best,config,progress,rank,world)
        model.eval();vv=np.zeros(len(KEYS));vc=0
        for rec,data in prefetch(distribute(vselected,world)[rank]):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):values,_=one_sequence(model,rec,data,False)
            vv+=values;vc+=1;del data
        validation=torch.tensor([*vv,vc],dtype=torch.float64,device='cuda');dist.all_reduce(validation);v=validation[:-1].cpu().numpy()/float(validation[-1]);t=total/count
        if rank==0:append(out/'metrics.jsonl',{'epoch':epoch,'train':dict(zip(KEYS,t.tolist())),'val':dict(zip(KEYS,v.tolist())),'updates':updates,'seconds':time.time()-t0,'limited_run':a.smoke,'train_sequences':count,'val_sequences':int(validation[-1])})
        improved=v[0]<best;best=min(best,float(v[0]));progress={'epoch_complete':True,'group_cursor':len(groups),'train_total':total.tolist(),'train_count':count}
        if not a.smoke:
            checkpoint(out/'last.pt',model,opt,epoch,updates,best,config,progress,rank,world)
            if improved:checkpoint(out/'best.pt',model,opt,epoch,updates,best,config,progress,rank,world)
            if epoch%5==0:checkpoint(out/f'epoch_{epoch:02d}.pt',model,opt,epoch,updates,best,config,progress,rank,world)
        if a.smoke:
            if rank==0:save(out/'smoke_completed.json',{'passed':True,'limited_run':True,'finite_train_and_val':True,'val_loss':float(v[0])})
            dist.destroy_process_group();return
        cursor0=0;tt0=np.zeros(len(KEYS));count0=0
    if rank==0:save(out/'completed.json',{'epochs':10,'limited_run':False,'updates':updates,'best_val_loss':best,'sampling':'4 seeds per instance per epoch'})
    dist.destroy_process_group()

if __name__=='__main__':main()
