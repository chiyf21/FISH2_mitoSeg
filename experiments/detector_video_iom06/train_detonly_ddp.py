"""Three-GPU detector-only fine-tuning; all old data + existing hard negatives."""
from common import *
import argparse, contextlib, random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
sys.path.insert(0, str(DET))
from prepare_hardneg import POS, NEG
from data.dataset_npz import NPZCropDataset, RareAwareRatioBatchSampler, compute_class_presence, npz_collate
from models.model_segdet3d import OrganelleSegDet3D
from models.box_ops import slice_boxes_from_masks
from models.losses import yolo_loss
sys.path.insert(0, str(CODE/'experiments/flyem2500_s1500x2_det_sam_dfl_jitter'))
from train_npz import _slice_det

OUT = ROOT/'detector_detonly_ft10'
KEYS = ['loss', 'det', 'cls', 'giou', 'dfl']

class DetectorOnly(OrganelleSegDet3D):
    def forward(self, images):
        features = list(self.neck(self.backbone(images)))
        return [head(features) for head in self.det_heads]

def dataset(split, source='both'):
    roots = [DET/'data'/POS.name, NEG] if source=='both' else [DET/'data'/POS.name] if source=='old' else [NEG]
    return NPZCropDataset(roots, split=split, dim='3d', scales=('s1500',))

def loss_fn(model, batch, device):
    images, masks, labels, _, _, _, _, sup, neg = batch
    predictions = model(images.to(device, non_blocking=True))
    raw = model.module if hasattr(model, 'module') else model
    terms = []
    for i, cid in enumerate(raw.instance_ids):
        eligible = [b for b in range(len(images)) if bool(sup[b,cid-1]) or bool(neg[b,cid-1])]
        if not eligible:
            continue
        boxes, targets = [], []
        for b in eligible:
            keep = labels[b].long()==cid
            bb, _, _ = slice_boxes_from_masks(masks[b][keep].to(device), labels[b][keep].to(device))
            boxes.append(bb)
            targets.append(torch.ones(len(bb), dtype=torch.long, device=device))
        p = _slice_det(predictions[i], torch.tensor(eligible, device=device), len(images))
        terms.append(yolo_loss(p, boxes, targets, images.shape[-1]))
    assert terms, 'No supervised detection class'
    total = torch.stack([t['loss'] for t in terms]).mean()
    values = torch.stack([total.detach(), total.detach(), *[torch.stack([t[k] for t in terms]).mean() for k in KEYS[2:]]])
    return total, values

def plans(sampler, rank, world, micro):
    # Preserve the historical batch16 sampler sequence and group four batches
    # into one update: exactly64 samples, except the actual short final group.
    original = list(iter(sampler))
    flat, groups = [], []
    for start in range(0,len(original),4):
        ids = sum(original[start:start+4], [])
        shards = [list(x) for x in np.array_split(ids, world)]
        chunks = max((len(x)+micro-1)//micro for x in shards)
        assert all(len(x) >= chunks for x in shards)
        # array_split gives equally many nonempty forwards on all ranks.
        parts = [list(map(int,x)) for x in np.array_split(shards[rank], chunks)]
        flat.extend(parts)
        groups.append((len(ids), [len(x) for x in parts]))
    return flat, groups

@torch.no_grad()
def evaluate(model, ds, device, rank, world, micro):
    model.eval()
    loader=DataLoader(Subset(ds,list(range(rank,len(ds),world))),batch_size=micro,num_workers=2,collate_fn=npz_collate,pin_memory=True)
    totals=torch.zeros(len(KEYS)+1, device=device, dtype=torch.float64)
    for batch in loader:
        loss, vals=loss_fn(model,batch,device)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite validation loss')
        n=len(batch[0]);totals[:-1]+=vals.double()*n;totals[-1]+=n
    dist.all_reduce(totals)
    return {**{f'val_{k}':float(v/totals[-1]) for k,v in zip(KEYS,totals[:-1])},'samples':int(totals[-1])}

@torch.no_grad()
def negative_fp(model, ds, device, rank, world, micro):
    ids=np.linspace(0,len(ds)-1,min(200,len(ds)),dtype=int).tolist()[rank::world]
    stats=torch.zeros(3,device=device,dtype=torch.float64)
    for k in range(0,len(ids),micro):
        images=torch.stack([ds[i]['image'] for i in ids[k:k+micro]]).to(device)
        counts=np.zeros(len(images),int)
        for prediction in model(images):
            _,probs=model._top_boxes_binary(prediction,topk=12)
            counts+=np.array([int((p[:,0]>.3).sum()) for p in probs])
        stats+=torch.tensor([len(images),int((counts>0).sum()),int(counts.sum())],device=device)
    dist.all_reduce(stats)
    return {'samples':int(stats[0]),'fraction_with_any_mito_or_lyso_detection':float(stats[1]/stats[0]),'mean_detection_count':float(stats[2]/stats[0]),'det_score_threshold':.3}

def atomic_checkpoint(path, value):
    tmp=Path(str(path)+'.partial');torch.save(value,tmp);os.replace(tmp,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--micro',type=int,default=22);a=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==3
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(3)
    dist.init_process_group('nccl')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    OUT.mkdir(exist_ok=True)
    model=DetectorOnly(mode='det_sam',dim='3d',det_box='slice2d',prompt_pe='rff',prompt_jitter=.1).to(device)
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('backbone.','neck.','det_heads.')))
    warm=torch.load(ROOT/'original30_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(warm['model'],strict=True);del warm
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    start=1;updates=seen=0;best=float('inf')
    if not a.smoke and (OUT/'last_det_sam.pt').exists():
        ck=torch.load(OUT/'last_det_sam.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;updates=ck['optimizer_updates'];seen=ck['samples_seen'];best=ck['best_positive_val']
        rng=ck['rng_by_rank'][rank];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python']);del ck
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    train=dataset('train');old=dataset('test','old');neg=dataset('test','neg')
    sampler=RareAwareRatioBatchSampler(train,batch_size=16,epochs=10,presence=compute_class_presence(train,num_classes=5),rare_thr=.1,seed=42)
    config={'epochs':10,'lr':1e-4,'effective_batch':64,'micro_batch_max_per_gpu':a.micro,'world_size':3,'physical_gpus':[5,6,7],'precision':'FP32','train_samples':len(train),'old_validation_samples':len(old),'negative_validation_samples':len(neg),'initialization':'original completed30 best epoch27; fresh optimizer','objective':'ONLY historical detection classification + box + DFL loss; no segmentation, prompt or ER computation','trainable_modules':['backbone','neck','det_heads'],'frozen_modules':['seg_head','prompt_decoders','er_head'],'checkpoint':'best by original complete validation DETECTION loss; last each epoch; numbered every5','sampling':'same batch16 rare/type ratio sampler; four global batches/update; rank partitions22/21/21; final group32','ddp_loss_weight':'local sample count * world_size / actual global update size','note':'RFF and jitter belong to inactive prompt branch and are not used in detector-only training'}
    if rank==0:
        save(OUT/'config.json',config)
        save(ROOT/'detector_detonly_process.json',{'pid':os.getpid(),'physical_gpus':[5,6,7],'time':stamp(),'config':config})
    for epoch in range(start,11):
        model.train();sampler.set_epoch(epoch-1)
        batches,groups=plans(sampler,rank,world,a.micro)
        loader=iter(DataLoader(train,batch_sampler=batches,num_workers=2,pin_memory=True,collate_fn=npz_collate))
        window=torch.zeros(len(KEYS),device=device);wn=0;totals=torch.zeros(len(KEYS),device=device);nupdates=0;t0=time.time();micro_step=0
        progress=tqdm(groups,desc=f'DET ONLY e{epoch}/10',disable=rank!=0)
        for gi,(global_n,counts) in enumerate(progress):
            opt.zero_grad(set_to_none=True);updatevals=torch.zeros(len(KEYS),device=device)
            for ci,n in enumerate(counts):
                batch=next(loader)
                with (ddp.no_sync() if ci<len(counts)-1 else contextlib.nullcontext()):
                    loss,values=loss_fn(ddp,batch,device)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    (loss*(n*world/global_n)).backward()
                updatevals+=values*n/global_n;micro_step+=1
            # The segmentation branches were never called, and must have no grads.
            assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith(('backbone.','neck.','det_heads.')))
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            dist.all_reduce(updatevals)
            if a.smoke:
                grad={prefix:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None) for prefix in ('backbone.','neck.','det_heads.')}
                assert all(v>0 for v in grad.values())
                peak=torch.cuda.max_memory_allocated()/1024**2
                checks=[None]*world;dist.all_gather_object(checks,{'rank':rank,'grad_l1':grad,'peak_allocated_mib':peak,'finite_loss':float(updatevals[0]),'inactive_branches_no_grad':True,'global_samples':global_n})
                if rank==0:save(OUT/'ddp_preflight.json',{'passed':True,'limited_run':True,'checks':checks,'config':config})
                dist.destroy_process_group();return
            opt.step();updates+=1;seen+=global_n;window+=updatevals;wn+=1;totals+=updatevals;nupdates+=1
            if rank==0 and (updates%10==0 or gi==len(groups)-1):
                rec={'epoch':epoch,'micro_step':micro_step,'micro_steps_in_epoch':len(batches),'optimizer_update':updates,'samples_seen':seen,'last_update_samples':global_n,'lr':1e-4,**dict(zip(KEYS,(window/wn).tolist())),'time':time.time(),'objective':'detector_only','physical_gpus':[5,6,7]}
                append(OUT/'steps.jsonl',rec);progress.set_postfix(loss=f'{rec["loss"]:.4f}',updates=updates);window.zero_();wn=0
        # All ranks use rank0 BN buffers for reproducible saved-model validation.
        for buffer in model.buffers():dist.broadcast(buffer,src=0)
        positive=evaluate(model,old,device,rank,world,a.micro);negative=evaluate(model,neg,device,rank,world,a.micro);fp=negative_fp(model,neg,device,rank,world,a.micro)
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()};rngs=[None]*world;dist.all_gather_object(rngs,rng)
        if rank==0:
            row={'epoch':epoch,'train':dict(zip(KEYS,(totals/nupdates).tolist())),'positive_validation':positive,'negative_validation':negative,'negative_detections':fp,'optimizer_updates':updates,'samples_seen':seen,'epoch_seconds':time.time()-t0,'objective':'detector_only'}
            append(OUT/'metrics.jsonl',row);print(json.dumps(row),flush=True)
            improved=positive['val_loss']<best;best=min(best,positive['val_loss'])
            ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'mode':'det_sam','dim':'3d','optimizer_updates':updates,'samples_seen':seen,'best_positive_val':best,'config':config,'metrics':row,'rng_by_rank':rngs}
            atomic_checkpoint(OUT/'last_det_sam.pt',ck)
            if improved:atomic_checkpoint(OUT/'best_det_sam.pt',ck)
            if epoch%5==0:atomic_checkpoint(OUT/f'e{epoch}_det_sam.pt',ck)
        dist.barrier()
    if rank==0:save(OUT/'completed.json',{'epochs':10,'limited_run':False,'optimizer_updates':updates,'samples_seen':seen,'best_positive_val_loss':best,'objective':'detector_only'})
    dist.destroy_process_group()

if __name__=='__main__':main()
