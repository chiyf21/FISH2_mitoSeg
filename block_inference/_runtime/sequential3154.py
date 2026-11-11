"""Causal box seeding with independent, adaptively moved SAM2 memories.

Only seeds whose centres lie in the selected ROI are admitted. Crops may read
real source context beyond the ROI. Published historical masks are immutable.
On a crop change replay the same seed through the current Z in the new crop.
"""
import json, math, time, os, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fast_runtime import ROOT, create_local
from infer_block import save
from sequential_rules import mask_box_iom

REF=ROOT.parent/'sam_video_hardcases_20261102'
TRAIN=ROOT.parent/'detector_video_iom06_20261031'
RUN=ROOT/'sequential3154_iom08_score05_20261103'
GALLERY=TRAIN/'dashboard/sam_video_hardcases_20261102/sequential3154_iom08_score05'
ROI=[703,0,8,1215,512,52]
DETECTIONS_PATH=ROOT/'detgraph113_iou05_iom08_score05_20261103/detections.json'
EXPERIMENT={}
SCORE_THRESHOLD=.5

def compact(mask, xy):
    yy,xx=np.nonzero(mask)
    if not len(xx):return np.zeros((1,1),bool),list(xy)
    return mask[yy.min():yy.max()+1,xx.min():xx.max()+1].copy(),[xy[0]+int(xx.min()),xy[1]+int(yy.min())]

def bounds(mask,xy):
    yy,xx=np.nonzero(mask)
    return None if not len(xx) else [xy[0]+int(xx.min()),xy[1]+int(yy.min()),xy[0]+int(xx.max())+1,xy[1]+int(yy.max())+1]

def union_bounds(a,b):
    if b is None:return a
    return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]

def relocate(crop, required):
    x,y,s=crop
    ns=int(math.ceil(max(s,required[2]-required[0]+202,required[3]-required[1]+202)/32)*32)
    # Preserve the current centre where possible; move the minimum distance.
    nx=min(max(x,math.ceil(required[2]+101-ns)),math.floor(required[0]-101))
    ny=min(max(y,math.ceil(required[3]+101-ns)),math.floor(required[1]-101))
    return [nx,ny,ns]

class Tracker:
    def __init__(self,engine,seed,tid,reverse=False,score_threshold=None):
        self.e=engine;self.seed=seed;self.id=tid;self.state={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}
        self.reverse=reverse;self.score_threshold=SCORE_THRESHOLD if score_threshold is None else score_threshold
        b=seed['box_xyxy'];s=int(math.ceil(max(100,3*max(b[2]-b[0],b[3]-b[1]))))
        self.crop=[math.floor((b[0]+b[2]-s)/2),math.floor((b[1]+b[3]-s)/2),s]
        self.required=list(b);self.history=[];self.frames=[];self.active=True;self.stop=None

    @torch.inference_mode()
    def decode(self,z,state):
        x,y,s=self.crop;m=self.e.model;j=z-ROI[2];cond=z==self.seed['z']
        enc=self.e.fast_encoder.encode((x,y,z,1),s)[0]
        steps=int(getattr(self.e,'cpu_prefetch_steps',0))
        for distance in range(1,steps+1):
            q=z-distance if self.reverse else z+distance
            if ROI[2]<=q<ROI[5]:self.e.fast_encoder.io.prefetch((x,y,q,1),s)
        point=None
        if cond:
            b=np.asarray(self.seed['box_xyxy'])-np.array([x,y,x,y])
            point=dict(point_coords=torch.tensor(b,dtype=torch.float32,device='cuda').reshape(1,2,2)/s*1024,point_labels=torch.tensor([[2,3]],dtype=torch.int32,device='cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,feat,pos,sizes=m._prepare_backbone_features(enc)
            cur,outs,_,_=m._track_step(j,cond,feat,pos,sizes,point,None,state,ROI[5]-ROI[2],self.reverse,None)
            _,_,ious,lo,hi,ptr,obj=outs
            score=float(ious.float().reshape(-1)[0])
            if not math.isfinite(score):raise ValueError('nonfinite score')
            if score<self.score_threshold:return None,score
            cur.update(pred_masks=lo,obj_ptr=ptr)
            m._encode_memory_in_output(feat,sizes,point,True,hi,obj,cur)
        mask=torch.nn.functional.interpolate(hi.float(),size=(s,s),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()>=.5
        mask[self.e.nuc_crop(z,x,y,s)]=False
        h,w=self.e.nuc.shape[1:];valid=np.zeros_like(mask)
        lx,ly=max(0,-x),max(0,-y);hx,hy=min(s,w-x),min(s,h-y)
        if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
        mask &= valid
        kept={k:cur[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
        if not getattr(self.e,'keep_memory_on_gpu',False):
            kept['maskmem_features']=kept['maskmem_features'].cpu()
            kept['maskmem_pos_enc']=[v.cpu() for v in kept['maskmem_pos_enc']]
        state['cond_frame_outputs' if cond else 'non_cond_frame_outputs'][j]=kept
        retain=max(m.max_obj_ptrs_in_encoder,m.num_maskmem*getattr(m,'memory_temporal_stride_for_eval',1))+2
        for k in list(state['non_cond_frame_outputs']):
            if (not self.reverse and k<j-retain) or (self.reverse and k>j+retain):del state['non_cond_frame_outputs'][k]
        return mask,score

    def advance(self,z):
        replay=False
        for attempt in range(32):
            state={'cond_frame_outputs':{},'non_cond_frame_outputs':{}} if replay else self.state
            required=list(self.required);mask=None;bad=None
            replay_order=range(self.seed['z'],z-1,-1) if self.reverse else range(self.seed['z'],z+1)
            if replay and getattr(self.e,'advanced_execution',None):self.e.advanced_execution.prefill(self,list(replay_order))
            if replay and getattr(self.e,'cpu_prefetch_steps',0):
                for q in list(replay_order)[:8]:self.e.fast_encoder.io.prefetch((*self.crop[:2],q,1),self.crop[2])
            for q in replay_order if replay else [z]:
                mask,score=self.decode(q,state)
                if mask is None:
                    self.active=False;self.stop=dict(reason='crop_replay_low_score' if replay and q!=z else 'low_score',z=q,score=score);self.state={};return None
                required=union_bounds(required,bounds(mask,self.crop[:2]))
                b=bounds(mask,self.crop[:2]);x,y,s=self.crop
                if b and min(b[0]-x,b[1]-y,x+s-b[2],y+s-b[3])<=100:bad=True
            if not bad:
                self.state=state;self.required=required;small,xy=compact(mask,self.crop[:2])
                f=dict(z=z,score=score,origin_xy=xy,crop=list(self.crop),replayed=replay,crop_attempts=attempt+1)
                self.frames.append(f);return small,f
            new=relocate(self.crop,required)
            self.history.append(dict(z=z,old_crop=self.crop,new_crop=new,required=required,replay_from_z=self.seed['z']))
            if new==self.crop or new[2]>4096:break
            self.crop=new;replay=True
        self.active=False;self.stop=dict(reason='unresolved_crop_margin',z=z);self.state={};return None

def publish(z,frames,tracks,decisions,started,completed=False):
    rgba=np.zeros((2500,2500,4),np.uint8);best=np.full((2500,2500),-1.,np.float32)
    for tid,mask,f in frames:
        x,y=f['origin_xy'];h,w=mask.shape;lx,ly=max(0,x),max(0,y);hx,hy=min(2500,x+w),min(2500,y+h)
        if hx<=lx or hy<=ly:continue
        piece=mask[ly-y:hy-y,lx-x:hx-x];take=piece&(f['score']>best[ly:hy,lx:hx]);best[ly:hy,lx:hx][take]=f['score']
        color=[45+(tid*73)%210,45+(tid*131)%210,45+(tid*197)%210,140];rgba[ly:hy,lx:hx][take]=color
    p=GALLERY/'masks'/f'z{z:03d}.png';p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.partial.png');Image.fromarray(rgba).save(tmp);tmp.replace(p)
    save(GALLERY/'decisions'/f'z{z:03d}.json',dict(boxes=decisions))
    save(GALLERY/'instances'/f'z{z:03d}.json',[dict(track_id=tid,**f) for tid,_,f in frames])
    progress=dict(z=z,completed_layers=z-ROI[2]+1,total_layers=ROI[5]-ROI[2],active=sum(t.active for t in tracks),created=len(tracks),suppressed=SUPPRESSED,seconds=time.time()-started,completed=completed)
    save(RUN/'progress.json',progress);save(GALLERY/'inference_progress.json',progress)
    for t in tracks:save(RUN/'tracks'/f'track_{t.id:05d}.json',dict(track_id=t.id,seed=t.seed,frames=t.frames,crop_history=t.history,stop=t.stop,active=t.active))
    print(json.dumps(progress),flush=True)

SUPPRESSED=0
def main():
    global SUPPRESSED
    import fcntl
    RUN.mkdir(exist_ok=True,parents=True);lock=open(RUN/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'completed.json').exists():return
    for name in ['tracks','frames']:(RUN/name).mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    started=time.time();assets=ROOT/'baseline/assets'
    save(RUN/'config.json',dict(roi=ROI,score_threshold=.5,box_mask_iom_suppression_strict_gt=.8,initial_crop='max(100,3*longest_bbox_side)',crop_margin=101,crop_change='replay original seed through current frame in new crop',same_frame_seeds_do_not_suppress_each_other=True,detections_path=str(DETECTIONS_PATH),**EXPERIMENT))
    e=create_local(REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif',REF/'instance_centered_20261102/nucleus.npy',assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True)
    detections=json.loads(DETECTIONS_PATH.read_text())['detections'];byz={}
    for b in detections:
        x0,y0,x1,y1=b['box_xyxy'];cx,cy=(x0+x1)/2,(y0+y1)/2
        if ROI[0]<=cx<ROI[3] and ROI[1]<=cy<ROI[4] and ROI[2]<=b['z']<ROI[5]:byz.setdefault(b['z'],[]).append(b)
    tracks=[]
    for z in range(ROI[2],ROI[5]):
        frames=[]
        for t in tracks:
            if t.active:
                r=t.advance(z)
                if r:frames.append((t.id,*r))
        snapshot=list(frames);decisions=[]
        for b in sorted(byz.get(z,[]),key=lambda v:v['global_detection_id']):
            options=[(mask_box_iom(mask,f['origin_xy'],b['box_xyxy']),-tid) for tid,mask,f in snapshot]
            iom,negid=max(options,default=(0.,0));d=dict(detection_id=b['global_detection_id'],iom=iom,suppressed=iom>.8,covered_by_track=-negid if iom>.8 else None)
            if d['suppressed']:SUPPRESSED+=1
            else:
                t=Tracker(e,b,len(tracks)+1);tracks.append(t);d['track_id']=t.id;r=t.advance(z)
                if r:frames.append((t.id,*r))
                else:d['seed_stop']=t.stop
            decisions.append(d)
        for tid,mask,f in frames:np.savez_compressed(RUN/'frames'/f'z{z:03d}_track{tid:05d}.npz',mask=mask,origin_xy=f['origin_xy'],score=f['score'])
        publish(z,frames,tracks,decisions,started)
    for t in tracks:
        if t.active:t.active=False;t.stop=dict(reason='requested_roi_z_boundary',z=ROI[5]-1);t.state={}
    publish(z,frames,tracks,decisions,started,True)
    save(RUN/'completed.json',dict(limited_run=False,roi=ROI,layers=44,tracks=len(tracks),seconds=time.time()-started))
    e.fast_encoder.io.close()

if __name__=='__main__':
    try:main()
    except Exception:
        RUN.mkdir(parents=True,exist_ok=True);save(RUN/'error.json',dict(trace=traceback.format_exc(),time=time.time()));raise
