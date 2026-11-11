"""Reference-seeded instance tracking; GT enters evaluation only."""
import os,sys,json,math,time,gc,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from functools import lru_cache
from scipy import ndimage as ndi
from infer_block import Engines,save,TRAIN,digest
from infer_edge_shift import rerun_track_adaptive
from shared_source import SharedRaw
from stitch_core import starts
ROOT=Path(__file__).resolve().parent
REF_MODE=os.environ.get('CENTERED_REF_MODE','fixed')
RUN=ROOT/os.environ.get('CENTERED_RUN_NAME','instance_centered_20261102')
GALLERY=TRAIN/'dashboard/sam_video_hardcases_20261102'/os.environ.get('CENTERED_GALLERY_NAME','instance_centered_113')
NUC_WEIGHT=Path('/path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt')

def select_reference_frames(block,detections,mode=None):
    mode=mode or REF_MODE;lo,hi=int(block[2]),int(block[5]);counts={z:0 for z in range(lo,hi)}
    for d in detections:
        if d['z'] in counts:counts[d['z']]+=1
    if mode=='fixed':selected=[z for z in counts if z>=10 and (z-10)%20==0];target=len(selected);remaining=[]
    elif mode=='ranked':
        target=int(math.ceil(.1*(hi-lo)));remaining=set(counts);selected=[]
        while remaining and len(selected)<target:
            z=min(remaining,key=lambda z:(-counts[z],z));selected.append(z)
            remaining={q for q in remaining if abs(q-z)>10}
    else:raise ValueError(mode)
    return {'mode':mode,'block_z_halfopen':[lo,hi],'total_frames':hi-lo,'target_reference_frames':target,'selected_z_in_priority_order':selected,'actual_reference_frames':len(selected),'queue_exhausted_before_target':len(selected)<target,'exclusion_each_direction':10 if mode=='ranked' else None,'tie_rule':'smaller source Z first','frame_box_counts':counts,'selected_frame_box_counts':{z:counts[z] for z in selected}}
def nuclear():
    RUN.mkdir(exist_ok=True)
    marker=RUN/'nucleus_completed.json';target=RUN/'nucleus.npy'
    if marker.exists():return
    sys.path.insert(0,'/path/to/JaneliaEM/organelle_detect/scripts')
    from train_nuclear_unet2d import UNet2D
    model=UNet2D(base=32).cuda().eval();ck=torch.load(NUC_WEIGHT,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    source=SharedRaw();source.column=lru_cache(maxsize=512)(lambda x,y:SharedRaw.column.__wrapped__(source,x,y));zsize,ysize,xsize=source.a.shape
    mask=np.lib.format.open_memmap(target,mode='r+' if target.exists() and (RUN/'nucleus_progress.json').exists() else 'w+',dtype=bool,shape=source.a.shape)
    windows=[(x,y) for y in starts(ysize,800,400) for x in starts(xsize,800,400)]
    resume=json.loads((RUN/'nucleus_progress.json').read_text()).get('z_done',0) if (RUN/'nucleus_progress.json').exists() else 0
    with torch.inference_mode():
        for z in range(resume,zsize):
            prob=np.zeros((ysize,xsize),np.float32);count=np.zeros_like(prob)
            for i in range(0,len(windows),8):
                batch=windows[i:i+8];a=np.stack([source.slice_crop(z,x,y,800)[0] for x,y in batch]).astype(np.float32)/255
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(torch.from_numpy(a[:,None]).cuda()).float().sigmoid().cpu().numpy()[:,0]
                for (x,y),pp in zip(batch,p):prob[y:y+800,x:x+800]+=pp;count[y:y+800,x:x+800]+=1
            mask[z]=prob/np.maximum(count,1)>.5;mask.flush()
            save(RUN/'nucleus_progress.json',{'z_done':z+1,'total':zsize})
            if z%10==0:print('nucleus',z+1,zsize,flush=True)
    save(marker,{'checkpoint':str(NUC_WEIGHT),'sha256':digest(NUC_WEIGHT),'threshold':.5,'window':800,'stride':400,'normalization':'uint8/255','fusion':'mean probabilities','source':'shared interpolated native EM','limited_run':False})
    del model,mask;gc.collect();torch.cuda.empty_cache()

class InstanceEngine(Engines):
    def __init__(self):
        super().__init__();self.nuc=np.load(RUN/'nucleus.npy',mmap_mode='r')
        self.raw.column=lru_cache(maxsize=512)(lambda x,y:SharedRaw.column.__wrapped__(self.raw,x,y))
    def nuc_crop(self,z,x,y,side):
        out=np.zeros((side,side),bool);h,w=self.nuc.shape[1:];lx=max(x,0);ly=max(y,0);hx=min(x+side,w);hy=min(y+side,h)
        if hx>lx and hy>ly:out[ly-y:hy-y,lx-x:hx-x]=self.nuc[z,ly:hy,lx:hx]
        return out
    @torch.inference_mode()
    def track(self,encoded,geometry,side,det):
        x,y,z,n=geometry;seed=det['z']-z;m=self.model;seed_out=None;frames=[];stops=[];evaluated=[]
        for reverse,order in [(False,list(range(seed,n))),(True,list(range(seed-1,-1,-1)))]:
            state={'cond_frame_outputs':{} if seed_out is None else {seed:seed_out},'non_cond_frame_outputs':{}};low=[]
            for j in order:
                features={k:[a.cuda(non_blocking=True) for a in encoded[j][k]] for k in ('backbone_fpn','vision_pos_enc')}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,feat,pos,sizes=m._prepare_backbone_features(features);point=None
                    if j==seed:
                        box=np.asarray(det['box_xyxy'])-np.array([x,y,x,y]);coords=torch.tensor(box,dtype=torch.float32,device='cuda').reshape(1,2,2)/side*1024
                        point={'point_coords':coords,'point_labels':torch.tensor([[2,3]],dtype=torch.int32,device='cuda')}
                    cur,outs,_,_=m._track_step(j,j==seed,feat,pos,sizes,point,None,state,n,reverse,None)
                    _,_,ious,lo,hi,ptr,obj=outs;cur.update(pred_masks=lo,obj_ptr=ptr)
                    m._encode_memory_in_output(feat,sizes,point,True,hi,obj,cur)
                quality=float(ious.float().reshape(-1)[0]);presence=float(obj.float().sigmoid().reshape(-1)[0]);assert math.isfinite(quality)
                evaluated.append({'z':z+j,'score':quality,'direction':'backward' if reverse else 'forward'})
                score_threshold=float(getattr(self,'score_threshold',.5))
                low=low+[z+j] if quality<score_threshold else []
                required_low_frames=int(getattr(self,'score_stop_count',2))
                if len(low)>=required_low_frames:
                    frames=[f for f in frames if f['z'] not in low]
                    stops.append({'direction':'backward' if reverse else 'forward','low_z':low,'threshold':score_threshold,'required_low_frames':required_low_frames,'excluded_both_low_frames':required_low_frames==2,'excluded_low_frames':True});break
                prob=torch.nn.functional.interpolate(hi.float(),size=(side,side),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
                nucleus=self.nuc_crop(z+j,x,y,side);prob[nucleus]=0
                h,w=self.nuc.shape[1:];valid=np.zeros_like(nucleus);lx=max(0,-x);ly=max(0,-y);hx=min(side,w-x);hy=min(side,h-y)
                if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
                prob[~valid]=0;binary=prob>=.5
                frames.append({'z':z+j,'origin_xy':[x,y],'mask':binary,'probability':prob,'score':quality,'object_probability':presence,'detection_score':det['slice_score'],'seed_z':det['z'],'detection_id':det['detection_id'],'touches_crop_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any())})
                kept={k:cur[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')};state['cond_frame_outputs' if j==seed else 'non_cond_frame_outputs'][j]=kept
                if j==seed:seed_out=kept
            if seed_out is None:break
        return {'geometry':list(geometry),'seed_detection':det,'frames':frames,'stop_events':stops,'evaluated_scores':evaluated,'crop_side':side}

def infer_seed(engine,det):
    cache=RUN/'seed_cache';cache.mkdir(exist_ok=True)
    key=hashlib.sha256(json.dumps({'z':det['z'],'box':det['box_xyxy'],'sam':engine.provenance['sam_sha256'],'source':engine.raw.source_tag,'version':'instance_centered_v1'} ,sort_keys=True).encode()).hexdigest()
    mp=cache/f'{key}.json';ap=cache/f'{key}.npz'
    shared=os.environ.get('CENTERED_SHARED_SEED_CACHE')
    if shared and not (mp.exists() and ap.exists()):
        shared_mp=Path(shared)/f'{key}.json';shared_ap=Path(shared)/f'{key}.npz'
        if shared_mp.exists() and shared_ap.exists():mp,ap=shared_mp,shared_ap
    if mp.exists() and ap.exists():
        t=json.loads(mp.read_text());a=np.load(ap);t['seed_detection']=det
        for i,f in enumerate(t['frames']):f.update(mask=a[f'm_{i}'],detection_id=det['detection_id'],detection_score=det['slice_score'])
        t['cache_reused']=True;t['cache_source']=str(mp)
        return t
    b=det['box_xyxy'];side=int(math.ceil(5*max(b[2]-b[0],b[3]-b[1])))
    x=int(math.floor((b[0]+b[2]-side)/2));y=int(math.floor((b[1]+b[3]-side)/2));z=max(0,det['z']-10);end=min(engine.nuc.shape[0],det['z']+11)
    track,history=rerun_track_adaptive(engine,det,(x,y,z,end-z),[0,0,0,*engine.raw.size.tolist()],initial_side=side)
    assert not history[-1].get('frames_nearest_edge_under_100'), 'Unresolved crop margin'
    track['crop_history']=history
    arrays={f'm_{i}':f['mask'] for i,f in enumerate(track['frames'])}
    np.savez_compressed(ap,**arrays)
    for f in track['frames']:f.pop('probability',None)
    save(mp,{**{k:v for k,v in track.items() if k!='frames'},'frames':[{k:v for k,v in f.items() if k!='mask'} for f in track['frames']]})
    return track

class TrackAssembly:
    """Prompt identities first; merge across prompts on true 26-neighbor contact."""
    def __init__(self):self.parent=[]
    def root(self,i):
        while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def add(self,tracks):
        i=len(tracks)-1;self.parent.append(i)
        for t in tracks:
            for f in t['frames']:
                if '_extent' not in f:
                    yy,xx=np.nonzero(f['mask']);x,y=f['origin_xy'];f['_extent']=[x+int(xx.min()),y+int(yy.min()),x+int(xx.max())+1,y+int(yy.max())+1] if len(xx) else None
        for j in range(i):
            if self.root(i)==self.root(j):continue
            hit=False
            for a in tracks[i]['frames']:
                if a['_extent'] is None:continue
                for b in tracks[j]['frames']:
                    if abs(a['z']-b['z'])>1 or b['_extent'] is None:continue
                    ax,ay,ahx,ahy=a['_extent'];bx,by,bhx,bhy=b['_extent']
                    lx=max(ax-1,bx);ly=max(ay-1,by);hx=min(ahx+1,bhx);hy=min(ahy+1,bhy)
                    if hx<=lx or hy<=ly:continue
                    # Dilate only a tight local region, preserving 8-neighbor XY.
                    fa=np.zeros((hy-ly+2,hx-lx+2),bool);x,y=a['origin_xy'];s=a['mask'].shape[0]
                    ux=max(lx-1,x);uy=max(ly-1,y);vx=min(hx+1,x+s);vy=min(hy+1,y+s)
                    if vx>ux and vy>uy:fa[uy-(ly-1):vy-(ly-1),ux-(lx-1):vx-(lx-1)]=a['mask'][uy-y:vy-y,ux-x:vx-x]
                    x,y=b['origin_xy'];near=ndi.binary_dilation(fa,structure=np.ones((3,3),bool))[1:-1,1:-1]
                    if np.any(near&b['mask'][ly-y:hy-y,lx-x:hx-x]):hit=True;break
                if hit:break
            if hit:self.parent[self.root(i)]=self.root(j)
        for k,t in enumerate(tracks):t['assembly_root']=self.root(k)

def box_vote(det,tracks):
    b=det['box_xyxy'];x0,y0,x1,y1=[int(math.ceil(v-.5)) for v in b]
    if x1<=x0 or y1<=y0:return {'iom':0.,'winner':None,'votes':0}
    lab=np.zeros((y1-y0,x1-x0),np.int32);q=np.full(lab.shape,-np.inf,np.float32);areas={}
    for ti,t in enumerate(tracks,1):
        for f in t['frames']:
            if f['z']!=det['z']:continue
            group=int(t.get('assembly_root',ti-1))+1
            areas[group]=areas.get(group,0)+int(f['mask'].sum());x,y=f['origin_xy'];h,w=f['mask'].shape;lx=max(x0,x);ly=max(y0,y);hx=min(x1,x+w);hy=min(y1,y+h)
            if hx<=lx or hy<=ly:continue
            crop=f['mask'][ly-y:hy-y,lx-x:hx-x];l=lab[ly-y0:hy-y0,lx-x0:hx-x0];s=q[ly-y0:hy-y0,lx-x0:hx-x0];take=crop&(f['score']>s);l[take]=group;s[take]=f['score']
    ids,counts=np.unique(lab[lab>0],return_counts=True)
    if not len(ids):return {'iom':0.,'winner':None,'votes':0}
    k=int(np.argmax(counts));winner=int(ids[k]);ff=[f for ti,t in enumerate(tracks) if int(t.get('assembly_root',ti))+1==winner for f in t['frames'] if f['z']==det['z'] and f['mask'].any()]
    xmin=min(f['origin_xy'][0] for f in ff);ymin=min(f['origin_xy'][1] for f in ff);xmax=max(f['origin_xy'][0]+f['mask'].shape[1] for f in ff);ymax=max(f['origin_xy'][1]+f['mask'].shape[0] for f in ff)
    merged=np.zeros((ymax-ymin,xmax-xmin),bool)
    for f in ff:
        x,y=f['origin_xy'];h,w=f['mask'].shape;merged[y-ymin:y-ymin+h,x-xmin:x-xmin+w]|=f['mask']
    lx=max(x0,xmin);ly=max(y0,ymin);hx=min(x1,xmax);hy=min(y1,ymax);inter=int(merged[ly-ymin:hy-ymin,lx-xmin:hx-xmin].sum()) if hx>lx and hy>ly else 0
    area=int(merged.sum());return {'iom':inter/max(1,area),'winner':winner-1,'votes':int(counts[k]),'intersection':inter,'full_mask_area':area,'denominator':'full union of connected prompt-instance slices'}

def run_case(engine,rec):
    idx=rec['index'];out=RUN/'cases'/f'instance_{idx:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    old=ROOT/'cases'/f'instance_{idx:03d}'/'video_repaired_overlap';cache=json.loads((old/'detections.json').read_text());dets=cache['detections']
    selection=select_reference_frames(rec['block'],dets);save(out/'reference_selection.json',selection)
    chosen=set(selection['selected_z_in_priority_order']);priority={z:i for i,z in enumerate(selection['selected_z_in_priority_order'])}
    tracks=[];assembly=TrackAssembly();decisions=[];reference=[d for d in dets if d['z'] in chosen];other=[d for d in dets if d['z'] not in chosen]
    for phase,pool in [('reference',reference),('non_reference_rescue',other)]:
        for i,d in enumerate(sorted(pool,key=lambda q:(priority[q['z']] if phase=='reference' else q['z'],-q['slice_score'],q['detection_id']))):
            vote={'iom':0.} if phase=='reference' else box_vote(d,tracks)
            start=phase=='reference' or vote['iom']<.6
            row={'phase':phase,'detection_id':d['detection_id'],'z':d['z'],'vote':vote,'new_instance':start}
            if start:
                t=infer_seed(engine,d);t['phase']=phase;tracks.append(t);assembly.add(tracks);row['track']=len(tracks)-1
            decisions.append(row)
            save(out/'progress.json',{'phase':phase,'boxes_done':i+1,'phase_boxes':len(pool),'tracks':len(tracks),'reference_frames':selection['actual_reference_frames'],'reference_target':selection['target_reference_frames'],'cache_reused_tracks':sum(bool(t.get('cache_reused')) for t in tracks),'time':time.time()})
            if i%10==0:print('case',idx,phase,i+1,len(pool),'tracks',len(tracks),flush=True)
    save(out/'decisions.json',decisions)
    # First preserve prompt identities, then assemble across references by 26CC.
    bb=rec['block'];w,h,zsize=map(int,engine.raw.size)
    x0=max(0,min([bb[0]]+[t['geometry'][0] for t in tracks]));y0=max(0,min([bb[1]]+[t['geometry'][1] for t in tracks]))
    x1=min(w,max([bb[3]]+[t['geometry'][0]+t['crop_side'] for t in tracks]));y1=min(h,max([bb[4]]+[t['geometry'][1]+t['crop_side'] for t in tracks]))
    z0=min([bb[2]]+[t['geometry'][2] for t in tracks]);z1=max([bb[5]]+[t['geometry'][2]+t['geometry'][3] for t in tracks]);shape=(z1-z0,y1-y0,x1-x0)
    union=np.lib.format.open_memmap(out/'union.npy',mode='w+',dtype=bool,shape=shape);union[:]=False
    score=np.lib.format.open_memmap(out/'score.npy',mode='w+',dtype=np.float32,shape=shape);score[:]=-np.inf
    for ti,t in enumerate(tracks):
        ff=out/'prompt_instances';ff.mkdir(exist_ok=True)
        # Sparse per-prompt frames are retained before any cross-ref union.
        np.savez_compressed(ff/f'track_{ti:05d}.npz',z=np.array([f['z'] for f in t['frames']]),masks=np.array([f['mask'] for f in t['frames']],dtype=bool),origin_xy=t['geometry'][:2])
        for f in t['frames']:
            x,y=f['origin_xy'];s=t['crop_side'];lx=max(x,x0);ly=max(y,y0);hx=min(x+s,x1);hy=min(y+s,y1)
            if hx<=lx or hy<=ly:continue
            a=f['mask'][ly-y:hy-y,lx-x:hx-x];j=f['z']-z0;u=union[j,ly-y0:hy-y0,lx-x0:hx-x0];q=score[j,ly-y0:hy-y0,lx-x0:hx-x0];u|=a;q[a]=np.maximum(q[a],f['score'])
    union.flush();score.flush()
    cc=np.lib.format.open_memmap(out/'instances.npy',mode='w+',dtype=np.uint32,shape=shape)
    from vote_view_core import build_components
    cc,info=build_components(union,output=cc);cc.flush()
    # Required invariant: nucleus overlap is zero after global assembly too.
    for j in range(shape[0]):assert not np.any(union[j]&engine.nuc[z0+j,y0:y1,x0:x1])
    save(out/'tracks.json',[{**{k:v for k,v in t.items() if k!='frames'},'frames':[{**{k:v for k,v in f.items() if k not in ('mask','probability')},'mask_area':int(f['mask'].sum())} for f in t['frames']]} for t in tracks])
    save(out/'completed.json',{'index':idx,'bbox_xyz':[x0,y0,z0,x1,y1,z1],'detections':len(dets),'reference_seeds':len(reference),'reference_selection':selection,'tracks':len(tracks),'cache_reused_tracks':sum(bool(t.get('cache_reused')) for t in tracks),'components':info,'provenance':engine.provenance,'nucleus_excluded':True,'limited_run':False})
    del cc,union,score,tracks;gc.collect()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--nucleus-only',action='store_true');a=ap.parse_args();torch.set_num_threads(3)
    from render_instance_centered import publish
    publish();nuclear()
    if a.nucleus_only:return
    engine=InstanceEngine();records=json.loads((ROOT/'cohort.json').read_text())
    # Two previously debugged instances first, then all remaining cohort IDs.
    records=sorted(records,key=lambda r:(r['index'] not in (3,113),r['index']))
    if a.limit:records=records[:a.limit]
    for rec in records:
        try:
            run_case(engine,rec)
            from render_instance_centered import render_case
            render_case(rec,engine.nuc)
        except Exception:
            import traceback
            save(RUN/'last_error.json',{'index':rec['index'],'traceback':traceback.format_exc()});raise
    save(RUN/'completed.json',{'instances':len(records),'limited_run':bool(a.limit),'time':time.time()})
if __name__=='__main__':main()
