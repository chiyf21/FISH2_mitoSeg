"""Frozen detector -> overlapping XY/Z videos -> global-coordinate instance graph.

CLI can segment any source block. --reference-dir is ONLY a pilot association
filter for an existing instance, never a SAM input, crop geometry or output mask.
"""
import os,sys,json,math,time,argparse,gc,itertools,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from stitch_core import starts,assemble,paste

BASE=Path('/path/to/project_data')
TRAIN=BASE/'JaneliaEM_experiments/detector_video_iom06_20261031'
SAM=BASE/'EM/point-MAE/runs/detsam_sam2base_cohort5k_20261031'
sys.path[:0]=[str(TRAIN),str(SAM)]
import train_video as training
import run_experiment as original


def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.partial.json');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for a in iter(lambda:f.read(2**20),b''):h.update(a)
    return h.hexdigest()


def clip_geometry(bbox,xy,depth,xy_stride,z_stride):
    x,y,z,hx,hy,hz=bbox
    return [(x+dx,y+dy,z+dz,min(depth,hz-z-dz)) for dz,dy,dx in itertools.product(starts(hz-z,depth,z_stride),starts(hy-y,xy,xy_stride),starts(hx-x,xy,xy_stride))]


def box_mask_match(det,mask,bbox):
    z=det['z']-bbox[2]
    if not 0<=z<len(mask):return 0.,0.
    m=mask[z];x0,y0,x1,y1=det['box_xyxy']
    # Pixel centres in half-open continuous detector box.
    lx=max(0,int(math.ceil(x0-.5))-bbox[0]);ly=max(0,int(math.ceil(y0-.5))-bbox[1]);hx=min(m.shape[1],int(math.ceil(x1-.5))-bbox[0]);hy=min(m.shape[0],int(math.ceil(y1-.5))-bbox[1])
    inter=int(m[ly:hy,lx:hx].sum()) if hx>lx and hy>ly else 0
    fg=int(m.sum());area=max(0.,(x1-x0)*(y1-y0));return inter/max(1,fg+area-inter),inter/max(1,fg)


class Engines:
    def __init__(self):
        torch.set_num_threads(3)
        self.raw=original.Raw()
        original.ARMS['ft10_block']={'weight':TRAIN/'detector_frozen.pt','name':'frozen detonly epoch3 best'}
        self.detector=original.load_detector('ft10_block')
        self.model=training.make_model();ck=torch.load(TRAIN/'video_fast_fullgt_ft10/best.pt',map_location='cpu',weights_only=False)
        self.model.load_state_dict(ck['model'],strict=True);self.model.eval()
        self.provenance={'detector_checkpoint':str(TRAIN/'detector_frozen.pt'),'detector_sha256':digest(TRAIN/'detector_frozen.pt'),'sam_checkpoint':str(TRAIN/'video_fast_fullgt_ft10/best.pt'),'sam_sha256':digest(TRAIN/'video_fast_fullgt_ft10/best.pt'),'sam_epoch':ck['epoch'],'sam_updates':ck['updates']};del ck

    def detections(self,bbox,out):
        p=out/'detections.json'
        if p.exists():
            v=json.loads(p.read_text());assert v['detector_sha256']==self.provenance['detector_sha256'];assert v['bbox_xyz']==bbox,'Output cache belongs to a different block';return v['detections']
        keys=original.keys_for_bbox(bbox);self.raw.prefetch(keys);rows=[]
        for ki,key in enumerate(keys):
            rr=original.detection_tile('ft10_block',key,self.detector,self.raw)
            for r in rr:
                x0,y0,x1,y1=r['box_xyxy'];cx=(x0+x1)/2;cy=(y0+y1)/2
                if bbox[0]<=cx<bbox[3] and bbox[1]<=cy<bbox[4] and bbox[2]<=r['z']<bbox[5]:rows.append(r)
            print(f'detection tile {ki+1}/{len(keys)}',flush=True)
        # Native detector tiles overlap; dedup only almost identical boxes on same Z.
        kept=[]
        for r in sorted(rows,key=lambda a:a['slice_score'],reverse=True):
            a=r['box_xyxy'];duplicate=False
            for b in kept:
                if b['z']!=r['z']:continue
                b=b['box_xyxy'];inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
                if inter/max(1,union)>.9:duplicate=True;break
            if not duplicate:kept.append({**r,'detection_id':len(kept)})
        save(p,{'detector_sha256':self.provenance['detector_sha256'],'bbox_xyz':bbox,'detections':kept});return kept

    @torch.inference_mode()
    def encode_clip(self,geometry,side):
        x,y,z,n=geometry;frames=[self.raw.slice_crop(z+j,x,y,side)[0] for j in range(n)];encoded=[]
        for start in range(0,n,8):
            aa=[]
            for im in frames[start:start+8]:
                rgb=np.asarray(Image.fromarray(im).convert('RGB').resize((1024,1024))).copy()
                aa.append(torch.from_numpy(rgb).permute(2,0,1).float()/255.)
            xx=torch.stack(aa).cuda();xx.sub_(torch.tensor([.485,.456,.406],device='cuda')[:,None,None]).div_(torch.tensor([.229,.224,.225],device='cuda')[:,None,None])
            with torch.autocast('cuda',dtype=torch.bfloat16):features=self.model.forward_image(xx)
            for j in range(len(aa)):encoded.append({k:[a[j:j+1].cpu() for a in features[k]] for k in ('backbone_fpn','vision_pos_enc')})
        return encoded

    @torch.inference_mode()
    def track(self,encoded,geometry,side,det):
        x,y,z,n=geometry;seed=det['z']-z;seed_out=None;frames=[];m=self.model
        for reverse,order in [(False,list(range(seed,n))),(True,list(range(seed-1,-1,-1)))]:
            state={'cond_frame_outputs':{} if seed_out is None else {seed:seed_out},'non_cond_frame_outputs':{}}
            for j in order:
                features={k:[a.cuda(non_blocking=True) for a in encoded[j][k]] for k in ('backbone_fpn','vision_pos_enc')}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,feat,pos,sizes=m._prepare_backbone_features(features);point=None
                    if j==seed:
                        box=np.asarray(det['box_xyxy'])-np.array([x,y,x,y]);coords=torch.tensor(box,dtype=torch.float32,device='cuda').reshape(1,2,2)/side*1024
                        point={'point_coords':coords,'point_labels':torch.tensor([[2,3]],dtype=torch.int32,device='cuda')}
                    cur,outs,_,_=m._track_step(j,j==seed,feat,pos,sizes,point,None,state,n,reverse,None)
                    _,_,ious,low,high,objptr,obj=outs;cur.update(pred_masks=low,obj_ptr=objptr)
                    m._encode_memory_in_output(feat,sizes,point,True,high,obj,cur)
                prob=torch.nn.functional.interpolate(high.float(),size=(side,side),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
                quality=float(ious.float().reshape(-1)[0]);presence=float(obj.float().sigmoid().reshape(-1)[0])
                if not np.isfinite(quality):raise ValueError('Nonfinite SAM quality score')
                binary=prob>=.5
                frames.append({'z':z+j,'origin_xy':[x,y],'mask':binary,'score':quality,'object_probability':presence,'detection_score':det['slice_score'],'seed_z':det['z'],'detection_id':det['detection_id'],'touches_crop_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any())})
                kept={k:cur[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')};state['cond_frame_outputs' if j==seed else 'non_cond_frame_outputs'][j]=kept
                if j==seed:seed_out=kept
                if len(state['non_cond_frame_outputs'])>32:del state['non_cond_frame_outputs'][next(iter(state['non_cond_frame_outputs']))]
        return {'geometry':list(geometry),'seed_detection':det,'frames':frames}


def infer(engine,bbox,out,xy=256,depth=32,xy_stride=128,z_stride=16,reference=None,match_threshold=.3,seeds_per_z=1,merge_iou=.8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);t=time.time();detections=engine.detections(bbox,out)
    # Reference-dependent matching is explicit and restricted to the known-instance pilot.
    if reference is not None:
        mask,refbbox=reference
        detections=[{**d,'old_mask_box_iou':q,'old_mask_box_iom':iom} for d in detections for q,iom in [box_mask_match(d,mask,refbbox)] if q>=match_threshold]
        by_segment=defaultdict(list)
        for gz in starts(bbox[5]-bbox[2],depth,z_stride):
            pool=[d for d in detections if bbox[2]+gz<=d['z']<min(bbox[2]+gz+depth,bbox[5])]
            for d in sorted(pool,key=lambda d:(d['old_mask_box_iou'],d['slice_score']),reverse=True)[:seeds_per_z]:by_segment[d['detection_id']]=d
        detections=list(by_segment.values())
    save(out/'selected_detections.json',{'count':len(detections),'matching_threshold':match_threshold if reference is not None else None,'known_instance_association':reference is not None,'detections':detections})
    geometries=clip_geometry(bbox,xy,depth,xy_stride,z_stride);jobs=defaultdict(list)
    for det in detections:
        b=det['box_xyxy'];found=[]
        for g in geometries:
            x,y,z,n=g
            if z<=det['z']<z+n and x<=b[0] and y<=b[1] and b[2]<=x+xy and b[3]<=y+xy:found.append(g)
        if not found:
            # Real source context, detection-centred fallback only for a box lacking a full-containing grid tile.
            if b[2]-b[0]>xy or b[3]-b[1]>xy:raise ValueError('Detection wider than XY tile; increase --xy')
            for gz in starts(bbox[5]-bbox[2],depth,z_stride):
                z=bbox[2]+gz;n=min(depth,bbox[5]-z)
                if z<=det['z']<z+n:found.append((int(math.floor((b[0]+b[2]-xy)/2)),int(math.floor((b[1]+b[3]-xy)/2)),z,n))
        for g in found:jobs[g].append(det)
    tracks=[]
    for ci,(geometry,seeds) in enumerate(sorted(jobs.items()),1):
        encoded=engine.encode_clip(geometry,xy)
        for det in seeds:tracks.append(engine.track(encoded,geometry,xy,det))
        save(out/'progress.json',{'clips':ci,'total_clips':len(jobs),'tracks':len(tracks),'selected_detections':len(detections),'seconds':time.time()-t})
        print(f'clip {ci}/{len(jobs)} tracks {len(tracks)}',flush=True);del encoded
    save(out/'track_metadata.json',[{'track_index':i,'geometry':track['geometry'],'seed_detection':track['seed_detection'],'frames':[{**{k:v for k,v in f.items() if k!='mask'},'mask_area':int(f['mask'].sum())} for f in track['frames']]} for i,track in enumerate(tracks)])
    objects,audit=assemble(tracks,merge_iou);save(out/'stitch_audit.json',audit)
    # Sparse per-object masks avoid allocating block_size * object_count dense RAM.
    object_info=[];x0,y0,z0,x1,y1,z1=bbox
    union=np.lib.format.open_memmap(out/'predicted_union.npy',mode='w+',dtype=bool,shape=(z1-z0,y1-y0,x1-x0));union[:]=False
    for oi,(root,frames) in enumerate(sorted(objects.items()),1):
        points=[]
        for f in frames:
            yy,xx=np.nonzero(f['mask'])
            if len(xx):points.append((f['origin_xy'][0]+int(xx.min()),f['origin_xy'][1]+int(yy.min()),f['origin_xy'][0]+int(xx.max())+1,f['origin_xy'][1]+int(yy.max())+1))
        if not points:continue
        ox0=max(x0,min(p[0] for p in points));oy0=max(y0,min(p[1] for p in points));ox1=min(x1,max(p[2] for p in points));oy1=min(y1,max(p[3] for p in points))
        if ox1<=ox0 or oy1<=oy0:continue
        ob=[ox0,oy0,z0,ox1,oy1,z1]
        zz=sorted({f['z'] for f in frames if z0<=f['z']<z1});arrays=[]
        for z in zz:
            a=np.zeros((oy1-oy0,ox1-ox0),bool)
            for f in frames:
                if f['z']==z:a|=paste(f,ob)
            arrays.append(a);union[z-z0,oy0-y0:oy1-y0,ox0-x0:ox1-x0]|=a
        if not arrays:continue
        array=np.stack(arrays);np.savez_compressed(out/f'object_{oi:04d}.npz',z_global=np.array(zz),masks=array,origin_xy=np.array([ox0,oy0]))
        selected=[{k:v for k,v in f.items() if k!='mask'} for f in frames]
        object_info.append({'object_id':oi,'root_track':root,'z_global':zz,'voxels':int(array.sum()),'selected_frames':selected,'file':f'object_{oi:04d}.npz'})
    union.flush()
    metadata={'bbox_xyz':bbox,'spacing_xyz_nm':[8,8,30],'xy':xy,'xy_stride':xy_stride,'depth':depth,'z_stride':z_stride,'mask_iou_threshold':merge_iou,'score':'SAM trained mask IoU prediction head; higher wins within each same-Z duplicate component; detector score tie-break only','selected_detections':len(detections),'clips':len(jobs),'tracks':len(tracks),'objects':object_info,'reference_used_only_for_seed_association':reference is not None,'no_detection':not detections,'seconds':time.time()-t,**engine.provenance}
    save(out/'completed.json',metadata);gc.collect();return union,metadata


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bbox',nargs=6,type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--xy',type=int,default=256);ap.add_argument('--depth',type=int,default=32);ap.add_argument('--xy-stride',type=int,default=128);ap.add_argument('--z-stride',type=int,default=16);ap.add_argument('--reference-dir',type=Path);ap.add_argument('--reference-bbox',nargs=6,type=int);a=ap.parse_args()
    reference=None
    if a.reference_dir:
        if not a.reference_bbox:raise ValueError('--reference-bbox required')
        reference=(np.load(a.reference_dir/'mask.npy',mmap_mode='r'),a.reference_bbox)
    if (a.out/'completed.json').exists():raise FileExistsError('Completed output exists; choose a new output directory')
    infer(Engines(),a.bbox,a.out,a.xy,a.depth,a.xy_stride,a.z_stride,reference)


if __name__=='__main__':main()
