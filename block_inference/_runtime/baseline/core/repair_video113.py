"""Single-existing-instance video repair, original mask -> detector prompt -> one CC."""
import os,json,math,time,gc
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from instance_centered import InstanceEngine,ROOT
from infer_edge_shift import rerun_track_adaptive
from infer_block import save,TRAIN,Engines
from repair_quality import inspect,reference_line_check
EXTEND=os.environ.get('REPAIR_EXTEND_BY_SCORE')=='1'
PROMPT_MODE=os.environ.get('REPAIR_PROMPT_MODE','median_detector')
RUN=ROOT/os.environ.get('REPAIR_RUN_NAME','repair_qc20k_113_20261102')
GALLERY=TRAIN/'dashboard/sam_video_hardcases_20261102'/os.environ.get('REPAIR_GALLERY_NAME','repair_video113')

class LazyClip:
    """Encode at most ten nearby frames on demand; decoder memory stays continuous."""
    def __init__(self,engine,geometry,side):
        from collections import OrderedDict
        self.engine=engine;self.geometry=geometry;self.side=side;self.cache=OrderedDict();self.last=None;self.batches=[]
    def __getitem__(self,j):
        x,y,z,n=self.geometry
        if j not in self.cache:
            reverse=self.last is not None and j<self.last
            lo=max(0,j-9) if reverse else j;hi=j+1 if reverse else min(n,j+10)
            encoded=Engines.encode_clip(self.engine,(x,y,z+lo,hi-lo),self.side)
            for k,f in enumerate(encoded):self.cache[lo+k]=f
            self.batches.append({'z_start':z+lo,'z_end_exclusive':z+hi,'direction':'backward' if reverse else 'forward'})
            while len(self.cache)>20:self.cache.popitem(last=False)
        self.last=j;return self.cache[j]

class ScoreStoppedEngine(InstanceEngine):
    def encode_clip(self,geometry,side):return LazyClip(self,geometry,side)
    def track(self,encoded,geometry,side,det):
        track=super().track(encoded,geometry,side,det);evaluated=track['evaluated_scores']
        track['source_video_geometry']=list(geometry);track['encoded_z_batches']=encoded.batches
        if evaluated:
            lo=min(f['z'] for f in evaluated);hi=max(f['z'] for f in evaluated)+1;track['geometry']=[geometry[0],geometry[1],lo,hi-lo]
        stopped={e['direction']:e for e in track['stop_events']}
        reason='low_score' if getattr(self,'score_stop_count',2)==1 else 'two_low_scores'
        track['direction_end_reasons']={direction:reason if direction in stopped else 'source_volume_boundary' for direction in ('forward','backward')}
        if not any(f['z']==det['z'] for f in track['frames']):
            for direction in ('forward','backward'):
                if direction not in stopped and not any(f['direction']==direction for f in evaluated):track['direction_end_reasons'][direction]='reference_not_retained'
        return track
def raster_box(box):return [int(math.ceil(v-.5)) for v in box]
def frame_box_iou(mask,origin,box):
    x,y=origin;x0,y0,x1,y1=raster_box(box);h,w=mask.shape;lx=max(x,x0);ly=max(y,y0);hx=min(x+w,x1);hy=min(y+h,y1)
    inter=int(mask[ly-y:hy-y,lx-x:hx-x].sum()) if hx>lx and hy>ly else 0
    return inter/max(1,int(mask.sum())+max(0,x1-x0)*max(0,y1-y0)-inter)

def maxarea_prompt(mask,origin,require_clean=False,straight_threshold=100):
    areas=mask.sum(axis=(1,2),dtype=np.int64)
    if not areas.any():return None,{'reason':'no_existing_mask'}
    x,y,z=map(int,origin);rejected=[];chosen=None;line_check=None
    candidates=sorted(np.flatnonzero(areas),key=lambda j:(-int(areas[j]),int(j)))
    for rank,j in enumerate(candidates,1):
        j=int(j);check=reference_line_check(mask[j],(x,y),threshold=straight_threshold) if require_clean else None
        if check is not None and not check['accepted']:
            rejected.append({'z':z+j,'area_px':int(areas[j]),'area_rank':rank,'line_check':check});continue
        chosen=(j,rank);line_check=check;break
    if chosen is None:return None,{'reason':'no_clean_reference','mode':'max_area_clean_pad10','candidates_checked':len(candidates),'rejected_candidates':rejected,'uses_gt':False}
    j,rank=chosen;yy,xx=np.nonzero(mask[j])
    tight=[x+int(xx.min()),y+int(yy.min()),x+int(xx.max())+1,y+int(yy.max())+1]
    box=[tight[0]-10,tight[1]-10,tight[2]+10,tight[3]+10]
    prompt={'z':z+j,'box_xyxy':box,'detection_id':-1,'slice_score':None,'linked_score':None,'is_detector_prediction':False,'prompt_source':'existing_mask_largest_eligible_area_tight_bbox_plus10'}
    selection={'mode':'max_area_clean_pad10' if require_clean else 'max_area_pad10','selected_z':z+j,'largest_mask_area_px':int(areas[j]),'absolute_max_mask_area_px':int(areas.max()),'selected_area_rank':rank,'rejected_candidates':rejected,'selected_line_check':line_check,'tied_largest_area_slices':int(np.count_nonzero(areas==areas[j])),'tie_rule':'lowest source Z','tight_bbox_xyxy_halfopen':tight,'prompt_bbox_xyxy_halfopen':box,'padding_each_side_native_pixels':10,'outside_source_policy':'retain full expanded prompt; image exterior uses existing padding','uses_gt':False,'uses_detector_box':False}
    return prompt,selection
def choose_prompt(mask,origin,dets):
    if PROMPT_MODE in ('max_area_pad10','max_area_clean_pad10'):return maxarea_prompt(mask,origin,require_clean=PROMPT_MODE=='max_area_clean_pad10',straight_threshold=int(os.environ.get('REPAIR_STRAIGHT_THRESHOLD','100')))
    nonempty=np.flatnonzero(mask.any((1,2)));x,y,z=origin
    if not len(nonempty):return None,{'reason':'no_existing_mask'}
    median=z+int(nonempty[(len(nonempty)-1)//2]);byz={}
    for d in dets:byz.setdefault(d['z'],[]).append(d)
    tried=[]
    for zz in sorted((z+int(i) for i in nonempty),key=lambda zz:(abs(zz-median),zz)):
        options=[(frame_box_iou(mask[zz-z],(x,y),d['box_xyxy']),float(d['slice_score']),d) for d in byz.get(zz,[])]
        options=[v for v in options if v[0]>0];tried.append({'z':zz,'overlapping_detections':len(options)})
        if options:
            iou,_,d=max(options,key=lambda v:(v[0],v[1],-v[2]['detection_id']))
            return d,{'median_existing_z':median,'selected_z':zz,'z_offset_from_median':zz-median,'match_iou_to_existing_mask':iou,'tried':tried,'rule':'search existing nonempty slices nearest to lower median; tie lower Z; detector box with highest positive IoU, then detector score'}
    return None,{'reason':'no_overlapping_detector_box_on_any_existing_mask_slice','median_existing_z':median,'tried':tried}
def select_component(binary,origin,ref_z,box):
    x,y,z=origin;parts=np.zeros(binary.shape,np.uint32)
    for i,p in enumerate(binary):parts[i]=ndi.label(p,structure=np.ones((3,3),bool))[0]
    cc,n=ndi.label(parts>0,structure=np.ones((3,3,3),bool));j=ref_z-z
    if not 0<=j<len(parts):return np.zeros_like(binary),parts,cc,{'reason':'reference_outside_prediction'}
    ids=np.unique(parts[j]);options=[]
    for i in ids:
        if not i:continue
        m=parts[j]==i;iou=frame_box_iou(m,(x,y),box)
        labels=np.unique(cc[j][m]);assert len(labels)==1
        options.append({'ref_2d_component':int(i),'component_3d':int(labels[0]),'ref_area':int(m.sum()),'bbox_iou':iou})
    if not options:return np.zeros_like(binary),parts,cc,{'reason':'no_retained_prediction_on_reference_slice','candidates':[]}
    best=max(options,key=lambda q:(q['bbox_iou'],-q['ref_2d_component']))
    if best['bbox_iou']<=0:return np.zeros_like(binary),parts,cc,{'reason':'no_prediction_overlaps_prompt_box','candidates':options}
    return cc==best['component_3d'],parts,cc,{'selected':best,'candidates':options,'all_3d_components':int(n),'rule':'8-connected 2D pieces -> 26-connected 3D pieces -> largest reference-slice 2D mask/prompt-box IoU; no GT selection and no extra size filter'}
def existing(rec):
    cache=ROOT/'vote_view_cache'/f'instance_{rec["index"]:03d}';meta=json.loads((cache/'old.json').read_text());winner=meta['matching']['winner']
    if winner is None:return None,None
    obj=next(o for o in meta['component_construction']['objects'] if o['id']==winner);s=obj['bbox_zyx'];a=np.load(cache/'old_cc.npy',mmap_mode='r');m=np.asarray(a[tuple(slice(lo,hi) for lo,hi in s)]==winner)
    x,y,z=meta['origin_xyz'];return m,[x+s[2][0],y+s[1][0],z+s[0][0]]
def run_case(engine,rec):
    start=time.time();out=RUN/'cases'/f'instance_{rec["index"]:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    old,origin=existing(rec)
    if old is None:
        save(out/'completed.json',{'index':rec['index'],'status':'skipped_no_existing_mask','elapsed_seconds':time.time()-start,'limited_run':False});return
    np.save(out/'existing_mask.npy',old);save(out/'existing_origin.json',origin)
    boundaries=[0,int(engine.raw.size[2])-1] if os.environ.get('REPAIR_PIPELINE_KIND')=='simple_maxclean' else [rec['block'][2],rec['block'][5]-1]
    if os.environ.get('REPAIR_PIPELINE_KIND')=='simple_maxclean':qc=inspect(old,origin,block_boundary_z=boundaries,straight_threshold=int(os.environ.get('REPAIR_STRAIGHT_THRESHOLD','50')),end_cap_threshold=500)
    else:qc=inspect(old,origin,block_boundary_z=boundaries)
    save(out/'input_quality.json',qc)
    dets=[] if PROMPT_MODE in ('max_area_pad10','max_area_clean_pad10') else json.loads((ROOT/'cases'/f'instance_{rec["index"]:03d}'/'video_repaired_overlap/detections.json').read_text())['detections']
    det,selection=choose_prompt(old,origin,dets);save(out/'reference_selection.json',selection)
    if det is None:
        save(out/'completed.json',{'index':rec['index'],'status':'skipped_no_clean_reference' if selection.get('reason')=='no_clean_reference' else 'skipped_no_matching_detector_box','reference_selection':selection,'elapsed_seconds':time.time()-start,'limited_run':False});return
    b=det['box_xyxy'];side=max(int(os.environ.get('REPAIR_MIN_CROP_SIDE','1')),int(math.ceil(3*max(b[2]-b[0],b[3]-b[1]))));x=int(math.floor((b[0]+b[2]-side)/2));y=int(math.floor((b[1]+b[3]-side)/2));z=0 if EXTEND else max(0,det['z']-10);n=int(engine.raw.size[2]) if EXTEND else min(250,det['z']+11)-z
    t0=time.time();track,history=rerun_track_adaptive(engine,det,(x,y,z,n),rec['block'],initial_side=side);inference_seconds=time.time()-t0
    x,y,z,n=track['geometry'];side=track['crop_side'];binary=np.zeros((n,side,side),bool)
    for f in track['frames']:binary[f['z']-z]=f['mask']
    t0=time.time();chosen,parts,cc,chosen_info=select_component(binary,[x,y,z],det['z'],det['box_xyxy']);assembly_seconds=time.time()-t0
    for j,p in enumerate(chosen):assert not (p&engine.nuc_crop(z+j,x,y,side)).any()
    np.save(out/'all_retained_masks.npy',binary);np.save(out/'slice_components.npy',parts);np.save(out/'connected_components_3d.npy',cc)
    zz,yy,xx=np.nonzero(chosen)
    if len(zz):
        slices=(slice(int(zz.min()),int(zz.max())+1),slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1));final=chosen[slices];final_origin=[x+int(xx.min()),y+int(yy.min()),z+int(zz.min())]
    else:final=np.zeros((1,1,1),bool);final_origin=[max(0,x),max(0,y),z]
    np.save(out/'mask.npy',final);save(out/'mask_origin.json',final_origin);save(out/'component_selection.json',chosen_info)
    details={**{k:v for k,v in track.items() if k!='frames'},'frames':[{**{k:v for k,v in f.items() if k not in ('mask','probability')},'mask_area':int(f['mask'].sum())} for f in track['frames']],'crop_history':history}
    save(out/'track.json',details)
    save(out/'completed.json',{'index':rec['index'],'status':'repaired' if final.any() else 'empty_repair','detector_prompt':det,'reference_selection':selection,'component_selection':chosen_info,'initial_side':history[0]['crop_side'],'final_side':side,'crop_geometry':track['geometry'],'crop_attempts':len(history),'nucleus_excluded':True,'nucleus_overlap_voxels':0,'inference_seconds':inference_seconds,'assembly_seconds':assembly_seconds,'elapsed_seconds':time.time()-start,'provenance':engine.provenance,'limited_run':False})
def main():
    RUN.mkdir(exist_ok=True);torch.set_num_threads(3)
    from render_repair113 import publish,render
    publish();engine=ScoreStoppedEngine() if EXTEND else InstanceEngine();records=json.loads((ROOT/'cohort.json').read_text());records=sorted(records,key=lambda r:(r['index'] not in (3,113),r['index']));start=time.time()
    save(RUN/'repair_config.json',{'instances':113,'input':'existing original DET-SAM GT-paired structures from prior panel, pairing already fixed; no GT used to choose ref, prompt or repaired component','median':'lower median of occupied slices','prompt':'actual frozen-detector box; search up/down existing slices if no positive overlap at median','initial_crop_factor':3,'margin_gt100':True,'adaptive_crop':True,'max_frames_each_direction':10,'stop':'two consecutive raw mask IoU head scores <0.5; discard both low frames','final_selection':'reference-slice 8CC with largest prompt-box IoU -> corresponding 26CC; no additional size filtering','nucleus_excluded':True,'gpu':5})
    if EXTEND:
        c=json.loads((RUN/'repair_config.json').read_text());c.update(max_frames_each_direction=None,encode_batch_frames=10,continuous_video_memory=True,stop_only_at='two low scores or true source boundary',gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','6')));save(RUN/'repair_config.json',c)
    if PROMPT_MODE=='defect_directed':
        c=json.loads((RUN/'repair_config.json').read_text());c.update(prompt_mode=PROMPT_MODE,median=None,reference='unclosed: maximum clean area within terminal five native Z layers; hole/straight: nearest other nonempty clean slice, lower Z on tie',prompt='existing-mask tight bbox plus10 on all XY sides; no detector or GT',end_cap_area_threshold_exclusive_px=int(os.environ.get("REPAIR_END_CAP_THRESHOLD","2000")),straight_threshold_exclusive_px=int(os.environ.get("REPAIR_STRAIGHT_THRESHOLD","10")),post_local_end_recheck=os.environ.get("REPAIR_POSTCHECK_ENDS","1")=="1",end_cap_boundaries='true source Z=0,249',repair='preserve original trunk; append outside unclosed tip; replace local successful propagation frames for hole/straight; per-frame highest SAM score for conflicting repairs',short_span_only_policy='preserve original; no whole-instance replacement',stop='unclosed: two consecutive mask scores below .5; hole/straight: target reached or same confidence stop',display_z='GT/original/final nonempty Z union plus2 context, clipped at true source boundary');save(RUN/'repair_config.json',c)
    if PROMPT_MODE in ('max_area_pad10','max_area_clean_pad10'):
        c=json.loads((RUN/'repair_config.json').read_text());c.update(median=None,reference='maximum existing predicted-mask area, earliest Z on tie',prompt='that existing-mask slice tight bbox expanded exactly 10 native pixels on each XY side',detector_used_for_prompt=False,prompt_uses_gt=False,prompt_mode=PROMPT_MODE);save(RUN/'repair_config.json',c)
        if PROMPT_MODE=='max_area_clean_pad10':c.update(reference='descending existing-mask area; reject exposed horizontal/vertical straight edge >100px or >100px straight overlap with tight bbox; take first eligible, earliest Z on area tie',no_eligible_policy='skip and record all rejected candidates');save(RUN/'repair_config.json',c)
    for k,r in enumerate(records):
        save(RUN/'repair_progress.json',{'done_before':k,'total':len(records),'current_index':r['index'],'elapsed_seconds':time.time()-start})
        run_case(engine,r);render(r,engine.raw);publish();print('REPAIR_PANEL_READY',k+1,len(records),'GT',r['gt_id'],flush=True);gc.collect()
    completed=[json.loads(p.read_text()) for p in (RUN/'cases').glob('instance_*/completed.json')]
    save(RUN/'repair_completed.json',{'processed':len(records),'status_counts':dict(__import__('collections').Counter(c['status'] for c in completed)),'elapsed_seconds':time.time()-start,'limited_run':False});publish()
if __name__=='__main__':main()
