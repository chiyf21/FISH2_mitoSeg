"""Frozen detector -> actual 2D boxes -> per-layer GT IoM>=.6 -> video seeds."""
from common import *
import argparse,math,numpy as np,torch
from source_data import POS,NEG,plans,validate_plan
from data.dataset_npz import NPZCropDataset
from models.model_segdet3d import OrganelleSegDet3D,det_cfg_from_state,_slice_detections
from models.box_ops import link_slice_boxes,nms_3d

def iom_match(box,masks):
 z=int(round(box[0]));y0,x0=max(0,int(math.ceil(box[1]-.5))),max(0,int(math.ceil(box[2]-.5)));y1,x1=min(188,int(math.ceil(box[4]-.5))),min(188,int(math.ceil(box[5]-.5)))
 if not 0<=z<10 or y1<=y0 or x1<=x0:return None
 best=None
 for mi,mask in masks:
  area=int(mask[z].sum())
  if not area:continue
  inter=int(mask[z,y0:y1,x0:x1].sum());iom=inter/area
  if iom<.6:continue
  iou=inter/max(1,area+(y1-y0)*(x1-x0)-inter);key=(iom,iou,-mi)
  if best is None or key>best[0]:best=(key,mi,iom,iou)
 return None if best is None else best[1:]

def thin_boxes(model,det,bi):
 boxes=det['boxes'][bi];score=det['logits'][bi].float().softmax(-1)[...,1]
 ids=_slice_detections(boxes,score);sb=boxes[ids].detach().float().cpu();ss=score[ids].detach().float().cpu();b3,s3,p3,tracks=link_slice_boxes(sb,ss,ss[:,None]);keep=nms_3d(b3,s3,.6)[:12];out=[]
 for t in keep.tolist():
  if s3[t]>.3:
   for k in tracks[t]:out.append({'box_zyxzyx':sb[k].tolist(),'slice_score':float(ss[k]),'track_score':float(s3[t]),'track':t})
 return out

@torch.inference_mode()
def main(checkpoint,out,limit):
 torch.set_num_threads(3);out.mkdir(exist_ok=True);weight_sha=sha(checkpoint)
 c=torch.load(checkpoint,map_location='cpu',weights_only=False);state={k.removeprefix('module.'):v for k,v in c['model'].items()};model=OrganelleSegDet3D(mode='det_sam',dim='3d',**det_cfg_from_state(state));model.load_state_dict(state,strict=True);model.cuda().eval();del c,state
 old=out/'definition.json'
 definition={'checkpoint':str(checkpoint),'checkpoint_sha256':weight_sha,'iou_rule':'one actual 2D box matches one GT mito instance on its Z layer iff intersection pixels / GT mask pixels >=0.6; ties choose IoM then IoU','threshold':.6,'detector_postprocess':'unchanged per-slice NMS, link, 3D NMS .6, top12, track score > .3','video_crop':'fixed for all10 frames, centered on seed actual 2D box; side ceil(3*longest edge); native source context and corresponding SAME source instance GT','full_run':limit is None,'train_test_split':'unchanged original positive/negative NPZ dataset split; test is validation only','one_gt_per_box':True}
 if old.exists():assert json.loads(old.read_text())==definition
 else:save(old,definition)
 counts={};sources=[('positive',POS),('negative',NEG)]
 for source,path in sources:
  for split in ('train','test'):
   ds=NPZCropDataset(path,split=split,dim='3d',scales=('s1500',));ids=list(range(min(len(ds),limit))) if limit is not None else list(range(len(ds)))
   def dest(i):return out/f'detections/{source}/{split}/{i//1000:03d}/{i:06d}.json'
   pending=[i for i in ids if not dest(i).exists()]
   for k in range(0,len(pending),8):
    ii=pending[k:k+8];samples=[ds[i] for i in ii];images=torch.stack([x['image'] for x in samples]).cuda();features=model.neck(model.backbone(images));det=model.det_heads[model.instance_ids.index(2)](list(features))
    for bi,(i,ex) in enumerate(zip(ii,samples)):
     fi,ci=ds.index[i];file=ds.files[fi];masks=ex['masks'].numpy();ml=ex['mask_labels'].numpy();mito=[(mi,masks[mi]) for mi in np.where(ml==2)[0]];detections=thin_boxes(model,det,bi);records=[];validated={}
     for di,d in enumerate(detections):
      match=iom_match(d['box_zyxzyx'],mito)
      if match is None:continue
      mi,iom,iou=match;plan=plans()[f'{file.name}:{ci}']
      if mi not in validated:validated[mi]=validate_plan(file,ci,masks[mi],plan)
      iid,error=validated[mi];b=d['box_zyxzyx'];side=max(3,int(math.ceil(3*max(b[4]-b[1],b[5]-b[2]))));assert side<=4096
      y0=int(math.floor((b[1]+b[4]-side)/2));x0=int(math.floor((b[2]+b[5]-side)/2))
      zb=json.loads((ROOT/'source_cache/mito_z_bounds.json').read_text())[str(iid)];global_seed=int(round(plan['center_xyz'][2]))-5+int(round(b[0]));assert zb[0]<=global_seed<zb[1]
      records.append({'sample_id':f'{source}_{split}_{i:06d}_{di:03d}','split':'val' if split=='test' else 'train','source':source,'source_npz':str(file),'crop_index':ci,'dataset_index':i,'npz_gt_mask_index':int(mi),'source_instance_id':iid,'plan':plan,'seed_frame':global_seed-zb[0],'detector_seed_frame':int(round(b[0])),'video_z_bounds':zb,'seed_box_xyxy':[b[2],b[1],b[5],b[4]],'seed_prompt_xyxy':[b[2]-x0,b[1]-y0,b[5]-x0,b[4]-y0],'crop_origin_yx':[y0,x0],'crop_side':side,'iom':iom,'iou':iou,'slice_score':d['slice_score'],'track_score':d['track_score'],'source_image_max_difference':error})
     save(dest(i),{'dataset_index':i,'source':source,'split':split,'source_npz':str(file),'crop_index':ci,'mito_gt_count':len(mito),'detections':detections,'records':records})
    save(out/'progress.json',{'stage':'filter','source':source,'split':split,'processed':len(ids)-len(pending)+min(k+8,len(pending)),'total':len(ids),'time':stamp()})
   counts[f'{source}_{split}']=len(ids)
 # Publish manifests only after every expected input was detected and evaluated.
 totals={'train':0,'val':0};seen=set();masks_with_seed=set()
 for split,origsplit in [('train','train'),('val','test')]:
  target=out/f'{split}.jsonl';temp=target.with_suffix('.partial')
  with temp.open('w') as f:
   for source,_ in sources:
    files=sorted((out/f'detections/{source}/{origsplit}').glob('*/*.json'));assert len(files)==counts[f'{source}_{origsplit}']
    for p in files:
     for rec in json.loads(p.read_text())['records']:
      assert rec['sample_id'] not in seen;seen.add(rec['sample_id']);masks_with_seed.add((rec['source_npz'],rec['crop_index'],rec['npz_gt_mask_index']));f.write(json.dumps(rec)+'\n');totals[split]+=1
  temp.replace(target)
 assert sha(checkpoint)==weight_sha
 save(out/'completed.json',{'full_run':limit is None,'input_counts':counts,'eligible_seed_samples':totals,'unique_gt_instance_crops':len(masks_with_seed),'checkpoint_sha256':weight_sha,'all_source_masks_match_exported_gt':True})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=ROOT/'detector_frozen.pt');ap.add_argument('--out',type=Path,default=ROOT/'filtered');ap.add_argument('--limit',type=int);a=ap.parse_args();main(a.checkpoint,a.out,a.limit)
