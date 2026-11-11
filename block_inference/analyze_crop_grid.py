import json,math,collections
from pathlib import Path
import numpy as np
from filter_boxes import nms
P=Path(__file__).resolve().parent;E=P.parent;old=E/'mito_video_repair_fast';out=P/'analysis/crop_grid_plan_20261104';out.mkdir(parents=True,exist_ok=True)
det_path=old/'detgraph113_iou05_iom08_score05_20261103/detections.json';obj=json.loads(det_path.read_text());raw=obj['detections'];filtered,_=nms(raw,.4)
run=old/'sequential3154_probe5_best_bidir045_cc26_20261103';local=json.loads((run/'detections_filtered.json').read_text())['detections']
bins=[0,100,128,192,256,320,384,512,768,1024,float('inf')]
def stats(a):
 a=np.asarray(a,float);h=np.histogram(a,bins)[0];return dict(n=len(a),min=float(a.min()),mean=float(a.mean()),quantiles={str(q):float(np.percentile(a,q)) for q in [10,25,50,75,90,95,99]},max=float(a.max()),histogram=[dict(interval=f'[{bins[i]},{bins[i+1]})',n=int(n),percent=float(n/len(a)*100)) for i,n in enumerate(h)])
def sizes(rows):return [3*max(r['box_xyxy'][2]-r['box_xyxy'][0],r['box_xyxy'][3]-r['box_xyxy'][1]) for r in rows]
sets=dict(raw_deduplicated=stats(sizes(raw)),same_z_nms04_without_nucleus_filter=stats(sizes(filtered)),related_regions_after_nucleus_and_nms=stats(sizes(local)))
frames=[];initial=[];prompts=[]
for f in sorted((run/'tracks').glob('*.json')):
 x=json.loads(f.read_text())
 if not isinstance(x,dict):continue
 b=x.get('initial_detection') or x.get('seed')
 if isinstance(b,dict) and 'box_xyxy' in b:initial.append(3*max(b['box_xyxy'][2]-b['box_xyxy'][0],b['box_xyxy'][3]-b['box_xyxy'][1]))
 for a in x.get('frames',[]):
  if 'crop' in a:frames.append(a['crop'][2])
  if 'tight_bbox' in a and a['tight_bbox']:
   q=a['tight_bbox'];prompts.append(3*max(q[2]-q[0],q[3]-q[1]))
if frames:sets['actual_recorded_crop_side']=stats(frames)
if initial:sets['initial_prompt_records_including_probe']=stats(initial)
if prompts:sets['three_times_frame_mask_tightbox']=stats(prompts)
policies=[]
for scales in [[256,384,512],[256,384,512,768],[256,320,384,512,768],[256,384,512,768,1024],[384,512,768],[320,512,768],[384,640],[320,448,640]]:
 for divisor in [2,3,4]:
  counts=collections.Counter();missing=0;areas=[];ids=collections.defaultdict(set)
  for r in local:
   x0,y0,x1,y1=r['box_xyxy'];m=max(x1-x0,y1-y0)
   for s in scales:
    if s<3*m:continue
    d=max(16,int(s/divisor)//16*16);lx=math.ceil((x1+101-s)/d);hx=math.floor((x0-101)/d);ly=math.ceil((y1+101-s)/d);hy=math.floor((y0-101)/d)
    if lx>hx or ly>hy:continue
    kx=min(hx,max(lx,round(((x0+x1-s)/2)/d)));ky=min(hy,max(ly,round(((y0+y1-s)/2)/d)))
    counts[s]+=1;areas.append((s/max(100,3*m))**2);ids[r['z']].add((s,kx,ky));break
   else:missing+=1
  whole_tiles=0;by_scale={}
  for s in scales:
   d=max(16,int(s/divisor)//16*16);nn=math.floor((2500-101)/d)-math.ceil((101-s)/d)+1;by_scale[s]=dict(stride=d,tiles_per_2500_square_slice=nn*nn);whole_tiles+=nn*nn
  policies.append(dict(scales=scales,stride_divisor=divisor,margin=101,coverage_percent=100*(len(local)-missing)/len(local),assignments=dict(counts),uncovered=missing,mean_crop_area_ratio_to_current_initial=float(np.mean(areas)),selected_unique_crops_across_covered_z=sum(map(len,ids.values())),whole_frame_tiles_per_z=whole_tiles,whole_250_z_encode_images=whole_tiles*250,by_scale=by_scale))
r=dict(sources=dict(detections=str(det_path),source_tag=obj.get('source_tag'),source_files_count=len(obj.get('source_files',[])),scope='existing detection cache covering the 113 validation cases related regions; NOT full-volume detection or 20k population'),statistics=sets,policies=policies,policy_scope='existing cache after nucleus exclusion and global same-Z NMS .4, before ROI centre selection and sequential mask suppression',units='native source pixels',required_crop_side='ceil(max(100,3*max(width,height)))',guard='box coverage plus101px margin is an initial geometry proxy, not a guarantee for future predicted masks',boundary='grid may extend beyond source, with original padding rule; source boundary is not a reason to shift grid')
(out/'distribution.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(statistics={k:{q:v for q,v in x.items() if q!='histogram'} for k,x in sets.items()},policies=policies),indent=2))
