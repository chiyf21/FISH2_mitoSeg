from common import *
import math,numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw
from source_data import labels,affine,sequence,raw

def box_iom(mask,origin_yx,box):
    yy,xx=np.nonzero(mask);yy=yy+origin_yx[0]+.5;xx=xx+origin_yx[1]+.5
    x0,y0,x1,y1=box
    inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    return int(inside.sum()),int(len(inside))

manifest=json.loads((ROOT/'dashboard/video_tracking_fast/manifest.json').read_text())
records={}
for line in (ROOT/'filtered/val.jsonl').open():
    r=json.loads(line);iid=r['source_instance_id'];quality=(r['iom'],r['track_score'])
    if iid not in records or quality>records[iid][0]:records[iid]=(quality,r)
results=[];out=ROOT/'dashboard/video_iom_audit';out.mkdir(exist_ok=True)
for item in manifest['instances']:
    rec=records[item['source_instance_id']][1]
    assert rec['seed_frame']==item['seed_frame'] and rec['video_z_bounds'][0]==item['z_start']
    z=rec['video_z_bounds'][0]+rec['seed_frame'];iid=rec['source_instance_id'];source=labels(2)[z]
    ys,xs=np.nonzero(source==iid);assert len(ys)
    _,matrix,offset=affine(rec['plan'])
    corners=np.array([[ys.min()-.5,xs.min()-.5],[ys.min()-.5,xs.max()+.5],[ys.max()+.5,xs.min()-.5],[ys.max()+.5,xs.max()+.5]])
    transformed=(matrix.T@(corners-offset).T).T
    lower=np.floor(transformed.min(0)).astype(int)-3;upper=np.ceil(transformed.max(0)).astype(int)+4
    full=ndi.affine_transform(source,matrix,offset=offset+matrix@lower,output_shape=tuple(upper-lower),order=0,mode='constant',cval=0,prefilter=False)==iid
    intersection,area=box_iom(full,lower,rec['seed_box_xyxy'])
    clipped=ndi.affine_transform(source,matrix,offset=offset,output_shape=(188,188),order=0,mode='constant',cval=0,prefilter=False)==iid
    inter188,area188=box_iom(clipped,(0,0),rec['seed_box_xyxy'])
    cy,cx=rec['crop_origin_yx'];side=rec['crop_side']
    cropped=ndi.affine_transform(source,matrix,offset=offset+matrix@np.array([cy,cx]),output_shape=(side,side),order=0,mode='constant',cval=0,prefilter=False)==iid
    interp,areap=box_iom(cropped,(cy,cx),rec['seed_box_xyxy'])
    row={'panel_index':item['index'],'instance_id':iid,'seed_global_z':z,'stored_iom':rec['iom'],'recomputed_detector188_iom':inter188/area188,'full_slice_iom':intersection/area,'panel_crop_iom':interp/areap if areap else 0,'detector188_mask_coverage_of_full':area188/area,'panel_crop_mask_coverage_of_full':areap/area,'full_gt_pixels':area,'inside_box_pixels':intersection,'source_sample_id':rec['sample_id'],'seed_box_xyxy':rec['seed_box_xyxy']}
    results.append(row)
    # Audit view covers both the complete target slice and the detection box.
    box=rec['seed_box_xyxy'];lo=np.minimum(lower,np.floor([box[1],box[0]]).astype(int))-15;hi=np.maximum(upper,np.ceil([box[3],box[2]]).astype(int))+15
    shape=tuple(hi-lo);gt=ndi.affine_transform(source,matrix,offset=offset+matrix@lo,output_shape=shape,order=0,mode='constant',cval=0,prefilter=False)==iid
    img=np.rint(ndi.affine_transform(raw()[z],matrix,offset=offset+matrix@lo,output_shape=shape,output=np.float32,order=1,mode='constant',cval=0,prefilter=False)).clip(0,255).astype(np.uint8)
    yy,xx=np.indices(shape);inside=(xx+lo[1]+.5>=box[0])&(xx+lo[1]+.5<box[2])&(yy+lo[0]+.5>=box[1])&(yy+lo[0]+.5<box[3])
    rgb=np.repeat(img[...,None],3,axis=2).astype(float);rgb[gt&inside]=.5*rgb[gt&inside]+.5*np.array([30,230,100]);rgb[gt&~inside]=.5*rgb[gt&~inside]+.5*np.array([250,70,70])
    canvas=Image.new('RGB',(900,980),'#111827');scale=min(900/shape[1],900/shape[0]);w,h=int(shape[1]*scale),int(shape[0]*scale)
    canvas.paste(Image.fromarray(rgb.astype(np.uint8)).resize((w,h)),(0,80));d=ImageDraw.Draw(canvas)
    d.text((10,8),f'Panel {item["index"]}: instance {iid}, seed Z={z}',fill='white');d.text((10,28),f'Stored IoM={rec["iom"]:.3f}; COMPLETE-slice IoM={intersection/area:.3f}; {intersection}/{area} pixels',fill='white');d.text((10,48),'Green: target inside detection box; red: target OUTSIDE; cyan: original188 input',fill='white')
    d.rectangle(((box[0]-lo[1])*scale,80+(box[1]-lo[0])*scale,(box[2]-lo[1])*scale,80+(box[3]-lo[0])*scale),outline='yellow',width=3)
    d.rectangle((-lo[1]*scale,80-lo[0]*scale,(188-lo[1])*scale,80+(188-lo[0])*scale),outline='cyan',width=2)
    canvas.save(out/f'instance_{item["index"]:02d}.png')
save(ROOT/'video_iom_audit.json',{'definition':'IoM is intersection / GT target pixels. Full-slice denominator includes SAME original instance outside detector188 and panel crop; nearest-neighbor affine preserves target grid. Only seed layer was filtered, not all Z.','results':results})
save(out/'audit.json',json.loads((ROOT/'video_iom_audit.json').read_text()))
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Seed IoM audit</title><style>body{background:#111827;color:white;font:16px system-ui}img{max-width:900px;width:100%}pre{white-space:pre-wrap}</style><h1>Seed IoM audit: complete GT slice</h1><pre>'+json.dumps(results,indent=2)+'</pre>'+''.join(f'<p>Panel {r["panel_index"]}</p><img src="instance_{r["panel_index"]:02d}.png">' for r in results))
print(json.dumps(results,indent=2))
