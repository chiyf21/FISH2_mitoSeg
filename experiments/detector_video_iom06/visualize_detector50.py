from common import *
import argparse,math,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from filter_samples import thin_boxes
from source_data import POS,NEG
from data.dataset_npz import NPZCropDataset
from models.model_segdet3d import OrganelleSegDet3D,det_cfg_from_state

@torch.inference_mode()
def main(checkpoint,out,smoke=False):
 out.mkdir(parents=True,exist_ok=True)
 c=torch.load(checkpoint,map_location='cpu',weights_only=False);s={k.removeprefix('module.'):v for k,v in c['model'].items()};m=OrganelleSegDet3D(mode='det_sam',dim='3d',**det_cfg_from_state(s));m.load_state_dict(s,strict=True);m.cuda().eval();del c,s
 rng=np.random.default_rng(20261031);pool=[];datasets={}
 for src,path in [('original',POS),('hardnegative',NEG)]:
  for split in ('train','test'):
   key=f'{src}_{split}';ds=NPZCropDataset(path,split=split,dim='3d',scales=('s1500',));datasets[key]=ds
   n={('original','train'):28,('original','test'):7,('hardnegative','train'):12,('hardnegative','test'):3}[(src,split)]
   pool.extend((key,int(i)) for i in rng.choice(len(ds),n,replace=False))
 rng.shuffle(pool)
 if smoke:pool=[('original_train',0),('hardnegative_test',0)]
 items=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for start in range(0,len(pool),8):
  pp=pool[start:start+8];xx=[datasets[k][i] for k,i in pp];f=m.neck(m.backbone(torch.stack([x['image'] for x in xx]).cuda()));dd={cid:m.det_heads[m.instance_ids.index(cid)](list(f)) for cid in (2,3)}
  for b,((key,i),x) in enumerate(zip(pp,xx)):
   sample=start+b+1;p=out/f'sample_{sample:02d}';p.mkdir(exist_ok=True);det=[]
   for cid in (2,3):
    det.extend({'class_id':cid,**v} for v in thin_boxes(m,dd[cid],b))
   a=x['image'][0].numpy();lo,hi=np.percentile(a,[.5,99.5]);gray=np.uint8(np.clip((a-lo)/max(hi-lo,1e-6),0,1)*255);masks=x['masks'].numpy();labs=x['mask_labels'].numpy();panel=Image.new('RGB',(1160,85+10*410),(15,21,30));pd=ImageDraw.Draw(panel)
   pd.text((15,15),f'Sample{sample:02d} | {key} | dataset index {i}',font=font,fill='white');pd.text((15,40),'Actual model input / frozen detector boxes / annotated GT masks | pink mito, cyan lyso',font=font,fill='white')
   frames=[]
   for z in range(10):
    im=Image.fromarray(gray[z]);rgb=np.repeat(gray[z,...,None],3,axis=-1).astype(float);boxes=Image.fromarray(rgb.astype(np.uint8)).resize((376,376));bd=ImageDraw.Draw(boxes)
    current=[d for d in det if int(round(d['box_zyxzyx'][0]))==z]
    for r in current:
     bb=r['box_zyxzyx'];col=(255,55,155) if r['class_id']==2 else (35,220,255);bd.rectangle(tuple(2*bb[t] for t in (2,1,5,4)),outline=col,width=2);bd.text((max(0,2*bb[2]),max(0,2*bb[1]-14)),f'{r["track_score"]:.2f}',font=font,fill=col)
    for mask,cid in zip(masks,labs):
     if cid not in (2,3):continue
     col=np.array((255,55,155) if cid==2 else (35,220,255));rgb[mask[z]]=rgb[mask[z]]*.55+col*.45
    gt=Image.fromarray(rgb.astype(np.uint8));im.save(p/f'raw_{z:02d}.png');boxes.save(p/f'boxes_{z:02d}.png');gt.save(p/f'gt_{z:02d}.png');y=85+z*410;pd.text((15,y),f'local z={z}, actual thin detections={len(current)}',font=font,fill='white');panel.paste(im.resize((376,376)),(10,y+25));panel.paste(boxes,(392,y+25));panel.paste(gt.resize((376,376)),(774,y+25));frames.append({'local_z':z,'detections':current})
   panel.save(p/'panel_all_z.png');fi,ci=datasets[key].index[i];record={'sample':sample,'source':key,'dataset_index':i,'source_npz':str(datasets[key].files[fi]),'crop_index':ci,'frames':frames};save(p/'metadata.json',record);items.append(record)
 save(out/'manifest.json',{'seed':20261031,'samples':items,'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'sampling':'28 original train +7 original val +12 hardnegative train +3 hardnegative val; no score selection; full merged data domains represented','threshold':'unchanged track score>0.3, actual per-slice boxes shown; labels show track score','display':'Model input zscore mapped to grayscale using crop percentiles for display only.'})
 template=ROOT/'detector_preview_template.html';(out/'index.html').write_text(template.read_text());save(out/'completed.json',{'samples':len(pool),'slices':10*len(pool),'checkpoint_sha256':sha(checkpoint),'limited_run':smoke})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=ROOT/'detector_frozen.pt');ap.add_argument('--out',type=Path,default=ROOT/'dashboard/detector_samples50');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();torch.set_num_threads(3);main(a.checkpoint,a.out,a.smoke)
