"""Visualize six distinct validation mitochondria from an actual trained checkpoint."""
from common import *
import argparse,csv,html,math
import numpy as np
import torch
from PIL import Image,ImageDraw
import train_video as training
from source_data import sequence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,default=ROOT/'dashboard/video_tracking_fullgt');ap.add_argument('--count',type=int,default=6);ap.add_argument('--data',type=Path,default=ROOT/'filtered_fullgt');a=ap.parse_args()
    assert a.checkpoint.exists(), 'No trained checkpoint exists; never substitute official initial weights'
    torch.set_num_threads(3)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert 'model' in ck and ck.get('updates',0)>0, 'Requires a checkpoint with actual optimizer updates'
    metadata={'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'epoch':ck.get('epoch'),'optimizer_updates':ck['updates'],'evaluation':'eval mode; actual seed detection box only; full Z; forward and backward from seed; GT only for comparison'}
    model=training.make_model();model.load_state_dict(ck['model'],strict=True);model.eval();del ck
    rows=training.records(a.data/'val.jsonl')
    # Choose distinct source instances, a mix of shorter/medium/longer sequences.
    best={}
    for r in rows:
        assert r.get('full_gt_verified') and r['iom']>=.6
        if r['video_z_bounds'][1]-r['video_z_bounds'][0]<3:continue
        iid=r['source_instance_id'];quality=(r['iom'],r['track_score'])
        if iid not in best or quality>best[iid][0]:best[iid]=(quality,r)
    candidates=sorted((x[1] for x in best.values()),key=lambda r:r['video_z_bounds'][1]-r['video_z_bounds'][0])
    chosen=[candidates[int(i)] for i in np.linspace(0,max(0,int(.85*(len(candidates)-1))),min(a.count,len(candidates)),dtype=int)]
    a.out.mkdir(parents=True,exist_ok=True);manifest=[];metrics=[]
    original_loss=training.frame_losses
    for index,rec in enumerate(chosen,1):
        predictions=[]
        def capture(logits,gt,ious,obj):
            predictions.append(logits.detach().float().sigmoid()[0,0].cpu().numpy())
            return original_loss(logits,gt,ious,obj)
        training.frame_losses=capture
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            vals,n=training.one_sequence(model,rec)
        training.frame_losses=original_loss
        order=list(range(rec['seed_frame'],n))+list(range(rec['seed_frame']-1,-1,-1))
        probs={z:p for z,p in zip(order,predictions)}
        images,gt=sequence(rec);folder=a.out/f'instance_{index:02d}';folder.mkdir(exist_ok=True)
        item={'index':index,'source_instance_id':rec['source_instance_id'],'seed_frame':rec['seed_frame'],'z_start':rec['video_z_bounds'][0],'frames':n,'seed_iom_complete_gt':rec['iom'],'seed_gt_pixels_full':rec['seed_gt_pixels_full'],'seed_gt_pixels_in_box':rec['seed_gt_pixels_in_box'],'seed_sample_id':rec['sample_id'],'mean_frame_iou':float(vals[-1]),'files':[]}
        for z in range(n):
            side=rec['crop_side'];prob=np.asarray(Image.fromarray(probs[z],mode='F').resize((side,side),Image.Resampling.BILINEAR));binary=prob>=.5
            true=gt[z].astype(bool);inter=int((binary&true).sum());union=int((binary|true).sum());iou=inter/union if union else 1.
            metrics.append({'instance':index,'source_instance_id':rec['source_instance_id'],'local_z':z,'global_z':rec['video_z_bounds'][0]+z,'seed':z==rec['seed_frame'],'iou_native_crop':iou,'gt_pixels':int(true.sum()),'pred_pixels':int(binary.sum())})
            rgb=np.repeat(images[z][...,None],3,axis=2).astype(np.float32)
            gt_overlay=rgb.copy();gt_overlay[true]=.55*rgb[true]+.45*np.array([40,220,100])
            pred_overlay=rgb.copy();pred_overlay[binary]=.55*rgb[binary]+.45*np.array([240,80,150])
            heat=np.stack([prob*255,np.clip(1-np.abs(prob-.5)*2,0,1)*210,(1-prob)*255],-1)
            canvas=Image.new('RGB',(1536,562),'#111827');draw=ImageDraw.Draw(canvas)
            titles=['Raw + target GT (green)','Raw + predicted mask (pink)','Predicted probability: blue0 / red1']
            for k,(array,title) in enumerate(zip((gt_overlay,pred_overlay,heat),titles)):
                canvas.paste(Image.fromarray(array.clip(0,255).astype(np.uint8)).resize((512,512)),(k*512,50));draw.text((k*512+8,8),title,fill='white')
            draw.text((8,28),f'GT instance {rec["source_instance_id"]} | Z={rec["video_z_bounds"][0]+z} | Pred-GT IoU={iou:.3f} | Seed FULL-GT IoM={rec["iom"]:.3f}'+(' | SEED BOX' if z==rec['seed_frame'] else ''),fill='white')
            if z==rec['seed_frame']:
                x0,y0,x1,y1=rec['seed_prompt_xyxy'];draw.rectangle((x0/side*512,50+y0/side*512,x1/side*512,50+y1/side*512),outline='yellow',width=2)
            name=f'z_{z:03d}.jpg';canvas.save(folder/name,quality=90);item['files'].append(f'{folder.name}/{name}')
        manifest.append(item);save(a.out/'manifest.json',{'metadata':metadata,'instances':manifest});print(f'Finished {index}/{len(chosen)}',flush=True)
    with (a.out/'frame_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    page='''<!doctype html><meta charset="utf-8"><title>SAM2 Video tracking</title><style>body{background:#10151e;color:white;font:16px system-ui;margin:24px}select,button,input{margin:8px;padding:8px}img{width:100%}pre{white-space:pre-wrap}</style><h1>SAM2 Video: validation tracking</h1><pre id="meta"></pre><select id="instance"></select><button id="prev">Previous Z</button><input id="z" type="range" min="0" value="0"><button id="next">Next Z</button><span id="label"></span><img id="panel"><script>let d;const select=document.getElementById('instance'),slider=document.getElementById('z');function draw(){const r=d.instances[+select.value],z=+slider.value;document.getElementById('panel').src=r.files[z];document.getElementById('label').textContent=`Z ${z+r.z_start} (${z+1}/${r.frames}) ${z===r.seed_frame?'SEED':''}`;}fetch('manifest.json',{cache:'no-store'}).then(r=>r.json()).then(x=>{d=x;document.getElementById('meta').textContent=JSON.stringify(d.metadata,null,2);d.instances.forEach((r,i)=>select.add(new Option(`Instance ${r.source_instance_id}, ${r.frames} slices`,i)));select.onchange=()=>{slider.max=d.instances[+select.value].frames-1;slider.value=d.instances[+select.value].seed_frame;draw()};slider.oninput=draw;document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,+slider.value-1);draw()};document.getElementById('next').onclick=()=>{slider.value=Math.min(+slider.max,+slider.value+1);draw()};select.onchange()});</script>'''
    (a.out/'index.html').write_text(page)
    save(a.out/'completed.json',{'instances':len(manifest),'frames':len(metrics),'limited_run':False,**metadata})

if __name__=='__main__':main()
