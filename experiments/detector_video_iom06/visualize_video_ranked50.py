"""Visualize six distinct validation mitochondria from an actual trained checkpoint."""
from common import *
import argparse,csv,html,math
import numpy as np
import torch
from PIL import Image,ImageDraw
import train_video as training
import train_video_fast as fast
from source_data import sequence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,default=ROOT/'dashboard/video_tracking_fullgt');ap.add_argument('--count',type=int,default=50);ap.add_argument('--data',type=Path,default=ROOT/'filtered_fullgt');a=ap.parse_args()
    assert a.checkpoint.exists(), 'No trained checkpoint exists; never substitute official initial weights'
    torch.set_num_threads(3)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert 'model' in ck and ck.get('updates',0)>0, 'Requires a checkpoint with actual optimizer updates'
    metadata={'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'epoch':ck.get('epoch'),'optimizer_updates':ck['updates'],'evaluation':'eval mode; actual seed detection box only; full Z; forward and backward from seed; GT only for comparison'}
    model=training.make_model();model.load_state_dict(ck['model'],strict=True);model.eval();del ck
    rows=training.records(a.data/'val.jsonl')
    # Use one of the actual fixed validation seeds per source ID; no selection by prediction quality.
    from collections import defaultdict
    groups=defaultdict(list)
    for rec in rows:
        assert rec.get('full_gt_verified') and rec['iom']>=.6
        groups[rec['source_instance_id']].append(rec)
    fixed=fast.select_records(groups,0,validation=True)
    per_id={}
    for rec in fixed:per_id.setdefault(rec['source_instance_id'],rec)
    candidates=list(per_id.values())
    rng=np.random.default_rng(20261101)
    chosen=[candidates[int(i)] for i in rng.permutation(len(candidates))[:min(a.count,len(candidates))]]
    metadata.update(selection='Random distinct validation IDs, RNG20261101; first of four fixed validation seeds per ID; not selected by predicted IoU',target_instances=len(chosen),metric='Native-resolution fixed XY crop pooled IoU across all Z: sum intersection / sum union, threshold0.5. GT outside crop not evaluated. Empty/empty slices excluded from nonempty mean. Good/middle/poor are relative thirds of selected cohort, not clinical thresholds.')
    a.out.mkdir(parents=True,exist_ok=True);manifest=[];metrics=[]
    (a.out/'index.html').write_text((ROOT/'ranked_tracking.html').read_text())
    save(a.out/'manifest.json',{'metadata':metadata,'instances':[]})
    original_loss=training.frame_losses
    for index,rec in enumerate(chosen,1):
        predictions=[]
        def capture(logits,gt,ious,obj):
            predictions.append(logits.detach().float().sigmoid()[0,0].cpu().numpy())
            return original_loss(logits,gt,ious,obj)
        training.frame_losses=capture
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            vals,n=fast.one_sequence(model,rec,training.load_record(rec),False)
        training.frame_losses=original_loss
        order=list(range(rec['seed_frame'],n))+list(range(rec['seed_frame']-1,-1,-1))
        probs={z:p for z,p in zip(order,predictions)}
        images,gt=sequence(rec);folder=a.out/f'instance_{index:02d}';folder.mkdir(exist_ok=True)
        item={'index':index,'source_instance_id':rec['source_instance_id'],'seed_frame':rec['seed_frame'],'z_start':rec['video_z_bounds'][0],'frames':n,'seed_iom_complete_gt':rec['iom'],'seed_gt_pixels_full':rec['seed_gt_pixels_full'],'seed_gt_pixels_in_box':rec['seed_gt_pixels_in_box'],'seed_sample_id':rec['sample_id'],'mean_frame_iou':float(vals[-1]),'files':[]}
        sum_inter=0;sum_union=0;nonempty_ious=[];frame_ious=[]
        for z in range(n):
            side=rec['crop_side'];prob=np.asarray(Image.fromarray(probs[z],mode='F').resize((side,side),Image.Resampling.BILINEAR));binary=prob>=.5
            true=gt[z].astype(bool);inter=int((binary&true).sum());union=int((binary|true).sum());iou=inter/union if union else 1.
            sum_inter+=inter;sum_union+=union;frame_ious.append(iou)
            if union:nonempty_ious.append(iou)
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
        item.update(pooled_iou_native_crop=sum_inter/sum_union if sum_union else 1.,mean_nonempty_frame_iou=float(np.mean(nonempty_ious)) if nonempty_ious else None,mean_frame_iou_native=float(np.mean(frame_ious)),intersection_voxels=sum_inter,union_voxels=sum_union,frame_ious=frame_ious)
        manifest.append(item);save(a.out/'manifest.json',{'metadata':metadata,'instances':manifest});print(f'Finished {index}/{len(chosen)}',flush=True)
    with (a.out/'frame_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    with (a.out/'instance_metrics.csv').open('w',newline='') as f:
        keys=['source_instance_id','frames','seed_iom_complete_gt','pooled_iou_native_crop','mean_nonempty_frame_iou','mean_frame_iou_native','intersection_voxels','union_voxels']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(sorted(manifest,key=lambda x:x['pooled_iou_native_crop']))
    (a.out/'README.md').write_text('SAM2 best epoch8 evaluation. Random50 unique validation instances, fixed validation seed per instance. IoU: native crop threshold0.5, pooled intersection/union over all Z. Outside-crop GT is not included; Seed FULL-GT IoM is a different eligibility metric. Relative thirds for browsing good/middle/poor. Full Z visualized; GT only for comparison. See manifest.json for checkpoint SHA and configuration. No training performed.')
    save(a.out/'completed.json',{'instances':len(manifest),'frames':len(metrics),'limited_run':False,**metadata})

if __name__=='__main__':main()
