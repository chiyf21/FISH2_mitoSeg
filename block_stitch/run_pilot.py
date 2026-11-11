"""Known-instance pilot: matched real frozen-detector seeds, no GT SAM prompts."""
import os,json,time,argparse,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from infer_block import Engines,infer,save,BASE,TRAIN
from audit_instances import inspect

ROOT=BASE/'JaneliaEM_experiments/block_video_stitch_20261101'
DATA=BASE/'EM/neuron_mito_dataset'
GALLERY=TRAIN/'dashboard/block_video_stitch_pilot'


def old_in_bbox(mask,oldbb,bb):
    out=np.zeros((bb[5]-bb[2],bb[4]-bb[1],bb[3]-bb[0]),bool)
    lo=np.maximum(bb[:3],oldbb[:3]);hi=np.minimum(bb[3:],oldbb[3:])
    if (hi>lo).all():
        ss=tuple(slice(int(a-b),int(c-b)) for a,c,b in zip(lo[::-1],hi[::-1],bb[:3][::-1]));tt=tuple(slice(int(a-b),int(c-b)) for a,c,b in zip(lo[::-1],hi[::-1],oldbb[:3][::-1]));out[ss]=mask[tt]
    return out


def overlay(raw,mask,color):
    a=np.repeat(raw[...,None],3,axis=2).astype(float);a[mask]=.55*a[mask]+.45*np.array(color);return np.uint8(a)


def publish():
    GALLERY.mkdir(parents=True,exist_ok=True);selected=json.loads((ROOT/'audit/selected.json').read_text());items=[]
    for i,r in enumerate(selected,1):
        p=GALLERY/f'instance_{i:02d}/case.json'
        if p.exists():items.append(json.loads(p.read_text()))
    save(GALLERY/'manifest.json',{'target':len(selected),'revision':time.time(),'instances':items,'note':'Old masks are model-generated reference, not manual GT. No old mask used as SAM prompt or to trim predictions. All frames are actual source planes. Good agreement is not proof of correctness.'})
    (GALLERY/'index.html').write_text((ROOT/'pilot_gallery.html').read_text())
    if len(items)==len(selected):save(GALLERY/'completed.json',{'instances':len(items),'inference_completed':sum(x['status']=='completed' for x in items),'no_detection':sum(x['status']=='no_detection' for x in items),'errors':sum(x['status']=='error' for x in items),'all_selected_processed':True})


def render(engine,rec,index,bb,out,pred,info):
    folder=GALLERY/f'instance_{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    oldmask=np.load(DATA/rec['mask_path'],mmap_mode='r');old=old_in_bbox(oldmask,rec['bbox_xyz'],bb);inter=int((old&pred).sum());union=int((old|pred).sum())
    newdef=inspect(out,{'mask_path':'predicted_union.npy','bbox_xyz':json.dumps(bb)}) if pred.any() else {'hole_slices':[],'straight_slices':[],'seam_slices':[],'z_span':0,'occupied_z':0,'hole_voxels_2d':0}
    objects=info['objects'];objarr=[];best_iou=0.;bestid=None
    for ob in objects:
        a=np.load(out/ob['file']);mi=mu=0
        ox,oy=map(int,a['origin_xy']);local={}
        for z,m in zip(a['z_global'],a['masks']):
            expanded=np.zeros(old.shape[1:],bool);expanded[oy-bb[1]:oy-bb[1]+m.shape[0],ox-bb[0]:ox-bb[0]+m.shape[1]]=m;local[int(z)]=expanded
        for j in range(len(old)):
            m=local.get(bb[2]+j,np.zeros(old.shape[1:],bool));mi+=int((m&old[j]).sum());mu+=int((m|old[j]).sum())
        iou=mi/max(1,mu)
        if iou>best_iou:best_iou=iou;bestid=ob['object_id']
        objarr.append((ob['object_id'],local))
    palette=[(255,70,145),(250,180,40),(100,170,255),(180,90,255),(40,230,180)]
    files=[];frame_metrics=[];x0,y0,z0,x1,y1,z1=bb;side=x1-x0
    selected=json.loads((out/'selected_detections.json').read_text())['detections']
    for j,z in enumerate(range(z0,z1)):
        raw,_=engine.raw.slice_crop(z,x0,y0,side);p=pred[j];r=old[j]
        new=np.repeat(raw[...,None],3,axis=2).astype(float)
        for oi,local in objarr:
            m=local.get(z)
            if m is not None:new[m]=.55*new[m]+.45*np.array(palette[(oi-1)%len(palette)])
        diff=np.repeat(raw[...,None],3,axis=2).astype(float);diff[r&p]=.55*diff[r&p]+.45*np.array([210,210,210]);diff[p&~r]=.4*diff[p&~r]+.6*np.array([255,70,145]);diff[r&~p]=.4*diff[r&~p]+.6*np.array([0,220,255])
        canv=Image.new('RGB',(1536,566),'#111827');d=ImageDraw.Draw(canv)
        for k,(a,title) in enumerate(zip((overlay(raw,r,(30,220,100)),new.clip(0,255).astype(np.uint8),diff.clip(0,255).astype(np.uint8)),('Old segmentation reference (green)','New SAM video: colors = 3D object IDs','Difference: new-only pink / old-only cyan'))):
            canv.paste(Image.fromarray(a).resize((512,512)),(k*512,54));d.text((k*512+5,8),title,fill='white')
        for det in selected:
            if det['z']==z:
                a,b,c,e=det['box_xyxy'];d.rectangle((512+(a-x0)/side*512,54+(b-y0)/side*512,512+(c-x0)/side*512,54+(e-y0)/side*512),outline='yellow',width=2)
        fi=int((r&p).sum());fu=int((r|p).sum());iou=fi/fu if fu else None
        d.text((6,29),f"Case {index} | {rec['selected_group']} | global Z {z} | old/new agreement IoU {iou if iou is not None else 'empty'} | objects {len(objects)}",fill='white')
        name=f'z_{j:03d}.jpg';canv.save(folder/name,quality=87);files.append(f'{folder.name}/{name}');frame_metrics.append({'global_z':z,'old_pixels':int(r.sum()),'new_pixels':int(p.sum()),'old_new_iou':iou})
    case={'index':index,'record_id':rec['record_id'],'body_id':rec['body_id'],'selected_group':rec['selected_group'],'status':'no_detection' if info['no_detection'] else 'completed','bbox_xyz':bb,'old_bbox_xyz':rec['bbox_xyz'],'frames':len(files),'z_start':z0,'files':files,'old_new_union_iou':inter/max(1,union),'best_component_old_new_iou':best_iou,'best_matching_object_id':bestid,'new_objects':len(objects),'selected_detection_count':info['selected_detections'],'clip_count':info['clips'],'old_hole_slices':len(rec['hole_slices']),'new_hole_slices':len(newdef['hole_slices']),'old_straight_slices':len(rec['straight_slices']),'new_straight_slices':len(newdef['straight_slices']),'old_z_span':rec['z_span'],'new_z_span':newdef['z_span'],'old_voxels':int(old.sum()),'new_voxels':int(pred.sum()),'new_foreground_at_block_z_boundary':bool(pred[0].any() or pred[-1].any()),'provenance':{k:info[k] for k in ('sam_checkpoint','sam_sha256','sam_epoch','detector_checkpoint','detector_sha256')},'frame_metrics':frame_metrics,'selected_detections':selected,'output_dir':str(out)}
    save(folder/'case.json',case);publish();return case


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1);a=ap.parse_args()
    while not (ROOT/'audit/audit_completed.json').exists():time.sleep(10)
    selected=json.loads((ROOT/'audit/selected.json').read_text());publish();engine=Engines()
    for index,rec in enumerate(selected,1):
        if (index-1)%a.world!=a.rank:continue
        folder=GALLERY/f'instance_{index:02d}'
        if (folder/'case.json').exists():continue
        try:
            ob=rec['bbox_xyz'];cx=(ob[0]+ob[3])/2;cy=(ob[1]+ob[4])/2
            # Larger block than each 256 tile; include real XY/Z context beyond old instance.
            x0=max(0,int(cx)-256);y0=max(0,int(cy)-256);z0=max(0,ob[2]-16);z1=min(int(engine.raw.size[2]),ob[5]+16);bb=[x0,y0,z0,x0+512,y0+512,z1]
            out=ROOT/'cases'/f'instance_{index:02d}';mask=np.load(DATA/rec['mask_path'],mmap_mode='r')
            pred,info=infer(engine,bb,out,reference=(mask,ob),match_threshold=.3)
            case=render(engine,rec,index,bb,out,pred,info);print(json.dumps({'case':index,'status':case['status'],'agreement_iou':case['old_new_union_iou']}),flush=True)
        except Exception as e:
            folder.mkdir(parents=True,exist_ok=True);save(folder/'case.json',{'index':index,'record_id':rec['record_id'],'selected_group':rec['selected_group'],'status':'error','error':repr(e),'traceback':traceback.format_exc(),'files':[],'frames':0});publish();print(traceback.format_exc(),flush=True)
    engine.raw.pool.shutdown(wait=True)


if __name__=='__main__':main()
