"""Ten-frame forward lookahead; max(mask coverage, box coverage)>.5 rejection."""
import json,time,traceback,argparse,hashlib
import numpy as np
import torch
import sequential3154 as engine
from sequential_rules import mask_box_overlap,global_slice_nms,box_foreground_fraction
from global_bbox_viewer import HTML
from infer_block import save

# Importing the NMS experiment sets its defaults; use separate outputs here.
engine.RUN=engine.ROOT/'sequential3154_nms04_nuc06_lookahead10_overlap05_20261103'
engine.GALLERY=engine.TRAIN/'dashboard/sam_video_hardcases_20261102/sequential3154_nms04_nuc06_lookahead10_overlap05'
engine.DETECTIONS_PATH=engine.RUN/'detections_filtered.json'
CONFIG_EXTRA={}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare():
    r,g=engine.RUN,engine.GALLERY
    original=json.loads((engine.ROOT/'detgraph113_iou05_iom08_score05_20261103/detections.json').read_text())
    nucleus_path=engine.REF/'instance_centered_20261102/nucleus.npy';nucleus=np.load(nucleus_path,mmap_mode='r');audit={};candidates=[]
    for b in original['detections']:
        ratio=box_foreground_fraction(nucleus[b['z']],b['box_xyxy'])
        # The nuclear full-plane area is irrelevant; only full box area divides.
        audit[b['global_detection_id']]=dict(nucleus_box_fraction=ratio,nucleus_suppressed=ratio>.6,nms_suppressed=False)
        if ratio<=.6:candidates.append(b)
    kept,nms_audit=global_slice_nms(candidates,.4)
    for a in nms_audit:audit[a['detection_id']].update(a)
    save(engine.DETECTIONS_PATH,{**original,'detections':kept,'global_nms_iou':.4,'nucleus_box_fraction_threshold':.6})
    save(r/'filter_audit.json',audit)
    byz={}
    for b in original['detections']:
        byz.setdefault(b['z'],[]).append(dict(id=b['global_detection_id'],box=b['box_xyxy'],score=b['slice_score'],**audit[b['global_detection_id']]))
    for z,rows in byz.items():save(g/'boxes'/f'z{z:03d}.json',rows)
    weight=engine.REF.__class__('/path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt');code=engine.REF.__class__('/path/to/JaneliaEM/organelle_detect/scripts/train_nuclear_unet2d.py')
    provenance=dict(weight=str(weight),weight_sha256=sha(weight),code=str(code),code_sha256=sha(code),mask=str(nucleus_path),mask_sha256=sha(nucleus_path),mask_completion=json.loads(nucleus_path.with_name('nucleus_completed.json').read_text()),reuse_existing_mask=True)
    save(r/'nucleus_provenance.json',provenance);save(g/'nucleus_provenance.json',provenance)
    roi=engine.ROI
    def inside(b):
        x0,y0,x1,y1=b['box_xyxy'];return roi[0]<=(x0+x1)/2<roi[3] and roi[1]<=(y0+y1)/2<roi[4] and roi[2]<=b['z']<roi[5]
    summary=dict(input=len(original['detections']),after_nucleus=len(candidates),after_nms=len(kept),roi_input=sum(inside(b) for b in original['detections']),roi_after_nucleus=sum(inside(b) for b in candidates),roi_after_nms=sum(inside(b) for b in kept),nms_threshold=.4,nucleus_threshold=.6)
    save(r/'filter_summary.json',summary);save(g/'filter_summary.json',summary);print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stop-rule',choices=['score','fixed'],required=True);a=p.parse_args()
    if a.stop_rule!='score':raise ValueError('Fixed-horizon score override not implemented; do not silently change stopping')
    import fcntl
    r=engine.RUN;g=engine.GALLERY;r.mkdir(parents=True,exist_ok=True);g.mkdir(parents=True,exist_ok=True)
    lock=open(r/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (r/'completed.json').exists():return
    prior=g.parent/'sequential3154_iom08_score05'
    for name in ['raw','gt']:
        if not (g/name).exists():(g/name).symlink_to(prior/name,target_is_directory=True)
    for name in ['tracks','frames']:(r/name).mkdir(exist_ok=True)
    html=HTML.replace('逐帧补种追踪：3154所在块＋全局检测框','Video向后10层＋NMS 0.4＋核框过滤0.6＋双向覆盖率0.5')
    html=html.replace('每个本层框若覆盖任一已有mask超过80%就跳过，否则新建独立实例；第一帧所有框都作为候选。','每个新seed先向后推10层（另含seed层），score&lt;0.5提前停止。逐层检查框：交集/mask面积与交集/框面积的最大值&gt;0.5就舍弃；新seed的未来预测参与后续层去框，同层新seed仍互不抑制。')
    html=html.replace('已有实例先续推一层。','先排除细胞核占框面积&gt;0.6的框，再同层跨滑窗NMS 0.4。')
    html=html.replace('<label><input id="labels"','<label><input id="rejected" type="checkbox">显示过滤框（核=红，NMS=橙）</label><label><input id="labels"')
    html=html.replace('for(let b of boxes){let v=decisions[b.id];',"for(let b of boxes){if((b.nms_suppressed||b.nucleus_suppressed)&&!e('rejected').checked)continue;let v=decisions[b.id];")
    html=html.replace('ctx.strokeStyle=b.id===selected?',"ctx.strokeStyle=b.nucleus_suppressed?'#ff4455':b.nms_suppressed?'#ff8744':b.id===selected?")
    html=html.replace("'boxes','labels','masks','gt','crops'","'boxes','labels','masks','gt','crops','rejected'")
    html=html.replace("let at=boxes.filter(b=>x>=", "let at=boxes.filter(b=>(e('rejected').checked||(!b.nms_suppressed&&!b.nucleus_suppressed))&&x>=")
    (g/'index.html').write_text(html)
    prepare()
    config=dict(roi=engine.ROI,global_nms_iou=.4,nucleus_box_fraction_threshold=.6,nucleus_model_record='nucleus_provenance.json',forward_steps=10,seed_plus_forward_max_frames=11,score_threshold=.5,stop_rule=a.stop_rule,reject='max(intersection/full_mask_area,intersection/full_box_pixel_area)>0.5',same_frame_snapshot=True,adaptive_crop='max(100,3*longest_box_edge), shift/grow to margin>100 and replay memory',detections_path=str(engine.DETECTIONS_PATH))
    config.update(CONFIG_EXTRA);save(r/'config.json',config);save(g/'config.json',config)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25);started=time.time();assets=engine.ROOT/'baseline/assets'
    e=engine.create_local(engine.REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif',engine.REF/'instance_centered_20261102/nucleus.npy',assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True)
    detections=json.loads(engine.DETECTIONS_PATH.read_text())['detections'];byz={};roi=engine.ROI
    for b in detections:
        x0,y0,x1,y1=b['box_xyxy']
        if roi[0]<=(x0+x1)/2<roi[3] and roi[1]<=(y0+y1)/2<roi[4] and roi[2]<=b['z']<roi[5]:byz.setdefault(b['z'],[]).append(b)
    pending={};tracks=[];engine.SUPPRESSED=0
    for z in range(roi[2],roi[5]):
        frames=pending.pop(z,[]);snapshot=list(frames);decisions=[]
        for b in sorted(byz.get(z,[]),key=lambda d:d['global_detection_id']):
            best=dict(iom=0.,opm=0.,overlap_max=0.,intersection=0,mask_area=0,box_area=0);winner=None
            for tid,mask,f in snapshot:
                v=mask_box_overlap(mask,f['origin_xy'],b['box_xyxy'])
                if v['overlap_max']>best['overlap_max']:best=v;winner=tid
            suppressed=best['overlap_max']>.5
            d=dict(detection_id=b['global_detection_id'],suppressed=suppressed,covered_by_track=winner if suppressed else None,**best)
            if suppressed:engine.SUPPRESSED+=1
            else:
                t=engine.Tracker(e,b,len(tracks)+1);tracks.append(t);d['track_id']=t.id
                stop=min(z+10,roi[5]-1)
                for q in range(z,stop+1):
                    out=t.advance(q)
                    if out is None:break
                    mask,f=out
                    np.savez_compressed(r/'frames'/f'z{q:03d}_track{t.id:05d}.npz',mask=mask,origin_xy=f['origin_xy'],score=f['score'])
                    if q==z:frames.append((t.id,mask,f))
                    else:pending.setdefault(q,[]).append((t.id,mask,f))
                if t.active:t.stop=dict(reason='requested_roi_z_boundary' if stop==roi[5]-1 else 'ten_forward_steps',z=stop)
                t.active=False;t.state={}
                d['seed_stop']=t.stop
            decisions.append(d)
        engine.publish(z,frames,tracks,decisions,started)
        progress=json.loads((r/'progress.json').read_text());progress['active']=len({tid for vs in pending.values() for tid,_,_ in vs});save(g/'inference_progress.json',progress)
    engine.publish(z,frames,tracks,decisions,started,True)
    save(r/'completed.json',dict(limited_run=False,roi=roi,layers=44,tracks=len(tracks),seconds=time.time()-started))
    e.fast_encoder.io.close()

if __name__=='__main__':
    try:main()
    except Exception:
        save(engine.RUN/'error.json',dict(trace=traceback.format_exc(),time=time.time()));raise
