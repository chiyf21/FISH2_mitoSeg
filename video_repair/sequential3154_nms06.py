"""Independent experiment; only change is same-Z global NMS IoU>.6."""
import json,traceback,time
import sequential3154 as engine
from sequential_rules import global_slice_nms
from global_bbox_viewer import HTML
from infer_block import save

engine.RUN=engine.ROOT/'sequential3154_nms06_iom08_score05_20261103'
engine.GALLERY=engine.TRAIN/'dashboard/sam_video_hardcases_20261102/sequential3154_nms06_iom08_score05'
engine.EXPERIMENT=dict(global_nms_iou=.6,global_nms_score='slice_score',nms_before_roi_selection=True)
engine.DETECTIONS_PATH=engine.RUN/'detections_nms06.json'

def prepare():
    r=engine.RUN;g=engine.GALLERY;r.mkdir(parents=True,exist_ok=True);g.mkdir(parents=True,exist_ok=True)
    original=json.loads((engine.ROOT/'detgraph113_iou05_iom08_score05_20261103/detections.json').read_text())
    kept,audit=global_slice_nms(original['detections'],.6)
    save(engine.DETECTIONS_PATH,{**original,'detections':kept,'global_nms_iou':.6})
    save(r/'nms_audit.json',audit);byz={};meta={d['detection_id']:d for d in audit}
    for d in original['detections']:
        byz.setdefault(d['z'],[]).append(dict(id=d['global_detection_id'],box=d['box_xyxy'],score=d['slice_score'],**meta[d['global_detection_id']]))
    for z,rows in byz.items():save(g/'boxes'/f'z{z:03d}.json',rows)
    # Reuse identical raw/GT images; this experiment has its own masks/decisions.
    prior=g.parent/'sequential3154_iom08_score05'
    for name in ['raw','gt']:
        if not (g/name).exists():(g/name).symlink_to(prior/name,target_is_directory=True)
    html=HTML.replace('逐帧补种追踪：3154所在块＋全局检测框','逐帧补种追踪＋全局NMS 0.6：3154所在块')
    html=html.replace('已有实例先续推一层。','先在同层所有缓存滑窗框上做全局NMS（IoU&gt;0.6，保留slice_score最高的原框），再选择ROI内seed。已有实例先续推一层。')
    html=html.replace('<label><input id="labels"','<label><input id="rejected" type="checkbox">显示NMS删除框（橙色）</label><label><input id="labels"')
    html=html.replace("for(let b of boxes){let v=decisions[b.id];", "for(let b of boxes){if(b.nms_suppressed&&!e('rejected').checked)continue;let v=decisions[b.id];")
    html=html.replace("ctx.strokeStyle=b.id===selected?", "ctx.strokeStyle=b.nms_suppressed?'#ff8744':b.id===selected?")
    html=html.replace("'boxes','labels','masks','gt','crops'", "'boxes','labels','masks','gt','crops','rejected'")
    html=html.replace("本层 '+boxes.length+' 个缓存检测框", "本层 '+boxes.filter(b=>!b.nms_suppressed).length+' 个NMS保留框 / '+boxes.length+' 个原框")
    (g/'index.html').write_text(html)
    def inroi(d):
        b=d['box_xyxy'];roi=engine.ROI
        return roi[0]<=(b[0]+b[2])/2<roi[3] and roi[1]<=(b[1]+b[3])/2<roi[4] and roi[2]<=d['z']<roi[5]
    summary=dict(input_boxes=len(original['detections']),retained_boxes=len(kept),suppressed_boxes=len(audit)-len(kept),roi_before=sum(inroi(d) for d in original['detections']),roi_after=sum(inroi(d) for d in kept),threshold=.6)
    save(r/'nms_summary.json',summary);save(g/'nms_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    try:prepare();engine.main()
    except Exception:
        save(engine.RUN/'error.json',dict(trace=traceback.format_exc(),time=time.time()));raise
