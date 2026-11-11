"""Union predictions -> 26CC -> filtered 3D instances, with independent GT view."""
import json,time,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from fast_runtime import ROOT
from shared_source import SharedRaw
from infer_block import save
from refresh5_rules import assemble26
from instance_frame_status import annotate

TRAIN=ROOT.parent/'detector_video_iom06_20261031'

def rgba(mask,color,path):
    out=np.zeros((*mask.shape,4),np.uint8);out[mask]=[*color,145];Image.fromarray(out).save(path)

def main(run,gallery):
    started=time.time();files=sorted((run/'frames').glob('z*_track*.npz'));boxes=[]
    config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
    eval_roi=config.get('iou_bbox_xyz') if config.get('iou_scope')=='detection_roi' else None
    for p in files:
        with np.load(p) as f:
            if f['mask'].any():
                x,y=map(int,f['origin_xy']);h,w=f['mask'].shape;z=int(p.name[1:4]);boxes.append([x,y,z,x+w,y+h,z+1])
    gallery.mkdir(parents=True,exist_ok=True);(gallery/'instances').mkdir(exist_ok=True)
    if not boxes:
        save(run/'completed.json',dict(instances=0,inference=json.loads((run/'inference_completed.json').read_text()),limited_run=False));return
    bb=[min(b[j] for b in boxes) for j in range(3)]+[max(b[j] for b in boxes) for j in range(3,6)];x0,y0,z0,x1,y1,z1=bb
    binary=np.zeros((z1-z0,y1-y0,x1-x0),bool)
    for p in files:
        with np.load(p) as f:
            m=f['mask'];x,y=map(int,f['origin_xy']);z=int(p.name[1:4]);h,w=m.shape
            if m.any():binary[z-z0,y-y0:y-y0+h,x-x0:x-x0+w]|=m
    labels,keep,all_count=assemble26(binary);del binary
    lut=np.zeros(all_count+1,np.int32)
    for new,(old,_,_) in enumerate(keep,1):lut[old]=new
    labels=lut[labels];np.save(run/'instances_26cc.npy',labels);save(run/'instances_origin.json',[x0,y0,z0])
    support=defaultdict(set);scores=defaultdict(dict);source_scores=defaultdict(lambda:defaultdict(list));track_meta={}
    for p in (run/'tracks').glob('track_*.json'):
        t=json.loads(p.read_text());track_meta[t['track_id']]=t
    for p in files:
        with np.load(p) as f:
            m=f['mask'];x,y=map(int,f['origin_xy']);z=int(p.name[1:4]);tid=int(p.stem.split('track')[1]);h,w=m.shape
            if not m.any():continue
            ids=np.unique(labels[z-z0,y-y0:y-y0+h,x-x0:x-x0+w][m])
            for cid in ids[ids>0]:
                support[int(cid)].add(tid);scores[int(cid)][z]=max(scores[int(cid)].get(z,0),float(f['score']))
                source_scores[int(cid)][z].append(dict(track_id=tid,score=float(f['score'])))
    gt=np.load(TRAIN/'source_cache/labels_2.npy',mmap_mode='r');objects=ndi.find_objects(gt)
    raw=SharedRaw(ROOT.parent.parent/'flyEM/2500x2500x250/raw.ome.tif');manifest=[]
    global_view=gallery/'global.html'
    if not global_view.exists():(gallery/'index.html').replace(global_view)
    viewer_html=HTML
    if config.get('probe_min_forward_steps'):
        viewer_html=viewer_html.replace('每5层以预测mask tight bbox重新提示 → 26邻域3D实例','原框试推 → 最高分mask tight bbox → 双向Video 0.45 → 3D实例')
        viewer_html=re.sub(r'(<h2>.*?</h2>)<p>.*?</p>',lambda m:m[1]+'<p>原检测框至少向后试推5帧，首次低分若更晚则试推到该层；以最高分非空mask的零外扩tight bbox分别向前/向后推理。score＜0.45停止，无10帧上限，每5层更新tight bbox。所有预测取并集后26连通组装，过滤小于1000体素或单层结构。IoU只计算检测块XYZ [703,0,8,1215,512,52]内的GT与预测，范围外仅作上下文展示。</p>',viewer_html,count=1)
    if config.get('limited_run'):
        cutoff=config.get('snapshot_cutoff_z')
        viewer_html=viewer_html.replace('<h2>','<p style="background:#65471b;padding:12px;color:#fff1ba;font-weight:bold">阶段快照：仅包含已处理到检测层Z'+str(cutoff)+'的已完成track。正式推理仍在继续；后续3D实例可能连接或合并。</p><h2>',1)
    (gallery/'index.html').write_text(viewer_html)
    for new,(_,sl,voxels) in enumerate(keep,1):
        d=gallery/'instances'/f'instance_{new:04d}';d.mkdir(exist_ok=True)
        pred=labels[sl]==new;o=[x0+sl[2].start,y0+sl[1].start,z0+sl[0].start];np.save(d/'mask.npy',pred);save(d/'origin.json',o)
        zz,yy,xx=pred.shape
        if eval_roi:
            rx0,ry0,rz0,rx1,ry1,rz1=eval_roi
            ax0,ay0,az0=max(o[0],rx0),max(o[1],ry0),max(o[2],rz0);ax1,ay1,az1=min(o[0]+xx,rx1),min(o[1]+yy,ry1),min(o[2]+zz,rz1)
            if ax1>ax0 and ay1>ay0 and az1>az0:
                evaluated=pred[az0-o[2]:az1-o[2],ay0-o[1]:ay1-o[1],ax0-o[0]:ax1-o[0]]
                votes=Counter(map(int,gt[az0:az1,ay0:ay1,ax0:ax1][evaluated]));pred_eval_area=int(evaluated.sum())
            else:votes=Counter();pred_eval_area=0
        else:
            votes=Counter(map(int,gt[o[2]:o[2]+zz,o[1]:o[1]+yy,o[0]:o[0]+xx][pred]));pred_eval_area=voxels
        votes.pop(0,None)
        gid=max(votes,key=votes.get) if votes else None;gs=objects[gid-1] if gid else None
        gbb=[gs[2].start,gs[1].start,gs[0].start,gs[2].stop,gs[1].stop,gs[0].stop] if gs else None
        views=[[o[0],o[1],o[0]+xx,o[1]+yy]]
        if gbb:views.append([gbb[0],gbb[1],gbb[3],gbb[4]])
        vx0=max(0,min(b[0] for b in views)-24);vy0=max(0,min(b[1] for b in views)-24);vx1=min(2500,max(b[2] for b in views)+24);vy1=min(2500,max(b[3] for b in views)+24)
        occupied=set(range(o[2],o[2]+zz))
        if gbb:occupied.update(range(gbb[2],gbb[5]))
        lo=max(0,min(occupied)-2);hi=min(249,max(occupied)+2)
        while hi-lo+1<5 and (lo>0 or hi<249):
            if lo>0:lo-=1
            if hi-lo+1<5 and hi<249:hi+=1
        inter=votes.get(gid,0) if gid else 0
        gtarea=int((gt[rz0:rz1,ry0:ry1,rx0:rx1]==gid).sum()) if gid and eval_roi else (int((gt[gs]==gid).sum()) if gid else 0)
        iou=inter/max(1,pred_eval_area+gtarea-inter) if gid else None
        tids=sorted(support[new]);prompts=[];crops={}
        for tid in tids:
            pp=run/'tracks'/f'prompts_{tid:05d}.json'
            if pp.exists():prompts.extend([dict(track_id=tid,**p) for p in json.loads(pp.read_text())])
            for f in track_meta[tid]['frames']:crops.setdefault(str(f['z']),[]).append(dict(track_id=tid,crop=f['crop']))
        row=dict(id=new,gt_id=gid,gt_votes=dict(votes),iou=iou,iou_scope='detection_roi' if eval_roi else 'full_volume',iou_bbox_xyz=eval_roi,pred_voxels_in_evaluation=pred_eval_area,gt_voxels_in_evaluation=gtarea,intersection_in_evaluation=inter,voxels=voxels,z_span=zz,origin=o,view_xy=[vx0,vy0,vx1,vy1],source_tracks=tids,prompts=prompts,crops=crops,frames=[])
        for z in range(lo,hi+1):
            shape=(vy1-vy0,vx1-vx0);pm=np.zeros(shape,bool)
            if o[2]<=z<o[2]+zz:pm[o[1]-vy0:o[1]-vy0+yy,o[0]-vx0:o[0]-vx0+xx]=pred[z-o[2]]
            gm=(gt[z,vy0:vy1,vx0:vx1]==gid) if gid else np.zeros(shape,bool)
            im,_=raw.slice_crop(z,vx0,vy0,max(shape));Image.fromarray(im[:shape[0],:shape[1]]).save(d/f'z{z:03d}.jpg',quality=86)
            rgba(pm,[240,55,165],d/f'z{z:03d}_pred.png');rgba(gm,[40,225,100],d/f'z{z:03d}_gt.png')
            base=f'instances/instance_{new:04d}/z{z:03d}'
            row['frames'].append(dict(z=z,raw=base+'.jpg',pred=base+'_pred.png',gt=base+'_gt.png',pred_area=int(pm.sum()),gt_area=int(gm.sum()),score=scores[new].get(z),source_scores=sorted(source_scores[new].get(z,[]),key=lambda a:-a['score']),context=not(pm.any() or gm.any())))
        annotate(row,track_meta,run)
        save(d/'instance.json',row);manifest.append(row);save(gallery/'manifest.json',dict(instances=manifest,ready=len(manifest),total=len(keep),completed=False))
        print('rendered instance',new,'/',len(keep),'GT',gid,'IoU',iou,flush=True)
    save(gallery/'manifest.json',dict(instances=manifest,ready=len(manifest),total=len(keep),completed=True))
    result=dict(instances=len(keep),raw_components=all_count,min_voxels=1000,min_z_layers=2,iou_scope='detection_roi' if eval_roi else 'full_volume',iou_bbox_xyz=eval_roi,assembly_render_seconds=time.time()-started,inference=json.loads((run/'inference_completed.json').read_text()),limited_run=bool(config.get('limited_run',False)))
    save(run/'completed.json',result);save(gallery/'completed.json',result)

HTML='''<!doctype html><meta charset="utf-8"><title>每5层tight bbox续推：3D实例</title><style>body{background:#0b1221;color:#eef3ff;font:15px system-ui;margin:15px}.bar{position:sticky;top:0;background:#172339;padding:12px;z-index:5}select,input,button{font:inherit;margin:5px}#z{width:65%}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.frame{position:relative}.raw{width:100%;display:block}.overlay,svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}a{color:#7ecfff}</style><h2>每5层以预测mask tight bbox重新提示 → 26邻域3D实例</h2><p>新seed当层即分割；每seed向后最多10层，score&lt;0.5停止，第5层以预测mask的零外扩tight bbox重新提示。所有预测逐层取并集，再做26连通域；保留≥1000体素且至少2层的结构。每个选项是一个最终预测3D实例。GT仅用于对照与IoU计算，不参与推理。Z展示GT与预测mask并集＋前后2层context。</p><a href="global.html">全局检测框/逐层视图</a><div class="bar"><span id="status"></span><br>预测实例<select id="inst"></select><br>Z<input type="range" id="z" min="0" max="0"><span id="zl"></span><strong id="score" style="display:inline-block;background:#254561;border:1px solid #70ceff;color:#fff;padding:8px 14px;margin:6px;font-size:20px"></strong><br><label><input id="prompt" type="checkbox" checked>实际prompt</label><label><input id="project" type="checkbox">投影其他层prompt</label><label><input id="crop" type="checkbox">实际crop</label></div><p id="frameStatus" style="font-size:18px;font-weight:bold;color:#ffd477"></p><p id="trackStatus" style="white-space:pre-wrap"></p><p id="scoreSources"></p><p>显示的是本层参与组装的来源预测最高score（SAM IoU评分头）；各来源分数列在下方。3D连通组装本身没有新增评分头。</p><p id="info"></p><div class="grid" id="grid"></div><details><summary>实例及prompt记录</summary><pre id="details"></pre></details><script>
const e=id=>document.getElementById(id);let rows=[],cur=null,token=0;e('grid').innerHTML=['原图','人工GT','组装后预测实例'].map((t,i)=>'<div><p>'+t+'</p><div class="frame"><img class="raw" id="r'+i+'"><img class="overlay" id="m'+i+'"><svg id="s'+i+'" preserveAspectRatio="none"></svg></div></div>').join('');
function select(){cur=rows.find(x=>x.id===+e('inst').value);if(!cur)return;e('z').max=cur.frames.length-1;e('z').value=cur.frames.findIndex(f=>f.pred_area>0);draw()}
async function draw(){if(!cur)return;let t=++token,c=cur,f=c.frames[+e('z').value];e('zl').textContent='Z '+f.z+(f.context?' · CONTEXT':' · UNION');e('frameStatus').textContent=f.inference_status_label||'';e('trackStatus').textContent=(f.track_statuses||[]).map(v=>'track '+v.track_id+'：'+v.label).join(String.fromCharCode(10));e('score').textContent=f.score===null||f.score===undefined?'Score：—（本层无保留mask）':'Score：'+f.score.toFixed(4)+'（来源最高）';e('scoreSources').textContent=(f.source_scores||[]).length?'本层来源：'+f.source_scores.map(v=>'track '+v.track_id+' = '+v.score.toFixed(4)).join('；'):'本层没有参与该3D实例组装的保留预测；拒绝分数和停止原因见上方。';e('info').textContent='预测实例 '+c.id+' | GT '+(c.gt_id??'无匹配')+' | '+(c.iou_scope==='detection_roi'?'检测范围内3D IoU ':'完整体积3D IoU ')+(c.iou===null?'无GT':c.iou.toFixed(4))+' | '+c.voxels+'体素 / '+c.z_span+'层 | 来源track '+c.source_tracks.join(',')+' | 本层score '+(f.score?.toFixed(4)??'无预测');e('details').textContent=JSON.stringify(c,null,2);
const urls=[f.raw,f.gt,f.pred];const imgs=await Promise.all(urls.map(u=>new Promise(resolve=>{let im=new Image();im.onload=()=>resolve(im.src);im.onerror=()=>resolve('');im.src=u})));if(t!==token||cur!==c)return;
let [x0,y0,x1,y1]=c.view_xy;function rect(b,col){return '<rect x="'+b[0]+'" y="'+b[1]+'" width="'+(b[2]-b[0])+'" height="'+(b[3]-b[1])+'" fill="none" stroke="'+col+'" stroke-width="2" vector-effect="non-scaling-stroke"/>'}
for(let i=0;i<3;i++){e('r'+i).src=imgs[0];e('m'+i).src=i===0?'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=':imgs[i];let s='';if(i!==1){if(e('prompt').checked)for(let p of c.prompts)if(p.z===f.z||e('project').checked)s+=rect(p.box_xyxy,p.source==='detector'?'#ffe050':'#60e0ff');if(e('crop').checked)for(let q of c.crops[String(f.z)]||[])s+=rect([q.crop[0],q.crop[1],q.crop[0]+q.crop[2],q.crop[1]+q.crop[2]],'#ff9740')}e('s'+i).setAttribute('viewBox',[x0,y0,x1-x0,y1-y0].join(' '));e('s'+i).innerHTML=s}}
async function load(){try{let m=await(await fetch('manifest.json?'+Date.now())).json();e('status').textContent='可查看 '+m.ready+'/'+m.total+(m.completed?' · 完成':' · 逐实例出图');let selected=cur?.id;rows=m.instances;e('inst').innerHTML=rows.map(x=>'<option value="'+x.id+'">实例 '+x.id+' · GT '+(x.gt_id??'无')+' · '+x.z_span+'层 · IoU '+(x.iou?.toFixed(3)??'无GT')+'</option>').join('');if(rows.length){e('inst').value=rows.some(x=>x.id===selected)?selected:rows[0].id;if(!cur)select()}}catch(err){e('status').textContent='正在组装/出图：'+err.message}setTimeout(load,5000)}e('inst').onchange=select;e('z').oninput=draw;for(let id of ['prompt','project','crop'])e(id).onchange=draw;load();</script>'''
