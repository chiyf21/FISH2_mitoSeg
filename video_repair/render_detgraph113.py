"""Fixed113 target/group associations; full-volume IoU; continuous unionZ+2."""
import json,time,os
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from fast_runtime import ROOT
from shared_source import SharedRaw
from infer_block import save
REF=ROOT.parent/'sam_video_hardcases_20261102';TRAIN=ROOT.parent/'detector_video_iom06_20261031';RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103';GALLERY=TRAIN/'dashboard/sam_video_hardcases_20261102/detgraph113_iou05_iom08_score05'

def plane(m,o,z,b):
    x,y,z0=o;x0,y0,x1,y1=b;out=np.zeros((y1-y0,x1-x0),bool)
    if not 0<=z-z0<len(m):return out
    lx=max(x,x0);ly=max(y,y0);hx=min(x+m.shape[2],x1);hy=min(y+m.shape[1],y1)
    if hx>lx and hy>ly:out[ly-y0:hy-y0,lx-x0:hx-x0]=m[z-z0,ly-y:hy-y,lx-x:hx-x]
    return out

def overlay(m,size,color,path):
    if not m.any():return 'empty.png'
    alpha=Image.fromarray((m*128).astype('uint8')).resize(size,Image.Resampling.NEAREST);out=Image.new('RGBA',size,(*color,0));out.putalpha(alpha);out.save(path);return str(path.relative_to(GALLERY))

def make_group_page(meta,raw):
    gid=meta['group_id'];out=GALLERY/f'group_{gid:06d}';out.mkdir(exist_ok=True);p=meta['seed'];g=meta['crop_geometry'];side=meta['final_side'];path=out/'seed.jpg'
    if not path.exists():
        rawcrop,_=raw.slice_crop(p['z'],g[0],g[1],side);pic=Image.fromarray(rawcrop);pic.thumbnail((768,768));pic.save(path,quality=89)
    (out/'index.html').write_text('<meta charset="utf-8"><body style="background:#111827;color:white;font:16px system-ui"><h2>框组'+str(gid)+'的实际seed</h2><pre>'+json.dumps(dict(seed=p,crop_geometry=g,crop_side=side,member_count=meta['member_count']),ensure_ascii=False,indent=2)+'</pre><img src="seed.jpg"></body>')

def render_case(rec,meta,raw,labels):
    idx=rec['index'];gid=rec['matched_group_id'];out=GALLERY/f'instance_{idx:03d}';out.mkdir(exist_ok=True);case=out/'case.json'
    if case.exists():return json.loads(case.read_text())
    oldd=REF/'simple113_component500_union_tight_score04_20261103/cases'/out.name
    if (oldd/'existing_mask.npy').exists():old=np.load(oldd/'existing_mask.npy',mmap_mode='r');oo=json.loads((oldd/'existing_origin.json').read_text())
    else:old=np.zeros((1,1,1),bool);oo=rec['gt_bbox'][:3]
    if meta:
        pred=RUN/'predictions'/f'group_{gid:06d}';new=np.load(pred/'mask.npy',mmap_mode='r');no=json.loads((pred/'origin.json').read_text())
    else:new=np.zeros((1,1,1),bool);no=oo
    gtbb=rec['gt_bbox'];boxes=[gtbb]
    for m,o in [(old,oo),(new,no)]:
        if m.any():boxes.append([*o,o[0]+m.shape[2],o[1]+m.shape[1],o[2]+len(m)])
    b=[max(0,min(q[0] for q in boxes)-24),max(0,min(q[1] for q in boxes)-24),min(2500,max(q[3] for q in boxes)+24),min(2500,max(q[4] for q in boxes)+24)]
    gt_nonempty=(labels[gtbb[2]:gtbb[5],gtbb[1]:gtbb[4],gtbb[0]:gtbb[3]]==rec['gt_id']).any((1,2));union={gtbb[2]+int(j) for j in np.flatnonzero(gt_nonempty)}
    for m,o in [(old,oo),(new,no)]:union.update(o[2]+int(j) for j in np.flatnonzero(m.any((1,2))))
    lo=max(0,min(union)-2);hi=min(249,max(union)+2)
    while hi-lo+1<5 and (lo>0 or hi<249):
        if lo>0:lo-=1
        if hi-lo+1<5 and hi<249:hi+=1
    rev=out/f'group_{gid or 0:06d}';rev.mkdir(exist_ok=True);frames=[];oi=ou=ni=nu=0;x0,y0,x1,y1=b
    for z in range(lo,hi+1):
        a,_=raw.slice_crop(z,x0,y0,max(x1-x0,y1-y0));a=a[:y1-y0,:x1-x0];pic=Image.fromarray(a);pic.thumbnail((512,512));rawfile=rev/f'z{z:03d}_raw.jpg';pic.save(rawfile,quality=88)
        g=labels[z,y0:y1,x0:x1]==rec['gt_id'];o=plane(old,oo,z,b);n=plane(new,no,z,b)
        iold=int((g&o).sum());uold=int((g|o).sum());inew=int((g&n).sum());unew=int((g|n).sum());oi+=iold;ou+=uold;ni+=inew;nu+=unew
        frames.append(dict(z=z,kind='UNION' if z in union else 'CONTEXT',raw=str(rawfile.relative_to(GALLERY)),gt=overlay(g,pic.size,(35,225,105),rev/f'z{z:03d}_gt.png'),old=overlay(o,pic.size,(255,170,35),rev/f'z{z:03d}_old.png'),new=overlay(n,pic.size,(240,55,165),rev/f'z{z:03d}_new.png'),gt_area=int(g.sum()),old_area=int(o.sum()),new_area=int(n.sum()),old_iou=iold/max(1,uold),new_iou=inew/max(1,unew),score=(meta or {}).get('frame_scores',{}).get(str(z))))
    row=dict(index=idx,gt_id=rec['gt_id'],matched_group_id=gid,group_size=(meta or {}).get('member_count'),gt_match_votes=rec['votes'],gt_box_coverage=rec['coverage'],seed=(meta or {}).get('seed'),crop_geometry=(meta or {}).get('crop_geometry'),crop_side=(meta or {}).get('final_side'),old_iou=oi/max(1,ou),new_iou=ni/max(1,nu),frames=frames,view_xy=b,z_values=list(range(lo,hi+1)),status='empty_prediction' if meta and not meta['voxels'] else ('predicted' if meta else 'unmatched'),stop_reasons=(meta or {}).get('stop_reasons'),context_before=min(union)-lo,context_after=hi-max(union),clipped_source_boundary=lo==0 or hi==249,seen_in_sam_train=rec.get('seen_in_sam_train'),group_details=f'group_{gid:06d}.json' if gid else None)
    save(case,row);return row

def publish_once(raw,labels):
    GALLERY.mkdir(parents=True,exist_ok=True)
    if not (GALLERY/'empty.png').exists():Image.new('RGBA',(1,1),(0,0,0,0)).save(GALLERY/'empty.png')
    matches=json.loads((RUN/'target_matches.json').read_text());selected=json.loads((RUN/'selected_groups.json').read_text());complete={}
    for p in (RUN/'predictions').glob('group_*/completed.json'):m=json.loads(p.read_text());complete[m['group_id']]=m
    final=(RUN/'inference_completed.json').exists();rows=[];dets=json.loads((RUN/'detections.json').read_text())['detections']
    for g in selected:
        if g['index'] not in complete:continue
        m=complete[g['index']];path=GALLERY/f'group_{g["index"]:06d}.json'
        if not path.exists():
            byz={}
            for i in g['member_ids']:
                d=dets[i];byz.setdefault(str(d['z']),[]).append(d['box_xyxy'])
            save(path,dict(group_id=g['index'],seed=m['seed'],boxes_by_z=byz,member_count=g['member_count'],matched_gt_ids=g['matched_gt_ids'],component_selection=m['component_selection'],stop_reasons=m['stop_reasons']));make_group_page(m,raw)
    for r in matches:
        gid=r['matched_group_id']
        if gid in complete or (final and gid is None):rows.append(render_case(r,complete.get(gid),raw,labels))
    progress=json.loads((RUN/'inference_progress.json').read_text()) if (RUN/'inference_progress.json').exists() else {}
    save(GALLERY/'manifest.json',dict(instances=rows,ready_instances=len(rows),target_instances=113,done_groups=len(complete),selected_groups=len(selected),all_groups=1094,inference_complete=final,progress=progress));(GALLERY/'index.html').write_text(HTML)
    if final and len(rows)==113:
        result=dict(instances=113,groups=len(selected),old_mean_iou=float(np.mean([r['old_iou'] for r in rows])),new_mean_iou=float(np.mean([r['new_iou'] for r in rows])),empty_instances=sum(r['status']=='empty_prediction' for r in rows),score_threshold=.5,limited_run=False)
        save(RUN/'completed.json',result);save(GALLERY/'completed.json',result);return True
    return False

def main():
    raw=SharedRaw(REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif');raw.column=lru_cache(maxsize=512)(lambda x,y:SharedRaw.column.__wrapped__(raw,x,y));labels=np.load(TRAIN/'source_cache/labels_2.npy',mmap_mode='r')
    while True:
        if publish_once(raw,labels):break
        if (RUN/'blocked.json').exists():break
        time.sleep(15)

HTML='''<!doctype html><meta charset="utf-8"><title>113例：检测框图→SAM Video</title><style>body{background:#0b1221;color:#edf3ff;font:15px system-ui;margin:16px}.bar{position:sticky;top:0;background:#172339;padding:12px;z-index:5}a{color:#80cfff}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.frame{position:relative;overflow:hidden}.raw{display:block;width:100%;height:auto}.mask,svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}input,select{margin:6px;font:inherit}#z{width:60%}p{line-height:1.6}</style><h2>113例：相邻层框组最大框prompt → SAM Video</h2><p>框IoU&gt;0.5或交集/较小框面积&gt;0.8，相邻Z传递合并。只推113目标匹配的64个唯一组；每组用面积最大检测框作prompt，零外扩；各组独立Video记忆，score&lt;0.5停止。GT用于选验证目标组及评估，不生成prompt、不修改预测。多个GT若匹配同组，会显示同一个预测，便于检查粘连。橙=旧版，绿=人工GT，洋红=新版。</p><div class="bar"><span id="status">准备中</span><br><label>GT实例<select id="instance"></select></label><br>Z<input id="z" type="range" min="0" max="0"><span id="zl"></span><button id="ref">跳到seed</button><br><label><input id="prompt" type="checkbox" checked>实际seed框</label><label><input id="project" type="checkbox">投影seed到其他层</label><label><input id="boxes" type="checkbox">本组本层检测框</label><label><input id="crop" type="checkbox">实际crop</label></div><p id="info"></p><div class="grid" id="grid"></div><p id="metrics"></p><a id="seedlink" target="_blank">查看本组seed原图及参数</a><details><summary>分组、GT匹配与停止记录</summary><pre id="details"></pre></details><script>
const e=id=>document.getElementById(id);let data=[],cur=null,group=null,cache=new Map(),selectionGeneration=0;
const titles=['Raw EM','人工GT','旧DET-SAM实例','框组SAM Video'];e('grid').innerHTML=titles.map((t,i)=>'<div><p>'+t+'</p><div class="frame"><img class="raw" id="raw'+i+'"><img class="mask" id="mask'+i+'"><svg id="svg'+i+'" preserveAspectRatio="none"></svg></div></div>').join('');
async function select(keepZ=null){const generation=++selectionGeneration;cur=data.find(r=>r.index===+e('instance').value);if(!cur)return;const target=cur;group=null;e('z').max=cur.frames.length-1;let zi=keepZ===null?cur.z_values.indexOf(cur.seed?.z):cur.z_values.indexOf(keepZ);e('z').value=zi<0?0:zi;if(cur.group_details){const file=cur.group_details;if(!cache.has(file))cache.set(file,await(await fetch(file)).json());if(generation!==selectionGeneration||cur.index!==target.index)return;group=cache.get(file)}e('details').textContent=JSON.stringify({case:cur,group:group},null,2);e('seedlink').href='group_'+String(cur.matched_group_id).padStart(6,'0')+'/';draw()}
function draw(){if(!cur)return;let f=cur.frames[+e('z').value],z=f.z;e('zl').textContent='Z '+z+' · '+f.kind+' · '+(+e('z').value+1)+'/'+cur.frames.length;e('info').textContent='GT '+cur.gt_id+' | 框组 '+cur.matched_group_id+' ('+cur.group_size+'框) | seed Z'+cur.seed?.z+' | 整体3D IoU '+cur.old_iou.toFixed(4)+' → '+cur.new_iou.toFixed(4)+' | 本层score '+(f.score===null||f.score===undefined?'未推理/低分未保留':f.score.toFixed(4))+' | '+cur.status;e('metrics').textContent=JSON.stringify({gt:f.gt_area,old:f.old_area,new:f.new_area,old_iou:f.old_iou,new_iou:f.new_iou,context_before:cur.context_before,context_after:cur.context_after,boundary_clipped:cur.clipped_source_boundary});
let [x0,y0,x1,y1]=cur.view_xy;function rect(b,c,label){return '<rect x="'+b[0]+'" y="'+b[1]+'" width="'+(b[2]-b[0])+'" height="'+(b[3]-b[1])+'" fill="none" stroke="'+c+'" stroke-width="2" vector-effect="non-scaling-stroke"/>'}
for(let i=0;i<4;i++){e('raw'+i).src=f.raw;e('mask'+i).src=i===0?'empty.png':i===1?f.gt:i===2?f.old:f.new;let s='';if(i===0||i===3){if(e('boxes').checked)for(let b of group?.boxes_by_z?.[String(z)]||[])s+=rect(b,'#65bcff');if(cur.seed&&e('prompt').checked&&(z===cur.seed.z||e('project').checked))s+=rect(cur.seed.box_xyxy,'#ffd050');let g=cur.crop_geometry;if(g&&e('crop').checked&&z>=g[2]&&z<g[2]+g[3])s+=rect([g[0],g[1],g[0]+cur.crop_side,g[1]+cur.crop_side],'#3ee2da')}e('svg'+i).setAttribute('viewBox',[x0,y0,x1-x0,y1-y0].join(' '));e('svg'+i).innerHTML=s}}
async function load(){try{let m=await(await fetch('manifest.json?'+Date.now())).json();e('status').textContent='框组推理 '+m.done_groups+'/'+m.selected_groups+'；可查看 '+m.ready_instances+'/113'+(m.inference_complete?' · 推理完成':' · 逐组更新');let old=cur?.index,z=cur?.frames[+e('z').value]?.z;data=m.instances;e('instance').innerHTML=data.map(r=>'<option value="'+r.index+'">GT '+r.gt_id+' · 组'+r.matched_group_id+' · '+r.old_iou.toFixed(3)+'→'+r.new_iou.toFixed(3)+'</option>').join('');if(data.length){let target=data.find(r=>r.index===old)||data.find(r=>r.gt_id===3154)||data[0];e('instance').value=target.index;if(!cur||cur.index!==target.index)await select(z);else cur=target}}catch(err){e('status').textContent='准备中 '+err.message}setTimeout(load,15000)}
e('instance').onchange=()=>select();e('z').oninput=draw;for(let id of ['prompt','project','boxes','crop'])e(id).onchange=draw;e('ref').onclick=()=>{let j=cur?.z_values.indexOf(cur.seed?.z);if(j>=0){e('z').value=j;draw()}else if(cur)window.open(e('seedlink').href,'_blank')};load();</script>'''
if __name__=='__main__':main()
