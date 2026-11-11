"""Optional native-coordinate original-vs-repaired viewer; no GT dependency."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from infer_block import save

def plane(mask,o,z,box):
    x,y,z0=o;x0,y0,x1,y1=box;out=np.zeros((y1-y0,x1-x0),bool)
    if not 0<=z-z0<len(mask):return out
    lx=max(x,x0);ly=max(y,y0);hx=min(x+mask.shape[2],x1);hy=min(y+mask.shape[1],y1)
    if hx>lx and hy>ly:out[ly-y0:hy-y0,lx-x0:hx-x0]=mask[z-z0,ly-y:hy-y,lx-x:hx-x]
    return out

def render_panel(raw,folder,panel,index):
    panel.mkdir(parents=True,exist_ok=True);dest=panel/f'instance_{index:03d}';dest.mkdir(exist_ok=True)
    old=np.load(folder/'existing_mask.npy',mmap_mode='r');new=np.load(folder/'mask.npy',mmap_mode='r');oo=json.loads((folder/'existing_origin.json').read_text());no=json.loads((folder/'mask_origin.json').read_text());meta=json.loads((folder/'completed.json').read_text())
    boxes=[[o[0],o[1],o[0]+m.shape[2],o[1]+m.shape[1]] for m,o in [(old,oo),(new,no)]]
    box=[max(0,min(b[0] for b in boxes)-16),max(0,min(b[1] for b in boxes)-16),min(int(raw.size[0]),max(b[2] for b in boxes)+16),min(int(raw.size[1]),max(b[3] for b in boxes)+16)]
    union=set()
    for m,o in [(old,oo),(new,no)]:union.update(o[2]+int(j) for j in np.flatnonzero(m.any((1,2))))
    lo=max(0,min(union)-2);hi=min(int(raw.size[2])-1,max(union)+2)
    while hi-lo+1<5 and (lo>0 or hi<int(raw.size[2])-1):
        if lo>0:lo-=1
        if hi-lo+1<5 and hi<int(raw.size[2])-1:hi+=1
    files=[];scores={}
    for i,a in enumerate(meta.get('repair_actions',[])):
        t=json.loads((folder/f'action_{i:03d}/track.json').read_text())
        for f in t['evaluated_scores']:scores[f['z']]=max(scores.get(f['z'],-1),f['score'])
    for z in range(lo,hi+1):
        x0,y0,x1,y1=box;em,_=raw.slice_crop(z,x0,y0,max(x1-x0,y1-y0));em=em[:y1-y0,:x1-x0];im=Image.new('RGB',(1536,570),'#111827');draw=ImageDraw.Draw(im)
        for k,(mask,title,color) in enumerate([(None,'Raw EM',None),(plane(old,oo,z,box),'Original',(255,170,35)),(plane(new,no,z,box),'Repaired: union ->26CC',(240,55,165))]):
            rgb=np.repeat(em[...,None],3,2).astype(float)
            if mask is not None:rgb[mask]=.5*rgb[mask]+.5*np.array(color)
            pic=Image.fromarray(rgb.astype('uint8'));pic.thumbnail((512,512));px=k*512+(512-pic.width)//2;py=58+(512-pic.height)//2;im.paste(pic,(px,py));draw.text((k*512+5,8),title,fill='white')
            scale=pic.width/(x1-x0)
            for a in meta.get('repair_actions',[]):
                if a['seed_z']!=z:continue
                b=a['prompt']['box_xyxy'];draw.rectangle([px+(b[0]-x0)*scale,py+(b[1]-y0)*scale,px+(b[2]-x0)*scale,py+(b[3]-y0)*scale],outline='#ffd050',width=2)
        draw.text((5,30),f'Instance {index} Z{z} '+('UNION' if z in union else 'CONTEXT')+f' | max evaluated route score: {scores.get(z)}',fill='white');file=dest/f'z{z:04d}.jpg';im.save(file,quality=89);files.append(str(file.relative_to(panel)))
    save(panel/f'case_{index:03d}.json',dict(index=index,z_values=list(range(lo,hi+1)),files=files,connectivity=meta.get('component_selection'),view_xy=box))
    rows=[json.loads(p.read_text()) for p in sorted(panel.glob('case_*.json'))];save(panel/'manifest.json',rows);(panel/'index.html').write_text(HTML)

HTML='''<!doctype html><meta charset="utf-8"><title>Mito repair v1</title><style>body{background:#111827;color:white;font:16px system-ui}img{width:100%}input{width:65%}</style><h2>Mito repair v1: original vs repaired</h2><p>Yellow boxes: actual zero-padding tight prompt, shown only on its seed slice. Score is max evaluated route score, not confidence of the OR mask. Z=mask union plus2 context.</p><select id="i"></select><input id="z" type="range"><span id="t"></span><img id="p"><pre id="a"></pre><script>let rows=[],cur;const e=id=>document.getElementById(id);function select(){cur=rows[+e('i').value];e('z').min=0;e('z').max=cur.files.length-1;e('z').value=0;e('a').textContent=JSON.stringify(cur.connectivity,null,2);draw()}function draw(){let j=+e('z').value;e('p').src=cur.files[j];e('t').textContent='Z '+cur.z_values[j]}fetch('manifest.json').then(r=>r.json()).then(r=>{rows=r;e('i').innerHTML=r.map((x,i)=>'<option value="'+i+'">Instance '+x.index+'</option>').join('');select()});e('i').onchange=select;e('z').oninput=draw;</script>'''
