"""Native128x128x32 detector,10-layer windows stride3; same canonical heads."""
import argparse,json,time,sys,itertools
from pathlib import Path
from common import ROOT,config,gpu,guard,save,signature,roi_for
from block_raw import BlockRaw
import numpy as np

def run(d):
    out=guard(Path(d['output'])/'detection',d,'detector')
    if (out/'completed.json').exists():return
    gpu(d)
    import torch
    sys.path.insert(0,str(ROOT/'vendor/detector'))
    from models.model_segdet3d import OrganelleSegDet3D,det_cfg_from_state,_slice_detections
    from models.box_ops import link_slice_boxes,nms_3d
    raw=BlockRaw(d['raw'],d.get('raw_cache_mib',512));roi=roi_for(raw,d);weight=d['models']['detector'];ck=torch.load(weight,map_location='cpu',weights_only=False);state={k.removeprefix('module.'):v for k,v in ck['model'].items()};m=OrganelleSegDet3D(mode='det_sam',dim='3d',**det_cfg_from_state(state));m.load_state_dict(state,strict=True);m.cuda().eval();del ck,state
    # Native neighbour tiles overlap through reflect padding; no boxes are
    # discarded at tile borders. Global same-Z NMS happens before Video seeding.
    keys=list(itertools.product(range(roi[0]//128*128,roi[3],128),range(roi[1]//128*128,roi[4],128),range(roi[2]//32*32,roi[5],32)));allrows=[];started=time.time();windows=list(range(0,23,3));windows=sorted(set(windows+[22]))
    with torch.inference_mode():
        for ordinal,key in enumerate(keys):
            tilefile=out/'tiles'/f'x{key[0]}_y{key[1]}_z{key[2]}.json'
            if tilefile.exists():rows=json.loads(tilefile.read_text())
            else:
                aa=raw.tile(key).transpose(2,1,0);aa=np.pad(aa,((0,0),(30,30),(30,30)),mode='reflect');inp=np.stack([aa[z:z+10] for z in windows]).astype(np.float32);inp=np.clip((inp-16)/228,0,1);inp=(inp-inp.mean((1,2,3),keepdims=True))/(inp.std((1,2,3),keepdims=True)+1e-6);rows=[]
                for st in range(0,len(windows),9):
                    with torch.autocast('cuda',dtype=torch.bfloat16):f=m.neck(m.backbone(torch.from_numpy(inp[st:st+9,None]).cuda()));det=m.det_heads[m.instance_ids.index(2)](list(f))
                    boxes=det['boxes'];scores=det['logits'].float().softmax(-1)[...,1]
                    for bi in range(len(boxes)):
                        ids=_slice_detections(boxes[bi],scores[bi]);sb=boxes[bi][ids].detach().float().cpu();ss=scores[bi][ids].detach().float().cpu();b3,s3,_,tracks=link_slice_boxes(sb,ss,ss[:,None]);kept=nms_3d(b3,s3,.6)[:12]
                        for ti in kept.tolist():
                            if s3[ti]<=.3:continue
                            for member in tracks[ti]:
                                b=sb[member].numpy();z=int(round(float(b[0])))+key[2]+windows[st+bi];box=[float(b[2]+key[0]-30),float(b[1]+key[1]-30),float(b[5]+key[0]-30),float(b[4]+key[1]-30)]
                                if roi[2]<=z<roi[5]:rows.append(dict(z=z,box_xyxy=box,slice_score=float(ss[member]),linked_score=float(s3[ti]),tile_xyz=list(key),window_z=key[2]+windows[st+bi]))
                save(tilefile,rows)
            allrows.extend(rows);save(out/'progress.json',dict(tiles_done=ordinal+1,total_tiles=len(keys),boxes=len(allrows)));print('detection tiles',ordinal+1,'/',len(keys),flush=True)
    # Exact duplicate removal only. Global NMS uses every window box before
    # centre-in-ROI seed selection, including boxes centred outside ROI.
    unique={}
    for row in allrows:
        key=(row['z'],*row['box_xyxy'])
        if key not in unique or row['slice_score']>unique[key]['slice_score']:unique[key]=row
    rows=sorted(unique.values(),key=lambda r:(r['z'],*r['box_xyxy']))
    for i,b in enumerate(rows):b['global_detection_id']=i;b['detection_id']=i
    save(out/'detections.json',dict(source_tag=raw.source_tag,detector=signature(weight),bbox_xyz=roi,detections=rows))
    save(out/'black_plane_interpolation.json',list(raw.repair_log.values()));save(out/'completed.json',dict(boxes=len(rows),tiles=len(keys),seconds=time.time()-started,limited_run=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);run(config(p.parse_args().config))
