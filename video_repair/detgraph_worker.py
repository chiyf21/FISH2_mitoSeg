"""One largest-box prompt per detection graph component; no old-mask union/GT."""
import argparse,json,time,os,urllib.request,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT,create_local
from repair_three_prompt import shared_geometry
from repair_directional_routes import compact,frame_meta
from repair_video113 import frame_box_iou
from infer_edge_shift import rerun_track_adaptive
from infer_block import save
from scipy import ndimage as ndi
import numpy as np
import torch
REF=ROOT.parent/'sam_video_hardcases_20261102';RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103'

def select_seed_component(mask,origin,seed):
    labels,n=ndi.label(mask,structure=np.ones((3,3,3),bool));j=seed['z']-origin[2]
    if not 0<=j<len(mask) or not mask[j].any():return np.zeros((1,1,1),bool),origin,dict(reason='no_retained_mask_at_prompt_frame',raw_components=int(n))
    plane,k=ndi.label(mask[j],structure=np.ones((3,3),bool));options=[]
    for q in range(1,k+1):
        p=plane==q;overlap=frame_box_iou(p,origin[:2],seed['box_xyxy']);cid=int(labels[j][p][0]);options.append((overlap,int(p.sum()),-q,cid))
    best=max(options)
    if best[0]<=0:return np.zeros((1,1,1),bool),origin,dict(reason='prompt_does_not_overlap_retained_mask',raw_components=int(n))
    keep=labels==best[3];z,y,x=np.nonzero(keep);o=[origin[0]+int(x.min()),origin[1]+int(y.min()),origin[2]+int(z.min())];m=keep[z.min():z.max()+1,y.min():y.max()+1,x.min():x.max()+1]
    return m,o,dict(selected_component=best[3],seed_2d_mask_prompt_iou=best[0],raw_components=int(n),rule='seed2D component maxIoU with original detector prompt -> its26CC; noGT/no extra size filter')

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--port',type=int,default=18089);a=p.parse_args();torch.cuda.set_per_process_memory_fraction(.25);token=json.loads((RUN/'queue/queue_access.json').read_text())['token'];assets=ROOT/'baseline/assets';started=time.perf_counter();e=create_local(REF.parent.parent/'flyEM/2500x2500x250/raw.ome.tif',REF/'instance_centered_20261102/nucleus.npy',assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True);assert e.score_threshold==.5;post=ThreadPoolExecutor(max_workers=2);pending=[]
    source=json.loads((RUN/'detections.json').read_text());assert source['source_tag']==e.raw.source_tag
    def rpc(path,**data):
        req=urllib.request.Request(f'http://127.0.0.1:{a.port}'+path,data=json.dumps(dict(token=token,worker=a.worker,**data)).encode())
        with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
    def finish(g,track,hist,seconds):
        out=RUN/'predictions'/f'group_{g["index"]:06d}';out.mkdir(parents=True,exist_ok=True);seed=g['seed'];m,o=compact(track['frames'],track['geometry'][:2],seed['z']);t=time.perf_counter();m,o,selection=select_seed_component(m,o,seed);cc_time=time.perf_counter()-t
        tmp=out/'mask.partial.npy';np.save(tmp,m);tmp.replace(out/'mask.npy');save(out/'origin.json',o)
        info={k:v for k,v in track.items() if k not in ('frames','routes')};info.update(frames=[frame_meta(f) for f in track['frames']],crop_history=hist);save(out/'track.json',info)
        scores={int(f['z']):float(f['score']) for f in track['frames']};nz=np.flatnonzero(m.any((1,2)));summary=dict(group_id=g['index'],seed=seed,member_count=g['member_count'],member_ids=g['member_ids'],stack_z=g['z_values'],mask_origin=o,mask_shape_zyx=list(m.shape),bbox_xyz=[*o,o[0]+m.shape[2],o[1]+m.shape[1],o[2]+m.shape[0]],voxels=int(m.sum()),nonempty_z=[o[2]+int(j) for j in nz],frame_scores=scores,crop_geometry=track['geometry'],final_side=track['crop_side'],initial_crop=hist[0],crop_attempts=len(hist),stop_events=track['stop_events'],stop_reasons=track['direction_end_reasons'],component_selection=selection,score_threshold=.5,prompt_padding=0,inference_seconds=seconds,cc_seconds=cc_time,limited_run=False)
        save(out/'completed.json',summary);rpc('/complete',index=g['index'],result=dict(group_id=g['index'],voxels=summary['voxels'],seconds=seconds));return summary
    while True:
        g=rpc('/claim')['job']
        if g is None:break
        out=RUN/'predictions'/f'group_{g["index"]:06d}'
        if (out/'completed.json').exists():rpc('/complete',index=g['index'],result=dict(reused_existing=True));continue
        try:
            seed={**g['seed'],'roles':['maxclean']};det={**seed,'anchors':[seed],'all_prompt_boxes':[seed['box_xyxy']],'directional_routes':True,'route_specs':[('maxclean',['forward','backward'])]};geometry,side=shared_geometry([seed],int(e.raw.size[2]));save(RUN/f'worker_{a.worker}_progress.json',dict(group=g['index'],seed_z=seed['z'],stack_boxes=g['member_count'],initial_side=side,pid=os.getpid(),time=time.time()))
            t=time.perf_counter();track,hist=rerun_track_adaptive(e,det,geometry,[0,0,0,*map(int,e.raw.size)],initial_side=side);pending.append(post.submit(finish,g,track,hist,time.perf_counter()-t))
            for f in pending:
                if f.done():f.result()
            while len([f for f in pending if not f.done()])>=2:pending[0].result();pending=[f for f in pending if not f.done()]
        except Exception as exc:rpc('/fail',index=g['index'],result=dict(error=repr(exc),trace=traceback.format_exc()));raise
    [f.result() for f in pending];post.shutdown(wait=True);e.fast_encoder.io.close();save(RUN/f'worker_{a.worker}_finished.json',dict(pid=os.getpid(),seconds=time.perf_counter()-started,fast_stats=dict(e.fast_stats),limited_run=False))
if __name__=='__main__':main()
