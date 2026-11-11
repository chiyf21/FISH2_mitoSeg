"""Rerun selected GT-associated instances with crop shifts on mask/crop contact.

Detector boxes are kept fixed. The inference window moves and grows as needed.
Each shifted crop is re-encoded and the same detector box is re-applied at its
seed Z; the resulting tracks are stitched with the unchanged stitch_core rules.
"""
import gc, json, math, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from infer_block import Engines, save
from stitch_core import assemble, paste

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
EDGE_DISTANCE_THRESHOLD = 100
TARGET_EDGE_DISTANCE = 101
MAX_SHIFTS = 24
XY = 256
DEPTH = 32


def edge_profile(track):
    profiles = []
    for f in track["frames"]:
        m = f["mask"]
        e = {
            "left": int(m[:, 0].sum()),
            "right": int(m[:, -1].sum()),
            "top": int(m[0, :].sum()),
            "bottom": int(m[-1, :].sum()),
        }
        yy, xx = np.nonzero(m)
        if len(xx):
            distances = {"left": int(xx.min()), "right": int(m.shape[1]-1-xx.max()),
                         "top": int(yy.min()), "bottom": int(m.shape[0]-1-yy.max())}
            nearest_side = min(distances, key=lambda k: (distances[k], k))
            nearest = distances[nearest_side]
        else:
            distances = {"left": None, "right": None, "top": None, "bottom": None}
            nearest_side = None; nearest = None
        profiles.append({"z": int(f["z"]), "edge_pixels": e,
                         "distance_to_each_edge_px": distances,
                         "nearest_edge": nearest_side, "nearest_distance_px": nearest})
    maxima = {k: max((p["edge_pixels"][k] for p in profiles), default=0)
              for k in ("left", "right", "top", "bottom")}
    under = [{"z": p["z"], "side": p["nearest_edge"], "distance_px": p["nearest_distance_px"]}
             for p in profiles if p["nearest_distance_px"] is not None
             and p["nearest_distance_px"] <= EDGE_DISTANCE_THRESHOLD]
    return profiles, maxima, under


def constrain_origin(x, y, box, bbox):
    bx0, by0, _, bx1, by1, _ = bbox
    xlo = max(bx0, int(math.ceil(box[2] - XY)))
    xhi = min(bx1 - XY, int(math.floor(box[0])))
    ylo = max(by0, int(math.ceil(box[3] - XY)))
    yhi = min(by1 - XY, int(math.floor(box[1])))
    if xlo > xhi or ylo > yhi:
        raise ValueError(f"prompt box cannot fit in block: {box}, {bbox}")
    return min(max(int(x), xlo), xhi), min(max(int(y), ylo), yhi)


def shift_direction(side, distance_px, x, y, box, bbox):
    # A mask at the left/top edge is brought into the crop by moving the crop
    # left/up; a right/bottom edge is handled by moving it right/down.
    bx0, by0, _, bx1, by1, _ = bbox
    xlo = max(bx0, int(math.ceil(box[2] - XY)))
    xhi = min(bx1 - XY, int(math.floor(box[0])))
    ylo = max(by0, int(math.ceil(box[3] - XY)))
    yhi = min(by1 - XY, int(math.floor(box[1])))
    amount = TARGET_EDGE_DISTANCE - int(distance_px)
    delta = {"left": (-amount, 0), "right": (amount, 0),
             "top": (0, -amount), "bottom": (0, amount)}
    dx, dy = delta[side]
    nx, ny = constrain_origin(x + dx, y + dy, box, bbox)
    if (nx, ny) == (x, y):
        return None
    return side, nx, ny


def rerun_track(engine, det, geometry, bbox):
    x, y, z, n = geometry
    history = []
    visited = {}
    best_track = None
    best_key = None
    for attempt in range(MAX_SHIFTS + 1):
        geom = (x, y, z, n)
        encoded = engine.encode_clip(geom, XY)
        track = engine.track(encoded, geom, XY, det)
        profiles, maxima, over = edge_profile(track)
        distances=[p['nearest_distance_px'] for p in profiles if p['nearest_distance_px'] is not None]
        key = (len(over), -min(distances, default=XY), -float(np.mean(distances)) if distances else -XY,
               -sum(f["score"] for f in track["frames"]))
        visited[(x, y)] = (track, key)
        if best_key is None or key < best_key:
            best_key, best_track = key, track
        entry = {"attempt": attempt, "geometry": list(geom),
                 "edge_pixels_by_side_max_over_frames": maxima,
                 "per_frame_edge_pixels": profiles, "frames_nearest_edge_under_100": over}
        if not over:
            entry["action"] = "accepted_no_edge_contact"
            history.append(entry)
            return track, history
        if attempt == MAX_SHIFTS:
            entry["action"] = "shift_limit_reached"
            history.append(entry)
            return best_track, history
        # A blocked/cyclic request from one frame must not prevent the other
        # frames and directions from being tried. Use real source bounds.
        source_bounds = [0, 0, 0, int(engine.raw.size[0]), int(engine.raw.size[1]), int(engine.raw.size[2])]
        candidates = []
        for p in profiles:
            for side, distance in p['distance_to_each_edge_px'].items():
                if distance is not None and distance <= EDGE_DISTANCE_THRESHOLD:
                    candidates.append({'z':p['z'], 'side':side, 'distance_px':distance})
        move = None
        for trigger in sorted(candidates, key=lambda q:(q['distance_px'], q['z'], q['side'])):
            proposed = shift_direction(trigger['side'], trigger['distance_px'], x, y,
                                       det['box_xyxy'], source_bounds)
            if proposed is not None and tuple(proposed[1:]) not in visited:
                move = proposed
                break
        if move is None:
            entry['action'] = 'all_frame_direction_candidates_exhausted_or_bound'
            history.append(entry)
            return best_track, history
        side, nx, ny = move
        entry["action"] = "shift_directly_to_over_100px_margin_and_rerun"
        entry["shift_side"] = side
        entry["trigger_frame_z"] = trigger["z"]
        entry["trigger_distance_px"] = trigger["distance_px"]
        entry["requested_margin_px"] = TARGET_EDGE_DISTANCE
        entry["next_origin_xy"] = [nx, ny]
        history.append(entry)
        x, y = nx, ny
    return best_track, history


def rerun_track_adaptive(engine, det, geometry, bbox, initial_side=None):
    x,y,z,n=geometry;side=int(initial_side or XY);history=[];visited=set()
    width,height=map(int,engine.raw.size[:2])
    max_side=int(math.ceil((max(width,height)+204)/32)*32)
    for attempt in range(32):
        geom=(x,y,z,n); visited.add((x,y,side))
        encoded=engine.encode_clip(geom,side)
        track=engine.track(encoded,geom,side,det);track['crop_side']=side
        # Outside-volume padding is context, never a predicted biological voxel.
        for f in track['frames']:
            valid=np.zeros_like(f['mask'])
            lx=max(0,-x);ly=max(0,-y);hx=min(side,width-x);hy=min(side,height-y)
            if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
            f['mask'] &= valid;f['probability'][~valid]=0
        profiles,maxima,bad=edge_profile(track)
        entry={'attempt':attempt,'geometry':list(geom),'crop_side':side,
               'edge_pixels_by_side_max_over_frames':maxima,
               'per_frame_edge_pixels':profiles,'frames_nearest_edge_under_100':bad}
        if not bad:
            entry['action']='accepted_all_frames_margin_gt100' if any(f['mask'].any() for f in track['frames']) else 'no_nonempty_retained_mask_margin_not_evaluable';history.append(entry)
            return track,history
        extents=[]
        for f in track['frames']:
            yy,xx=np.nonzero(f['mask'])
            if len(xx):extents.append((x+int(xx.min()),y+int(yy.min()),x+int(xx.max()),y+int(yy.max())))
        boxes=det.get('all_prompt_boxes',[det['box_xyxy']])
        b=[min(q[0] for q in boxes),min(q[1] for q in boxes),max(q[2] for q in boxes),max(q[3] for q in boxes)]
        left=min([e[0] for e in extents]+[math.floor(b[0])]);right=max([e[2] for e in extents]+[math.ceil(b[2])])
        top=min([e[1] for e in extents]+[math.floor(b[1])]);bottom=max([e[3] for e in extents]+[math.ceil(b[3])])
        required=max(right-left+1,bottom-top+1)+2*TARGET_EDGE_DISTANCE
        new_side=min(max_side,max(side,int(math.ceil(required/32)*32)))
        nx=int(math.floor((left+right-(new_side-1))/2));ny=int(math.floor((top+bottom-(new_side-1))/2))
        if (nx,ny,new_side) in visited:
            new_side=min(max_side,new_side+128)
            nx=int(math.floor((left+right-(new_side-1))/2));ny=int(math.floor((top+bottom-(new_side-1))/2))
        if new_side==max_side:
            nx=(width-new_side)//2;ny=(height-new_side)//2
        if (nx,ny,new_side) in visited:
            entry['action']='cannot_improve_at_full_source_context';history.append(entry)
            return track,history
        entry.update(action='adaptive_shift_resize_and_rerun',next_origin_xy=[nx,ny],next_crop_side=new_side,requested_margin_px=101)
        history.append(entry);x,y,side=nx,ny,new_side
        del encoded
    raise RuntimeError('Adaptive margin failed to converge; no success marker written')


def materialize(out, tracks, bbox, provenance, selected):
    objects, audit = assemble(tracks, .8)
    save(out / "stitch_audit.json", audit)
    x0, y0, z0, x1, y1, z1 = bbox
    shape = (z1-z0, y1-y0, x1-x0)
    union = np.lib.format.open_memmap(out/"predicted_union.npy", mode="w+", dtype=bool, shape=shape); union[:] = False
    labels = np.lib.format.open_memmap(out/"instance_labels.npy", mode="w+", dtype=np.uint32, shape=shape); labels[:] = 0
    scores = np.lib.format.open_memmap(out/"instance_winner_score.npy", mode="w+", dtype=np.float32, shape=shape); scores[:] = -np.inf
    prob = np.lib.format.open_memmap(out/"predicted_probability.npy", mode="w+", dtype=np.float32, shape=shape); prob[:] = 0
    object_info = []
    for oi, (root, frames) in enumerate(sorted(objects.items()), 1):
        points=[]
        for f in frames:
            yy,xx=np.nonzero(f['mask'])
            if len(xx):points.append((f['origin_xy'][0]+int(xx.min()),f['origin_xy'][1]+int(yy.min()),f['origin_xy'][0]+int(xx.max())+1,f['origin_xy'][1]+int(yy.max())+1))
        if not points: continue
        ox0=max(x0,min(p[0] for p in points)); oy0=max(y0,min(p[1] for p in points))
        ox1=min(x1,max(p[2] for p in points)); oy1=min(y1,max(p[3] for p in points))
        if ox1<=ox0 or oy1<=oy0:continue
        zz=sorted({f['z'] for f in frames if z0<=f['z']<z1}); arrays=[]
        for z in zz:
            a=np.zeros((oy1-oy0,ox1-ox0),bool)
            for f in frames:
                if f['z']==z: a|=paste(f,[ox0,oy0,z0,ox1,oy1,z1])
            arrays.append(a); union[z-z0,oy0-y0:oy1-y0,ox0-x0:ox1-x0]|=a
        if not arrays: continue
        for f in frames:
            if not z0<=f['z']<z1:continue
            mask=paste(f,bbox); zi=f['z']-z0; take=mask&(float(f['score'])>scores[zi])
            labels[zi][take]=oi; scores[zi][take]=float(f['score'])
        array=np.stack(arrays); fn=f'object_{oi:04d}.npz'
        np.savez_compressed(out/fn,z_global=np.array(zz),masks=array,origin_xy=np.array([ox0,oy0]))
        object_info.append({'object_id':oi,'root_track':root,'z_global':zz,'voxels':int(array.sum()),'selected_frames':[{k:v for k,v in f.items() if k not in ('mask','probability')} for f in frames],'file':fn})
    for frames in objects.values():
        for f in frames:
            if not z0<=f['z']<z1:continue
            fx,fy=f['origin_xy']; pp=f['probability']; lx=max(x0,fx);ly=max(y0,fy);hx=min(x1,fx+pp.shape[1]);hy=min(y1,fy+pp.shape[0])
            if hx>lx and hy>ly:
                view=prob[f['z']-z0,ly-y0:hy-y0,lx-x0:hx-x0];np.maximum(view,pp[ly-fy:hy-fy,lx-fx:hx-fx],out=view)
    union.flush();labels.flush();scores.flush();prob.flush()
    assert np.array_equal(union, prob>=.5)
    assert np.array_equal(union, labels>0)
    save(out/'track_metadata.json',[{'track_index':i,'geometry':t['geometry'],'crop_side':t.get('crop_side',XY),'seed_detection':t['seed_detection'],'frames':[{**{k:v for k,v in f.items() if k not in ('mask','probability')},'mask_area':int(f['mask'].sum())} for f in t['frames']]} for i,t in enumerate(tracks)])
    metadata={'bbox_xyz':bbox,'xy':XY,'depth':DEPTH,'xy_stride':128,'z_stride':16,'mask_iou_threshold':.8,
              'selected_detections':len(selected),'clips':len(tracks),'tracks':len(tracks),'objects':object_info,
              'crop_policy':'rerun same fixed detection box after shifting a crop away from mask-contacted edges',**provenance}
    save(out/'completed.json',metadata)


def main():
    torch.set_num_threads(3)
    engine=Engines()
    aggregate=[]
    for idx in (3,113):
        base=CASES/f'instance_{idx:03d}'/'video_repaired_overlap'
        out=CASES/f'instance_{idx:03d}'/'video_edge_shift_v10'
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        source=json.loads((base/'completed.json').read_text()); bbox=source['bbox_xyz']
        selected=json.loads((base/'selected_detections.json').read_text())['detections']
        oldtracks=json.loads((base/'track_metadata.json').read_text())
        bykey={}
        for t in oldtracks:
            d=t['seed_detection']; key=(int(d['detection_id']),tuple(t['geometry']))
            bykey[key]=d
        # Preserve every baseline detection/clip combination, but rerun with
        # an adaptive crop origin.  Original detector boxes are never altered.
        jobs=[]
        for t in oldtracks:
            d=t['seed_detection']; jobs.append((d,tuple(t['geometry'])))
        tracks=[]; audit=[]
        for n,(det,geom) in enumerate(jobs,1):
            tr,hist=rerun_track_adaptive(engine,det,geom,bbox);tracks.append(tr)
            final_profile,final_maxima,final_over=edge_profile(tr)
            audit.append({'track':n-1,'detection_id':int(det['detection_id']),'seed_z':int(det['z']),
                          'fixed_prompt_box':det['box_xyxy'],'history':hist,
                          'final_geometry':tr['geometry'],'final_crop_side':tr.get('crop_side',XY),'final_per_frame_edge_pixels':final_profile,
                          'final_edge_pixels_by_side_max_over_frames':final_maxima,
                          'final_frames_nearest_edge_under_100':final_over})
            save(out/'progress.json',{'instance_index':idx,'done_tracks':n,'total_tracks':len(jobs),'last':audit[-1]})
            print(json.dumps({'instance':idx,'track':n,'total':len(jobs),'det':det['detection_id'],'moves':len(hist)-1,'geometry':tr['geometry'],'crop_side':tr.get('crop_side',XY),'remaining_bad_frames':len(final_over)}),flush=True)
            gc.collect()
        save(out/'edge_shift_audit.json',audit)
        # Output FOV must include the enlarged crops; clipping back to the old
        # pilot block would create a new artificial straight mask boundary.
        lx=min([bbox[0]]+[t['geometry'][0] for t in tracks]);ly=min([bbox[1]]+[t['geometry'][1] for t in tracks])
        hx=max([bbox[3]]+[t['geometry'][0]+t.get('crop_side',XY) for t in tracks]);hy=max([bbox[4]]+[t['geometry'][1]+t.get('crop_side',XY) for t in tracks])
        w,h=map(int,engine.raw.size[:2]);lx=max(0,lx);ly=max(0,ly);hx=min(w,hx);hy=min(h,hy)
        side=max(hx-lx,hy-ly);lx=max(0,min(lx,w-side));ly=max(0,min(ly,h-side))
        expanded_bbox=[lx,ly,bbox[2],lx+side,ly+side,bbox[5]]
        materialize(out,tracks,expanded_bbox,engine.provenance,selected)
        original=np.load(base/'predicted_union.npy',mmap_mode='r'); shifted=np.load(out/'predicted_union.npy',mmap_mode='r')
        aggregate.append({'index':idx,'gt_id':2891 if idx==3 else 10796,'baseline_voxels':int(original.sum()),'shifted_voxels':int(shifted.sum()),'baseline_output':str(base),'shifted_output':str(out),'tracks':len(tracks),'shifted_tracks':sum(any(a.get('action')=='adaptive_shift_resize_and_rerun' for a in row['history']) for row in audit)})
        save(out/'completed.json',{**json.loads((out/'completed.json').read_text()),'limited_run':False,'baseline_output':str(base),'edge_shift_audit':'edge_shift_audit.json'})
    save(ROOT/'edge_shift_two_instances_v10_completed.json',{'completed':True,'limited_run':False,'method':'per-frame nearest-mask-to-crop-edge distance: if <=100 px, translate crop in one calculated move so the nearest pixel is >100 px (target 101 px), then rerun with original detector box; recheck every frame; stop at geometric bound, shift limit, or cycle while retaining best-seen crop; stitching unchanged','instances':aggregate})
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()



