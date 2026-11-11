"""Read-only defect screen over the entire original instance dataset."""
import argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage as ndi


def longest_run(a):
    if not a.any():return 0
    edges=np.diff(np.r_[False,a,False].astype(np.int8));return int((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).max())


def inspect(root,row):
    a=np.load(root/row['mask_path'],mmap_mode='r');bb=json.loads(row['bbox_xyz'])
    occupied=np.any(a,axis=(1,2));zz=np.flatnonzero(occupied);span=int(zz[-1]-zz[0]+1)
    hole_slices=[];straight_slices=[];seam_slices=[];total_hole=0;largest_line=0
    for z in zz:
        m=np.asarray(a[z],bool);fg=int(m.sum())
        h=ndi.binary_fill_holes(m)&~m;lab,n=ndi.label(h);sizes=np.bincount(lab.ravel())[1:];nlarge=int((sizes>=4).sum());hv=int(h.sum())
        if nlarge>=3 and hv>=max(12,.02*fg):hole_slices.append(int(z))
        total_hole+=hv
        # Exposed horizontal/vertical boundaries, including internal straight cuts.
        pad=np.pad(m,1);horizontal=m&(~pad[:-2,1:-1]|~pad[2:,1:-1]);vertical=m&(~pad[1:-1,:-2]|~pad[1:-1,2:])
        maxline=0;seam=False
        for y in np.flatnonzero(horizontal.sum(1)>=12):
            run=longest_run(horizontal[y]);maxline=max(maxline,run)
            if run>=12 and ((bb[1]+int(y))%128 in (0,127)):seam=True
        for x in np.flatnonzero(vertical.sum(0)>=12):
            run=longest_run(vertical[:,x]);maxline=max(maxline,run)
            if run>=12 and ((bb[0]+int(x))%128 in (0,127)):seam=True
        largest_line=max(largest_line,maxline)
        if maxline>=max(16,int(.25*min(m.shape))):straight_slices.append(int(z))
        if seam:seam_slices.append(int(z))
    return {**row,'bbox_xyz':bb,'z_span':span,'occupied_z':len(zz),'hole_slices':hole_slices,'straight_slices':straight_slices,'seam_slices':seam_slices,'hole_voxels_2d':total_hole,'largest_straight_run_px':largest_line,'short':span<10,'holes':len(hole_slices)>=2,'straight':len(straight_slices)>=2}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--per-group',type=int,default=4);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader((args.data/'metadata/instances.csv').open()));results=[];errors=[];t=time.time()
    def task(row):
        try:return inspect(args.data,row)
        except Exception as e:return {'record_id':row['record_id'],'error':repr(e)}
    with ThreadPoolExecutor(args.workers) as pool,(args.out/'audit.jsonl').open('w') as f:
        for i,r in enumerate(pool.map(task,rows),1):
            f.write(json.dumps(r)+'\n');f.flush()
            (errors if 'error' in r else results).append(r)
            if i%200==0:
                progress={'processed':i,'total':len(rows),'seconds':time.time()-t};(args.out/'audit_progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    # Bounded first pilot; raw reads and inference never modify original masks.
    selected=[];seen=set();bodies={}
    for group in ('short','holes','straight'):
        candidates=[r for r in results if r[group] and max(r['bbox_xyz'][3]-r['bbox_xyz'][0],r['bbox_xyz'][4]-r['bbox_xyz'][1])<=300 and r['z_span']<=64]
        def priority(r):
            severity={'short':-r['z_span'],'holes':len(r['hole_slices']),'straight':len(r['seam_slices'])*10+len(r['straight_slices'])}[group]
            return (-severity,r['record_id'])
        count=0
        for r in sorted(candidates,key=priority):
            if r['record_id'] in seen or bodies.get(r['body_id'],0)>=2:continue
            selected.append({**r,'selected_group':group});seen.add(r['record_id']);bodies[r['body_id']]=bodies.get(r['body_id'],0)+1;count+=1
            if count>=args.per_group:break
    (args.out/'selected.json').write_text(json.dumps(selected,indent=2))
    summary={'total':len(rows),'inspected':len(results),'errors':errors,'flag_counts':{k:sum(bool(r[k]) for r in results) for k in ('short','holes','straight')},'pilot_selected':len(selected),'per_group_requested':args.per_group,'definitions':{'short':'occupied Z span <10','holes':'>=2 slices each with >=3 enclosed 2D holes >=4 pixels and total hole pixels >=max(12,2% foreground)','straight':'>=2 slices with axis-aligned exposed boundary run >=max(16px,25% minimum XY side); heuristic, not proof of stitch failure','seam':'straight run>=12px on native 128-grid boundary; extra priority only'},'seconds':time.time()-t}
    (args.out/'audit_completed.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
