"""Native XY quality candidates; no GT and no edits to existing masks."""
import numpy as np
from scipy import ndimage as ndi
def longest_run(a):
    d=np.diff(np.r_[False,a,False].astype(np.int8));starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
    return int((ends-starts).max()) if len(starts) else 0

def reference_line_check(mask,origin_xy=(0,0),threshold=100):
    p=np.asarray(mask,dtype=bool);yy,xx=np.nonzero(p)
    if not len(xx):return {'accepted':False,'reason':'empty_mask'}
    xmin,xmax=int(xx.min()),int(xx.max());ymin,ymax=int(yy.min()),int(yy.max());tight=p[ymin:ymax+1,xmin:xmax+1];x0,y0=map(int,origin_xy)
    edge_runs={'top':longest_run(tight[0]),'bottom':longest_run(tight[-1]),'left':longest_run(tight[:,0]),'right':longest_run(tight[:,-1])}
    up=p.copy();up[1:]&=~p[:-1];down=p.copy();down[:-1]&=~p[1:];left=p.copy();left[:,1:]&=~p[:,:-1];right=p.copy();right[:,:-1]&=~p[:,1:]
    lines=[]
    for side,a in [('top',up),('bottom',down),('left',left.T),('right',right.T)]:
        for coordinate in np.flatnonzero(a.sum(axis=1)>threshold):
            row=a[coordinate];d=np.diff(np.r_[False,row,False].astype(np.int8));starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
            for lo,hi in zip(starts,ends):
                if hi-lo>threshold:lines.append({'side':side,'coordinate':int(coordinate)+(y0 if side in ('top','bottom') else x0),'start':int(lo)+(x0 if side in ('top','bottom') else y0),'end_exclusive':int(hi)+(x0 if side in ('top','bottom') else y0),'length_px':int(hi-lo)})
    return {'accepted':not lines and not any(v>threshold for v in edge_runs.values()),'straight_threshold_exclusive_px':threshold,'tight_bbox_xyxy_halfopen':[x0+xmin,y0+ymin,x0+xmax+1,y0+ymax+1],'tight_bbox_side_overlap_max_runs_px':edge_runs,'has_long_straight_edge':bool(lines),'has_long_straight_overlap_with_tight_bbox':any(v>threshold for v in edge_runs.values()),'qualifying_straight_lines':lines}
def inspect(mask,origin_xyz,block_boundary_z=None,grid=None,straight_threshold=100,end_cap_threshold=1000,hole_min_area=1,end_cap_mode='gt'):
    x0,y0,z0=map(int,origin_xyz);areas=[];lines=[];holes=[]
    for j,raw in enumerate(mask):
        p=np.asarray(raw,dtype=bool);area=int(p.sum());areas.append(area)
        if not area:continue
        hole=ndi.binary_fill_holes(p)&~p
        if hole.any():
            labels,n=ndi.label(hole);sizes=np.bincount(labels.ravel())[1:];keep=sizes[sizes>=hole_min_area]
            if len(keep):holes.append({'z':z0+j,'holes':int(len(keep)),'hole_pixels':int(keep.sum()),'largest_hole_pixels':int(keep.max()),'minimum_individual_hole_area':hole_min_area})
        up=p.copy();up[1:]&=~p[:-1]
        down=p.copy();down[:-1]&=~p[1:]
        left=p.copy();left[:,1:]&=~p[:,:-1]
        right=p.copy();right[:,:-1]&=~p[:,1:]
        longest=0
        for a in (up,down,left.T,right.T):
            # Only rows with >100 exposed pixels can contain a >100 run.
            for row in a[a.sum(axis=1)>straight_threshold]:longest=max(longest,longest_run(row))
        if longest>straight_threshold:lines.append({'z':z0+j,'longest_exposed_axis_aligned_run_px':longest})
    occupied=np.flatnonzero(areas);caps=[]
    if len(occupied):
        for j in sorted(set([int(occupied[0]),int(occupied[-1])])):
            z=z0+j;boundary=None
            if block_boundary_z is not None:boundary=z in block_boundary_z
            elif grid and grid.get('found') and grid.get('z_origin_mod_depth'):
                depth=grid['shape_xyz'][2];boundary=any((z-phase)%depth in (0,depth-1) for phase in grid['z_origin_mod_depth'])
            hit=areas[j]<end_cap_threshold if end_cap_mode=='lt' else areas[j]>end_cap_threshold and boundary is False
            caps.append({'z':z,'area':areas[j],'block_boundary':boundary,'unclosed_confirmed':bool(hit),'unclosed_boundary_unknown':areas[j]>end_cap_threshold and boundary is None and end_cap_mode=='gt','area_comparison':end_cap_mode,'area_threshold':end_cap_threshold})
    end_key=f'unclosed_cap_lt{end_cap_threshold}' if end_cap_mode=='lt' else f'unclosed_cap_gt{end_cap_threshold}_not_block_boundary'
    flags={f'straight_gt{straight_threshold}':bool(lines),'holes':bool(holes),'occupied_layers_lt10':len(occupied)<10,end_key:any(c['unclosed_confirmed'] for c in caps)}
    return {'flags':flags,'needs_repair':any(flags.values()),'occupied_z_layers':int(len(occupied)),'z_span':int(occupied[-1]-occupied[0]+1) if len(occupied) else 0,'nonempty_z':[z0+int(j) for j in occupied],'median_nonempty_z':z0+int(occupied[(len(occupied)-1)//2]) if len(occupied) else None,'end_caps':caps,'hole_layers':holes,'straight_layers':lines,'boundary_unknown_cap_candidate':any(c['unclosed_boundary_unknown'] for c in caps)}
