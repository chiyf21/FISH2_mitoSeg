from common import *
from collections import Counter
import numpy as np

result={}
for split in ('train','val'):
    instances=Counter();views=set();exact_crops=set();frames=0;count=0
    for line in (ROOT/f'filtered/{split}.jsonl').open():
        r=json.loads(line);count+=1;iid=r['source_instance_id'];instances[iid]+=1
        views.add((r['source_npz'],r['crop_index'],iid))
        exact_crops.add((iid,tuple(r['plan']['center_xyz']),r['plan']['angle'],tuple(r['crop_origin_yx']),r['crop_side']))
        frames+=r['video_z_bounds'][1]-r['video_z_bounds'][0]
    result[split]={'seed_videos':count,'frames':frames,'unique_source_instances':len(instances),'instance_augmented_crop_views':len(views),'distinct_fixed_xy_video_crops':len(exact_crops),'videos_per_instance_percentiles':np.percentile(list(instances.values()),[0,50,90,99,100]).tolist(),'mean_frames_per_video':frames/count}
save(ROOT/'video_workload_audit.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
