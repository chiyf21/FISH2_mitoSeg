import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;source=ROOT.parent/'sam_video_hardcases_20261102/repair20k_prefill200_plan_20261103/flagged_manifest.json';rows=json.loads(source.read_text());out=ROOT/'global_smoke_inputs';(out/'masks').mkdir(parents=True,exist_ok=True)
# Include short, median, long and large-XY cases; choose deterministically by
# strata and proximity to the first body to exercise spatial caching as well.
group=[r for r in rows if r['body_id']=='100000128'];group.sort(key=lambda r:r['original_z_span']);chosen=[group[int(i)] for i in np.linspace(0,len(group)-1,4,dtype=int)]
manifest=[]
for r in chosen:
    path=out/'masks'/f'{r["index"]:06d}.npy';np.save(path,np.asarray(np.load(r['mask_path'],mmap_mode='r'),dtype=bool));manifest.append({**r,'mask_path':str(path.relative_to(out))})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps([dict(index=r['index'],z=r['original_z_span'],flags=r['flags']) for r in manifest]))
