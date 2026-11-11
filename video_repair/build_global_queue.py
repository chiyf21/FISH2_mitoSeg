"""Prepare spatially ordered distributed jobs. Does NOT start production repair."""
import csv,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True);a=p.parse_args();rows=json.loads(Path(a.plan).read_text());original={r['record_id']:r for r in csv.DictReader(Path(a.csv).open())};jobs=[]
for r in rows:
    src=original[r['record_id']];jobs.append(dict(index=r['index'],record_id=r['record_id'],body_id=r['body_id'],mask_path=r['mask_path'],original_mask_path112=str(Path(src['source_instance_dir'])/'mask.npy'),origin_xyz=r['origin_xyz'],prefill_lt200=True,expected_prepared_voxels=int(np.load(r['mask_path'],mmap_mode='r').sum()),flags=r['flags']))
jobs.sort(key=lambda r:(r['body_id'],r['origin_xyz'][2]//32,r['origin_xyz'][1]//512,r['origin_xyz'][0]//512));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(jobs));print(json.dumps(dict(jobs=len(jobs),spatial_order=True,video_launched=False)))
