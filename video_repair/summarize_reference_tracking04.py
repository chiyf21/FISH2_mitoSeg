import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;source=ROOT.parent/'sam_video_hardcases_20261102/simple113_component500_union_tight_score04_20261103';rows=[]
for d in source.glob('cases/instance_*'):
    if not (d/'existing_mask.npy').exists():continue
    meta=json.loads((d/'completed.json').read_text());evalz=set();keepz=set();routes=[]
    for i,a in enumerate(meta['repair_actions']):
        ad=d/f'action_{i:03d}';tr=json.loads((ad/'track.json').read_text());zz={r['z'] for r in tr['evaluated_scores']};evalz|=zz;m=np.load(ad/'selected_mask.npy',mmap_mode='r');o=json.loads((ad/'origin.json').read_text());kk={o[2]+int(j) for j in np.flatnonzero(m.any((1,2)))};keepz|=kk;routes.append(dict(evaluated_layers=len(zz),nonempty_layers=len(kk)))
    old=np.load(d/'existing_mask.npy',mmap_mode='r');rows.append(dict(index=int(d.name.split('_')[-1]),original_layers=int(old.any((1,2)).sum()),evaluated_layers=len(evalz),video_nonempty_layers=len(keepz),routes=routes))
report=dict(dataset='historical113 annotated local volume; not full-global20k',valid_instances=len(rows),score_threshold=.4,true_source_z_size=250,mean_original_layers=float(np.mean([r['original_layers'] for r in rows])),mean_distinct_evaluated_layers=float(np.mean([r['evaluated_layers'] for r in rows])),median_distinct_evaluated_layers=float(np.median([r['evaluated_layers'] for r in rows])),mean_video_nonempty_layers=float(np.mean([r['video_nonempty_layers'] for r in rows])),mean_evaluated_layers_per_route=float(np.mean([a['evaluated_layers'] for r in rows for a in r['routes']])),quantiles={str(q):float(np.quantile([r['evaluated_layers'] for r in rows],q)) for q in (.5,.9,.95,1)},rows=rows,limited_run=False)
(ROOT/'REFERENCE_113_TRACKING04.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
