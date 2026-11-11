from common import *
import math,numpy as np
from filter_samples import iom_match
g=json.loads((ROOT/'video_smoke/gradient_check.json').read_text());assert g['frozen_encoder_no_grad'] and all(x>0 and math.isfinite(x) for x in g['trainable_grad_l1'].values())
metrics=[json.loads(x) for x in (ROOT/'video_smoke/metrics.jsonl').read_text().splitlines() if x.strip()];assert metrics and all(math.isfinite(x[k]['loss']) for x in metrics for k in ('train','val'))
aug=json.loads((ROOT/'augmentation_verification.json').read_text());assert aug['samples']==42012 and aug['all_metadata_match']
m=np.zeros((10,188,188),bool);m[0,0,:10]=True
hit=iom_match([0,0,0,1,1,6],[(0,m)]);assert hit is not None and hit[1]==.6
assert iom_match([0,0,0,1,1,5],[(0,m)]) is None
filtered=json.loads((ROOT/'smoke_original30/completed.json').read_text());assert filtered['all_source_masks_match_exported_gt'] and filtered['eligible_seed_samples']['train']>0 and filtered['eligible_seed_samples']['val']>0
save(ROOT/'pipeline_preflight_verified.json',{'passed':True,'scope':'small non-final preflight using original30 best, never counted as final fine-tuning','all_42012_augmentation_metadata_exact':True,'source_masks_equal_exported_gt':True,'iom_exact_06_accepts_05_rejects':True,'video_gradients':g['trainable_grad_l1'],'frozen_encoder_no_grad':True,'finite_train_and_val':True,'peak_cuda_mb':g['peak_cuda_allocated_mb'],'time':stamp()})
print((ROOT/'pipeline_preflight_verified.json').read_text())
