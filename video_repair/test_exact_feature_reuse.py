import time,json
import torch
from fast_runtime import ROOT,create_local
from infer_block import Engines,save
ref=ROOT.parent/'sam_video_hardcases_20261102';assets=ROOT/'baseline/assets'
e=create_local(ref.parent.parent/'flyEM/2500x2500x250/raw.ome.tif',ref/'instance_centered_20261102/nucleus.npy',assets/'models/video_sam_best.pt',assets/'models/sam2.1_hiera_base_plus.pt',assets/'sam2',True)
geometry=(500,500,80,10);side=256
baseline=Engines.encode_clip(e,geometry,side);first=e.fast_encoder.encode(geometry,side);before=e.fast_stats['encoder_batches'];t=time.perf_counter();second=e.fast_encoder.encode(geometry,side);seconds=time.perf_counter()-t
for a,b,c in zip(baseline,first,second):
    for key in a:
        for x,y,z in zip(a[key],b[key],c[key]):assert torch.equal(x,y.cpu()) and torch.equal(y,z)
assert e.fast_stats['encoder_batches']==before and e.fast_stats['encoded_frame_cache_hits']==10
e.fast_encoder.io.close();save(ROOT/'EXACT_FEATURE_REUSE_VERIFIED.json',dict(original_and_optimized_features_bit_exact=True,cached_frames=10,second_call_encoder_batches=0,second_call_seconds=seconds,limited_run=False));print('PASS:original/GPU-resident/cached features bit-exact;10 cache-hit frames require0 additional encoder batches')
