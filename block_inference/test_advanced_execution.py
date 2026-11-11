"""CPU tests for cache validity, independence and batch-compatible layouts."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from common import ROOT
import advanced_execution as a

def main():
    calls=[]
    def original(t,z,state):
        calls.append((t.seed['z'],z,t.reverse,tuple(t.crop)))
        kept=dict(maskmem_features=torch.ones(1,2,2,2),maskmem_pos_enc=[torch.zeros(1,2,2,2)],obj_ptr=torch.zeros(1,4))
        state['cond_frame_outputs' if z==t.seed['z'] else 'non_cond_frame_outputs'][z]=kept
        return np.ones((4,4),bool),.8
    a._ORIGINAL_DECODE=original
    encoder=NS(encode=lambda geometry,side:[])
    e=NS(raw=NS(source_tag='raw'),fast_encoder=encoder,model=NS(max_obj_ptrs_in_encoder=16,num_maskmem=7,memory_temporal_stride_for_eval=1))
    method=NS(engine=NS(ROI=[0,0,0,100,100,30]))
    manager=a.AdvancedExecution(e,method,dict(prompt_initialization_reuse=True,prefix_cache=True))
    def tracker(reverse=False,box=None):return NS(crop=[0,0,256],seed=dict(z=10,box_xyxy=box or [5,5,15,15]),reverse=reverse,score_threshold=.45)
    def state():return dict(cond_frame_outputs={},non_cond_frame_outputs={})
    t,s=tracker(),state();manager.decode(t,10,s);manager.decode(t,11,s);assert len(calls)==2
    tt,ss=tracker(),state();manager.decode(tt,10,ss);manager.decode(tt,11,ss);assert len(calls)==2
    assert s['cond_frame_outputs'][10] is not ss['cond_frame_outputs'][10]
    tr,sr=tracker(True),state();manager.decode(tr,10,sr);assert len(calls)==2
    manager.decode(tr,9,sr);assert len(calls)==3
    changed,sc=tracker(box=[5,5,16,15]),state();manager.decode(changed,10,sc);manager.decode(changed,11,sc);assert len(calls)==5
    changed.crop=[-1,0,256];manager.decode(changed,10,state());assert len(calls)==6
    print('PASS prompt/prefix reuse, independent dictionaries, direction/prompt/crop invalidation')
    packed=manager._pack([dict(state=s),dict(state=ss)])
    assert packed['cond_frame_outputs'][10]['maskmem_features'].shape[0]==2
    assert '__adv_token' not in packed['cond_frame_outputs'][10]
    print('PASS independent-state batch packing ignores cache metadata')
    requests=[]
    def encode(geometry,side):
        requests.append((geometry,side));return [dict(backbone_fpn=[torch.full((1,1,2,2),float(z))],vision_pos_enc=[torch.zeros(1,1,2,2)]) for z in range(geometry[2],geometry[2]+geometry[3])]
    fp=a.FeatureProvider(NS(fast_encoder=NS(encode=encode)),dict(replay_batch_encoder=True,replay_batch_size=4),__import__('collections').Counter())
    replay=NS(crop=[0,0,256]);fp.prefill(replay,[3,4,5,6,7]);assert len(requests)==1
    fp.ensure(replay,7);assert len(requests)==2
    assert fp.signature(0,0,3,256)==('batch',3,4,0)
    assert fp.signature(0,0,7,256)==('single',7,1,0)
    assert fp.encode((0,0,3,1),256)[0]['backbone_fpn'][0].flatten()[0]==3 and len(requests)==2
    print('PASS replay batch mapping and arithmetic-provenance cache keys')

if __name__=='__main__':main()
