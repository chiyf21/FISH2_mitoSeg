"""Exercise the actual Video route control flow on CPU with a tiny mock SAM."""
from contextlib import nullcontext
from unittest.mock import patch
import types,json
import numpy as np
import torch
from fast_runtime import ROOT
from policy import load_policy,apply_policy
from repair_three_prompt import MultiPromptEngine
class Model:
    max_cond_frames_in_attn=-1
    def __init__(self,scores):self.scores=scores;self.seen=[]
    def _prepare_backbone_features(self,f):return None,None,None,None
    def _track_step(self,j,*args):
        self.seen.append(j);hi=torch.ones((1,1,8,8));score=torch.tensor([[self.scores[j]]]);zero=torch.zeros(1)
        return {},(None,None,score,hi,hi,zero,zero),None,None
    def _encode_memory_in_output(self,*args):args[-1].update(maskmem_features=torch.zeros(1),maskmem_pos_enc=[torch.zeros(1)])
class Encoded:
    batches=[]
    def __getitem__(self,j):return {k:[torch.zeros(1)] for k in ('backbone_fpn','vision_pos_enc')}
def check(role,directions,scores,expected,low):
    e=MultiPromptEngine.__new__(MultiPromptEngine);apply_policy(e);e.model=Model(scores);e.nuc=np.zeros((5,8,8),bool);e.nuc_crop=lambda *a:np.zeros((8,8),bool)
    prompt=dict(z=2,roles=[role],box_xyxy=[1,1,6,6]);det=dict(**prompt,anchors=[prompt],directional_routes=True,route_specs=[(role,directions)])
    tensor=torch.tensor
    def cpu_tensor(*args,**kw):
        if kw.get('device')=='cuda':kw['device']='cpu'
        return tensor(*args,**kw)
    with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),patch('torch.autocast',lambda *a,**k:nullcontext()),patch('torch.tensor',cpu_tensor):r=e.track(Encoded(),[0,0,0,5],8,det)
    assert [f['z'] for f in r['frames']]==expected and r['stop_events'][0]['low_z']==[low]
    assert r['stop_events'][0]['threshold']==.5 and r['stop_events'][0]['required_low_frames']==1
    return e.model.seen
back=check('head',['backward'],{2:.6,1:.5,0:.499},[1,2],0)
forward=check('tail',['forward'],{2:.5,3:.499,4:.9},[2],3)
assert 4 not in forward
result=dict(score_threshold=.5,stop_count=1,equal05_retained=True,below05_excluded=True,stop_at_first_low_frame=True,backward_test=True,forward_test=True,limited_run=False)
(ROOT/'STOP_POLICY_05_VERIFIED.json').write_text(json.dumps(result,indent=2));print('PASS:actual route flow keeps0.5; stops at first<0.5; excludes low frame; both directions verified')
