from common import *
import torch
from sam2.build_sam import build_sam2

out=ROOT/'video_fast_fullgt_ft10'
candidates=[p for p in (out/'recovery.pt',out/'last.pt',out/'first_preview.pt') if p.exists()]
path=max(candidates,key=lambda p:torch.load(p,map_location='cpu',weights_only=False)['updates'])
ck=torch.load(path,map_location='cpu',weights_only=False)
source=SAM/'checkpoints/sam2.1_hiera_base_plus.pt'
model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(source),device='cpu',apply_postprocessing=False,hydra_overrides_extra=['++model.multimask_output_in_sam=false','++model.multimask_output_for_tracking=false'])
prefixes=('sam_mask_decoder.','memory_attention.','memory_encoder.','obj_ptr_proj.','obj_ptr_tpos_proj.')
trainable=[(n,p) for n,p in model.named_parameters() if n.startswith(prefixes)]
ids=ck['optimizer']['param_groups'][0]['params'];assert len(ids)==len(trainable)
results=[]
for parameter_id,(name,param) in zip(ids,trainable):
    if 'iou_prediction_head' not in name and 'pred_obj_score_head' not in name:continue
    state=ck['optimizer']['state'].get(parameter_id,{})
    difference=(ck['model'][name].float()-param.detach().float()).abs()
    results.append({'name':name,'included_in_optimizer':True,'optimizer_step':float(state.get('step',0)),'adam_exp_avg_l1':float(state.get('exp_avg',torch.zeros(1)).abs().sum()),'max_abs_change_from_official':float(difference.max())})
assert results and any('iou_prediction_head' in r['name'] for r in results) and any('pred_obj_score_head' in r['name'] for r in results)
assert all(r['optimizer_step']>0 and r['adam_exp_avg_l1']>0 and r['max_abs_change_from_official']>0 for r in results)
result={'checkpoint':str(path),'updates':ck['updates'],'all_score_head_parameters_updated':True,'parameters':results}
save(ROOT/'score_heads_training_verified.json',result)
print(json.dumps(result,indent=2))
