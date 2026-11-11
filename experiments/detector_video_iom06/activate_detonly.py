"""Record the explicitly authorized objective change; preserve historical output."""
from common import *
import shutil
assert json.loads((ROOT/'detector_detonly_ft10/ddp_preflight.json').read_text())['passed']
assert json.loads((ROOT/'detector_detonly_ft10/ddp_preflight.json').read_text())['config']['micro_batch_max_per_gpu']==22
config=json.loads((ROOT/'pipeline_config.json').read_text())
config.update(detector_objective='detector_only',detector_output='detector_detonly_ft10',detector_entry='launch_detonly.py',detector_physical_gpus=[5,6,7],detector_effective_batch=64,detector_trainable=['backbone','neck','det_heads'],detector_inactive=['seg_head','prompt_decoders','er_head'],detector_checkpoint_selection='original complete validation detection loss')
save(ROOT/'pipeline_config.json',config)
append(ROOT/'events.jsonl',{'time':stamp(),'event':'user_requested_detector_only_restart','warm_start':str(ROOT/'original30_best.pt'),'old_output_retained':'detector_ft10','new_output':'detector_detonly_ft10','gpus':[5,6,7],'old_pids_stopped':[1118719,1140338]})
save(ROOT/'state.json',{'stage':'detector_detonly_finetune10','status':'running','message':'已切换仅检测损失训练，GPU5/6/7三卡，从原epoch27 best重新开始10轮；旧联合结果保留。','time':stamp()})
shutil.copy2(ROOT/'dashboard.html', ROOT/'dashboard/index.html')
shutil.copy2(ROOT/'README.md', ROOT/'dashboard/README.md')
