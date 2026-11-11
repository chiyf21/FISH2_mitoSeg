from common import *
import shutil
dashboard=ROOT/'dashboard';dashboard.mkdir(exist_ok=True)
shutil.copyfile(ROOT/'dashboard.html',dashboard/'index.html');shutil.copyfile(ROOT/'README.md',dashboard/'README.md')
gallery=SAM/'gallery/training_pipeline'
if gallery.is_symlink():assert gallery.resolve()==dashboard
elif not gallery.exists():gallery.symlink_to(dashboard,target_is_directory=True)
else:raise RuntimeError('Unexpected existing gallery target')
save(ROOT/'pipeline_config.json',{'detector_warm_start':'original30_best.pt','detector_additional_epochs':10,'detector_lr':1e-4,'detector_train_samples':46830,'detector_validation_samples':13182,'negative_dataset':'previous exported v1 z000-079 exclusion-based18k views','old_data':'entire original type1 and type2; no positive-only selection','iom_threshold':.6,'iom_mode':'per 2D box and GT mask on one slice','video_epochs':10,'video_lr':1e-5,'full_instance_z':True,'seed_prompt':'one actual accepted2D box','crop':'centered at seed box, side3*longest edge, fixed all Z','publisher_interval_seconds':20})
if not (ROOT/'state.json').exists():save(ROOT/'state.json',{'stage':'detector_finetune10','status':'running','message':'Warm-start original30 best(epoch27);10 new epochs on all old training data + existing hard negatives.','time':stamp()})
