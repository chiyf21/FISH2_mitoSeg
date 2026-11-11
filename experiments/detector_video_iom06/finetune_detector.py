"""Warm-start original completed30 RFF+jitter detector, all old data + hard negatives,10epochs."""
from common import *
import fcntl,shutil,importlib.util,torch
ROOT.mkdir(exist_ok=True);lock=open(ROOT/'detector_ft10.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
original=Path('/path/to/JaneliaEM/outputs/train_det_sam_flyem2500_s1500x2_slice2d_dfl_rffjit/best_det_sam.pt')
copied=ROOT/'original30_best.pt'
if not copied.exists():shutil.copy2(original,copied)
assert sha(original)==sha(copied)
src=(DET/'train_hardneg.py').read_text()
changes=[("out=ROOT/'training'",f"out=Path({str(ROOT/'detector_ft10')!r})",1),("epochs=30","epochs=10",1),("'epochs':30","'epochs':10",2),("range(start,31)","range(start,11)",1),("e{epoch}/30","e{epoch}/10",1),("TRAINING_COMPLETED_30","TRAINING_COMPLETED_10",1),("dump(ROOT/'training_config.json',config)",f"dump(Path({str(ROOT/'detector_ft10_config.json')!r}),config)",1),("'initialization':'from scratch'","'initialization':'original completed30 best checkpoint epoch27; fresh optimizer'",1)]
for a,b,n in changes:
 assert src.count(a)==n,(a,src.count(a),n);src=src.replace(a,b)
anchor="start=1;updates=0;best=float('inf');seen=0"
assert src.count(anchor)==1
src=src.replace(anchor,f"warm=torch.load({str(copied)!r},map_location='cpu',weights_only=False);model.load_state_dict(warm['model'],strict=True);del warm\n    "+anchor)
generated=ROOT/'code_snapshot/train_detector_ft10.py';generated.parent.mkdir(exist_ok=True);generated.write_text(src)
save(ROOT/'detector_ft10_process.json',{'pid':os.getpid(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':stamp(),'warm_start':str(original),'warm_sha256':sha(original),'new_epochs':10,'data':'all original type1 and type2 plus existing hardneg exports; original train/val split','source_sha256':sha(generated)})
sys.path.insert(0,str(DET));spec=importlib.util.spec_from_file_location('train_detector_ft10',generated);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.train()
