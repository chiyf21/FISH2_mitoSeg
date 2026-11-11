from common import *
import fcntl,subprocess
lock=open(ROOT/'detector_detonly.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(line.split(',')[0]):int(line.split(',')[1]) for line in usage.splitlines()}
assert all(free[g]>=16000 for g in (5,6,7)), ('Insufficient free VRAM; leave other tasks alone',free)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'5,6,7','OMP_NUM_THREADS':'3'}
command=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(ROOT/'train_detonly_ddp.py'),*sys.argv[1:]]
save(ROOT/'detector_detonly_launcher.json',{'pid':os.getpid(),'time':stamp(),'command':command,'physical_gpus':[5,6,7]})
raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))
