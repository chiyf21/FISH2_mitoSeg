from common import *
import fcntl,subprocess
lock=open(ROOT/'video_fast_fullgt.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()};assert all(free[g]>24000 for g in (5,6,7)),free
env={**os.environ,'CUDA_VISIBLE_DEVICES':'5,6,7','OMP_NUM_THREADS':'3'}
command=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(ROOT/'train_video_fast.py'),*sys.argv[1:]]
save(ROOT/'video_fast_launcher.json',{'pid':os.getpid(),'time':stamp(),'command':command,'gpus':[5,6,7]})
raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))
