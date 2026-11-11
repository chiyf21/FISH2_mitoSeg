from common import *
import subprocess,fcntl
lock=open(ROOT/'video_fullgt_preview.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
marker=ROOT/'dashboard/video_tracking_fullgt/completed.json'
attempts=0
while not marker.exists():
    checkpoint=ROOT/'video_fast_fullgt_ft10/first_preview.pt'
    if not checkpoint.exists():time.sleep(15);continue
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in usage.splitlines()}
    if free[0]<16000:time.sleep(30);continue
    attempts+=1
    with (ROOT/'video_tracking_fullgt_preview.log').open('a') as log:
        result=subprocess.run([str(PYTHON),'-u',str(ROOT/'visualize_video_tracking.py'),'--checkpoint',str(checkpoint),'--out',str(marker.parent),'--count','6'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT)
    if marker.exists():break
    if attempts>=3:
        save(ROOT/'video_tracking_preview_error.json',{'time':stamp(),'attempts':attempts,'exit_code':result.returncode});break
    time.sleep(60)
