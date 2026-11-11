import json,os,time,subprocess,sys,sqlite3,fcntl
from pathlib import Path
from fast_runtime import ROOT
from infer_block import save
RUN=ROOT/'detgraph113_iou05_iom08_score05_20261103'
def main():
    RUN.mkdir(exist_ok=True);lock=(RUN/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);selected=json.loads((RUN/'selected_groups.json').read_text());assert len(selected)<=113
    for name in ('launched.json','inference_completed.json'):
        if (RUN/name).exists():raise RuntimeError('Existing run: inspect before any restart, avoid duplicate inference')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    qdir=RUN/'queue';qdir.mkdir(exist_ok=True);log=(RUN/'queue.log').open('a');queue=subprocess.Popen([sys.executable,'-u',str(ROOT/'queue_service.py'),'--manifest',str(RUN/'selected_groups.json'),'--out',str(qdir),'--port','18089'],stdout=log,stderr=log)
    deadline=time.time()+20
    while not (qdir/'queue_access.json').exists():
        if queue.poll() is not None or time.time()>deadline:raise RuntimeError('Queue failed to start')
        time.sleep(.2)
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);mem={int(x.split(',')[0]):int(x.split(',')[1]) for x in stats.splitlines()};assert mem[7]<20000,'GPU7 no longer has planned headroom'
    attempts=0;worker=None;started=time.time()
    save(RUN/'launched.json',dict(controller_pid=os.getpid(),queue_pid=queue.pid,time=started,selected_groups=len(selected),target_instances=113,gpu=7,per_process_memory_fraction=.25,other_processes_not_stopped=True))
    while True:
        if worker is None:
            attempts+=1;log=(RUN/f'gpu7_attempt{attempts}.log').open('a');worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'detgraph_worker.py'),'--worker','gpu7','--port','18089'],cwd=ROOT,env=env,stdout=log,stderr=log);log.close()
        with sqlite3.connect(qdir/'queue.sqlite') as c:counts=dict(c.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
        save(RUN/'inference_progress.json',dict(done=counts.get('done',0),total=len(selected),counts=counts,worker_pid=worker.pid,worker_returncode=worker.poll(),attempt=attempts,gpu=7,elapsed_seconds=time.time()-started))
        if worker.poll() is not None:
            if counts.get('done',0)==len(selected):break
            if attempts>=3:save(RUN/'blocked.json',dict(reason='worker exited3times before all selected groups done',counts=counts));return
            # Only reset this verified exited worker's incomplete claims.
            with sqlite3.connect(qdir/'queue.sqlite') as c:
                c.execute("UPDATE jobs SET state='pending' WHERE worker='gpu7' AND state IN ('running','failed') AND attempt<3")
            worker=None
        time.sleep(5)
    save(RUN/'inference_completed.json',dict(groups=len(selected),limited_run=False,elapsed_seconds=time.time()-started));queue.terminate();queue.wait(timeout=10)
if __name__=='__main__':main()
