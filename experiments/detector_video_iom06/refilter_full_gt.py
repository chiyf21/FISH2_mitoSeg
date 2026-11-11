"""Recheck every previously admitted actual detection seed using complete GT."""
from common import *
import argparse,concurrent.futures,multiprocessing,os,fcntl
from collections import Counter
from full_gt_iom import prepare_bounds,measure

OUT=ROOT/'filtered_fullgt'

def work(job):
    split,index,lines=job;dest=OUT/f'shards/{split}/{index:05d}.jsonl';meta=dest.with_suffix('.meta.json')
    if meta.exists():return json.loads(meta.read_text())
    kept=[];old_overestimated=0
    for line in lines:
        rec=json.loads(line);old=rec['iom'];measurement=measure(rec)
        if old-measurement['iom']>1e-8:old_overestimated+=1
        if measurement['iom']<.6:continue
        rec.update(iom_detector188=old,**measurement,full_gt_verified=True)
        kept.append(rec)
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial')
    with tmp.open('w') as f:
        for rec in kept:f.write(json.dumps(rec)+'\n')
    tmp.replace(dest);result={'split':split,'shard':index,'input':len(lines),'kept':len(kept),'rejected':len(lines)-len(kept),'old_overestimated':old_overestimated};save(meta,result);return result

def jobs():
    for split in ('train','val'):
        block=[];index=0
        for line in (ROOT/f'filtered/{split}.jsonl').open():
            block.append(line)
            if len(block)==1000:yield split,index,block;block=[];index+=1
        if block:yield split,index,block

def worker_init():
    os.nice(10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=open(OUT/'export.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare_bounds()
    definition={'threshold':.6,'denominator':'complete original GT instance on seed XY slice, resampled on exact same affine grid; includes pixels outside188 detector input and outside3x video crop','box':'unchanged actual detector seed box, pixel-center convention','candidate_scope':'every previously admitted actual detection seed in filtered/train.jsonl and val.jsonl; no detector rerun or new GT-derived boxes','original_pool_sha256':{s:sha(ROOT/f'filtered/{s}.jsonl') for s in ('train','val')},'checkpoint_sha256':sha(ROOT/'detector_frozen.pt'),'train_val_split':'unchanged','full_instance_z':True}
    definition['measurement_code_sha256']=sha(ROOT/'full_gt_iom.py')
    if (OUT/'definition.json').exists():assert json.loads((OUT/'definition.json').read_text())==definition
    else:save(OUT/'definition.json',definition)
    # Regression: reproduce the six audit results from the independent full
    # source-slice implementation before trusting the optimized tight transform.
    audit=json.loads((ROOT/'video_iom_audit.json').read_text())['results'];wanted={r['source_sample_id']:r for r in audit};checked=[]
    for line in (ROOT/'filtered/val.jsonl').open():
        rec=json.loads(line)
        if rec['sample_id'] in wanted:
            m=measure(rec);expected=wanted[rec['sample_id']]
            assert m['seed_gt_pixels_full']==expected['full_gt_pixels'] and m['seed_gt_pixels_in_box']==expected['inside_box_pixels'],(rec['sample_id'],m,expected)
            checked.append({'sample_id':rec['sample_id'],**m})
    assert len(checked)==6
    save(OUT/'audit_regression.json',{'passed':True,'checks':checked})
    stats={s:Counter() for s in ('train','val')}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=worker_init) as pool:
        for finished in pool.map(work,jobs(),chunksize=1):
            for key in ('input','kept','rejected','old_overestimated'):stats[finished['split']][key]+=finished[key]
            save(OUT/'progress.json',{'time':stamp(),'stats':{s:dict(v) for s,v in stats.items()},'last_shard':finished})
    instances={};counts={};minimum={}
    for split in ('train','val'):
        target=OUT/f'{split}.jsonl';tmp=target.with_suffix('.partial');ids=set();n=0;lowest=1.
        with tmp.open('w') as f:
            for path in sorted((OUT/f'shards/{split}').glob('*.jsonl')):
                for line in path.open():
                    r=json.loads(line);assert r['full_gt_verified'] and r['iom']>=.6
                    ids.add(r['source_instance_id']);lowest=min(lowest,r['iom']);n+=1;f.write(line)
        assert n==stats[split]['kept'] and n>0;tmp.replace(target);instances[split]=len(ids);counts[split]=n;minimum[split]=lowest
    save(OUT/'completed.json',{'full_run':True,'threshold':.6,'denominator':'complete_source_gt_slice','eligible_seed_samples':counts,'eligible_source_instances':instances,'minimum_iom':minimum,'audit_regression_passed':True,'stats':{s:dict(v) for s,v in stats.items()},'checkpoint_sha256':definition['checkpoint_sha256']})
    print(json.dumps(json.loads((OUT/'completed.json').read_text()),indent=2),flush=True)

if __name__=='__main__':main()
