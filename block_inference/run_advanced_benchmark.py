"""Sequential same-GPU fresh-output ablation and numerical equivalence audit."""
import argparse,json,subprocess,sys,time,traceback,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def write(path,value):Path(path).write_text(json.dumps(value,indent=2))

def compare(base,out):
    a=np.load(base/'video/assembly/labels.npy',mmap_mode='r');b=np.load(out/'video/assembly/labels.npy',mmap_mode='r');af=a>0;bf=b>0
    result=dict(labels_exact=bool(np.array_equal(a,b)),foreground_difference_voxels=int(np.count_nonzero(af!=bf)),foreground_iou_to_baseline=int((af&bf).sum())/max(1,int((af|bf).sum())))
    files={p.name for p in (base/'video/frames').glob('*.npz')};other={p.name for p in (out/'video/frames').glob('*.npz')};same=0;different=0;delta=0.
    for name in files&other:
        with np.load(base/'video/frames'/name) as x,np.load(out/'video/frames'/name) as y:
            eq=all(np.array_equal(x[k],y[k]) for k in ['mask','origin_xy','score']);same+=int(eq);different+=int(not eq);delta=max(delta,abs(float(x['score'])-float(y['score'])))
    result.update(exact_frame_mask_origin_score=same,different_common_frames=different,frame_files_added=len(other-files),frame_files_missing=len(files-other),max_saved_score_difference=delta)
    result['seed_set_exact']=json.loads((base/'video/accepted_tracks.json').read_text())==json.loads((out/'video/accepted_tracks.json').read_text())
    records={p.name for p in (base/'video/tracks').glob('*.json') if not p.name.startswith('cached_')};oth={p.name for p in (out/'video/tracks').glob('*.json') if not p.name.startswith('cached_')}
    mismatches=[n for n in records&oth if json.loads((base/'video/tracks'/n).read_text())!=json.loads((out/'video/tracks'/n).read_text())]
    result.update(track_prompt_probe_record_mismatches=len(mismatches)+len(records^oth),mismatch_examples=sorted(mismatches)[:10])
    result['decisions_exact']=all((out/'video/decisions'/p.name).exists() and json.loads(p.read_text())==json.loads((out/'video/decisions'/p.name).read_text()) for p in (base/'video/decisions').glob('*.json'))
    for name in ['nucleus/nucleus.npy','detection/detections.json']:
        if name.endswith('.npy'):eq=np.array_equal(np.load(base/name,mmap_mode='r'),np.load(out/name,mmap_mode='r'))
        else:eq=json.loads((base/name).read_text())==json.loads((out/name).read_text())
        result[name+'_exact']=bool(eq)
    result['strict_exact']=bool(result['labels_exact'] and different==0 and files==other and result['track_prompt_probe_record_mismatches']==0 and result['decisions_exact'] and result['seed_set_exact'])
    return result

def evaluate(labels,gt):
    p=np.load(labels,mmap_mode='r');pf=p>0;gf=gt>0;areas=np.bincount(p.ravel());scores=[]
    for gid in np.unique(gt):
        if not gid:continue
        g=gt==gid;votes=np.bincount(p[g]);votes[0]=0;pid=int(votes.argmax())
        scores.append(float(votes[pid]/max(1,int(g.sum())+int(areas[pid])-int(votes[pid]))) if pid else 0.)
    return dict(foreground_iou_to_gt=int((pf&gf).sum())/max(1,int((pf|gf).sum())),fixed_gt_instances=len(scores),mean_gt_vote_matched_iou=float(np.mean(scores)) if scores else None)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--gt');a=p.parse_args()
    bench=Path(a.output).resolve();bench.mkdir(parents=True,exist_ok=True);source=json.loads(Path(a.config).read_text());source.pop('nucleus',None);source.pop('detections',None)
    common=dict(replay_batch_encoder=True,replay_batch_size=4,prompt_initialization_reuse=False,prefix_cache=False,multi_track_decoder=False,replay_cache_mib=512)
    variants=[('B0_current',False),('B1_replay_batch',dict(common)),('B2_plus_prompt',dict(common,prompt_initialization_reuse=True)),('B3_plus_prefix',dict(common,prompt_initialization_reuse=True,prefix_cache=True)),('B4_plus_multi',dict(common,prompt_initialization_reuse=True,prefix_cache=True,multi_track_decoder=True,track_workers=4,decode_batch_size=4,batch_wait_ms=1)),('S_cache_only',dict(common,replay_batch_encoder=False,prompt_initialization_reuse=True,prefix_cache=True))]
    gt=np.load(a.gt,mmap_mode='r') if a.gt else None;results=[];baseline=None
    for name,options in variants:
        out=bench/name;config=dict(source,output=str(out),advanced_optimizations=options);cfg=bench/(name+'.json');write(cfg,config)
        if out.exists():raise ValueError(f'Fresh output required: {out}')
        write(bench/'state.json',dict(stage=name,status='running'));start=time.perf_counter()
        try:
            with (bench/(name+'.log')).open('wb') as log:subprocess.run([sys.executable,'-u',str(ROOT/'infer_block.py'),'--config',str(cfg)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            external=time.perf_counter()-start;r=json.loads((out/'completed.json').read_text());stage={s['stage']:s['wall_seconds'] for s in r['stage_timings']}
            assert all(not s['reused_completed_output'] for s in r['stage_timings']);assert all(not s['reused'] for s in json.loads((out/'video/seed_timings.json').read_text()))
            row=dict(name=name,options=options,output=str(out),status='complete',pipeline_seconds=r['seconds'],external_wall_seconds=external,stage_seconds=stage,instances=r['instances'],execution_stats=json.loads((out/'video/execution_stats.json').read_text()))
            if (out/'video/advanced_execution_stats.json').exists():row['advanced_stats']=json.loads((out/'video/advanced_execution_stats.json').read_text())
            if baseline is None:baseline=out
            row['comparison']=compare(baseline,out)
            if gt is not None:row['gt_evaluation']=evaluate(out/'video/assembly/labels.npy',gt)
            row['speedup_vs_current']=results[0]['pipeline_seconds']/row['pipeline_seconds'] if results else 1.
            row['time_saved_percent']=(1-row['pipeline_seconds']/results[0]['pipeline_seconds'])*100 if results else 0.
        except Exception as exc:row=dict(name=name,status='failed',error=repr(exc),traceback=traceback.format_exc());print(row['traceback'],flush=True)
        results.append(row);write(bench/'results.json',results);print(json.dumps({k:v for k,v in row.items() if k not in ['execution_stats','advanced_stats','traceback']},indent=2),flush=True)
    write(bench/'state.json',dict(stage='all',status='complete'));print('ABLATION_COMPLETE',bench,flush=True)

if __name__=='__main__':main()
