"""Frozen v1 CLI. No detector, GT, historical cohort, or implicit raw dataset is used."""
import argparse,json,os,sys,time,hashlib,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
import numpy as np
from infer_block import save,digest
from union_connectivity import retain_connected_body

def read(path):return json.loads(Path(path).read_text())
def signature(path):
    p=Path(path).resolve();s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def origin(value):
    if isinstance(value,(str,Path)):value=read(value)
    if not isinstance(value,list) or len(value)!=3 or any(type(v)!=int for v in value):raise ValueError('origin_xyz must be three integer native XYZ coordinates')
    return value
def finalize_case(folder):
    meta=read(folder/'completed.json')
    union_path=folder/'union_mask.npy' if (folder/'union_mask.npy').exists() else folder/'mask.npy'
    if not union_path.exists():return None
    union=np.load(union_path);o=read(folder/'mask_origin.json');old=np.load(folder/'existing_mask.npy',mmap_mode='r');oo=read(folder/'existing_origin.json')
    main=next((a for a in meta.get('repair_actions',[]) if a['branch']=='maxclean'),None)
    z=main['seed_z'] if main else oo[2]+int(old.sum((1,2)).argmax())
    out,audit=retain_connected_body(union,o,old[z-oo[2]],oo[:2],z,old,oo)
    return out,o,audit

def configure_sam(code):
    code=Path(code).resolve()
    if not (code/'sam2').is_dir():raise ValueError(f'SAM2 source not found: {code}/sam2')
    sys.path[:0]=[str(code/'vendor_deps'),str(code)]

def runtime(raw_path,nucleus_path,weights,base,code):
    configure_sam(code)
    import torch
    from shared_source import SharedRaw
    from repair_three_prompt import MultiPromptEngine
    from sam2.build_sam import build_sam2
    torch.set_num_threads(3)
    class Engine(MultiPromptEngine):
        def __init__(self):
            self.raw=SharedRaw(raw_path);self.nuc=np.load(nucleus_path,mmap_mode='r')
            if self.nuc.shape!=self.raw.a.shape:raise ValueError('Nucleus mask must match full raw ZYX shape')
            if self.nuc.dtype!=bool:raise ValueError('Nucleus mask must be bool .npy; explicitly convert label maps before running')
            self.score_threshold=.4;self.score_stop_count=1
            self.model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(base),device='cuda',apply_postprocessing=False,hydra_overrides_extra=['++model.multimask_output_in_sam=false','++model.multimask_output_for_tracking=false'])
            ck=torch.load(weights,map_location='cpu',weights_only=False);self.model.load_state_dict(ck['model'],strict=True);self.model.eval()
            self.provenance=dict(sam_checkpoint=str(weights),sam_sha256=digest(weights),base_sha256=digest(base))
    return Engine()

def prepare_output(out,identity):
    out.mkdir(parents=True,exist_ok=True);p=out/'input_identity.json'
    if p.exists() and read(p)!=identity:raise ValueError('Output belongs to different inputs/code. Choose a new output folder.')
    save(p,identity)

def run(args):
    import repair_component_ends as pipeline
    from tqdm import tqdm
    out=Path(args.output).resolve();manifest_path=Path(args.manifest).resolve();entries=read(manifest_path)
    if isinstance(entries,dict):entries=entries['instances']
    base=manifest_path.parent;items=[];indices=set()
    for row in entries:
        idx=row['index']
        if type(idx)!=int or idx<0 or idx in indices:raise ValueError('index must be a unique nonnegative integer')
        indices.add(idx);p=Path(row['mask_path']);p=p if p.is_absolute() else base/p
        row={**row,'mask_path':str(p.resolve()),'origin_xyz':origin(row['origin_xyz'])}
        m=np.load(p,mmap_mode='r')
        if m.ndim!=3 or m.dtype!=bool:raise ValueError(f'{p}: require boolean ZYX .npy')
        if not m.any():raise ValueError(f'{p}: empty input instance')
        items.append(row)
    model=Path(args.weights).resolve();base_model=Path(args.base_weights).resolve()
    ident=dict(version=read(ROOT/'config.json'),code_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'core').glob('*.py'))},raw=signature(args.raw),nucleus=signature(args.nucleus),weights_sha256=digest(model),base_sha256=digest(base_model),instances=[{**r,'file':signature(r['mask_path'])} for r in items])
    prepare_output(out,ident);save(out/'config.json',read(ROOT/'config.json'));engine=runtime(args.raw,args.nucleus,model,base_model,args.sam2_code)
    lookup={r['index']:r for r in items};pipeline.RUN=out
    pipeline.existing=lambda rec:(np.load(lookup[rec['index']]['mask_path'],mmap_mode='r'),lookup[rec['index']]['origin_xyz'])
    os.environ['REPAIR_KEEP3D_COMPONENT']='0' # Save union before applying final selection below.
    start=time.perf_counter();summary=[]
    for row in tqdm(items,desc='mito repair',unit='instance'):
        d=out/f'cases/instance_{row["index"]:03d}'
        o=row['origin_xyz'];m=np.load(row['mask_path'],mmap_mode='r')
        if any(v<0 for v in o) or o[0]+m.shape[2]>engine.raw.size[0] or o[1]+m.shape[1]>engine.raw.size[1] or o[2]+len(m)>engine.raw.size[2]:raise ValueError('Instance mask lies outside raw volume')
        rec={**row,'gt_id':row.get('instance_id',row['index']),'block':[0,0,0,*map(int,engine.raw.size)]}
        if not (d/'finalized.json').exists():
            pipeline.run_case(engine,rec)
            result=finalize_case(d)
            if result:
                final,fo,audit=result
                # Keep raw union immutable for reproducible postprocessing/resume.
                if not (d/'union_mask.npy').exists():(d/'mask.npy').rename(d/'union_mask.npy')
                temp=d/'final_mask.partial.npy';np.save(temp,final);temp.replace(d/'mask.npy');save(d/'connectivity_audit.json',audit)
                meta=read(d/'completed.json');meta.update(final_connectivity=26,component_selection=audit,original_mask_preserved='union stage only; detached parts removed afterwards');save(d/'completed.json',meta)
            save(d/'finalized.json',dict(index=row['index'],limited_run=False))
        if args.panel:
            from panel import render_panel
            render_panel(engine.raw,d,out/'panel',row['index'])
        summary.append(dict(index=row['index'],mask_path=str(d/'mask.npy'),origin_xyz=read(d/'mask_origin.json'),connectivity=read(d/'connectivity_audit.json')))
    save(out/'results.json',dict(instances=summary,processed=len(summary),wall_seconds=time.perf_counter()-start,limited_run=False));save(out/'completed.json',dict(processed=len(summary),limited_run=False))
    print(json.dumps(dict(output=str(out),processed=len(summary)),ensure_ascii=False))

def finalize(args):
    from tqdm import tqdm
    source=Path(args.source_run).resolve();out=Path(args.output).resolve();folders=sorted((source/'cases').glob('instance_*'))
    if args.only:folders=[d for d in folders if int(d.name.split('_')[-1]) in args.only]
    if not folders:raise ValueError('No source cases found')
    prepare_output(out,dict(source=str(source),version=read(ROOT/'config.json'),source_identity=[signature(d/'completed.json') for d in folders]));start=time.perf_counter();records=[]
    for src in tqdm(folders,desc='union ->26CC',unit='instance'):
        if not (src/'completed.json').exists():raise ValueError(f'Incomplete inference: {src}')
        result=finalize_case(src);dst=out/'cases'/src.name;dst.mkdir(parents=True,exist_ok=True)
        if result is None:save(dst/'completed.json',dict(status='no_existing_mask',limited_run=False));continue
        final,o,audit=result;np.save(dst/'mask.npy',final);save(dst/'mask_origin.json',o);save(dst/'connectivity_audit.json',audit);save(dst/'completed.json',dict(source=str(src),inference_reused=True,limited_run=False))
        if args.expected_run:
            exp=Path(args.expected_run)/'cases'/src.name
            assert o==read(exp/'mask_origin.json') and np.array_equal(final,np.load(exp/'mask.npy',mmap_mode='r')),f'Regression mismatch: {src.name}'
        records.append(dict(instance=src.name,**audit))
    save(out/'results.json',dict(processed=len(folders),nonempty=len(records),records=records,wall_seconds=time.perf_counter()-start,inference_reused=True,regression_exact_match=bool(args.expected_run),limited_run=False));save(out/'completed.json',dict(processed=len(folders),limited_run=False))
    print(json.dumps(dict(processed=len(folders),nonempty=len(records),exact_match=bool(args.expected_run),output=str(out))))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run',help='Repair arbitrary existing instance masks; CUDA required')
    for name in ('raw','nucleus','manifest','output'):r.add_argument('--'+name,required=True)
    r.add_argument('--weights',default=str(ROOT/'assets/models/video_sam_best.pt'));r.add_argument('--base-weights',default=str(ROOT/'assets/models/sam2.1_hiera_base_plus.pt'));r.add_argument('--sam2-code',default=str(ROOT/'assets/sam2'));r.add_argument('--panel',action='store_true');r.set_defaults(func=run)
    f=sub.add_parser('finalize',help='CPU-only union->26CC from a cached source run');f.add_argument('--source-run',required=True);f.add_argument('--output',required=True);f.add_argument('--only',type=int,nargs='+');f.add_argument('--expected-run',help='Optional pixel-exact regression reference');f.set_defaults(func=finalize)
    args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.func(args)
if __name__=='__main__':main()
