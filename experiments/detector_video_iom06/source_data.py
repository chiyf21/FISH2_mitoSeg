"""Recover full-precision original augmentation plans and native image context."""
from common import *
import math, numpy as np
from functools import lru_cache
from scipy import ndimage as ndi
from scipy.special import cosdg,sindg
from data.dataset_npz import _stored_image
POS=DET/'data/npz_crops_flyem2500_s1500x2'
NEG=DET/'data/npz_hardneg_v1_z000_079'
FLY=BASE/'flyEM/2500x2500x250'

def labels(cid):return np.load(ROOT/f'source_cache/labels_{cid}.npy',mmap_mode='r')
@lru_cache(1)
def raw():
 import tifffile
 return tifffile.memmap(FLY/'raw.ome.tif')

@lru_cache(1)
def clipped_raw():
 lo,hi=json.loads((DET/'input_manifest.json').read_text())['clip']
 a=raw().astype(np.float32);a-=lo;a/=(hi-lo);np.clip(a,0,1,out=a);return a

def prepare():
 import zarr
 cache=ROOT/'source_cache';cache.mkdir(exist_ok=True)
 for cid,name in [(2,'mitochondria_0922.zarr'),(3,'lysosome_instances_v0924.zarr')]:
  p=cache/f'labels_{cid}.npy'
  if not p.exists():
   a=np.asarray(zarr.open(str(FLY/'annotation'/name),mode='r'));t=p.with_suffix('.partial.npy');np.save(t,a);t.replace(p)
 plans=ROOT/'augmentation_plans.json'
 if plans.exists() and (cache/'mito_z_bounds.json').exists():return
 exp=CODE/'experiments/flyem2500_s1500x2_det_sam_dfl_jitter/exporters';sys.path.insert(0,str(exp))
 import export_flyem2500_npz_ms1500_3000x10 as E
 rng=np.random.default_rng(42);shape=raw().shape;Z,Y,X=shape;centers=[];bounds={}
 for cid in (2,3):
  for iid,s in enumerate(ndi.find_objects(labels(cid)),1):
   if s is not None:
    centers.append(tuple(.5*(x.start+x.stop) for x in s))
    if cid==2:bounds[str(iid)]=[s[0].start,s[0].stop]
 save(cache/'mito_z_bounds.json',bounds)
 bases=[]
 for cz,cy,cx in centers:bases.append((int(np.clip(round(cz-5),0,Z-10)),int(np.clip(round(cy-94),0,Y-188)),int(np.clip(round(cx-94),0,X-188)),'type1'))
 for _ in range(1000):bases.append((int(rng.integers(0,Z-10)),int(rng.integers(0,Y-188)),int(rng.integers(0,X-188)),'type2'))
 groups={'type1':[],'type2':[]}
 for z,y,x,kind in bases:
  cx,cy,cz=x+94,y+94,z+5;common={'dataset':'fish2_2500','scale':'s1500','ds':1,'xy_nm_per_px':8}
  meta={**common,'crop':f's1500_s{E.stripe_of(cx,X)}','origin':[z,y,x],'kind':kind}
  groups[kind].append({'center_xyz':[cx,cy,cz],'angle':0.,'meta':meta})
  for _ in range(35 if kind=='type1' else 17):
   ncx,ncy,ncz,angle=E.plan_aug(rng,cx,cy,cz,shape,188,60.)
   meta={**common,'crop':f's1500_s{E.stripe_of(ncx,X)}','origin':[z,y,x],'kind':kind+'_aug','angle':round(angle,1),'trans':[round(ncx-cx,1),round(ncy-cy,1),round(ncz-cz,1)]}
   groups[kind].append({'center_xyz':[ncx,ncy,ncz],'angle':angle,'meta':meta})
 result={};count=0
 for kind in groups:
  rng.shuffle(groups[kind])
  cursor=0
  for p in sorted(POS.glob(kind+'_*.npz')):
   with np.load(p,allow_pickle=True) as a:metas=json.loads(str(a['meta']))
   for ci,m in enumerate(metas):
    plan=groups[kind][cursor];assert m==plan['meta'],f'Augmentation plan mismatch {p.name}/{ci}: actual={m}, expected={plan["meta"]}'
    result[f'{p.name}:{ci}']={k:v for k,v in plan.items() if k!='meta'};cursor+=1;count+=1
  assert cursor==len(groups[kind])
 save(plans,result);save(ROOT/'augmentation_verification.json',{'all_metadata_match':True,'samples':count,'seed':42,'full_precision_recovered':True,'note':'Rebuilt original deterministic sampling; rounded exported angles are not used to reconstruct context.'})

@lru_cache(1)
def plans():return json.loads((ROOT/'augmentation_plans.json').read_text())

def affine(plan,crop_y0=0,crop_x0=0):
 cx,cy,cz=plan['center_xyz'];c,s=cosdg(plan['angle']),sindg(plan['angle']);m=np.array([[c,s],[-s,c]])
 # Exactly reproduces original ndi.rotate(..., reshape=False) and central crop.
 offset=np.array([round(cy)-.5,round(cx)-.5])-m@np.array([93.5,93.5])+m@np.array([crop_y0,crop_x0])
 return int(round(cz))-5,m,offset

def source_label_patch(plan,side=188,y0=0,x0=0,iid=None,z_bounds=None):
 z0,m,off=affine(plan,y0,x0);a=labels(2)
 zz=range(z0,z0+10) if z_bounds is None else range(*z_bounds)
 out=np.stack([ndi.affine_transform(a[z],m,offset=off,output_shape=(side,side),order=0,mode='constant',cval=0,prefilter=False) for z in zz])
 return out if iid is None else out==iid

def sequence(record):
 plan=record['plan'];side=record['crop_side'];y0,x0=record['crop_origin_yx'];z0,m,off=affine(plan,y0,x0);a=raw();imgs=[]
 zz=record.get('video_z_bounds',[z0,z0+10])
 for z in range(*zz):
  imgs.append(np.rint(ndi.affine_transform(a[z],m,offset=off,output_shape=(side,side),output=np.float32,order=1,mode='constant',cval=0,prefilter=False)).clip(0,255).astype(np.uint8))
 gt=source_label_patch(plan,side,y0,x0,record['source_instance_id'],z_bounds=zz)
 return np.stack(imgs),gt

def validate_plan(path,ci,mask,plan):
 lab=source_label_patch(plan);ids=np.unique(lab[mask]);assert len(ids)==1 and ids[0]>0
 iid=int(ids[0]);assert np.array_equal(lab==iid,mask),'Restored source mask differs from actual NPZ mask'
 z0,m,off=affine(plan)
 em=np.stack([ndi.affine_transform(clipped_raw()[z],m,offset=off,output_shape=(188,188),order=1,mode='constant',cval=0,prefilter=False) for z in range(z0,z0+10)]);em=(em-em.mean())/(em.std()+1e-6)
 err=float(np.max(np.abs(em-_stored_image(str(path))[ci])));assert err<2e-5,f'Original augmentation image mismatch: {err}'
 return iid,err

if __name__=='__main__':prepare()
