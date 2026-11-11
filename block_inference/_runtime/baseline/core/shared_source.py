"""One repaired native-grid source used by BOTH the detector and SAM."""
from functools import lru_cache
import numpy as np
from gt_source import LocalRaw
from repair import interpolate_stack

class SharedRaw(LocalRaw):
    def __init__(self,path=None):
        if path is None:super().__init__()
        else:
            import tifffile
            from pathlib import Path
            self.a=np.load(path,mmap_mode='r') if Path(path).suffix=='.npy' else tifffile.memmap(path)
            if self.a.ndim!=3 or self.a.dtype!=np.uint8:raise ValueError('Require uint8 ZYX .npy or memory-mappable TIFF')
            self.size=np.array(self.a.shape[::-1])
        import hashlib
        source=__import__('pathlib').Path(self.a.filename)
        self.source_tag=hashlib.sha256(f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}'.encode()).hexdigest()
        self.repair_log={}
    @lru_cache(maxsize=64)
    def column(self,x,y):
        hx=min(x+128,int(self.size[0]));hy=min(y+128,int(self.size[1]))
        a=np.asarray(self.a[:,y:hy,x:hx]);fixed,audit=interpolate_stack(a)
        for r in audit:self.repair_log[(x,y,r['z'])]={'source_xyxy':[x,y,hx,hy],**r}
        return fixed
    @lru_cache(maxsize=96)
    def tile(self,key):
        x,y,z=map(int,key);out=np.zeros((128,128,32),np.uint8)
        if x<0 or y<0 or z<0:return out
        if x>=self.size[0] or y>=self.size[1] or z>=self.size[2]:return out
        a=self.column(x,y);n=min(32,len(a)-z)
        out[:a.shape[2],:a.shape[1],:n]=a[z:z+n].transpose(2,1,0)
        return out
    def slice_crop(self,z,x0,y0,side):
        out=np.zeros((side,side),np.uint8);valid=np.zeros_like(out,bool)
        lx=max(0,x0);ly=max(0,y0);hx=min(x0+side,int(self.size[0]));hy=min(y0+side,int(self.size[1]))
        if not 0<=z<self.size[2] or hx<=lx or hy<=ly:return out,valid
        for x in range(lx//128*128,(hx-1)//128*128+1,128):
            for y in range(ly//128*128,(hy-1)//128*128+1,128):
                a=self.column(x,y);ax=max(lx,x);ay=max(ly,y);bx=min(hx,x+128);by=min(hy,y+128)
                out[ay-y0:by-y0,ax-x0:bx-x0]=a[z,ay-y:by-y,ax-x:bx-x];valid[ay-y0:by-y0,ax-x0:bx-x0]=True
        return out,valid
