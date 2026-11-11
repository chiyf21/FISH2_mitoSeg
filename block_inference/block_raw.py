"""Bounded-memory native EM reader and exactly matched black-plane repair."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import threading
import numpy as np

class BlockRaw:
    def __init__(self,path,cache_mib=512):
        p=Path(path).resolve()
        if p.suffix.lower()=='.npy':self.a=np.load(p,mmap_mode='r')
        else:
            import tifffile
            self.a=tifffile.memmap(p)
        if self.a.ndim!=3 or self.a.dtype!=np.uint8:raise ValueError('Require uint8 ZYX .npy or memory-mappable TIFF; no implicit resampling/normalization')
        self.size=np.asarray(self.a.shape[::-1]);st=p.stat();self.source_tag=hashlib.sha256(f'{p}:{st.st_size}:{st.st_mtime_ns}'.encode()).hexdigest()
        self.limit=int(cache_mib*2**20);self.bytes=0;self.cache=OrderedDict();self.repair_log={};self.bad={};self.lock=threading.RLock()
    def _put(self,key,a):
        if key in self.cache:self.bytes-=self.cache.pop(key).nbytes
        self.cache[key]=a;self.bytes+=a.nbytes
        while self.bytes>self.limit and len(self.cache)>1:self.bytes-=self.cache.popitem(last=False)[1].nbytes
        return a
    def _plane(self,z,x,y):return self.a[z,y:min(y+128,int(self.size[1])),x:min(x+128,int(self.size[0]))]
    def _is_bad(self,z,x,y):
        k=(z,x,y)
        if k not in self.bad:self.bad[k]=bool(np.all(self._plane(z,x,y)<=1))
        return self.bad[k]
    def plane(self,z,x,y):
        with self.lock:return self._locked_plane(z,x,y)
    def _locked_plane(self,z,x,y):
        key=(z,x,y)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        if not self._is_bad(z,x,y):return self._put(key,np.array(self._plane(z,x,y)))
        l=z-1;h=z+1
        while l>=0 and self._is_bad(l,x,y):l-=1
        while h<self.size[2] and self._is_bad(h,x,y):h+=1
        if l<0 and h>=self.size[2]:raise ValueError(f'All source Z planes black in XY tile {x},{y}')
        if l<0:l=h
        if h>=self.size[2]:h=l
        w=(z-l)/(h-l) if h!=l else 0.
        a=np.rint((1-w)*self._plane(l,x,y).astype(np.float32)+w*self._plane(h,x,y).astype(np.float32)).astype(np.uint8)
        self.repair_log[key]=dict(z=z,lower_z=int(l),upper_z=int(h),upper_weight=float(w),source_xyxy=[x,y,min(x+128,int(self.size[0])),min(y+128,int(self.size[1]))])
        return self._put(key,a)
    def slice_crop(self,z,x,y,side):
        out=np.zeros((side,side),np.uint8);valid=np.zeros((side,side),bool);lx,ly=max(0,x),max(0,y);hx,hy=min(x+side,int(self.size[0])),min(y+side,int(self.size[1]))
        if z<0 or z>=self.size[2] or hx<=lx or hy<=ly:return out,valid
        for tx in range(lx//128*128,(hx-1)//128*128+1,128):
            for ty in range(ly//128*128,(hy-1)//128*128+1,128):
                a=self.plane(z,tx,ty);ax,ay=max(lx,tx),max(ly,ty);bx,by=min(hx,tx+a.shape[1]),min(hy,ty+a.shape[0])
                out[ay-y:by-y,ax-x:bx-x]=a[ay-ty:by-ty,ax-tx:bx-tx];valid[ay-y:by-y,ax-x:bx-x]=True
        return out,valid
    def tile(self,key):
        x,y,z=map(int,key);out=np.zeros((128,128,32),np.uint8)
        for q in range(z,min(z+32,int(self.size[2]))):out[:,:,q-z]=self.slice_crop(q,x,y,128)[0].T
        return out
    def prefetch(self,keys,wait=True):return []
