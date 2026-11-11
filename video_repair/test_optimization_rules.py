import sys,threading,json,time,subprocess,urllib.request,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fast_runtime import ROOT,ByteLRU
from repair import interpolate_stack
import numpy as np

def main():
    # Cross-chunk support must be equivalent to full-column interpolation.
    rng=np.random.default_rng(4);raw=rng.integers(10,255,(100,8,8),dtype=np.uint8);raw[[0,31,32,63,64,99]]=0
    expected,_=interpolate_stack(raw);actual=raw.copy()
    for z in range(len(raw)):
        if np.any(raw[z]>1):continue
        lo=z-1;hi=z+1
        while lo>=0 and not np.any(raw[lo]>1):lo-=1
        while hi<len(raw) and not np.any(raw[hi]>1):hi+=1
        lo=hi if lo<0 else lo;hi=lo if hi>=len(raw) else hi;w=(z-lo)/(hi-lo) if hi!=lo else 0
        actual[z]=np.rint((1-w)*raw[lo].astype(np.float32)+w*raw[hi].astype(np.float32)).astype(np.uint8)
    assert np.array_equal(expected,actual)
    c=ByteLRU(10);c.put(('cropA','batch8'),1,6);c.put(('cropA','batch2'),2,6);assert c.get(('cropA','batch8')) is None and c.get(('cropA','batch2'))==2
    with tempfile.TemporaryDirectory(prefix='mito_queue_') as tmp:
        p=Path(tmp);(p/'manifest.json').write_text(json.dumps([dict(index=i) for i in range(32)]))
        proc=subprocess.Popen([sys.executable,str(ROOT/'queue_service.py'),'--manifest',str(p/'manifest.json'),'--out',str(p),'--port','18087'],stdout=subprocess.DEVNULL)
        try:
            deadline=time.time()+10
            while not (p/'queue_access.json').exists():
                if time.time()>deadline:raise RuntimeError('Queue timeout')
                time.sleep(.1)
            token=json.loads((p/'queue_access.json').read_text())['token']
            def claim(i):
                r=urllib.request.Request('http://127.0.0.1:18087/claim',data=json.dumps(dict(token=token,worker=str(i))).encode())
                return json.load(urllib.request.urlopen(r))['job']['index']
            with ThreadPoolExecutor(max_workers=8) as pool:ids=list(pool.map(claim,range(32)))
            assert len(set(ids))==32
        finally:proc.terminate();proc.wait(timeout=10)
    print('PASS: cross32Z-chunk interpolation exact; batch-specific cache keys and byte eviction;32 concurrent dynamic claims have no duplicate')
if __name__=='__main__':main()
