"""Experimental replay batches, immutable initial/prefix reuse and decode broker.

Never changes crop/threshold/history rules. Batched arithmetic is explicitly
tracked because batch-size changes need not be bitwise equivalent on CUDA.
"""
import collections,hashlib,json,math,queue,threading,time
from concurrent.futures import Future
import numpy as np
import torch
from fast_runtime import ByteLRU

_ORIGINAL_DECODE=None

def tensor_bytes(x):
    if torch.is_tensor(x):return x.numel()*x.element_size()
    if isinstance(x,np.ndarray):return x.nbytes
    if isinstance(x,dict):return sum(tensor_bytes(v) for k,v in x.items() if not k.startswith('__'))
    if isinstance(x,(list,tuple)):return sum(tensor_bytes(v) for v in x)
    return 0

class FeatureProvider:
    def __init__(self,e,options,counters):
        self.e=e;self.options=options;self.counts=counters;self.original=e.fast_encoder.encode
        self.frames=ByteLRU(int(options.get('replay_cache_mib',512)*2**20))
        self.current_request=None;self.image_streams=None
    def key(self,x,y,z,side):return (int(x),int(y),int(z),int(side))
    def signature(self,x,y,z,side):
        value=self.frames.items.get(self.key(x,y,z,side))
        return value[0][1] if value else ('single',int(z),1,0)
    def encode(self,geometry,side):
        x,y,z,n=geometry
        r=self.current_request
        if n==1 and r is not None and (x,y,z,side)==(*r['t'].crop[:2],r['z'],r['t'].crop[2]):
            if r.get('feature_handle') is not None:self.counts['replay_feature_hits']+=1;return [r['feature_handle']]
            return self.original(geometry,side)
        if n==1:
            cached=self.frames.get(self.key(x,y,z,side))
            if cached is not None:self.counts['replay_feature_hits']+=1;return [cached[0]]
        return self.original(geometry,side)
    def prefill(self,t,zs):
        if not self.options.get('replay_batch_encoder'):return
        t._advanced_replay_plan=tuple(zs)
        if zs:self.ensure(t,zs[0])
    def ensure(self,t,z):
        if not self.options.get('replay_batch_encoder'):return
        plan=getattr(t,'_advanced_replay_plan',())
        if z not in plan:return
        x,y,side=t.crop;batch=int(self.options.get('replay_batch_size',4));batch=max(1,min(8,batch))
        if self.key(x,y,z,side) in self.frames.items:return
        zs=plan[plan.index(z):plan.index(z)+batch]
        # Reverse replay is still decoded backwards; forward_image is framewise.
        missing=sorted(z for z in zs if self.key(x,y,z,side) not in self.frames.items)
        if self.options.get('replay_mode')=='streams' and missing:
            if self.image_streams is None:self.image_streams=[torch.cuda.Stream() for _ in range(int(self.options.get('image_streams',2)))]
            default=torch.cuda.current_stream()
            for i,z in enumerate(missing):
                stream=self.image_streams[i%len(self.image_streams)];stream.wait_stream(default)
                with torch.cuda.stream(stream):encoded=self.original((x,y,z,1),side)[0]
                self.frames.put(self.key(x,y,z,side),(encoded,('single',z,1,0)),tensor_bytes(encoded))
            for stream in self.image_streams:default.wait_stream(stream)
            self.counts['replay_stream_groups']+=1;self.counts['replay_stream_frames']+=len(missing)
            return
        groups=[]
        for z in missing:
            if groups and len(groups[-1])<batch and z==groups[-1][-1]+1:groups[-1].append(z)
            else:groups.append([z])
        for zs in groups:
            encoded=self.original((x,y,zs[0],len(zs)),side);self.counts['replay_prefill_batches']+=1
            self.counts['replay_prefill_frames']+=len(zs)
            for i,(z,f) in enumerate(zip(zs,encoded)):
                sig=('single',z,1,0) if len(zs)==1 else ('batch',zs[0],len(zs),i)
                self.frames.put(self.key(x,y,z,side),(f,sig),tensor_bytes(f))

class AdvancedExecution:
    def __init__(self,e,method,options):
        self.e=e;self.method=method;self.o=dict(options);self.counts=collections.Counter();self.features=FeatureProvider(e,self.o,self.counts)
        self.namespace=(id(e.model),getattr(e,'provenance',{}).get('sam_sha256'),str(getattr(getattr(e,'nuc',None),'filename','per-engine-nucleus')))
        self.prompts=ByteLRU(int(self.o.get('prompt_cache_mib',128)*2**20));self.prefix=ByteLRU(int(self.o.get('prefix_cache_mib',512)*2**20))
        self.workers=int(self.o.get('track_workers',4)) if self.o.get('multi_track_decoder') else 1
        self.parallel_width=1;self.queue=queue.Queue();self.stop_event=threading.Event();self.thread=None
        self.decode_streams=None;self.warm_cond=False;self.warm_memory=False
        self.e.fast_encoder.encode=self.features.encode
        if self.workers>1:
            torch.cuda.synchronize()
            self.thread=threading.Thread(target=self._loop,name='sam-decode-broker',daemon=True);self.thread.start()
        self.lock=threading.RLock()
    def submit(self,kind,t,z=None,state=None,zs=None):
        request=dict(kind=kind,t=t,z=z,state=state,zs=zs,future=Future())
        if self.thread:self.queue.put(request);return request['future'].result()
        with self.lock:
            if kind=='prefill':return self.features.prefill(t,zs)
            return self._single(request)
    def prefill(self,t,zs):return self.submit('prefill',t,zs=zs)
    def decode(self,t,z,state):return self.submit('decode',t,z,state)
    def _keys(self,r):
        t,z,state=r['t'],r['z'],r['state'];x,y,s=t.crop;cond=z==t.seed['z'];j=z-self.method.engine.ROI[2]
        if 'feature_signature' not in r:
            self.features.ensure(t,z);item=self.features.frames.items.get(self.features.key(x,y,z,s))
            r['feature_signature']=item[0][1] if item else ('single',z,1,0)
            r['feature_handle']=item[0][0] if item else None
        base=(self.namespace,self.e.raw.source_tag,tuple(self.method.engine.ROI),x,y,s,t.seed['z'],tuple(t.seed['box_xyxy']),float(t.score_threshold),r['feature_signature'],('decoder_batch',r.get('decoder_batch',1)))
        if cond:full=('initial',base,j)
        else:
            prev=j+1 if t.reverse else j-1
            previous=state['non_cond_frame_outputs'].get(prev,state['cond_frame_outputs'].get(prev,{})).get('__adv_token')
            full=('prefix',base,j,bool(t.reverse),previous) if previous is not None else None
        token=hashlib.sha256(repr(full).encode()).hexdigest() if full is not None else None
        cache=self.prompts if cond and self.o.get('prompt_initialization_reuse') else (self.prefix if not cond and self.o.get('prefix_cache') else None)
        r.update(cond=cond,j=j,cache=cache,cache_key=full,token=token)
        return r
    def _attach(self,r,kept):
        t,state,j=r['t'],r['state'],r['j'];copy=dict(kept);copy['maskmem_pos_enc']=list(kept['maskmem_pos_enc']);copy['__adv_token']=r['token']
        state['cond_frame_outputs' if r['cond'] else 'non_cond_frame_outputs'][j]=copy
        m=self.e.model;retain=max(m.max_obj_ptrs_in_encoder,m.num_maskmem*getattr(m,'memory_temporal_stride_for_eval',1))+2
        for k in list(state['non_cond_frame_outputs']):
            if (not t.reverse and k<j-retain) or (t.reverse and k>j+retain):del state['non_cond_frame_outputs'][k]
    def _hit(self,r):
        self._keys(r);cache=r['cache'];key=r['cache_key']
        item=cache.get(key) if cache is not None and key is not None else None
        if item is None:return None
        mask,score,kept=item
        if kept is not None:self._attach(r,kept)
        self.counts['prompt_cache_hits' if r['cond'] else 'prefix_cache_hits']+=1
        return (mask,score)
    def _remember(self,r,mask,score,kept):
        if kept is not None:self._attach(r,kept)
        if r['cache'] is not None and r['cache_key'] is not None:
            stored=mask.copy() if mask is not None else None
            if stored is not None:stored.flags.writeable=False
            value=(stored,score,dict(kept) if kept is not None else None)
            r['cache'].put(r['cache_key'],value,max(2048,tensor_bytes(value)))
    @torch.inference_mode()
    def _single(self,r,looked_up=False):
        if not looked_up:
            hit=self._hit(r)
            if hit is not None:return hit
        self.features.current_request=r
        try:mask,score=_ORIGINAL_DECODE(r['t'],r['z'],r['state'])
        finally:self.features.current_request=None
        self.counts['single_decode_calls']+=1
        if r['cond'] and mask is not None:self.warm_cond=True
        if not r['cond']:self.warm_memory=True
        kept=r['state']['cond_frame_outputs' if r['cond'] else 'non_cond_frame_outputs'].get(r['j']) if mask is not None else None
        self._remember(r,mask,score,kept);return mask,score
    def _group_key(self,r):
        if self.o.get('decode_mode')=='streams':return ('independent_B1_streams',)
        t,state=r['t'],r['state']
        return (r['j'],r['cond'],t.reverse,tuple(sorted(state['cond_frame_outputs'])),tuple(sorted(state['non_cond_frame_outputs'])))
    def _pack(self,jobs):
        result={}
        for group in ['cond_frame_outputs','non_cond_frame_outputs']:
            result[group]={}
            for j in jobs[0]['state'][group]:
                values=[r['state'][group][j] for r in jobs]
                result[group][j]=dict(maskmem_features=torch.cat([v['maskmem_features'] for v in values],0),maskmem_pos_enc=[torch.cat([v['maskmem_pos_enc'][k] for v in values],0) for k in range(len(values[0]['maskmem_pos_enc']))],obj_ptr=torch.cat([v['obj_ptr'] for v in values],0))
        return result
    @torch.inference_mode()
    def _batch(self,jobs):
        if self.o.get('decode_mode')=='streams':return self._stream_group(jobs)
        m=self.e.model
        if len(jobs)==1 or getattr(m,'non_overlap_masks_for_mem_enc',False):return [self._single(r,True) for r in jobs]
        encoded=[r['feature_handle'] if r['feature_handle'] is not None else self.features.original((*r['t'].crop[:2],r['z'],1),r['t'].crop[2])[0] for r in jobs]
        merged={k:[torch.cat([f[k][i] for f in encoded],0) for i in range(len(encoded[0][k]))] for k in ['backbone_fpn','vision_pos_enc']}
        point=None
        if jobs[0]['cond']:
            points=[]
            for r in jobs:
                x,y,s=r['t'].crop;b=np.asarray(r['t'].seed['box_xyxy'])-np.array([x,y,x,y]);points.append(torch.tensor(b,dtype=torch.float32,device='cuda').reshape(1,2,2)/s*1024)
            point=dict(point_coords=torch.cat(points,0),point_labels=torch.tensor([[2,3]]*len(jobs),dtype=torch.int32,device='cuda'))
        state=self._pack(jobs)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,feat,pos,sizes=m._prepare_backbone_features(merged)
            cur,outs,_,_=m._track_step(jobs[0]['j'],jobs[0]['cond'],feat,pos,sizes,point,None,state,self.method.engine.ROI[5]-self.method.engine.ROI[2],jobs[0]['t'].reverse,None)
            _,_,ious,lo,hi,ptr,obj=outs;cur.update(pred_masks=lo,obj_ptr=ptr)
            m._encode_memory_in_output(feat,sizes,point,True,hi,obj,cur)
        scores=ious.float().reshape(len(jobs),-1)[:,0].cpu().tolist();result=[]
        for i,(r,score) in enumerate(zip(jobs,scores)):
            if not math.isfinite(score):raise ValueError('Nonfinite batch score')
            if score<r['t'].score_threshold:
                self._remember(r,None,score,None);result.append((None,score));continue
            x,y,s=r['t'].crop;mask=torch.nn.functional.interpolate(hi[i:i+1].float(),size=(s,s),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()>=.5
            mask[self.e.nuc_crop(r['z'],x,y,s)]=False;h,w=self.e.nuc.shape[1:];valid=np.zeros_like(mask);lx,ly=max(0,-x),max(0,-y);hx,hy=min(s,w-x),min(s,h-y)
            if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
            mask &= valid
            kept=dict(maskmem_features=cur['maskmem_features'][i:i+1].clone(),maskmem_pos_enc=[v[i:i+1].clone() for v in cur['maskmem_pos_enc']],obj_ptr=ptr[i:i+1].clone())
            if not getattr(self.e,'keep_memory_on_gpu',False):kept['maskmem_features']=kept['maskmem_features'].cpu();kept['maskmem_pos_enc']=[v.cpu() for v in kept['maskmem_pos_enc']]
            self._remember(r,mask,score,kept);result.append((mask,score))
        self.counts['batched_decode_calls']+=1;self.counts['batched_decode_items']+=len(jobs);self.counts['batch_size_'+str(len(jobs))]+=1
        return result
    @torch.inference_mode()
    def _stream_group(self,jobs):
        # All math retains B1 shapes. Warm lazy positional caches on the default
        # stream before independent forwards are allowed to overlap.
        if len(jobs)==1:return [self._single(jobs[0],True)]
        results={};pending=list(enumerate(jobs))
        for kind in ['cond','memory']:
            while pending and not (self.warm_cond if kind=='cond' else self.warm_memory):
                candidate=next(((i,r) for i,r in pending if r['cond']==(kind=='cond')),None)
                if candidate is None:break
                i,r=candidate;results[i]=self._single(r,True);pending.remove(candidate)
        if not pending:return [results[i] for i in range(len(jobs))]
        if self.decode_streams is None:self.decode_streams=[torch.cuda.Stream() for _ in range(int(self.o.get('decode_streams',4)))]
        m=self.e.model;default=torch.cuda.current_stream();records=[]
        for number,(i,r) in enumerate(pending):
            enc=r['feature_handle'] if r['feature_handle'] is not None else self.features.original((*r['t'].crop[:2],r['z'],1),r['t'].crop[2])[0]
            stream=self.decode_streams[number%len(self.decode_streams)];stream.wait_stream(default);point=None
            with torch.cuda.stream(stream):
                if r['cond']:
                    x,y,s=r['t'].crop;b=np.asarray(r['t'].seed['box_xyxy'])-np.array([x,y,x,y]);point=dict(point_coords=torch.tensor(b,dtype=torch.float32,device='cuda').reshape(1,2,2)/s*1024,point_labels=torch.tensor([[2,3]],dtype=torch.int32,device='cuda'))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,feat,pos,sizes=m._prepare_backbone_features(enc)
                    cur,outs,_,_=m._track_step(r['j'],r['cond'],feat,pos,sizes,point,None,r['state'],self.method.engine.ROI[5]-self.method.engine.ROI[2],r['t'].reverse,None)
            records.append(dict(i=i,r=r,stream=stream,point=point,feat=feat,sizes=sizes,cur=cur,outs=outs))
        for record in records:
            r=record['r'];stream=record['stream'];stream.synchronize();_,_,ious,lo,hi,ptr,obj=record['outs'];score=float(ious.float().reshape(-1)[0]);record['score']=score
            if not math.isfinite(score):raise ValueError('Nonfinite stream score')
            if score<r['t'].score_threshold:continue
            with torch.cuda.stream(stream):
                record['cur'].update(pred_masks=lo,obj_ptr=ptr)
                with torch.autocast('cuda',dtype=torch.bfloat16):m._encode_memory_in_output(record['feat'],record['sizes'],record['point'],True,hi,obj,record['cur'])
                record['prob']=torch.nn.functional.interpolate(hi.float(),size=(r['t'].crop[2],)*2,mode='bilinear',align_corners=False).sigmoid()[0,0]
        for record in records:
            i,r,score=record['i'],record['r'],record['score'];stream=record['stream']
            if score<r['t'].score_threshold:self._remember(r,None,score,None);results[i]=(None,score);continue
            stream.synchronize()
            with torch.cuda.stream(stream):mask=record['prob'].cpu().numpy()>=.5
            x,y,s=r['t'].crop;mask[self.e.nuc_crop(r['z'],x,y,s)]=False;h,w=self.e.nuc.shape[1:];valid=np.zeros_like(mask);lx,ly=max(0,-x),max(0,-y);hx,hy=min(s,w-x),min(s,h-y)
            if hx>lx and hy>ly:valid[ly:hy,lx:hx]=True
            mask &= valid;kept={k:record['cur'][k] for k in ['maskmem_features','maskmem_pos_enc','obj_ptr']}
            if not getattr(self.e,'keep_memory_on_gpu',False):kept['maskmem_features']=kept['maskmem_features'].cpu();kept['maskmem_pos_enc']=[v.cpu() for v in kept['maskmem_pos_enc']]
            self._remember(r,mask,score,kept);results[i]=(mask,score)
        self.counts['stream_decode_groups']+=1;self.counts['stream_decode_items']+=len(records)
        return [results[i] for i in range(len(jobs))]
    def _loop(self):
        while not self.stop_event.is_set():
            try:r=self.queue.get(timeout=.05)
            except queue.Empty:continue
            jobs=[r]
            deadline=time.perf_counter()+float(self.o.get('batch_wait_ms',1))/1000
            while len(jobs)<int(self.o.get('decode_batch_size',4)):
                try:jobs.append(self.queue.get(timeout=max(0,deadline-time.perf_counter()) if self.parallel_width>1 else 0))
                except queue.Empty:break
            groups=collections.defaultdict(list)
            try:
                for r in jobs:
                    if r['kind']=='prefill':self.features.prefill(r['t'],r['zs']);r['future'].set_result(None);continue
                    self._keys(r);groups[self._group_key(r)].append(r)
                for group in groups.values():
                    if getattr(self.e.model,'non_overlap_masks_for_mem_enc',False) and self.o.get('decode_mode')!='streams':
                        for r in group:r['decoder_batch']=1;r['future'].set_result(self._single(r))
                        continue
                    # Include actual arithmetic batch provenance in cache keys.
                    # Cache hits shrink the group; reconsider the remaining keys
                    # before computing a smaller batch.
                    pending=group
                    while pending:
                        missing=[]
                        for r in pending:
                            r['decoder_batch']=1 if self.o.get('decode_mode')=='streams' else len(pending);hit=self._hit(r)
                            if hit is not None:r['future'].set_result(hit)
                            else:missing.append(r)
                        if len(missing)==len(pending):
                            for r,result in zip(missing,self._batch(missing)):r['future'].set_result(result)
                            break
                        pending=missing
            except BaseException as exc:
                for r in jobs:
                    if not r['future'].done():r['future'].set_exception(exc)
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=30)
        return dict(options=self.o,counters=dict(self.counts),prompt_cache_bytes=self.prompts.bytes,prefix_cache_bytes=self.prefix.bytes,replay_cache_bytes=self.features.frames.bytes)

def install(e,method,options):
    global _ORIGINAL_DECODE
    cls=method.engine.Tracker
    if _ORIGINAL_DECODE is None:
        _ORIGINAL_DECODE=cls.decode
        def routed(t,z,state):
            manager=getattr(t.e,'advanced_execution',None)
            return manager.decode(t,z,state) if manager else _ORIGINAL_DECODE(t,z,state)
        cls.decode=routed
    manager=AdvancedExecution(e,method,options);e.advanced_execution=manager
    original_progress=method.progress;lock=threading.Lock()
    def progress(*args,**kwargs):
        with lock:return original_progress(*args,**kwargs)
    method.progress=progress
    return manager
