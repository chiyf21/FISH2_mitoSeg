"""Five-step mask-tight-box reconditioning and full 3D instance presentation."""
import json,time,traceback,copy
import sequential3154_lookahead10 as runner
from refresh5_rules import tight_prompt
from infer_block import save

engine=runner.engine
BaseTracker=engine.Tracker
engine.RUN=engine.ROOT/'sequential3154_nms04_nuc06_refresh5_cc26_20261103'
engine.GALLERY=engine.TRAIN/'dashboard/sam_video_hardcases_20261102/sequential3154_nms04_nuc06_refresh5_cc26'
engine.DETECTIONS_PATH=engine.RUN/'detections_filtered.json'
runner.CONFIG_EXTRA=dict(refresh_every_forward_steps=5,refresh_prompt='tight bbox of propagated mask, zero padding; same layer re-inferred as conditioning frame',refresh_memory='new independent segment memory; same track ID',assembly='union all same-Z masks,26CC,keep >=1000 voxels and >=2 Z slices')

class RefreshTracker:
    def __init__(self,e,seed,tid):
        self.e=e;self.seed=seed;self.id=tid;self.inner=BaseTracker(e,seed,tid)
        self.frames=[];self._history=[];self.active=True;self.stop=None;self.prompts=[dict(z=seed['z'],box_xyxy=seed['box_xyxy'],source='detector',segment=0)]
        self.segment=0;self.state={}

    def advance(self,z):
        if not self.active:return None
        result=self.inner.advance(z)
        if result is None:
            self.stop=self.inner.stop;self.active=False;return None
        mask,f=result
        offset=z-self.seed['z']
        if offset>0 and offset%5==0 and offset<10:
            box=tight_prompt(mask,f['origin_xy'])
            if box is None:
                self.stop=dict(reason='empty_mask_at_refresh',z=z);self.active=False;return None
            else:
                self._history.extend(self.inner.history);self.inner.state={};self.segment+=1
                newseed={**self.seed,'z':z,'box_xyxy':box}
                self.prompts.append(dict(z=z,box_xyxy=box,source='predicted_mask_tight_bbox',padding=0,segment=self.segment,derived_from_score=f['score']))
                self.inner=BaseTracker(self.e,newseed,self.id);refreshed=self.inner.advance(z)
                if refreshed is None:
                    self.stop=dict(reason='refresh_prompt_low_score',z=z,detail=self.inner.stop);self.active=False;return None
                else:mask,f=refreshed
        f=dict(f,segment=self.segment,is_prompt_frame=z==self.prompts[-1]['z'],conditioning_z=self.prompts[-1]['z'],prompt_box_xyxy=self.prompts[-1]['box_xyxy'])
        self.frames.append(f)
        # publish() expects crop movement history. Prompt history is saved here
        # independently because the common publisher only knows original seeds.
        save(engine.RUN/'tracks'/f'prompts_{self.id:05d}.json',self.prompts)
        return mask,f

    @property
    def history(self):return self._history+self.inner.history

def main():
    engine.Tracker=RefreshTracker
    runner.main()
    (engine.RUN/'completed.json').replace(engine.RUN/'inference_completed.json')
    from assemble_refresh5 import main as assemble
    assemble(engine.RUN,engine.GALLERY)

if __name__=='__main__':
    try:main()
    except Exception:
        save(engine.RUN/'error.json',dict(trace=traceback.format_exc(),time=time.time()));raise
