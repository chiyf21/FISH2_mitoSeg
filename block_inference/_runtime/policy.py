import json
from pathlib import Path
def load_policy():
    p=json.loads((Path(__file__).parent/'inference_policy.json').read_text())
    assert p['score_stop_count']==1 and p['stop_comparison']=='strictly_less_than' and p['exclude_low_score_frame']
    assert 0<=p['score_threshold']<=1
    return p
def apply_policy(engine):
    p=load_policy();engine.score_threshold=float(p['score_threshold']);engine.score_stop_count=int(p['score_stop_count']);engine.inference_policy=p
    return engine
