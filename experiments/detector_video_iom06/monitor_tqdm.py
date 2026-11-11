"""Read-only live monitor; closing this process never stops training."""
import json
import time
import urllib.request
from tqdm import tqdm

URL = 'http://127.0.0.1:18071/training_pipeline/curves.json'
print('113 | Detector + SAM2 Video | READ ONLY | refresh 10s', flush=True)
print('Closing this window does not stop training.', flush=True)
bar = None
key = None
last_report = None
while True:
    try:
        with urllib.request.urlopen(URL, timeout=15) as r:
            data = json.load(r)
        state = data.get('state', {})
        video = 'video' in state.get('stage', '').lower()
        rows = data.get('video10_steps' if video else 'detector10_steps', [])
        step = rows[-1] if rows else {}
        epoch = step.get('epoch', 0)
        total = step.get('sequences_total', step.get('micro_steps_in_epoch', step.get('steps_in_epoch', 0)))
        current = step.get('sequence_step', step.get('micro_step', step.get('step', 0)))
        newkey = (state.get('stage'), epoch, total)
        if key != newkey:
            if bar:
                bar.close()
            bar = tqdm(total=total or 1, desc=f"{state.get('stage')} epoch {epoch}/10", dynamic_ncols=True,
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}')
            key = newkey
        bar.n = current
        fields = {k: round(step[k], 5) if isinstance(step[k], float) else step[k]
                  for k in ('loss', 'dice', 'mask_iou', 'cls', 'giou', 'dfl', 'optimizer_update') if k in step}
        fields['updated'] = data.get('updated', '')[11:19]
        bar.set_postfix(fields, refresh=True)
        epochs = data.get('video10_epochs' if video else 'detector10_epochs', [])
        report = (state.get('stage'), len(epochs))
        if report != last_report:
            tqdm.write('Latest completed epoch: ' + (json.dumps(epochs[-1], ensure_ascii=False) if epochs else 'waiting for validation'))
            last_report = report
        if state.get('status') in ('completed', 'needs_attention', 'error'):
            tqdm.write(json.dumps(state, ensure_ascii=False))
    except KeyboardInterrupt:
        break
    except Exception as e:
        tqdm.write(f'Monitor connection unavailable: {e}; retrying. Training is independent.')
    time.sleep(10)
if bar:
    bar.close()
