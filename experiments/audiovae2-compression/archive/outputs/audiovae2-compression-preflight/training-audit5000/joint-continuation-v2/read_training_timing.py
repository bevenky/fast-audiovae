"""Summarize existing timing records without changing or evaluating the model."""
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('/tmp/fast-audiovae-joint-recovery-v2')
path = root / 'segment-5625-6625/train.jsonl'
lines = path.read_text().splitlines()
records = []
for i, line in enumerate(lines):
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        if i != len(lines)-1:
            raise
def summary(rows, start_elapsed):
    wall = rows[-1]['elapsed_seconds'] - start_elapsed
    update = sum(row['step_seconds'] for row in rows)
    return {'updates': len(rows), 'wall_seconds': wall,
            'measured_update_seconds': update,
            'other_seconds': wall-update,
            'update_fraction': update/wall,
            'mean_update_seconds': update/len(rows),
            'wall_seconds_per_update': wall/len(rows)}
log = (root / 'segment-5625-6625.log').read_text().splitlines()
events = []
for line in log:
    if line.startswith('{'):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
n = min(200,len(records))
start = records[-n-1]['elapsed_seconds'] if len(records)>n else 0
launch = json.loads((path.parent / 'launch.json').read_text())
result = {'observed_utc':datetime.now(timezone.utc).isoformat(),
          'last_step':records[-1]['step'],
          'batch': {k:launch[k] for k in ('execution_batch_size','gradient_accumulation','trainable_stages','learning_rate')},
          'whole_segment':summary(records,0),
          'last200_updates':summary(records[-n:],start),
          'data_wait_events':sum(e.get('stage')=='waiting_for_sealed_training_data' for e in events),
          'interpretation':'Update time includes model and loss execution, backpropagation, teacher/cache comparison and optimizer work. Other time combines data waits, fetches, warmup, evaluation/checkpointing and logging; it is not all GPU idle time. The producer also uses the same GPU.'}
(root / 'training-timing-snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
