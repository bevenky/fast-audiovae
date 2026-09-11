"""Read progress for the single selected continuation."""
from pathlib import Path
import json
import urllib.request

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
receipt = json.loads((root/'continuation-singleton-pid.json').read_text())
process = Path('/proc')/str(receipt['pid'])/'cmdline'
log = Path(receipt['log']).read_text()
rows = []
for line in log.splitlines():
    try:
        value = json.loads(line)
        if isinstance(value, dict): rows.append(value)
    except ValueError:
        pass
keys = {'stage','step','total','waveform','mel','feature','mae','nonquiet_cosine_mean',
        'quiet_residual_rms_mean','quiet_failed_windows','group_mse','elapsed_seconds'}
result = {'running':process.exists() and b'continue_settings.py' in process.read_bytes(),
          'completed':(root/'current-lowrate-continue1000-v1/completed.json').exists(),
          'last_events':[{k:v for k,v in row.items() if k in keys} for row in rows[-4:]]}
if 'Traceback' in log or not rows: result['log_tail'] = log[-2000:]
with urllib.request.urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags',timeout=3) as response:
    result['tensorboard_runs'] = list(json.load(response))
print(json.dumps(result))
