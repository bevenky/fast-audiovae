"""Small read-only status snapshot for the settings comparison."""
from pathlib import Path
import json
import subprocess
import urllib.request

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
log = (root/'pilot/screen.log').read_text()
rows = []
for line in log.splitlines():
    try:
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    except ValueError:
        pass
keep = {'stage', 'arm', 'event', 'step', 'elapsed_seconds', 'total', 'waveform',
        'mel', 'feature', 'group_mse', 'nonquiet_cosine_mean', 'quiet_residual_rms_mean',
        'gradient_norm', 'gpu_memory_gib'}
receipt = json.loads((root/'screen-singleton-pid.json').read_text())
process = Path('/proc')/str(receipt['pid'])/'cmdline'
result = {'running': process.exists() and b'settings_screen.py' in process.read_bytes(),
          'last_events': [{k:v for k,v in row.items() if k in keep} for row in rows[-4:]],
          'complete': (root/'settings-screen-v1/completed.json').exists()}
if not result['last_events'] or 'Traceback' in log:
    result['log_tail'] = log[-1800:]
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                        '--format=csv,noheader'], text=True).strip()
try:
    with urllib.request.urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags', timeout=3) as response:
        result['tensorboard_runs'] = list(json.load(response))
except Exception as error:
    result['tensorboard_error'] = str(error)
print(json.dumps(result))
