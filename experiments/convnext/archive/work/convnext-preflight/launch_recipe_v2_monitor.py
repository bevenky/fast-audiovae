"""Switch only TensorBoard to the audited reporting view; keep trainer intact."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

base = Path('/workspace/fast-audiovae-convnext-20260909-r9')
runtime = Path('/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin')
out = base / 'remediation/monitoring'
state = json.loads((out / 'state.json').read_text())
if state['step'] < 1 or not state['evaluations']:
    raise ValueError('Reporting view must pass its first saved-metrics load')
pid_file = out / 'launch.json'
if pid_file.exists():
    raise ValueError('Observer launch already exists; inspect before restarting')
command = [str(runtime/'python'), str(base/'run_recipe_v2_monitor.py'), '--base', str(base)]
env = dict(os.environ, PYTHONPATH=str(base), CUDA_VISIBLE_DEVICES='')
with (out/'observer.log').open('ab') as log:
    observer = subprocess.Popen(command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
time.sleep(1)
if observer.poll() is not None:
    raise RuntimeError('Observer exited before dashboard switch')
old = json.loads((base/'tensorboard-server.json').read_text())
(out/'previous-tensorboard-server.json').write_text(json.dumps(old, indent=2))
process_path = Path('/proc')/str(old['pid'])/'cmdline'
if process_path.exists():
    args = process_path.read_bytes().split(b'\0')
    decoded = [v.decode() for v in args if v]
    if '--logdir' not in decoded or decoded[decoded.index('--logdir')+1] != str(base/'tensorboard'):
        raise ValueError('Unexpected dashboard command; refusing to signal it')
    if not any('tensorboard' in arg for arg in decoded):
        raise ValueError('Identified process is not TensorBoard')
    os.kill(old['pid'], signal.SIGTERM)
    time.sleep(2)
logs = out/'tensorboard'
command = [str(runtime/'tensorboard'), '--logdir', str(logs), '--host', '0.0.0.0',
           '--port', '8888', '--reload_interval', '5', '--load_fast', 'false']
with (out/'tensorboard.log').open('ab') as log:
    dashboard = subprocess.Popen(command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
result = {'pid': dashboard.pid, 'observer_pid': observer.pid, 'logdir': str(logs),
          'original_training_logs': str(base/'tensorboard'), 'trainer_restarted': False,
          'url': 'https://34d6pb4ub5ldrz-8888.proxy.runpod.net/'}
for _ in range(60):
    if dashboard.poll() is not None:
        raise RuntimeError('New TensorBoard exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags', timeout=2) as response:
            tags = json.load(response)
        run_tags = tags.get('decoder-recipe-v2', {})
        if 'loss/reconstruction_only' not in run_tags:
            time.sleep(.5)
            continue
        if 'train/total' in run_tags:
            raise ValueError('Misleading total remains in new dashboard')
        result.update(visible_runs=list(tags), scalar_tags=len(run_tags),
                      reconstruction_label_verified=True, misleading_total_absent=True)
        break
    except OSError:
        time.sleep(.5)
else:
    raise RuntimeError('New dashboard did not expose the corrected scalar tags')
(base/'tensorboard-server.json').write_text(json.dumps(result, indent=2))
pid_file.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
