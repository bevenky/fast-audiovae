"""Reload only the reporting observer after a reporting-code update."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

base = Path('/workspace/fast-audiovae-convnext-20260909-r9')
out = base/'remediation/monitoring'
launch = json.loads((out/'launch.json').read_text())
pid = launch['observer_pid']
proc = Path('/proc')/str(pid)/'cmdline'
if proc.exists():
    args = proc.read_bytes().split(b'\0')
    if str(base/'run_recipe_v2_monitor.py').encode() not in args:
        raise ValueError('Observer PID no longer owns the expected command')
    os.kill(pid, signal.SIGINT)
    for _ in range(450):
        if not proc.exists() or not proc.read_bytes():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Observer did not stop; trainer remains untouched')
env = dict(os.environ, PYTHONPATH=str(base), CUDA_VISIBLE_DEVICES='')
command = ['/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python',
           str(base/'run_recipe_v2_monitor.py'), '--base', str(base)]
child = base/'training-runs/decoder-recipe-v2-expressive'
if (child/'run.json').is_file():
    command += ['--continuation-run', str(child)]
with (out/'observer.log').open('ab') as log:
    observer = subprocess.Popen(command, cwd=base, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(1)
if observer.poll() is not None:
    raise RuntimeError('Updated observer failed to start')
launch['observer_pid'] = observer.pid
(out/'launch.json').write_text(json.dumps(launch, indent=2))
(base/'tensorboard-server.json').write_text(json.dumps(launch, indent=2))
print(json.dumps({'old_observer_pid':pid,'observer_pid':observer.pid,'trainer_restarted':False}))
