"""Serve the validated continuation events and preserve the previous dashboard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

P = Path('/workspace/fast-audiovae-compression-20260910-v1')
R = Path('/tmp/fast-audiovae-joint-recovery-v2')
OLD = Path('/tmp/fast-audiovae-joint-recovery-v1')
PY = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
receipt = R/'display-launch.json'
if receipt.exists():
    raise FileExistsError('Continuation display is already launched')
check = json.loads((R/'segment-5625-6625/restore-check.json').read_text())
if not check['saved_quality']['passed'] or not all(v['equal'] for v in check['state'].values()):
    raise RuntimeError('Restored model must pass before switching the display')
env = dict(os.environ, PYTHONPATH=str(P/'code'))
command = [PY, '-u', str(P/'code/joint_recovery_display_v2.py'), '--source-logdir', str(R/'tensorboard/raw'),
           '--display-logdir', str(R/'tensorboard/display')]
with (R/'display.log').open('xb') as log:
    projector = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=P)
for _ in range(30):
    if (R/'tensorboard/display/projection-ready.json').is_file():
        break
    if projector.poll() is not None:
        raise RuntimeError('Projection failed; original server preserved')
    time.sleep(1)
else:
    raise TimeoutError('New projection did not become ready')
old_processes = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = [x.decode() for x in path.read_bytes().split(b'\0') if x]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        continue
    is_server = 'tensorboard.main' in args and str(OLD/'tensorboard/display') in args and '8888' in args
    is_projector = any(Path(x).name == 'joint_recovery_display_v1.py' for x in args) and str(OLD/'tensorboard/raw') in args
    if is_server or is_projector:
        old_processes.append({'pid':int(path.parent.name), 'command':args})
for item in old_processes:
    os.kill(item['pid'], signal.SIGTERM)
for _ in range(15):
    try:
        urllib.request.urlopen('http://127.0.0.1:8888/', timeout=1)
    except Exception:
        break
    time.sleep(1)
server_command = [PY, '-m', 'tensorboard.main', '--logdir', str(R/'tensorboard/display'),
    '--host', '0.0.0.0', '--port', '8888', '--reload_interval', '5', '--load_fast', 'false',
    '--reload_multifile', 'true', '--reload_multifile_inactive_secs', '86400']
with (R/'tensorboard.log').open('xb') as log:
    server = subprocess.Popen(server_command, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True, cwd=P)
for _ in range(30):
    if server.poll() is not None:
        raise RuntimeError('New TensorBoard server failed')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags',timeout=2) as response:
            tags = json.load(response)
        if len(tags) >= 15:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise TimeoutError('Continuation metric runs did not appear')
result = {'projector_pid':projector.pid,'tensorboard_pid':server.pid,'projector_command':command,
    'server_command':server_command,'historical_processes':old_processes,'historical_logs_preserved':True,
    'visible_runs':sorted(tags),'launched_unix':time.time()}
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
