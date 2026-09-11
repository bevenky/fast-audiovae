"""Point TensorBoard at safe display labels while raw training logs continue."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time
import urllib.request

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
display = root/'tensorboard-display-v2'
assert len(list(display.glob('*/events.out.tfevents.*'))) == 12
receipt_path = root/'tensorboard-unified-service.json'
previous = json.loads(receipt_path.read_text())
pid = previous['pid']
args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
assert b'tensorboard.main' in args and str(root/'tensorboard-display').encode() in args
backup = root/'tensorboard-service-before-display-v2.json'
assert not backup.exists()
backup.write_text(json.dumps(previous, indent=2)+'\n')
os.kill(pid, signal.SIGTERM)
for _ in range(100):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8888/data/environment', timeout=.2):
            pass
    except Exception:
        break
    time.sleep(.1)
else:
    raise RuntimeError('Previous TensorBoard still accepts connections')
command = ['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python', '-m', 'tensorboard.main',
    '--logdir', str(display), '--host', '0.0.0.0', '--port', '8888',
    '--reload_interval', '5', '--load_fast', 'false', '--reload_multifile', 'true',
    '--reload_multifile_inactive_secs', '86400']
log = root/'tensorboard-display-v2.log'
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log),
    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'previous_pid':pid,
    'raw_logdir':str(root/'tensorboard-unified'), 'display_logdir':str(display)}
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
