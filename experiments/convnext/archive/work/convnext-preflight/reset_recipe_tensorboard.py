import json, os, signal, subprocess, time, urllib.request
from pathlib import Path
base = Path('/workspace/fast-audiovae-convnext-20260909-r9')
old = Path('/workspace/fast-audiovae-convnext-20260908-r1')
base.mkdir(exist_ok=True)
logs = base / 'tensorboard'
logs.mkdir(exist_ok=True)
pid = 861340
proc = Path('/proc') / str(pid) / 'cmdline'
if proc.exists():
    command = proc.read_bytes().replace(b'\0', b' ').decode()
    if 'tensorboard' not in command or '--logdir ' + str(old / 'runs') not in command:
        raise ValueError('The identified dashboard process changed')
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        if not proc.exists():
            break
        time.sleep(.1)
command = [str(old / '.train-venv/bin/tensorboard'), '--logdir', str(logs),
           '--host', '0.0.0.0', '--port', '8888', '--reload_interval', '5', '--load_fast', 'false']
with (base / 'tensorboard-server.log').open('ab') as handle:
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
result = {'pid': process.pid, 'logdir': str(logs), 'historical_logs_preserved': str(old / 'runs'),
          'url': 'https://34d6pb4ub5ldrz-8888.proxy.runpod.net/'}
(base / 'tensorboard-server.json').write_text(json.dumps(result, indent=2))
for _ in range(50):
    if process.poll() is not None:
        raise RuntimeError('Dashboard exited; inspect server log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8888/data/runs', timeout=2) as response:
            result['visible_runs'] = json.load(response)
        break
    except OSError:
        time.sleep(.2)
else:
    raise RuntimeError('New dashboard did not become ready')
print(json.dumps(result))
