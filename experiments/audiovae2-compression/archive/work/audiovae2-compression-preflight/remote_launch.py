"""Task-local launcher for the approved bounded Runpod experiment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
assets = '/workspace/fast-audiovae-convnext-20260908-r1/assets'
mode = sys.argv[1]
if mode not in ('preflight', 'export', 'streaming', 'train', 'screen'):
    raise ValueError('Unknown experiment phase')
manifest = json.loads((root/'code-manifest-singleton.json').read_text())
for name, expected in manifest['files'].items():
    if hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() != expected:
        raise ValueError('Packaged source changed: '+name)
for phase in ('preflight', 'export', 'streaming', 'train', 'screen'):
    pid_file = root/(phase+'-singleton-pid.json')
    if pid_file.exists():
        prior = json.loads(pid_file.read_text())
        cmdline = Path('/proc')/str(prior['pid'])/'cmdline'
        if cmdline.exists() and str(root/'code').encode() in cmdline.read_bytes():
            raise RuntimeError('Experiment phase still running: '+phase)
out = root/'pilot'
out.mkdir(exist_ok=True)
common = ['--assets', assets, '--out', str(out)]
if mode == 'screen':
    command = [python, '-u', str(root/'code/settings_screen.py'), '--assets', assets,
               '--base-out', str(out), '--manifest', str(root/'pilot-selection-v1.json'),
               '--out', str(root/'settings-screen-v1'),
               '--tensorboard', str(root/'tensorboard/settings-screen-v1')]
elif mode in ('preflight', 'train'):
    command = [python, '-u', str(root/'code/run_pilot.py'), mode, *common,
               '--manifest', str(root/'pilot-selection-v1.json')]
    if mode == 'train':
        command += ['--steps', '1000', '--tensorboard', str(root/'tensorboard/group-width-v1')]
else:
    name = 'export_preflight.py' if mode == 'export' else 'streaming_preflight.py'
    command = [python, '-u', str(root/'code'/name), *common]
    if mode == 'streaming':
        command += ['--manifest', str(root/'pilot-selection-v1.json')]
log = out/(mode+'.log')
if log.exists():
    raise FileExistsError('Preserve the prior phase log before explicitly relaunching')
env = dict(os.environ, PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
if mode in ('export', 'streaming'):
    env['CUDA_VISIBLE_DEVICES'] = ''
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, cwd=root, env=env,
                               start_new_session=True)
receipt = {'phase': mode, 'pid': process.pid, 'command': command, 'log': str(log),
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'archive_sha256': manifest['archive_sha256']}
(root/(mode+'-singleton-pid.json')).write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
