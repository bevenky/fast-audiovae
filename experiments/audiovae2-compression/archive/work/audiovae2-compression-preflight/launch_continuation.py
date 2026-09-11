"""Manually selected continuation after the completed settings and quiet audit."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
manifest = json.loads((root/'code-manifest-singleton.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == expected, name
extra = {'continue_settings.py':'79a993bf152f3adc2b8d7fed3d74778a15d67490d363a08e56a68baa9f8f7017',
         'test_continue_settings.py':'b08552ad7d4cfe43984d9098a062278a754b366e8162f3f85e8de78a932230ec'}
for name, expected in extra.items():
    assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == expected, name
audit = json.loads((root/'reference-and-quiet-addendum-v1.json').read_text())
assert audit['complete'] and audit['passed'] and audit['training_updates']==0
for item in audit['arms']:
    assert item['common_reproduction']['passed']
    assert item['quiet_diagnostics']['teacher_replay']['aggregate']['failed_windows']==0
assert shutil.disk_usage(root).free > 400*1024**2
for receipt in root.glob('*-singleton-pid.json'):
    prior = json.loads(receipt.read_text())
    process = Path('/proc')/str(prior['pid'])/'cmdline'
    assert not process.exists() or str(root/'code').encode() not in process.read_bytes(), receipt
name = 'current_lr3e-5'
out = root/'current-lowrate-continue1000-v1'
assert not out.exists()
log = root/'pilot/current-lowrate-continue1000-v1.log'
command = [python, '-u', str(root/'code/continue_settings.py'),
    '--checkpoint', str(root/'settings-screen-v1'/name/'final.pt'),
    '--screen-out', str(root/'settings-screen-v1'), '--base-out', str(root/'pilot'),
    '--manifest', str(root/'pilot-selection-v1.json'),
    '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets',
    '--out', str(out), '--tensorboard', str(root/'tensorboard/continuation-v1')]
env = dict(os.environ, PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log),
           'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'manually_selected_arm':name, 'source_sha256':extra,
           'quiet_audit_sha256':hashlib.sha256((root/'reference-and-quiet-addendum-v1.json').read_bytes()).hexdigest(),
           'reason':'Best supported waveform/expressive continuation; both spectral definitions also favor current over reference at the same lower rate. Quiet failures remain, targets replay exactly.'}
(root/'continuation-singleton-pid.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
