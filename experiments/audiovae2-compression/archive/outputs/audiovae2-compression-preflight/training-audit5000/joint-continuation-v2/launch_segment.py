"""Launch only the explicitly authorized first continuation segment."""
import json
import os
from pathlib import Path
import subprocess
import time

P = Path('/workspace/fast-audiovae-compression-20260910-v1')
R = Path('/tmp/fast-audiovae-joint-recovery-v2')
PY = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
DATA = '/tmp/fast-audiovae-continuation-data-v2'
env = dict(os.environ, PYTHONPATH=DATA+':'+str(P/'code'))
out = R/'segment-5625-6625'
receipt = R/'launch-step5625.json'
if receipt.exists() or out.exists():
    raise FileExistsError('The first continuation segment has already been launched')
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError):
        continue
    if any(Path(arg.decode(errors='replace')).name in ('joint_recovery_v1.py', 'joint_recovery_v2.py') for arg in args):
        raise RuntimeError('A training worker is already running')
plan = Path('/tmp/fast-audiovae-continuation-plan-v3/plan.json')
if not plan.is_file():
    raise FileNotFoundError('The extended source plan must be sealed before launch')
R.mkdir(exist_ok=True)
command = [PY, '-u', str(P/'code/joint_recovery_v2.py'),
    '--checkpoint', str(P/'current-lowrate-fresh5000-v1/checkpoint-step4500.pt'),
    '--candidate', '/tmp/fast-audiovae-accumulation-comparison-v1/accumulation12/final.pt',
    '--anchor-checkpoint', str(P/'current-lowrate-continue1000-v1/checkpoint-step1000.pt'),
    '--screen-out', str(P/'settings-screen-v1'), '--base-out', str(P/'pilot'),
    '--manifest', str(P/'pilot-selection-v1.json'),
    '--fresh-manifest', str(P/'fresh-source-plan-v2/plan.json'),
    '--shards', '/dev/shm/fast-audiovae-compression-fresh-shards-v2',
    '--extended-manifest', str(plan),
    '--extended-shards', '/dev/shm/fast-audiovae-continuation-shards-v2',
    '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets',
    '--joint-run', '/tmp/fast-audiovae-joint-recovery-v1',
    '--tensorboard-logdir', str(R/'tensorboard/raw'), '--out', str(out), '--updates', '1000']
with (R/'segment-5625-6625.log').open('xb') as log:
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=P)
result = {'pid': process.pid, 'command': command, 'pythonpath': env['PYTHONPATH'],
          'launched_unix': time.time(), 'start_step': 5625, 'target_step': 6625,
          'training_settings_changed': False, 'later_segments_require_review': True}
receipt.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
