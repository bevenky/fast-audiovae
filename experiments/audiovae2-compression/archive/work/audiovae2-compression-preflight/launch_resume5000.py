"""Resume the selected student to the user's 5,000-update decision point."""
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
extra = {
    'continue_settings.py': '79a993bf152f3adc2b8d7fed3d74778a15d67490d363a08e56a68baa9f8f7017',
    'resume_settings.py': '91c95d3954199d7eea9bf028531e51326866367c52a6e3f73b1fdedb3ff35b16',
    'test_resume_settings.py': 'ffdd68777d91d36727d85e466ec26f2684fe6330c759cc46c9d6181bec45b9d1',
    'fresh_training_data.py': '164a0030e8d7986dd679d32a54bbc7440f1eda88efcb09ed237bc5f93f5e9aad',
    'unified_monitor.py': '04574ab7c9a62c0c472d338f6dc2557221244a9169a59e4761d74012ddb7c20c',
    'test_unified_monitor.py': '757d0e2e4cc840525f2720d972672beff55178dc615f0e716944442e6cea3547',
}
for name, expected in {**manifest['files'], **extra}.items():
    assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == expected, name
anchor = root/'current-lowrate-continue1000-v1/checkpoint-step1000.pt'
assert hashlib.sha256(anchor.read_bytes()).hexdigest() == 'bfecd759eb58db1eeeca3dd693660b26be91bd69a93e3457b9d91523c5de0573'
plan_path = root/'fresh-source-plan-v2/plan.json'
plan = json.loads(plan_path.read_text())
assert plan['identity_sha256'] == '3c65151fd2b37900257e179fee62218c3d33055c94d47dbb7c213593777e7c7d'
shards = Path('/dev/shm/fast-audiovae-compression-fresh-shards-v2')
seed = json.loads((shards/'000000-000300/receipt.json').read_text())
assert seed['complete'] and seed['plan_identity_sha256'] == plan['identity_sha256']
assert shutil.disk_usage(root).free > 600*1024**2
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    for name in ('resume_settings.py', 'continue_settings.py', 'settings_screen.py', 'run_pilot.py'):
        assert str(root/'code'/name).encode() not in args, f'Existing training process {proc.name}'
out = root/'current-lowrate-fresh5000-v1'
assert not out.exists()
log = root/'current-lowrate-fresh5000-v1.log'
command = [python, '-u', str(root/'code/resume_settings.py'),
    '--checkpoint', str(anchor), '--screen-out', str(root/'settings-screen-v1'),
    '--base-out', str(root/'pilot'), '--manifest', str(root/'pilot-selection-v1.json'),
    '--fresh-manifest', str(plan_path), '--shards', str(shards),
    '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets',
    '--out', str(out), '--tensorboard', str(root/'tensorboard-unified/active'),
    '--target-step', '5000', '--effective-batch-size', '3']
env = dict(os.environ, PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log),
    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'source_sha256':extra, 'fresh_plan_identity_sha256':plan['identity_sha256'],
    'target_step':5000, 'execution_batch_size':1, 'accumulation_steps':3,
    'automatic_promotion':False, 'stop_for_quality_decision':True}
(root/'resume5000-pid.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
