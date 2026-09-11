"""Launch only the approved first cut, preserving the historical run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/tmp/fast-audiovae-progressive-pruning-v1')
project = Path('/workspace/fast-audiovae-compression-20260910-v1')
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
out = root/'cut1-384-256'
receipt_path = root/'process-launch.json'
if receipt_path.exists() or out.exists():
    raise FileExistsError('Preserve an existing launch; inspect it instead of launching twice')
expected = '159571f9eba7b283e9360fb3882c64dbab7fe48b80b9bc7a773310a41871ea25'
actual = hashlib.sha256((root/'code/progressive_train.py').read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError('Runner differs from the reviewed and tested source')
if shutil.disk_usage(root).free < 1_500_000_000:
    raise RuntimeError('Insufficient space for the first cut checkpoints')
for entry in Path('/proc').iterdir():
    if entry.name.isdigit():
        try:
            cmdline = (entry/'cmdline').read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if str(root/'code/progressive_train.py').encode() in cmdline:
            raise RuntimeError('A progressive trainer is already active')
command = [python, '-u', str(root/'code/progressive_train.py'),
    '--manifest', str(project/'pilot-selection-v1.json'),
    '--source-plan', str(project/'fresh-source-plan-v2/plan.json'),
    '--shards', '/dev/shm/fast-audiovae-compression-fresh-shards-v2',
    '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets',
    '--recipe', '/tmp/fast-audiovae-joint-recovery-v2/segment-5625-6625/launch.json',
    '--schedule', str(root/'preparation/schedule.json'),
    '--out', str(out), '--tensorboard', str(root/'tensorboard/cut1'), '--cut-index', '1']
environment = os.environ.copy()
environment.update({'PYTHONPATH':str(root/'code')+':'+str(project/'code'),
                    'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
with (root/'training.log').open('xb') as log:
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True, cwd=root)
receipt = {'pid':process.pid,'command':command,
    'environment_overrides':{k:environment[k] for k in ('PYTHONPATH','PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS')},
    'runner_sha256':actual,'launched_unix':time.time(), 'automatic_next_cut':False,
    'target_updates':1000,'quality_steps':[0,500,1000],
    'old_checkpoint_preserved':'/tmp/fast-audiovae-joint-recovery-v2/segment-5625-6625/checkpoint-step6625.pt'}
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'pid':process.pid,'out':str(out),'target_updates':1000,'automatic_next_cut':False}))
