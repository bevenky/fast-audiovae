"""Start the reviewed same-width continuation after its CPU checks pass."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parent=Path('/tmp/fast-audiovae-progressive-pruning-v1')
root=parent/'recovery-1000-2000'
project=Path('/workspace/fast-audiovae-compression-20260910-v1')
python='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
out=root/'segment-1000-2000'
if (root/'process-launch.json').exists() or out.exists():
    raise FileExistsError('Inspect the existing recovery instead of starting a duplicate')
expected={'progressive_continue.py':'d3a7be85429e54f6dc8199174238847d9eb5acef8dbe42ebd44feaf2f511906c',
          'progressive_continue_monitor.py':'9e57cc0854c326e657fae92dc398cc2cf8a1fd2e57be7efac5b18ad3278c362a'}
for name,value in expected.items():
    if hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()!=value:
        raise RuntimeError('Reviewed source changed: '+name)
if '32 passed' not in (root/'cpu-tests.txt').read_text():
    raise RuntimeError('Run the focused CPU tests in the qualified runtime first')
if shutil.disk_usage(root).free<1_500_000_000:
    raise RuntimeError('Insufficient checkpoint space')
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:args=(entry/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if any(str(path).encode() in args for path in (root/'code/progressive_continue.py',parent/'code/progressive_train.py')):
        raise RuntimeError('A progressive trainer is already running')
command=[python,'-u',str(root/'code/progressive_continue.py'),
    '--parent-checkpoint',str(parent/'cut1-384-256/checkpoint-step1000.pt'),
    '--manifest',str(project/'pilot-selection-v1.json'),
    '--source-plan',str(project/'fresh-source-plan-v2/plan.json'),
    '--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2',
    '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets',
    '--recipe','/tmp/fast-audiovae-joint-recovery-v2/segment-5625-6625/launch.json',
    '--schedule',str(parent/'preparation/schedule.json'),
    '--out',str(out),'--tensorboard',str(root/'tensorboard')]
environment=os.environ.copy()
overrides={'PYTHONPATH':str(root/'code')+':'+str(parent/'code')+':'+str(project/'code'),
           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
environment.update(overrides)
with (root/'training.log').open('xb') as log:
    process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True,cwd=root)
receipt={'pid':process.pid,'command':command,'environment_overrides':overrides,
    'source_sha256':expected,'launched_unix':time.time(),'starting_step':1000,'target_step':2000,
    'parent_checkpoint_sha256':'4272198a8d76168564651c49960ef370690859981a5591ba0de39d42053eef56',
    'automatic_next_cut':False,'optimizer_reset':False,'width_changed':False}
(root/'process-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pid':process.pid,'starting_step':1000,'target_step':2000,'optimizer_reset':False,'width_changed':False}))
