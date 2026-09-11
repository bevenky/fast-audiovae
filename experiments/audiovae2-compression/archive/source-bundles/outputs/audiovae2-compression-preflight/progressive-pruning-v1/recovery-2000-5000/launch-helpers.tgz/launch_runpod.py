"""Launch the approved continuation and its separate frozen-target producer."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path

r=Path('/tmp/fast-audiovae-progressive-pruning-v1/recovery-2000-5000')
prior=r.parent/'recovery-1000-2000'
project=Path('/workspace/fast-audiovae-compression-20260910-v1')
py='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
parent=prior/'segment-1000-2000/checkpoint-step2000.pt'
expected={'progressive_continue_5000.py':'57acc6eb495a4c683c1e43d6b22dd80880c0bac1280125972d00dbfef5a66b39',
 'progressive_continue_5000_monitor.py':'28bf716d60e321e3b280683f1951810a0e3d232e381480e7b8d3dc523d461307',
 'progressive_extended_data.py':'9d8733934bf28ac3ab371968ddbf85ffaeaaf3c8d6f54821e0f03f73a9a52346'}
if (r/'process-launch.json').exists() or (r/'segment-2000-5000').exists():raise FileExistsError('Inspect existing launch before any retry')
for name,checksum in expected.items():
 if hashlib.sha256((r/'code'/name).read_bytes()).hexdigest()!=checksum:raise RuntimeError('Reviewed source changed: '+name)
if '71 passed' not in (r/'cpu-tests.txt').read_text():raise RuntimeError('Run the focused runtime tests first')
if shutil.disk_usage(r).free<6*parent.stat().st_size+200_000_000:raise RuntimeError('Insufficient space for all six checkpoints and logs')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if any(str(x).encode() in args for x in (r/'code/progressive_continue_5000.py',prior/'code/progressive_continue.py',r.parent/'code/progressive_train.py')):
  raise RuntimeError('A progressive trainer is already active')
env=os.environ.copy();overrides={'PYTHONPATH':str(r/'code')+':'+str(prior/'code')+':'+str(r.parent/'code')+':'+str(project/'code'),
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'};env.update(overrides)
cmd=[py,'-u',str(r/'code/progressive_continue_5000.py'),'--parent-checkpoint',str(parent),
 '--original-dir',str(r.parent/'cut1-384-256'),'--manifest',str(project/'pilot-selection-v1.json'),
 '--source-plan',str(project/'fresh-source-plan-v2/plan.json'),'--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2',
 '--extension-plan',str(r/'source-plan/plan.json'),'--extension-shards','/dev/shm/fast-audiovae-progressive-extension-5000',
 '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets',
 '--recipe','/tmp/fast-audiovae-joint-recovery-v2/segment-5625-6625/launch.json',
 '--schedule',str(r.parent/'preparation/schedule.json'),'--out',str(r/'segment-2000-5000'),
 '--tensorboard',str(r/'tensorboard')]
with (r/'training.log').open('xb') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=r)
receipt={'pid':p.pid,'command':cmd,'environment_overrides':overrides,'source_sha256':expected,'launched_unix':time.time(),
 'starting_step':2000,'target_step':5000,'parent_checkpoint_sha256':'a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f',
 'automatic_next_cut':False,'optimizer_reset':False,'width_changed':False}
(r/'process-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pid':p.pid,'starting_step':2000,'target_step':5000,'optimizer_reset':False}))
