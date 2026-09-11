"""Complete missing old targets, then prepare the separate appended target cache."""
import json,os,subprocess,time
from pathlib import Path
r=Path('/tmp/fast-audiovae-progressive-pruning-v1/recovery-2000-5000')
project=Path('/workspace/fast-audiovae-compression-20260910-v1')
py='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(r/'code')+':'+str(r.parent/'recovery-1000-2000/code')+':'+str(r.parent/'code')+':'+str(project/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
commands=[
 [py,'-u',str(r/'produce_progressive_tail_5000.py'),'--plan',str(project/'fresh-source-plan-v2/plan.json'),
  '--original-manifest',str(project/'pilot-selection-v1.json'),'--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2',
  '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets'],
 [py,'-u',str(r/'produce_progressive_extension_5000.py'),'--plan',str(r/'source-plan/plan.json'),
  '--parent-plan',str(project/'fresh-source-plan-v2/plan.json'),'--original-manifest',str(project/'pilot-selection-v1.json'),
  '--original-shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2','--shards','/dev/shm/fast-audiovae-progressive-extension-5000',
  '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets']]
for phase,cmd in enumerate(commands,1):
 p=subprocess.Popen(cmd,env=env)
 (r/'producer-current.json').write_text(json.dumps({'phase':phase,'pid':p.pid,'command':cmd,'started_unix':time.time()})+'\n')
 code=p.wait()
 if code:
  (r/'producer-failed.json').write_text(json.dumps({'phase':phase,'returncode':code})+'\n')
  raise SystemExit(code)
(r/'producer-completed.json').write_text(json.dumps({'complete':True,'phases':2,'training_updates':0})+'\n')
