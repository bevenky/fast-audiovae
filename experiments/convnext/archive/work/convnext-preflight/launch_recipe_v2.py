import hashlib, json, os, subprocess
from pathlib import Path
base=Path('/workspace/fast-audiovae-convnext-20260909-r9')
py=Path('/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python')
report=json.loads((base/'full-model-preflight.json').read_text())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if report.get('state')!='passed' or report['source_archive_sha256']!=sha(base/'source.tgz') or report['launcher_sha256']!=sha(base/'run_recipe_v2_stage.py'):
    raise ValueError('Validation does not match launch source')
if (base/'training-launch.json').exists() or (base/'training-runs/decoder-recipe-v2').exists():
    raise ValueError('This fresh run already has launch state; do not start duplicate training')
env=os.environ.copy()
env.update(PYTHONPATH=str(base), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
command=[str(py),str(base/'run_recipe_v2_stage.py')]
with (base/'training.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':process.pid,'command':command,'source_archive_sha256':sha(base/'source.tgz'),
       'launcher_sha256':sha(base/'run_recipe_v2_stage.py'),'requested_updates':10000,
       'run_name':'decoder-recipe-v2','dashboard_logdir':str(base/'tensorboard'),
       'preflight_sha256':sha(base/'full-model-preflight.json')}
(base/'training-launch.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
