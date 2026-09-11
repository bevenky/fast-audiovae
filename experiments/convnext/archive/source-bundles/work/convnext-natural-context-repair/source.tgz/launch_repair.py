import os,json,subprocess,time
from pathlib import Path
base=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
code=base/'diagnostic-natural-repair-code';out=base/'diagnostics-v1'
if (out/'natural-repair-launch.json').exists():raise RuntimeError('Already launched')
env=dict(os.environ,PYTHONPATH=str(code)+':'+str(base/'diagnostic-supplement-code')+':'+str(base/'diagnostic-code')+':'+str(base/'corrected-code'),
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
cmd=['/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python',str(code/'run_repair.py')]
with (out/'natural-repair-runner.log').open('xb') as log:
    p=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':p.pid,'command':cmd,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(out/'natural-repair-launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
