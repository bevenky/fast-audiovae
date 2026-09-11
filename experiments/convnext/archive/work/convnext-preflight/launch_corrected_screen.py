from pathlib import Path
import json,os,subprocess,time

base=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
out=base/'corrected-screen';code=base/'corrected-code'
if (out/'launch.json').exists() or (out/'identity.json').exists():
    raise RuntimeError('Refusing to launch a duplicate experiment')
if not (out/'targeted-data.json').exists():raise RuntimeError('Sealed data is required')
env=dict(os.environ,PYTHONPATH=str(code),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
command=['/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python',str(code/'run_corrected_screen.py')]
with (out/'runner.log').open('xb') as log:
    process=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':process.pid,'command':command,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'arms':4,'generator_updates_per_arm':400,'parent_step':8090,'original_run':'paused'}
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
