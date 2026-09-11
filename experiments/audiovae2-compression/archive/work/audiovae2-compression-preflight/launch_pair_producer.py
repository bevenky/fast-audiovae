import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
script=P/'produce_continuation_pairs.py'
loader=P/'code/fresh_training_data.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='3cb47b3241e5b26f20b00d50a785a583a62d4e713509f0e8ced03e0b7aff7ade'
assert hashlib.sha256(loader.read_bytes()).hexdigest()=='164a0030e8d7986dd679d32a54bbc7440f1eda88efcb09ed237bc5f93f5e9aad'
log=P/'fresh-pair-producer-v2.log'
receipt=P/'fresh-pair-producer-v2-pid.json'
if receipt.exists():
    raise RuntimeError('Existing producer launch receipt; inspect before retrying')
env=dict(os.environ, PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(script),'--plan',str(P/'fresh-source-plan-v2/plan.json'),'--original-manifest',str(P/'pilot-selection-v1.json'),'--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2','--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets','--stop-index','12000']
with log.open('xb') as output:
    process=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':process.pid,'log':str(log),'approved_stop_index':12000,'command':cmd,'launch_time_unix':time.time(),'producer_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'loader_sha256':hashlib.sha256(loader.read_bytes()).hexdigest()}
receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
