"""Launch only the newly authorized12000:24000 immutable target-shard writer."""
import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
script=P/'produce_reserved_continuation_pairs_v1.py'
loader=P/'code/fresh_training_data.py'
expected='92f9fe089271e0ebb8aaf6e089facf619c8b642893210a4cf6837a9269087efd'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(script)!=expected:raise RuntimeError('Reviewed reserve producer source differs')
if sha(loader)!='164a0030e8d7986dd679d32a54bbc7440f1eda88efcb09ed237bc5f93f5e9aad':raise RuntimeError('Frozen target loader differs')
if sha(P/'produce_continuation_pairs.py')!='3cb47b3241e5b26f20b00d50a785a583a62d4e713509f0e8ced03e0b7aff7ade':raise RuntimeError('Original producer changed')
if sha(P/'fresh-source-plan-v2/plan.json')!='0036eae310e38ee1789e6830a2e1f7fc174c13772c267417522d51e0b9df92e8':raise RuntimeError('Original sealed plan changed')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU must be idle for initial target preparation: '+active)
log=P/'fresh-pair-producer-reserve-v1.log';receipt=P/'fresh-pair-producer-reserve-v1-pid.json'
if receipt.exists() or log.exists():raise FileExistsError('Do not overwrite original reserve launch')
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(script),
 '--plan',str(P/'fresh-source-plan-v2/plan.json'),'--original-manifest',str(P/'pilot-selection-v1.json'),
 '--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2','--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets',
 '--start-index','12000','--stop-index','24000']
with log.open('xb') as output:
 process=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':process.pid,'log':str(log),'authorized_source_interval':[12000,24000],'command':cmd,
 'launch_time_unix':time.time(),'producer_sha256':sha(script),'loader_sha256':sha(loader),'original_plan_preserved':True,
 'authorization':'User-approved joint stages2–4 continuation: at most1000 optimizer updates,12 new sources per update, review every250 updates.',
 'training_launched':False}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
