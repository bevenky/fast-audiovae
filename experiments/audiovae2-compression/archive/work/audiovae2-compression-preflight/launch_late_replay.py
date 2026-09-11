import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
code=P/'code/replay_late_segment.py'
expected='52394ae4e8b780a0a43631cb54329aa32fd37f1848f67b6fb48d993249bddd42'
assert hashlib.sha256(code.read_bytes()).hexdigest()==expected
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU is not idle before exclusive diagnostic replay: '+active)
log=Path('/tmp/fast-audiovae-replay4500-5000-v1.log');out=Path('/tmp/fast-audiovae-replay4500-5000-v1')
receipt=Path('/tmp/fast-audiovae-replay4500-5000-v1-pid.json')
if out.exists() or receipt.exists():raise RuntimeError('Replay output or launcher receipt already exists')
run=P/'current-lowrate-fresh5000-v1'
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(code),
 '--checkpoint',str(run/'checkpoint-step4500.pt'),'--expected',str(run/'checkpoint-step5000.pt'),
 '--anchor-checkpoint',str(P/'current-lowrate-continue1000-v1/checkpoint-step1000.pt'),
 '--screen-out',str(P/'settings-screen-v1'),'--base-out',str(P/'pilot'),
 '--manifest',str(P/'pilot-selection-v1.json'),'--fresh-manifest',str(P/'fresh-source-plan-v2/plan.json'),
 '--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2',
 '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets','--out',str(out)]
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('xb') as handle:
 proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':proc.pid,'log':str(log),'out':str(out),'source_sha256':expected,'command':cmd,'launch_time_unix':time.time(),'replay_only_steps':[4501,5000]}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
