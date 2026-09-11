"""Start the explicitly approved frozen contribution diagnostic only."""
import hashlib,json,os,subprocess,time
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
code=P/'code/diagnose_channel_contributions.py'
expected='67a5ba9ec8a0bb49475330645bd70770a223d8b2d3fb3e268fa712b9921c2cf8'
if hashlib.sha256(code.read_bytes()).hexdigest()!=expected:raise RuntimeError('Reviewed source changed')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU is not idle: '+active)
out=Path('/tmp/fast-audiovae-channel-contribution-v1');log=Path('/tmp/fast-audiovae-channel-contribution-v1.log')
receipt=Path('/tmp/fast-audiovae-channel-contribution-v1-pid.json')
if out.exists() or receipt.exists():raise RuntimeError('Use a new diagnostic output path')
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(code),
 '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets',
 '--base-out',str(P/'pilot'),'--manifest',str(P/'pilot-selection-v1.json'),'--out',str(out)]
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('xb') as handle:
 proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':proc.pid,'out':str(out),'log':str(log),'source_sha256':expected,'command':cmd,
      'launch_time_unix':time.time(),'no_neural_training':True,'main_training_unchanged':True}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
