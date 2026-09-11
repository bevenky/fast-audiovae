"""Launch only after protocol review, tests and explicit task approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--expected-sha256',required=True)
parser.add_argument('--expected-helper-sha256',required=True)
args=parser.parse_args()
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
code=P/'code/diagnose_native_up4_v1.py'
if hashlib.sha256(code.read_bytes()).hexdigest()!=args.expected_sha256:
    raise RuntimeError('Reviewed comparison source differs')
helper=P/'code/native_upsampler_fit_v1.py'
if hashlib.sha256(helper.read_bytes()).hexdigest()!=args.expected_helper_sha256:raise RuntimeError('Reviewed hint helper differs')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPU is not idle before exclusive native-up4 diagnostic: '+active)
out=Path('/tmp/fast-audiovae-native-up4-reconstruction-v1')
log=Path('/tmp/fast-audiovae-native-up4-reconstruction-v1.log')
receipt=Path('/tmp/fast-audiovae-native-up4-reconstruction-v1-pid.json')
if out.exists() or receipt.exists():raise RuntimeError('Comparison output or receipt already exists')
run=P/'current-lowrate-fresh5000-v1'
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(code),
 '--checkpoint',str(run/'checkpoint-step4500.pt'),'--candidate','/tmp/fast-audiovae-accumulation-comparison-v1/accumulation12/final.pt',
 '--anchor-checkpoint',str(P/'current-lowrate-continue1000-v1/checkpoint-step1000.pt'),
 '--screen-out',str(P/'settings-screen-v1'),'--base-out',str(P/'pilot'),
 '--manifest',str(P/'pilot-selection-v1.json'),'--fresh-manifest',str(P/'fresh-source-plan-v2/plan.json'),
 '--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2',
 '--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets','--out',str(out)]
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('xb') as handle:
    proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':proc.pid,'log':str(log),'out':str(out),'source_sha256':args.expected_sha256,'command':cmd,
      'launch_time_unix':time.time(),'no_neural_training':True,'calibration_sources':72,'development_sources':96,'trained_base_optimizer_step':4625,'helper_sha256':args.expected_helper_sha256,'main_training_unchanged':True}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
