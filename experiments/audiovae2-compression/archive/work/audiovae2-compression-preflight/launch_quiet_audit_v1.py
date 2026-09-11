"""One exclusive read-only checkpoint audit; no optimizer/training launch."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--expected-sha256',required=True);args=p.parse_args()
P=Path('/workspace/fast-audiovae-compression-20260910-v1');code=P/'code/audit_quiet_windows_v1.py'
if hashlib.sha256(code.read_bytes()).hexdigest()!=args.expected_sha256:raise ValueError('Reviewed audit source differs')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
workers=[s for s in processes if any(k in s for k in ('/code/joint_recovery_v1.py','/code/resume_settings.py','/code/compare_','/produce_reserved_continuation_pairs_v1.py'))]
if active or workers:raise RuntimeError('Audit requires idle GPU and no training/target worker: '+str((active,workers)))
out=Path('/tmp/fast-audiovae-quiet-audit-v1');log=Path(str(out)+'.log');receipt=Path(str(out)+'-pid.json')
if out.exists() or log.exists() or receipt.exists():raise FileExistsError('Preserve any earlier audit')
run=P/'current-lowrate-fresh5000-v1'
cmd=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-u',str(code),
 '--checkpoint',str(run/'checkpoint-step4500.pt'),'--candidate','/tmp/fast-audiovae-accumulation-comparison-v1/accumulation12/final.pt',
 '--anchor-checkpoint',str(P/'current-lowrate-continue1000-v1/checkpoint-step1000.pt'),'--screen-out',str(P/'settings-screen-v1'),
 '--base-out',str(P/'pilot'),'--manifest',str(P/'pilot-selection-v1.json'),'--fresh-manifest',str(P/'fresh-source-plan-v2/plan.json'),
 '--shards','/dev/shm/fast-audiovae-compression-fresh-shards-v2','--assets','/workspace/fast-audiovae-convnext-20260908-r1/assets',
 '--joint-run','/tmp/fast-audiovae-joint-recovery-v1','--out',str(out)]
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('xb') as f:proc=subprocess.Popen(cmd,cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':proc.pid,'out':str(out),'log':str(log),'command':cmd,'launch_time_unix':time.time(),'source_sha256':args.expected_sha256,
 'prelaunch_gpu':gpu,'prelaunch_gpu_processes':active,'prelaunch_training_or_target_workers':workers,'no_training':True,'no_model_intervention':True}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
