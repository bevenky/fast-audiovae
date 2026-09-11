import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
CMD=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python', '-u', '/workspace/fast-audiovae-compression-20260910-v1/code/joint_recovery_v1.py', '--checkpoint', '/workspace/fast-audiovae-compression-20260910-v1/current-lowrate-fresh5000-v1/checkpoint-step4500.pt', '--candidate', '/tmp/fast-audiovae-accumulation-comparison-v1/accumulation12/final.pt', '--anchor-checkpoint', '/workspace/fast-audiovae-compression-20260910-v1/current-lowrate-continue1000-v1/checkpoint-step1000.pt', '--screen-out', '/workspace/fast-audiovae-compression-20260910-v1/settings-screen-v1', '--base-out', '/workspace/fast-audiovae-compression-20260910-v1/pilot', '--manifest', '/workspace/fast-audiovae-compression-20260910-v1/pilot-selection-v1.json', '--fresh-manifest', '/workspace/fast-audiovae-compression-20260910-v1/fresh-source-plan-v2/plan.json', '--shards', '/dev/shm/fast-audiovae-compression-fresh-shards-v2', '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets', '--out', '/tmp/fast-audiovae-joint-recovery-v1']
HASHES={'joint_recovery_v1.py': '9eb351dbe9dc9a3ef756b573e45ddc017a04498919d730c8758d58ce02a18079', 'joint_recovery_gates_v1.py': '1462f633e1024a9b938a113220c45eaac9966197715ef80aef3388faf27fba92', 'joint_recovery_display_v1.py': 'de988cedec993038dd4ebd2df0f451380ab42bb6438e06fcec190f11eb60dc9a', 'test_joint_recovery_v1.py': '0737ade84949a7f27762faff3d01f809552a8aefdb217debad9b349cbdb86774', 'test_joint_recovery_gates_v1.py': '3d9a1af774998b752ea461329230e2adc9b3a0b8de3c9a9c15f470d55a2d930b'}
out=Path('/tmp/fast-audiovae-joint-recovery-v1')
receipt=P/'joint-recovery-v1-launch.json'
if out.exists() or receipt.exists():raise FileExistsError('Recovery already exists')
if shutil.disk_usage('/tmp').free<1500*1024**2:raise RuntimeError('Need1.5GiB scratch')
for name,expected in HASHES.items():
 if hashlib.sha256((P/'code'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Code hash mismatch:'+name)
log=Path('/tmp/fast-audiovae-joint-recovery-v1.log')
env=dict(os.environ,PYTHONPATH=str(P/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('xb') as handle:
 child=subprocess.Popen(CMD,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,cwd=P)
result={'pid':child.pid,'command':CMD,'source_hashes':HASHES,'log':str(log),'out':str(out),'launched_unix':time.time()}
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
