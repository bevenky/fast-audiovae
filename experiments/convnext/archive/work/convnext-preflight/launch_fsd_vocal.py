from pathlib import Path
import json, os, subprocess, time
base=Path('/workspace/fast-audiovae-convnext-20260908-r1')
launch=base/'corpus-fsd-vocal.launch.json'
if launch.exists():
 raise SystemExit('Existing FSD launch record: inspect before starting another process')
root=base/'data-fsd-vocal';root.mkdir(exist_ok=True)
command=[str(base/'.train-venv/bin/python'),'-u','-m','audiovae_student.acquire_fsd_vocal','acquire','--metadata-dir',str(base/'assets/fsd50k-metadata'),'--output-dir',str(root),'--selection-sha256','1aa018e4364532f4b16bfc7f802bf8e458f160c47a0a3c35f7347c23e40398af','--exclude-training-manifest',str(base/'data-bootstrap/train.jsonl'),'--reserved-manifest',str(base/'data-bootstrap/dev.jsonl'),'--reserved-evaluation-manifest',str(base/'reserved-evaluation.json'),'--max-gib','2','--min-free-gib','8']
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
with (root/'run.log').open('ab',buffering=0) as log:
 process=subprocess.Popen(command,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.2)
actual=Path(f'/proc/{process.pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
if actual!=command:raise RuntimeError('Detached process argv did not match')
stat=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()
btime=next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
started_at=btime+int(stat[19])/os.sysconf('SC_CLK_TCK')
value={'pid':process.pid,'argv':actual,'command':actual,'started_at':started_at,'recorded_at':time.time(),'record_source':'actual /proc argv and process start ticks','cwd':str(base),'log':str(root/'run.log'),'readiness':str(root/'prepared/provenance/complete.json'),'preparation_version':2}
part=launch.with_suffix('.json.tmp');part.write_text(json.dumps(value,indent=2)+'\n');os.replace(part,launch)
print(json.dumps(value,indent=2))
