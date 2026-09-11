import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
P=Path('/workspace/fast-audiovae-compression-20260910-v1')
R=Path('/tmp/fast-audiovae-joint-recovery-v1')
PY='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
receipt=P/'joint-recovery-display-launch.json'
if receipt.exists():raise FileExistsError('Display already launched')
env=dict(os.environ,PYTHONPATH=str(P/'code'))
projector_cmd=[PY,'-u',str(P/'code/joint_recovery_display_v1.py'),'--source-logdir',str(R/'tensorboard/raw'),'--display-logdir',str(R/'tensorboard/display')]
with (R/'display.log').open('xb') as log:
 projector=subprocess.Popen(projector_cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=P)
for _ in range(30):
 if (R/'tensorboard/display/projection-ready.json').exists():break
 if projector.poll() is not None:raise RuntimeError('New display projector failed; original dashboard retained')
 time.sleep(1)
else:raise TimeoutError('New display not ready; original dashboard retained')
old=[]
for pid,marker in [(1019903,'tensorboard-display-v2'),(1019827,'tensorboard_display_v2.py')]:
 path=Path('/proc')/str(pid)/'cmdline'
 if path.exists():
  command=path.read_bytes().replace(b'\0',b' ').decode()
  if marker not in command:raise RuntimeError('Old process identity changed')
  old.append({'pid':pid,'command':command})
for item in old:os.kill(item['pid'],signal.SIGTERM)
for _ in range(10):
 try:urllib.request.urlopen('http://127.0.0.1:8888/',timeout=1)
 except Exception:break
 time.sleep(1)
tb_cmd=[PY,'-m','tensorboard.main','--logdir',str(R/'tensorboard/display'),'--host','0.0.0.0','--port','8888','--reload_interval','5','--load_fast','false']
with (R/'tensorboard.log').open('xb') as log:
 tb=subprocess.Popen(tb_cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=P)
result={'projector_pid':projector.pid,'tensorboard_pid':tb.pid,'projector_command':projector_cmd,'tensorboard_command':tb_cmd,'historical_processes':old,'historical_logs_preserved':True,'launched_unix':time.time()}
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
