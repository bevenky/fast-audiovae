"""Serve the new cut's event files, retaining every historical event file."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

root=Path('/tmp/fast-audiovae-progressive-pruning-v1')
old=Path('/tmp/fast-audiovae-joint-recovery-v2')
python='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
logdir=root/'tensorboard/cut1'
receipt_path=root/'display-launch.json'
if receipt_path.exists():raise FileExistsError('Inspect the existing dashboard launch instead')
if not (root/'cut1-384-256/train.jsonl').exists() or not (logdir/'latest-validation.json').exists():
    raise RuntimeError('Wait for actual training and the first-cut quality baseline')
stopped=[]
for pid,required in (
    (1044978,['tensorboard.main',str(old/'tensorboard/display'),'8888']),
    (1044899,['/workspace/fast-audiovae-compression-20260910-v1/code/joint_recovery_display_v2.py',str(old/'tensorboard/raw')])):
    cmdfile=Path('/proc')/str(pid)/'cmdline'
    if not cmdfile.exists():continue
    args=cmdfile.read_bytes().decode().split('\0')
    if not all(item in args for item in required):
        raise RuntimeError('Recorded old dashboard PID has another identity')
    os.kill(pid,signal.SIGTERM)
    stopped.append({'pid':pid,'command':args})
for _ in range(30):
    cmdfile=Path('/proc/1044978/cmdline')
    if not cmdfile.exists() or not cmdfile.read_bytes():break
    time.sleep(.2)
else:raise RuntimeError('The old dashboard has not stopped')
command=[python,'-m','tensorboard.main','--logdir',str(logdir),'--host','0.0.0.0',
    '--port','8888','--reload_interval','5','--load_fast','false',
    '--reload_multifile','true','--reload_multifile_inactive_secs','86400']
with (root/'tensorboard-server.log').open('xb') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'pid':process.pid,'command':command,'stopped_previous_display_processes':stopped,
         'historical_events_preserved':True,'launched_unix':time.time()}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
for _ in range(40):
    if process.poll() is not None:raise RuntimeError('New TensorBoard exited')
    try:
        with urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags',timeout=2) as response:
            tags=json.load(response)
        runs=[name for name,values in tags.items() if 'overview/Percent' in values]
        if len(runs)==13:
            receipt.update({'ready':True,'overview_metric_runs':runs,'total_runs':len(tags)})
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({'pid':process.pid,'ready':True,'overview_metrics':len(runs),'total_runs':len(tags),'historical_events_preserved':True}))
            break
    except (OSError,ValueError):pass
    time.sleep(.5)
else:raise RuntimeError('Dashboard did not expose the new overview metrics in time')
