"""Switch to the preserved-history recovery display after a real resumed update."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

parent=Path('/tmp/fast-audiovae-progressive-pruning-v1')
root=parent/'recovery-2000-5000'
python='/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
logdir=root/'tensorboard'
receipt_path=root/'display-launch.json'
if receipt_path.exists():raise FileExistsError('Inspect the existing display launch')
training=root/'segment-2000-5000/train.jsonl'
if not training.exists() or not training.read_text().strip():
    raise RuntimeError('Wait for a real resumed optimizer update')
old=json.loads((parent/'recovery-1000-2000/display-launch.json').read_text())
pid=old['pid']
cmdfile=Path('/proc')/str(pid)/'cmdline'
stopped=[]
if cmdfile.exists():
    args=cmdfile.read_bytes().decode().split('\0')
    if not all(value in args for value in ('tensorboard.main',str(parent/'recovery-1000-2000/tensorboard'),'8888')):
        raise RuntimeError('Old display process no longer has the expected identity')
    os.kill(pid,signal.SIGTERM)
    stopped.append({'pid':pid,'command':args})
    for _ in range(30):
        if not cmdfile.exists() or not cmdfile.read_bytes():break
        time.sleep(.2)
    else:raise RuntimeError('Old display did not stop')
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
    if process.poll() is not None:raise RuntimeError('Recovery TensorBoard exited')
    try:
        with urlopen('http://127.0.0.1:8888/data/plugin/scalars/tags',timeout=2) as response:tags=json.load(response)
        runs=[name for name,values in tags.items() if 'overview/Percent' in values]
        if len(runs)==15:
            receipt.update({'ready':True,'overview_metric_runs':runs,'total_runs':len(tags)})
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({'pid':process.pid,'ready':True,'overview_metrics':len(runs),'total_runs':len(tags)}))
            break
    except (OSError,ValueError):pass
    time.sleep(.5)
else:raise RuntimeError('Recovery dashboard did not expose its 15 metrics')
