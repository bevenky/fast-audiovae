"""Backfill the selected model's single overview and show only its active run."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

from torch.utils.tensorboard import SummaryWriter
from unified_monitor import UnifiedMonitor
import run_pilot as base

root=Path('/workspace/fast-audiovae-compression-20260910-v1')
directory=root/'tensorboard-unified/active'
assert not directory.exists(), 'Preserve an existing active run'
read=lambda p:json.loads(p.read_text())
baseline=read(root/'pilot/development-step0.json')
manifest=read(root/'pilot-selection-v1.json')
metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
current=next(a for a in read(root/'reference-and-quiet-addendum-v1.json')['arms'] if a['arm']=='current_lr3e-5')
final=read(root/'continuation-quiet-addendum-v1.json')
reports=[(0,root/'pilot/development-step0.json'),
         (128,root/'settings-screen-v1/current_lr3e-5/development-step128.json'),
         (256,root/'settings-screen-v1/current_lr3e-5/development-step256.json'),
         (500,root/'current-lowrate-continue1000-v1/development-step500.json'),
         (1000,root/'current-lowrate-continue1000-v1/development-step1000.json')]
with SummaryWriter(str(directory)) as writer:
    monitor=UnifiedMonitor(writer,baseline['aggregate'],metadata)
    monitor.initialize_layout()
    for step,path in reports:
        report=read(path)
        if step in (256,1000):
            details=current if step==256 else final
            near=[w for w in details['quiet_diagnostics']['student']['windows'] if w['teacher_rms']<=1e-5]
            report['overview_window_metrics']={'near_silence_windows':len(near),
                'near_silence_failed_windows':sum(w['passed'] is False for w in near)}
        base.log_validation(writer,report,step,metadata)
        monitor.log_validation(report,step)
    monitor.log_waiting('Checkpoint 1,000 is saved. Preparing fresh targets and resuming to 5,000; '
                        'the earlier audio is not being replayed. The 10,000-step extension requires a quality review.',1000)
    writer.flush()

# Stop only this experiment's prior TensorBoard service. Keep all its events.
prior=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:parts=(entry/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'tensorboard.main' in parts and str(root/'tensorboard').encode() in parts:
        prior.append({'pid':int(entry.name),'command':[p.decode() for p in parts if p]})
for process in prior:os.kill(process['pid'],signal.SIGTERM)
command=['/tmp/fast-audiovae-recovery-20260909/venv214/bin/python','-m','tensorboard.main',
         '--logdir',str(root/'tensorboard-unified'),'--host','0.0.0.0','--port','8888',
         '--reload_interval','5','--load_fast','false']
with (root/'tensorboard-unified.log').open('xb') as handle:
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                             start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
receipt={'pid':process.pid,'command':command,'prior_services':prior,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'active_run':'active','overview':'Custom Scalars / Progress / All quality metrics'}
(root/'tensorboard-unified-service.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
