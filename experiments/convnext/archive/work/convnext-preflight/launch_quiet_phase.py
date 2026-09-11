"""Single launch of the source-pinned 250-update paired comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

base = Path('/workspace/fast-audiovae-convnext-20260909-r7')
expected = 'f06443009e9d2f0db11bf4ae0801bf0c0b9b5096882048cfbea238b9c875ffeb'
archive = base / 'source.tgz'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('Comparison source archive changed')
with tarfile.open(archive, 'r:gz') as handle:
    for member in handle.getmembers():
        if member.isfile() and handle.extractfile(member).read() != (base / member.name).read_bytes():
            raise ValueError('Runtime source differs from preflight archive: ' + member.name)
ready = json.loads((base/'data/quiet-phase-v1/ready.json').read_text())
if ready['identity_sha256'] != 'bbd230387be870bda39f677339238ec2ab7e9466a8a84ea32daf058a2ed9677b':
    raise ValueError('Approved data plan identity changed')
if hashlib.sha256((base/'run_quiet_phase_stage.py').read_bytes()).hexdigest() != 'e7ac3eb659691c5c24b9d5d0816a1765d31fd11de9d651e900793bb67c8b514a':
    raise ValueError('Reviewed training launcher changed')
command = ['/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python', '-u',
           str(base/'run_quiet_phase_stage.py')]
env = dict(os.environ, PYTHONPATH=str(base), CUBLAS_WORKSPACE_CONFIG=':4096:8')
record = base/'launch.json'
if record.exists():
    raise RuntimeError('An existing launch must be inspected before another is attempted')
with (base/'launch.log').open('xb') as output:
    process = subprocess.Popen(command, cwd=base, env=env, stdin=subprocess.DEVNULL,
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
result = {'pid':process.pid,'started_unix':time.time(),'command':command,
          'source_archive_sha256':expected,'launcher_sha256':hashlib.sha256((base/'run_quiet_phase_stage.py').read_bytes()).hexdigest(),
          'log':str(base/'launch.log'),'updates_per_arm':250,'automatic_gan':False}
record.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
