"""Single launch of the source-pinned 500-update paired comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

base = Path('/workspace/fast-audiovae-convnext-20260909-r6')
expected = 'd287db3ba197a9071fa299cfa797dd37f32aad7eb2a70631216e3857a270de60'
archive = base / 'source.tgz'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('Comparison source archive changed')
with tarfile.open(archive, 'r:gz') as handle:
    for member in handle.getmembers():
        if member.isfile() and handle.extractfile(member).read() != (base / member.name).read_bytes():
            raise ValueError('Runtime source differs from preflight archive: ' + member.name)
ready = json.loads((base/'data/comparison-v2/ready.json').read_text())
if ready['identity_sha256'] != 'df1a86a25ecda864b1e0843330d5956562176b3cad06174049ce3bcfb749ea7b':
    raise ValueError('Approved data plan identity changed')
command = ['/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python', '-u',
           str(base/'run_balance_stage.py')]
env = dict(os.environ, PYTHONPATH=str(base), CUBLAS_WORKSPACE_CONFIG=':4096:8')
record = base/'launch.json'
if record.exists():
    raise RuntimeError('An existing launch must be inspected before another is attempted')
with (base/'launch.log').open('xb') as output:
    process = subprocess.Popen(command, cwd=base, env=env, stdin=subprocess.DEVNULL,
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
result = {'pid':process.pid,'started_unix':time.time(),'command':command,
          'source_archive_sha256':expected,'launcher_sha256':hashlib.sha256((base/'run_balance_stage.py').read_bytes()).hexdigest(),
          'log':str(base/'launch.log'),'updates_per_arm':500,'automatic_gan':False}
record.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
