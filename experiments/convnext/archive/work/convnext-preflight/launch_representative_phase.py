"""Start one explicit bounded phase from the verified pilot source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

base = Path('/workspace/fast-audiovae-convnext-20260909-r5')
old = Path('/workspace/fast-audiovae-convnext-20260908-r1')
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
parser.add_argument('--max-updates', type=int, default=200)
args = parser.parse_args()
phase = 'remaining' if args.resume else 'first200'
record = base / f'launch-{phase}.json'
if record.exists():
    raise RuntimeError('Phase launch already recorded; inspect it instead of duplicating training')
expected = '6ececd6f8ad08a5badf06b1f06684cfbabe0c8a1007e7f4e9ca4cbc6eee628f4'
archive = base / 'source.tgz'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('Pilot source archive changed')
with tarfile.open(archive, 'r:gz') as handle:
    for member in handle.getmembers():
        if member.isfile() and handle.extractfile(member).read() != (base / member.name).read_bytes():
            raise ValueError('Pilot source differs from verified archive: ' + member.name)
command = [str(old / '.train-venv/bin/python'), '-u', str(base / 'run_representative_stage.py'),
           '--max-updates', str(args.max_updates)]
if args.resume:
    command.append('--resume')
env = dict(os.environ)
env['PYTHONPATH'] = str(base)
env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
log = base / f'launch-{phase}.log'
with log.open('xb') as output:
    process = subprocess.Popen(command, cwd=base, env=env, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
result = {'pid': process.pid, 'started_unix': time.time(), 'phase': phase, 'command': command,
          'log': str(log), 'source_archive_sha256': expected,
          'launcher_sha256': hashlib.sha256((base / 'run_representative_stage.py').read_bytes()).hexdigest()}
record.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
