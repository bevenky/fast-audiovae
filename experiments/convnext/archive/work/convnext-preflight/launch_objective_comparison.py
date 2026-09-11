"""Launch the two bounded arms once from an exact source archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--archive-sha256', required=True)
args = parser.parse_args()
base = Path('/workspace/fast-audiovae-convnext-20260909-r3')
previous = Path('/workspace/fast-audiovae-convnext-20260909-r2')
runtime = Path('/workspace/fast-audiovae-convnext-20260908-r1')
archive = base / 'source.tgz'
if hashlib.sha256(archive.read_bytes()).hexdigest() != args.archive_sha256:
    raise RuntimeError('Comparison source archive hash mismatch')
with tarfile.open(archive, 'r:gz') as source:
    for member in source.getmembers():
        path = PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unexpected source archive entry')
        if source.extractfile(member).read() != (base / member.name).read_bytes():
            raise RuntimeError('Extracted comparison source changed: ' + member.name)
parent = previous / 'training-runs/corrected-waveform-preflight-v1/checkpoint-step000500.pt'
parent_sha = 'df21671f774892cf5f2af0179da9fdfdc3df62ae6b195d66a11de125bb1a9fdd'
if hashlib.sha256(parent.read_bytes()).hexdigest() != parent_sha:
    raise RuntimeError('Original step-500 checkpoint changed')
command = [str(runtime / '.train-venv/bin/python'), '-u', '-m',
    'audiovae_student.run_objective_comparison', '--parent-checkpoint', str(parent),
    '--parent-sha256', parent_sha, '--cache-dir', str(previous / 'teacher-cache-preflight-v1'),
    '--output-dir', str(base / 'training-runs/objective-comparison-v1'),
    '--log-dir', str(runtime / 'runs'), '--run-name', 'objective-comparison-v1', '--device', 'cuda']
record = base / 'launch-objective-comparison-v1.json'
initial = {'state': 'launching', 'command': command, 'started_at_unix': time.time(),
           'source_archive_sha256': args.archive_sha256, 'parent_checkpoint_sha256': parent_sha,
           'kind': 'two_bounded_2000_update_objective_arms', 'main_training_started': False}
with record.open('x') as handle:
    json.dump(initial, handle, indent=2)
environment = dict(os.environ)
environment.update(PYTHONPATH=str(base), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
log_path = base / 'launch-objective-comparison-v1.log'
try:
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {**initial, 'state': 'launched', 'pid': process.pid, 'log': str(log_path)}
except BaseException as error:
    result = {**initial, 'state': 'launch_error', 'error': str(error)}
    record.write_text(json.dumps(result, indent=2) + '\n')
    raise
record.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
