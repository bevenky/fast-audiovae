"""Launch only the approved bounded diagnostic from a verified source snapshot."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import time

base = Path('/workspace/fast-audiovae-convnext-20260909-r2')
old = Path('/workspace/fast-audiovae-convnext-20260908-r1')
archive = base / 'source-final.tgz'
expected = 'eef05f8e1c1970cee32a376aeab465a5d1c184397414145a584336c5f00b12c1'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Corrected source archive hash mismatch')
with tarfile.open(archive, 'r:gz') as source:
    for member in source.getmembers():
        if member.isfile():
            if source.extractfile(member).read() != (base / member.name).read_bytes():
                raise RuntimeError('Extracted source differs: ' + member.name)
record = base / 'launch-corrected-waveform-preflight-v1.json'
if record.exists():
    raise RuntimeError('A launch record already exists; inspect it instead of starting another job')
selection = base / 'data/selection-v1'
command = [str(old / '.train-venv/bin/python'), '-u', '-m', 'audiovae_student.preflight_distillation',
    '--diagnostic-manifest', str(selection / 'diagnostic.jsonl'),
    '--sentinel-manifest', str(selection / 'sentinel.jsonl'),
    '--sample-counts', str(selection / 'input-sample-counts.json'),
    '--selection-audit', str(selection / 'ready.json'),
    '--teacher-source', str(old / 'assets/audio_vae_v2.py'),
    '--teacher-checkpoint', str(old / 'assets/audiovae.pth'),
    '--cache-dir', str(base / 'teacher-cache-preflight-v1'),
    '--output-dir', str(base / 'training-runs/corrected-waveform-preflight-v1'),
    '--log-dir', str(old / 'runs'), '--run-name', 'corrected-waveform-preflight-v1', '--device', 'cuda']
environment = dict(os.environ)
environment['PYTHONPATH'] = str(base)
environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
log_path = base / 'launch-corrected-waveform-preflight-v1.log'
with log_path.open('xb') as log:
    process = subprocess.Popen(command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
result = {'pid': process.pid, 'command': command, 'log': str(log_path),
          'started_at_unix': time.time(), 'source_archive_sha256': expected,
          'kind': 'bounded_500_update_diagnostic', 'main_training_started': False}
with record.open('x') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
