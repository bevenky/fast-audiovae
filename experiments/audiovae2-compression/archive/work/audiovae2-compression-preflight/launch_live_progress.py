"""Launch the read-only live status after verifying multifile event ingestion."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import time

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
expected = {
    'live_progress.py':'8b8014d4eea6ce5c666f193ddd6ad72bf6d11aef79e092c9e7addbb51fbda22a',
    'test_live_progress.py':'8ea2f3c50b6a0389d597ace62ad4bc433ff51445c94fe41c75b5e49b0d9e60ac',
}
with tarfile.open(root/'live-progress-code.tar.gz') as archive:
    assert set(archive.getnames()) == set(expected)
    for name in expected:
        data = archive.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest() == expected[name]
        (root/'code'/name).write_bytes(data)
server = json.loads((root/'tensorboard-unified-service.json').read_text())
args = (Path('/proc')/str(server['pid'])/'cmdline').read_bytes().split(b'\0')
assert args[args.index(b'--reload_multifile')+1] == b'true'
assert args[args.index(b'--load_fast')+1] == b'false'
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root/'code'), OMP_NUM_THREADS='1')
tests = subprocess.run([python, '-m', 'pytest', '-q', str(root/'code/test_live_progress.py')],
    cwd=root, env=env, capture_output=True, text=True)
(root/'live-progress-tests.txt').write_text(tests.stdout+tests.stderr)
assert tests.returncode == 0, tests.stdout+tests.stderr
training = json.loads((root/'resume5000-pid.json').read_text())
assert (Path('/proc')/str(training['pid'])/'cmdline').exists()
command = [python, '-u', str(root/'code/live_progress.py'),
    '--run-dir', str(root/'current-lowrate-fresh5000-v1'),
    '--producer-progress', '/dev/shm/fast-audiovae-compression-fresh-shards-v2/producer-progress.json',
    '--logdir', str(root/'tensorboard-unified/active'), '--target-step', '5000',
    '--training-pid', str(training['pid'])]
receipt_path = root/'live-progress-pid.json'
assert not receipt_path.exists()
log = root/'live-progress.log'
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log), 'source_sha256':expected,
    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'tests':tests.stdout.strip(), 'model_inference':False, 'optimizer_updates':0}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
