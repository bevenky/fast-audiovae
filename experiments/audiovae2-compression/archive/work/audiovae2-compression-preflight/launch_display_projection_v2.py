"""Launch the tested read-only projection and wait for initial display history."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import time

root = Path('/workspace/fast-audiovae-compression-20260910-v1')
expected = {
    'tensorboard_display_v2.py':'4edee5a8d072a4e1b4b65ee30bdc1843f4c7c93642c6bb8d81d5f2f0b3597ac6',
    'test_tensorboard_display_v2.py':'838b786fc1d6a2b7e5ecf9eb3d08196eeb0b9ad234fa5b989f06412d2002966a',
}
with tarfile.open(root/'display-v2-code.tar.gz') as archive:
    assert set(archive.getnames()) == set(expected)
    for name in expected:
        data = archive.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest() == expected[name]
        (root/'code'/name).write_bytes(data)
assert hashlib.sha256((root/'code/tensorboard_display.py').read_bytes()).hexdigest() == '25f2ea5d31f488f4f8151845eb25e250d3895cbd1bec486028b83d5a4386d794'
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root/'code'))
tests = subprocess.run([python, '-m', 'pytest', '-q', str(root/'code/test_tensorboard_display_v2.py')],
    cwd=root, env=env, capture_output=True, text=True)
(root/'tensorboard-display-v2-tests.txt').write_text(tests.stdout+tests.stderr)
assert tests.returncode == 0, tests.stdout+tests.stderr
command = [python, '-u', str(root/'code/tensorboard_display_v2.py'),
    '--source-logdir', str(root/'tensorboard-unified/active'),
    '--display-logdir', str(root/'tensorboard-display-v2')]
receipt_path = root/'display-projection-v2-pid.json'
assert not receipt_path.exists()
log = root/'display-projection-v2.log'
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log), 'source_sha256':expected,
    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'tests':tests.stdout.strip(), 'source_logs_read_only':True}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
ready_path = root/'tensorboard-display-v2/projection-ready.json'
for _ in range(40):
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        assert ready['ready'] and len(ready['counts_by_run']) == 11 and min(ready['counts_by_run'].values()) > 0
        print(json.dumps({'launch':receipt, 'ready':ready}))
        break
    if process.poll() is not None:
        raise RuntimeError(log.read_text())
    time.sleep(.5)
else:
    raise RuntimeError('Display projection is not ready; leave current dashboard in place')
