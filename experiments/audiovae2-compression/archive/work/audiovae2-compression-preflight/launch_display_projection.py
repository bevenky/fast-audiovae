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
    'tensorboard_display.py':'25f2ea5d31f488f4f8151845eb25e250d3895cbd1bec486028b83d5a4386d794',
    'test_tensorboard_display.py':'bbebae237bf2f24636908186adfdffb53a95ec3390cebe13b865fdb1aafe136c',
}
with tarfile.open(root/'display-code.tar.gz') as archive:
    assert set(archive.getnames()) == set(expected)
    for name in expected:
        data = archive.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest() == expected[name]
        (root/'code'/name).write_bytes(data)
python = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root/'code'))
tests = subprocess.run([python, '-m', 'pytest', '-q', str(root/'code/test_tensorboard_display.py')],
    cwd=root, env=env, capture_output=True, text=True)
(root/'tensorboard-display-tests.txt').write_text(tests.stdout+tests.stderr)
assert tests.returncode == 0, tests.stdout+tests.stderr
command = [python, '-u', str(root/'code/tensorboard_display.py'),
    '--source-logdir', str(root/'tensorboard-unified/active'),
    '--display-logdir', str(root/'tensorboard-display/active')]
receipt_path = root/'display-projection-pid.json'
assert not receipt_path.exists()
log = root/'display-projection.log'
with log.open('xb') as handle:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
        stderr=subprocess.STDOUT, cwd=root, env=env, start_new_session=True)
receipt = {'pid':process.pid, 'command':command, 'log':str(log), 'source_sha256':expected,
    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'tests':tests.stdout.strip(), 'source_logs_read_only':True}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
ready_path = root/'tensorboard-display/active/projection-ready.json'
for _ in range(40):
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        assert ready['ready'] and ready['overview_scalar_values_written'] > 0
        print(json.dumps({'launch':receipt, 'ready':ready}))
        break
    if process.poll() is not None:
        raise RuntimeError(log.read_text())
    time.sleep(.5)
else:
    raise RuntimeError('Display projection is not ready; leave current dashboard in place')
