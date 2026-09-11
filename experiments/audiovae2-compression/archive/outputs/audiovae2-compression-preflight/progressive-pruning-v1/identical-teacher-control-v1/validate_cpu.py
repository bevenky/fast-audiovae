"""Validate the isolated control on the qualified runtime with GPUs hidden."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = '/tmp/fast-audiovae-recovery-20260909/venv214/bin/python'
ACTIVE = ROOT.parent / 'recovery-2000-5000'
PROJECT = Path('/workspace/fast-audiovae-compression-20260910-v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if sys.executable != PYTHON:
    raise RuntimeError('Use the qualified Python executable')
frozen = json.loads((ROOT / 'source-freeze.json').read_text())
for name, expected in frozen.items():
    if sha(ROOT / name) != expected:
        raise RuntimeError('Source freeze differs: ' + name)
if sha(ROOT / 'code/group_model.py') != sha(PROJECT / 'code/group_model.py'):
    raise RuntimeError('The copied group implementation differs from the pinned project')

env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONPATH=':'.join(map(str, (ROOT / 'code', ACTIVE / 'code',
               ROOT.parent / 'recovery-1000-2000/code', ROOT.parent / 'code', PROJECT / 'code'))))
results = []
for test, logfile in [('code/test_identical_teacher_control.py', 'cpu-tests.txt'),
                      ('test_launch_control.py', 'launcher-cpu-tests.txt')]:
    with (ROOT / logfile).open('x') as handle:
        result = subprocess.run([PYTHON, '-m', 'pytest', '-q', '--disable-warnings',
                                 '-p', 'no:cacheprovider', test], cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    output = (ROOT / logfile).read_text()
    match = re.search(r'(\d+) passed', output)
    results.append({'test': test, 'exit_code': result.returncode,
                    'passed': int(match.group(1)) if match else 0,
                    'log': logfile, 'log_sha256': sha(ROOT / logfile)})
passed = all(r['exit_code'] == 0 and r['passed'] > 0 for r in results)
receipt = {'version': 'identical_teacher_control_cpu_tests_v1', 'passed': passed,
           'exit_code': 0 if passed else 1, 'python': PYTHON,
           'tests_passed': results[0]['passed'],
           'test_files': ['code/test_identical_teacher_control.py'],
           'source_freeze_sha256': sha(ROOT / 'source-freeze.json'),
           'test_log': 'cpu-tests.txt', 'test_log_sha256': results[0]['log_sha256'],
           'launch_helper_sha256': sha(ROOT / 'launch_control.py'),
           'launcher_tests_passed': results[1]['passed'],
           'launcher_test_log_sha256': results[1]['log_sha256'],
           'cuda_visible_devices': '', 'real_model_started': False}
with (ROOT / 'cpu-tests-receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps({'passed': passed, 'control_tests': results[0]['passed'],
                  'launcher_tests': results[1]['passed'], 'gpu_hidden': True,
                  'real_model_started': False}))
raise SystemExit(0 if passed else 1)
