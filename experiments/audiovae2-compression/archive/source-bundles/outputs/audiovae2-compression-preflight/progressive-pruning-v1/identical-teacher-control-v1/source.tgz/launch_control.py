"""Launch the sealed post5000 identity control once, only when prerequisites hold.

This CPU-only launcher never imports model code, waits for a process, or edits
the completed training run. A refusal requires a fresh explicit invocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/tmp/fast-audiovae-progressive-pruning-v1/identical-teacher-control-v1')
ACTIVE = ROOT.parent / 'recovery-2000-5000'
PROJECT = Path('/workspace/fast-audiovae-compression-20260910-v1')
PYTHON = Path('/tmp/fast-audiovae-recovery-20260909/venv214/bin/python')
FROZEN_FILES = {'code/identical_teacher_control.py', 'code/test_identical_teacher_control.py',
                'code/test_group_model.py', 'code/group_model.py'}


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def readiness(root=ROOT, active=ACTIVE, project=PROJECT, python=PYTHON, proc_root=Path('/proc')):
    """Read-only checks; completed checkpoints get deeper validation in the control."""
    root, active, project, python = map(Path, (root, active, project, python))
    for name in ('process-launch.json', 'results', 'control.log'):
        if (root / name).exists():
            raise FileExistsError('Inspect the existing control artifact before retrying: ' + name)
    frozen = read_json(root / 'source-freeze.json')
    if set(frozen) != FROZEN_FILES:
        raise ValueError('Expected exactly the four reviewed control/test files')
    for name, expected in frozen.items():
        if sha(root / name) != expected:
            raise ValueError('Frozen control file differs: ' + name)
    if sha(project / 'code/group_model.py') != frozen['code/group_model.py']:
        raise ValueError('The bundled group model differs from the pinned project copy')
    tests = read_json(root / 'cpu-tests-receipt.json')
    count = tests.get('tests_passed')
    helper_sha = sha(Path(__file__))
    if (tests.get('version') != 'identical_teacher_control_cpu_tests_v1'
            or tests.get('passed') is not True or tests.get('exit_code') != 0
            or tests.get('python') != str(python) or type(count) is not int or count < 1
            or tests.get('test_files') != ['code/test_identical_teacher_control.py']
            or tests.get('source_freeze_sha256') != sha(root / 'source-freeze.json')
            or tests.get('test_log') != 'cpu-tests.txt'
            or tests.get('test_log_sha256') != sha(root / 'cpu-tests.txt')
            or tests.get('launch_helper_sha256') != helper_sha
            or sha(root / 'launch_control.py') != helper_sha
            or not re.search(r'(?<!\d)' + str(count) + r' passed\b', (root / 'cpu-tests.txt').read_text())):
        raise ValueError('Qualified CPU test receipt or its sealed evidence differs')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Qualified runtime executable is unavailable')
    if shutil.disk_usage(root).free < 200_000_000:
        raise RuntimeError('At least200MB free is required for control evidence and logs')

    run = active / 'segment-2000-5000'
    complete = read_json(run / 'completed.json')
    receipt = read_json(run / 'checkpoint-step5000.json')
    checkpoint = run / 'checkpoint-step5000.pt'
    if (complete.get('status') != 'awaiting_review' or complete.get('failure') is not None
            or complete.get('step') != 5000 or complete.get('cut_updates') != 5000
            or complete.get('sources_seen') != 60000
            or complete.get('frozen_state_preserved') is not True
            or complete.get('original_files_preserved') is not True
            or receipt.get('cut_updates') != 5000 or receipt.get('global_updates') != 5000
            or receipt.get('sources_seen') != 60000 or receipt.get('frozen_state_preserved') is not True):
        raise ValueError('The original5000-step run has not completed intact')
    checkpoint_sha = sha(checkpoint)
    if complete.get('last_checkpoint_sha256') != checkpoint_sha or receipt.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Completed checkpoint checksum differs')
    produced = read_json(active / 'producer-completed.json')
    if (produced.get('complete') is not True or produced.get('phases') != 2
            or produced.get('training_updates') != 0 or (active / 'producer-failed.json').exists()):
        raise ValueError('Both immutable target-production phases must have completed')

    scripts = {active / 'code/progressive_continue_5000.py',
               active.parent / 'recovery-1000-2000/code/progressive_continue.py',
               active.parent / 'code/progressive_train.py', active / 'produce_targets.py',
               active / 'produce_progressive_tail_5000.py', active / 'produce_progressive_extension_5000.py',
               root / 'code/identical_teacher_control.py'}
    exact_tokens = {os.fsencode(str(path)) for path in scripts}
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if exact_tokens.intersection(argv):
            raise RuntimeError('An exact training, producer or control command is still active: PID ' + process.name)
    return {'checkpoint_sha256': checkpoint_sha, 'source_sha256': frozen,
            'source_freeze_sha256': sha(root / 'source-freeze.json'),
            'cpu_tests_receipt_sha256': sha(root / 'cpu-tests-receipt.json'),
            'launch_helper_sha256': helper_sha, 'training_step': 5000, 'sources': 60000}


def launch(root=ROOT, active=ACTIVE, project=PROJECT, python=PYTHON,
           *, proc_root=Path('/proc'), popen=subprocess.Popen):
    root, active, project, python = map(Path, (root, active, project, python))
    lock = root / '.launch.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    started = False
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump({'launcher_pid': os.getpid(), 'created_unix': time.time()}, handle)
            handle.flush(); os.fsync(handle.fileno())
        verified = readiness(root, active, project, python, proc_root)
        overrides = {'PYTHONPATH': ':'.join(map(str, (root / 'code', active / 'code',
            active.parent / 'recovery-1000-2000/code', active.parent / 'code', project / 'code'))),
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
        environment = os.environ.copy(); environment.update(overrides)
        command = [str(python), '-u', str(root / 'code/identical_teacher_control.py'),
            '--checkpoint', str(active / 'segment-2000-5000/checkpoint-step5000.pt'),
            '--manifest', str(project / 'pilot-selection-v1.json'),
            '--source-plan', str(project / 'fresh-source-plan-v2/plan.json'),
            '--shards', '/dev/shm/fast-audiovae-compression-fresh-shards-v2',
            '--extension-plan', str(active / 'source-plan/plan.json'),
            '--extension-shards', '/dev/shm/fast-audiovae-progressive-extension-5000',
            '--assets', '/workspace/fast-audiovae-convnext-20260908-r1/assets',
            '--out', str(root / 'results')]
        with (root / 'control.log').open('xb') as log:
            process = popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=root)
            started = True
        result = {**verified, 'pid': process.pid, 'command': command, 'environment_overrides': overrides,
                  'launched_unix': time.time(), 'automatic_promotion': False}
        with (root / 'process-launch.json').open('x') as handle:
            json.dump(result, handle, indent=2); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        return result
    finally:
        # Once Popen succeeds, keep the lock even if writing its receipt fails.
        # Any ambiguous launch then needs inspection instead of spawning twice.
        if not started:
            lock.unlink()


if __name__ == '__main__':
    launched = launch()
    print(json.dumps({'pid': launched['pid'], 'completed5000_guard_passed': True,
                      'control_status': 'started', 'expected_sources': 60000,
                      'stability_updates_max': 12}, sort_keys=True))
