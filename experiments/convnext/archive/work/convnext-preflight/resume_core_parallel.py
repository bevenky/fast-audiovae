"""Resume the existing core acquisition with four archive workers on Runpod."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_command(pid):
    try:
        return [value.decode() for value in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if value]
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    record_path = base / 'corpus-core-440h.launch.json'
    lock = (base / 'corpus-core-restart.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = json.loads(record_path.read_text())
    command = record['command']
    if command[0] != str(base / '.train-venv/bin/python') or command[1:3] != ['-m', 'audiovae_student.acquire']:
        raise ValueError('Unexpected recorded acquisition executable')
    if '--workers' in command:
        raise ValueError('The core acquisition has already been configured for parallel execution')
    if '--resume' not in command or command[command.index('--output-root') + 1] != str(base / 'data-expanded-core'):
        raise ValueError('Unexpected source root or missing resume option')
    if process_command(record['pid']) != command:
        raise RuntimeError('The original recorded process no longer matches; inspect before restarting')
    state_path = base / 'data-expanded-core/provenance/sources.json'
    original = json.loads(state_path.read_text())
    if original['state'] == 'ready':
        raise RuntimeError('Core acquisition already completed; no restart needed')
    history = base / 'acquisition-history' / str(time.time_ns())
    history.mkdir(parents=True)
    (history / 'core-launch.json').write_text(json.dumps(record, indent=2) + '\n')
    (history / 'core-before-stop.json').write_text(json.dumps(original, indent=2) + '\n')
    os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 25
    while process_command(record['pid']) == command:
        if time.monotonic() >= deadline:
            raise RuntimeError('Original process did not exit; no replacement was started')
        time.sleep(0.1)
    completed = json.loads(state_path.read_text())
    if completed['state'] == 'ready':
        print(json.dumps({'state': 'completed_before_restart', 'hours': completed['fresh_training_hours']}))
        return
    previous = {(item['dataset'], item['partition'], item['language']): item['manifest_sha256']
                for item in original['source_receipts']}
    current = {(item['dataset'], item['partition'], item['language']): item['manifest_sha256']
               for item in completed['source_receipts']}
    if any(current.get(key) != value for key, value in previous.items()):
        raise RuntimeError('A completed source changed during shutdown; replacement was not started')
    (history / 'core-after-stop.json').write_text(json.dumps(completed, indent=2) + '\n')
    next_command = [*command, '--workers', '4']
    environment = dict(os.environ, PYTHONPATH=str(base), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log_path = base / 'corpus-core-440h-parallel.log'
    with log_path.open('a') as log:
        child = subprocess.Popen(next_command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    next_record = dict(pid=child.pid, command=next_command, started_at=time.time(), log=str(log_path),
                       prior_launch=str(history / 'core-launch.json'),
                       preserved_completed_sources=len(current),
                       preserved_training_hours=completed['fresh_training_hours'])
    temporary = record_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(next_record, indent=2) + '\n')
    temporary.replace(record_path)
    time.sleep(1)
    if process_command(child.pid) != next_command:
        raise RuntimeError('Replacement acquisition exited; inspect its log')
    print(json.dumps(next_record))


if __name__ == '__main__':
    main()
