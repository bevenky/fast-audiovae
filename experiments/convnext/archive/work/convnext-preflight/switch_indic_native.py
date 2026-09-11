"""Safely switch the existing Indic acquisition transport without changing data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command_of(pid):
    try:
        return [x.decode() for x in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:
        return []


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def start(base, command, log_path):
    environment = dict(os.environ, PYTHONPATH=str(base), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log_path.open('ab') as log:
        child = subprocess.Popen(command, cwd=base, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, command=command, started_at=time.time(), log=str(log_path))
    time.sleep(1)
    if command_of(child.pid) != command:
        raise RuntimeError('Replacement process exited during startup; inspect its log')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pause', 'status', 'resume'])
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    audit_path = base / 'indic-native-switch.json'
    manifest = base / 'data-expanded-indic/train.jsonl'
    policy = base / 'data-expanded-indic/acquisition-policy.json'
    source_record = base / 'corpus-indic-44h.launch.json'
    controller_record = base / 'expanded-pilot.launch.json'
    if args.mode == 'pause':
        if audit_path.exists():
            raise RuntimeError('A migration audit already exists; inspect before repeating')
        controller, source = read(controller_record), read(source_record)
        if read(base / 'expanded-pilot/status.json')['phase'] != 'waiting_for_sources':
            raise RuntimeError('Pipeline is no longer waiting; do not interrupt training or verification')
        for record in [controller, source]:
            if command_of(record['pid']) != record['command']:
                raise RuntimeError('The recorded process changed; inspect before signaling')
        if '--download-method' in source['command'] or source['command'][source['command'].index('--workers') + 1] != '6':
            raise ValueError('Unexpected original Indic download configuration')
        child_path = base / 'expanded-pilot/training.launch.json'
        if child_path.exists():
            child = read(child_path)
            if command_of(child['pid']) == child['command']:
                raise RuntimeError('Training is already active; no process was stopped')
        history = base / 'acquisition-history' / ('indic-native-' + str(time.time_ns()))
        history.mkdir(parents=True)
        snapshot = history / 'train-before.jsonl'
        snapshot.write_bytes(manifest.read_bytes())
        audit = dict(state='pausing', original_source=source, original_controller=controller,
                     before_manifest=str(snapshot), before_rows=len(rows(snapshot)),
                     policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(), started_at=time.time())
        write(audit_path, audit)
        os.kill(controller['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while command_of(controller['pid']) == controller['command']:
            if time.monotonic() > deadline:
                raise RuntimeError('Controller did not stop; Indic remains untouched')
            time.sleep(0.1)
        os.kill(source['pid'], signal.SIGTERM)
        print(json.dumps({'state': 'pausing', 'preserved_rows': audit['before_rows']}))
        return
    audit = read(audit_path)
    old = audit['original_source']
    current = command_of(old['pid'])
    if args.mode == 'status':
        print(json.dumps({'old_source_running': current == old['command'],
                          'source_status': read(base / 'data-expanded-indic/progress.json')['status']}))
        return
    if audit['state'] != 'pausing' or current:
        raise RuntimeError('Original workers remain active or migration already resumed')
    if read(base / 'data-expanded-indic/progress.json')['status'] != 'paused':
        raise RuntimeError('A clean paused source snapshot is required')
    before, after = rows(Path(audit['before_manifest'])), rows(manifest)
    lookup = {item['source_id']: item for item in after}
    if len(lookup) != len(after) or any(lookup.get(item['source_id']) != item for item in before):
        raise RuntimeError('Previously collected source rows changed or disappeared')
    if hashlib.sha256(policy.read_bytes()).hexdigest() != audit['policy_sha256']:
        raise RuntimeError('The data-selection policy changed')
    command = [*old['command'], '--download-method', 'native']
    source = start(base, command, base / 'data-expanded-indic/run-native.log')
    write(source_record, source)
    audit.update(state='source_resumed', new_source=source, all_previous_rows_preserved=True,
                 paused_rows=len(after), resumed_at=time.time())
    write(audit_path, audit)
    controller = start(base, audit['original_controller']['command'], base / 'expanded-pilot.log')
    write(controller_record, controller)
    audit.update(state='resumed', new_controller=controller)
    write(audit_path, audit)
    print(json.dumps({'state': 'resumed', 'source_pid': source['pid'], 'controller_pid': controller['pid'],
                      'all_previous_rows_preserved': True, 'paused_rows': len(after)}))


if __name__ == '__main__':
    main()
