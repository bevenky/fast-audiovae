"""Stop only a fully committed training generation, then seal its data successor.

A failed boundary inspection resumes the same live process. No journal is
truncated and no older checkpoint is replayed. This tool never starts training.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import torch

from audiovae_student.checkpoint_retention import atomic_json
from audiovae_student.comparison_data import load_comparison_plan
from audiovae_student.restart_data import FixedWindowSampler, digest, file_sha


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')
RUN = BASE / 'training-runs/decoder-recipe-v2'
PYTHON = Path('/workspace/fast-audiovae-convnext-20260908-r1/.train-venv/bin/python')


def proc(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    return {'state': fields[0], 'start_ticks': fields[19],
            'argv': (root / 'cmdline').read_bytes().split(b'\0')[:-1]}


def frozen_boundary(plan):
    if (RUN / 'inflight.json').exists():
        return None
    payload = torch.load(RUN / 'latest.pt', map_location='cpu', weights_only=True)
    step = payload['engine']['step']
    journal = [json.loads(line) for line in (RUN / 'exposure.jsonl').read_text().splitlines()]
    metrics = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    if len(journal) != step or len(metrics) != step:
        return None
    identity = payload['identity']
    if json.loads((RUN / 'run.json').read_text()) != identity:
        raise ValueError('Parent run identity changed')
    sampler = FixedWindowSampler(plan['windows'])
    sampler.load_state_dict(payload['sampler'])
    batch = identity['batch_size']
    chain = digest([])
    for index, record in enumerate(journal):
        selected = plan['windows'][index * batch:(index + 1) * batch]
        body = {key: value for key, value in record.items() if key != 'sha256'}
        if (record['step'] != index + 1 or record['cursor'] != min((index + 1) * batch, len(plan['windows']))
                or record['window_identity'] != sampler.identity_sha256
                or record['batch_windows_sha256'] != digest([w.to_dict() for w in selected])
                or record['previous_sha256'] != chain or record['sha256'] != digest(body)):
            raise ValueError('Parent journal changed')
        chain = record['sha256']
    if (chain != payload['journal_sha256'] or digest(metrics) != payload['metrics_sha256']
            or any(row['step'] != index + 1 for index, row in enumerate(metrics))
            or sampler.cursor != min(step * batch, len(plan['windows']))
            or payload['teacher_coverage']['valid_samples'] != sum(
                w.valid_output_samples48k for w in plan['windows'][:sampler.cursor])):
        raise ValueError('Parent checkpoint and exposure disagree')
    return {'step': step, 'cursor': sampler.cursor, 'checkpoint_sha256': file_sha(RUN / 'latest.pt'),
        'journal_file_sha256': file_sha(RUN / 'exposure.jsonl'),
        'metrics_file_sha256': file_sha(RUN / 'metrics.jsonl'),
        'cache_identity': identity['data']['source_corpus']}


def evict_old_targets(expected_identity):
    cache = BASE / 'teacher-cache-train'
    if cache.is_symlink() or json.loads((cache / 'identity.json').read_text()) != expected_identity:
        raise ValueError('Old cache identity changed')
    with (cache / '.writer.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        targets = cache / 'targets'
        if targets.is_symlink():
            raise ValueError('Cache target directory cannot be a symlink')
        paths = list(targets.glob('*.pt'))
        if any(p.is_symlink() or not p.is_file() or not re.fullmatch('[a-f0-9]{64}\\.pt', p.name) for p in paths):
            raise ValueError('Unexpected cached-target entry')
        receipt = {'kind': 'regenerable_teacher_cache_eviction', 'target_count': len(paths),
                   'bytes': sum(p.stat().st_size for p in paths),
                   'files': [p.name for p in paths], 'source_audio_changed': False,
                   'checkpoint_or_journal_changed': False}
        atomic_json(BASE / 'remediation/old-cache-eviction.json', receipt)
        for path in paths:
            path.unlink()
        # SourceCorpus removes these missing cache entries from SQLite on any
        # later legitimate reopen. Its identity and statistics remain intact.
        return {key: value for key, value in receipt.items() if key != 'files'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=900)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f'Handoff interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    launch = json.loads((BASE / 'training-launch.json').read_text())
    pid = launch['pid']
    expected = [str(PYTHON).encode(), str(BASE / 'run_recipe_v2_stage.py').encode()]
    original = proc(pid)
    if original['argv'] != expected:
        raise ValueError('Refusing to signal an unverified process')
    term_mask = 1 << (signal.SIGTERM - 1)
    dispositions = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                    for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines() if ':' in line}
    if any(int(dispositions[key], 16) & term_mask for key in ('SigBlk', 'SigIgn', 'SigCgt')):
        raise ValueError('Parent SIGTERM disposition is not unblocked default termination')
    pidfd = os.pidfd_open(pid)
    def send(signum):
        current = proc(pid)
        if current['start_ticks'] != original['start_ticks'] or current['argv'] != expected:
            raise ValueError('Parent identity changed before signal')
        signal.pidfd_send_signal(pidfd, signum)
    if (BASE / 'remediation/committed-handoff.json').exists():
        raise ValueError('A committed handoff already exists')
    torch.set_num_threads(1)
    plan = load_comparison_plan(BASE / 'data/recipe-v2/optimization')
    deadline, seen_inode = time.monotonic() + args.wait_seconds, None
    stopped, terminated = False, False
    try:
        while time.monotonic() < deadline:
            current = proc(pid)
            if current['start_ticks'] != original['start_ticks'] or current['argv'] != expected:
                raise ValueError('Parent process identity changed')
            inode = (RUN / 'latest.pt').stat().st_ino
            if inode == seen_inode:
                time.sleep(.1)
                continue
            seen_inode = inode
            stopped = True
            send(signal.SIGSTOP)
            for _ in range(100):
                if proc(pid)['state'] == 'T':
                    break
                time.sleep(.01)
            if proc(pid)['state'] != 'T':
                raise RuntimeError('Parent did not enter a verified suspended state')
            boundary = frozen_boundary(plan)
            if boundary is None:
                send(signal.SIGCONT)
                stopped = False
                continue
            stage = BASE / 'remediation' / f"continuation-step{boundary['step']:06d}"
            env = dict(os.environ, PYTHONPATH=str(args.code))
            command = [str(PYTHON), str(args.code / 'prepare_recipe_continuation.py'),
                '--destination', str(stage), '--expected-checkpoint-sha', boundary['checkpoint_sha256']]
            preparation_started = time.monotonic()
            subprocess.run(command, env=env, check=True, timeout=300)
            preparation_seconds = time.monotonic() - preparation_started
            if frozen_boundary(plan) != boundary:
                raise ValueError('Suspended parent changed during preparation')
            # SIGTERM is pending while stopped, then is delivered before user
            # code runs. The verified saved boundary is retained unchanged.
            send(signal.SIGTERM)
            send(signal.SIGCONT)
            for _ in range(200):
                try:
                    if proc(pid)['state'] == 'Z':
                        break
                except FileNotFoundError:
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('Parent did not terminate; do not start a successor')
            terminated, stopped = True, False
            with (RUN.parent / ('.' + RUN.name + '.runner.lock')).open('a+b') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (file_sha(RUN / 'latest.pt') != boundary['checkpoint_sha256']
                        or file_sha(RUN / 'exposure.jsonl') != boundary['journal_file_sha256']
                        or file_sha(RUN / 'metrics.jsonl') != boundary['metrics_file_sha256']
                        or (RUN / 'inflight.json').exists()):
                    raise ValueError('Parent changed after verified termination')
                cache = evict_old_targets(boundary.pop('cache_identity'))
                receipt = {**boundary, 'parent_pid': pid, 'parent_start_ticks': original['start_ticks'],
                    'plan_path': str(stage), 'parent_stopped_at_committed_boundary': True,
                    'expected_plan_identity_sha256': json.loads((stage / 'handoff.json').read_text())['expected_plan_identity_sha256'],
                    'new_optimizer_updates': 0, 'old_target_cache': cache,
                    'metadata_preparation_seconds': preparation_seconds}
                atomic_json(BASE / 'remediation/committed-handoff.json', receipt)
                atomic_json(RUN / 'status-before-handoff.json', json.loads((RUN / 'status.json').read_text()))
                atomic_json(RUN / 'status.json', {'state': 'continued_in_versioned_data_segment',
                    'step': boundary['step'], 'consumed_windows': boundary['cursor'],
                    'successor': str(BASE / 'training-runs/decoder-recipe-v2-expressive')})
            print(json.dumps(receipt, sort_keys=True), flush=True)
            return
        raise TimeoutError('No committed boundary captured; parent continues unchanged')
    finally:
        try:
            if stopped and not terminated:
                try:
                    send(signal.SIGCONT)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        finally:
            os.close(pidfd)


if __name__ == '__main__':
    main()
