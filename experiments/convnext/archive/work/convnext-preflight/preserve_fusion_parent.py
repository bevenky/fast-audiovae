"""Verify a paused continuation and preserve its committed generation.

Run on the pod with the original continuation package on PYTHONPATH. This
script does not signal a process, train, delete files, or alter the source run.
Checkpoint immutability relies on the trainer's atomic replacement protocol;
never write to parent.pt in place, since it deliberately shares the inode.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid


BASE = Path('/workspace/fast-audiovae-convnext-20260909-r9')


def regular(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Expected a regular, nonsymlink file: ' + str(path))
    return info


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def checkpoint_position(payload, *, expected_identity=None):
    """Validate the committed continuation cursor without newer helper APIs."""
    identity, sampler = payload['identity'], payload['sampler']
    if expected_identity is not None and identity != expected_identity:
        raise ValueError('Checkpoint run identity changed')
    kind = payload.get('format_version')
    if kind != 'recipe_v2_continuation_v1' or identity.get('kind') != kind:
        raise ValueError('Expected a versioned continuation checkpoint')
    step, batch, cursor = payload['engine']['step'], identity['batch_size'], sampler['cursor']
    start, total, count = (identity['global_start_step'], identity['global_total_steps'],
                           identity['segment_window_count'])
    if (any(type(value) is not int for value in (step, batch, cursor, start, total, count))
            or min(step, cursor) < 0 or min(batch, start, count) < 1
            or identity['parent']['step'] != start or identity['parent']['batch_size'] != batch
            or total != start + (count + batch - 1) // batch or not start <= step <= total
            or cursor != min((step - start) * batch, count)
            or sampler.get('identity_sha256') != identity['window_identity']):
        raise ValueError('Checkpoint global budget, parent binding or exposure cursor disagrees')
    canonical = (json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return {'format_version': kind, 'step': step, 'cursor': cursor,
        'global_start_step': start, 'segment_step': step - start,
        'segment_window_count': count, 'run_identity_sha256': sha_bytes(canonical)}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(path, raw):
    """Atomic no-clobber publication. Keep the tiny staging inode for recovery."""
    if path.exists() or path.is_symlink():
        regular(path)
        if path.read_bytes() != raw:
            raise ValueError('Refusing to overwrite a different preserved file: ' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.prepared-' + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path, follow_symlinks=False)
    sync_directory(path.parent)


@contextmanager
def paused_run_lock(run):
    path = run.parent / ('.' + run.name + '.runner.lock')
    regular(path)
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('The continuation runner is still active; do not snapshot it') from error
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--code', type=Path)
    parser.add_argument('--expected-step', type=int, default=8090)
    parser.add_argument('--expected-cursor', type=int, default=70080)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    run = args.run or base / 'training-runs/decoder-recipe-v2-expressive'
    plan_dir = args.plan or base / 'remediation/continuation-step005900'
    destination = args.destination or base / 'remediation/fusion-screen'
    code = args.code or base / 'remediation/continuation-code'
    if destination.is_symlink():
        raise ValueError('Destination cannot be a symlink')
    resolved_run, resolved_destination = run.resolve(strict=True), destination.resolve()
    if (resolved_run == resolved_destination or resolved_run in resolved_destination.parents
            or resolved_destination in resolved_run.parents):
        raise ValueError('Destination must be outside the source run')
    sys.path.insert(0, str(code.resolve(strict=True)))
    import torch
    from audiovae_student.continuation_data import load_continuation_plan
    from audiovae_student.recipe_v2 import _norm_fingerprint, _report_hash
    from audiovae_student.recipe_v2_continuation import implementation_identity, verify_loaded_plan
    from audiovae_student.restart_data import FixedWindowSampler, digest, file_sha

    torch.set_num_threads(1)
    with paused_run_lock(run):
        if (run / 'inflight.json').exists() or (run / 'inflight.json').is_symlink():
            raise ValueError('Uncommitted in-flight exposure remains')
        names = ('run.json', 'status.json', 'pause.request.json', 'metrics.jsonl', 'exposure.jsonl')
        source = {}
        for name in names:
            regular(run / name)
            source[name] = (run / name).read_bytes()
        status = json.loads(source['status.json'])
        if (status.get('state') != 'paused_by_request'
                or json.loads(source['pause.request.json']) != {'action': 'pause'}):
            raise ValueError('The explicit pause request is not acknowledged')
        latest = run / 'latest.pt'
        before_stat = regular(latest)
        with latest.open('rb') as handle:
            checkpoint_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            handle.seek(0)
            payload = torch.load(handle, map_location='cpu', weights_only=True)
        identity, state = payload['identity'], payload['engine']
        position = checkpoint_position(payload, expected_identity=json.loads(source['run.json']))
        if position['format_version'] != 'recipe_v2_continuation_v1':
            raise ValueError('Expected the expressive continuation checkpoint')
        step, cursor, batch = state['step'], payload['sampler']['cursor'], identity['batch_size']
        if (step != args.expected_step or cursor != args.expected_cursor
                or status.get('step') != step or status.get('consumed_windows') != cursor
                or status.get('segment_updates') != position['segment_step']):
            raise ValueError('Requested boundary, status, checkpoint and sampler disagree')
        actual_code = implementation_identity()
        if identity.get('implementation') != actual_code:
            raise ValueError('Bound continuation implementation differs from its recorded source')
        launcher = code / 'run_recipe_continuation.py'
        if file_sha(launcher) != identity['data'].get('continuation_launcher_sha256'):
            raise ValueError('Bound continuation launcher changed')
        plan = load_continuation_plan(plan_dir)
        verify_loaded_plan(plan)
        if plan['identity'] != identity['plan']:
            raise ValueError('Continuation plan changed')
        sampler = FixedWindowSampler(plan['windows'])
        sampler.load_state_dict(payload['sampler'])
        if sampler.identity_sha256 != identity['window_identity']:
            raise ValueError('Fixed window identity changed')
        records = [json.loads(line) for line in source['exposure.jsonl'].splitlines()]
        metrics = [json.loads(line) for line in source['metrics.jsonl'].splitlines()]
        chain, start = digest([]), identity['global_start_step']
        for index, record in enumerate(records):
            chosen = plan['windows'][index * batch:(index + 1) * batch]
            body = {key: value for key, value in record.items() if key != 'sha256'}
            if (record.get('step') != start + index + 1
                    or record.get('cursor') != min((index + 1) * batch, len(plan['windows']))
                    or record.get('window_identity') != sampler.identity_sha256
                    or record.get('batch_windows_sha256') != digest([w.to_dict() for w in chosen])
                    or record.get('previous_sha256') != chain or record.get('sha256') != digest(body)):
                raise ValueError('Exposure journal differs from the immutable selected intervals')
            chain = record['sha256']
        if (len(records) != step - start or len(metrics) != step - start
                or any(row.get('step') != start + i + 1 for i, row in enumerate(metrics))
                or chain != payload['journal_sha256'] or digest(metrics) != payload['metrics_sha256']
                or metrics[-1] != payload['latest_metrics'] or status.get('latest_metrics') != metrics[-1]
                or metrics[-1].get('consumed_windows') != cursor):
            raise ValueError('Checkpoint is not the fully committed metrics/journal boundary')
        coverage = payload['teacher_coverage']
        if (digest(coverage) != records[-1]['teacher_coverage_sha256']
                or coverage['valid_samples'] != sum(w.valid_output_samples48k for w in plan['windows'][:cursor])
                or status.get('teacher_coverage') != coverage
                or status.get('parent_teacher_coverage') != payload['parent_teacher_coverage']):
            raise ValueError('Exposure coverage differs from checkpoint/status')
        calibration = state.get('calibration')
        if (not calibration or digest(calibration) != identity['calibration_sha256']
                or calibration['report_sha256'] != _report_hash(calibration['report'])
                or calibration['fixed_buffers_sha256'] != _norm_fingerprint(state['model'])
                or state['discriminator_updates'] != step - state['recipe']['reconstruction_warmup_steps']):
            raise ValueError('Calibration or discriminator clock changed')
        if (state.get('recipe') != identity.get('recipe')
                or state.get('model_config') != identity.get('model_config')):
            raise ValueError('Checkpoint recipe/model identity changed')
        for name, raw in source.items():
            if (run / name).read_bytes() != raw:
                raise ValueError('Paused source metadata changed during verification')
        after_stat = regular(latest)
        if ((before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns)
                != (after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns)
                or file_sha(latest) != checkpoint_sha or (run / 'inflight.json').exists()):
            raise ValueError('Paused checkpoint changed during verification')
        destination.mkdir(parents=True, exist_ok=True)
        snapshot = destination / 'parent.pt'
        if snapshot.exists() or snapshot.is_symlink():
            regular(snapshot)
            if not os.path.samefile(latest, snapshot) or file_sha(snapshot) != checkpoint_sha:
                raise ValueError('Refusing to overwrite a different parent snapshot')
        else:
            os.link(latest, snapshot, follow_symlinks=False)
            sync_directory(destination)
        for name, raw in source.items():
            publish_bytes(destination / ('parent-' + name), raw)
        receipt_path = destination / 'parent-receipt.json'
        stable = {**position, 'checkpoint_sha256': checkpoint_sha,
            'checkpoint_bytes': before_stat.st_size, 'source_run': str(run), 'source_checkpoint': str(latest),
            'snapshot': str(snapshot), 'plan_path': str(plan_dir),
            'plan_identity_sha256': plan['identity']['identity_sha256'],
            'run_identity_sha256': digest(identity), 'status': status,
            'journal_sha256': chain, 'metrics_sha256': payload['metrics_sha256'],
            'source_file_sha256': {name: sha_bytes(raw) for name, raw in source.items()},
            'implementation_sha256': actual_code, 'launcher_sha256': file_sha(launcher),
            'calibration_sha256': identity['calibration_sha256'],
            'snapshot_shares_source_inode': True, 'source_run_modified': False,
            'optimizer_updates': 0, 'files_deleted': 0,
            'checkpoint_immutability': 'Atomic replacement protocol; never write either hard link in place'}
        if receipt_path.exists() or receipt_path.is_symlink():
            regular(receipt_path)
            receipt = json.loads(receipt_path.read_bytes())
            if any(receipt.get(key) != value for key, value in stable.items()):
                raise ValueError('Existing preservation receipt differs; no files overwritten')
        else:
            receipt = {**stable, 'preserved_at_utc': datetime.now(timezone.utc).isoformat(),
                'free_bytes_after_preservation': shutil.disk_usage(destination).free,
                'preservation_script_sha256': file_sha(Path(__file__))}
            publish_bytes(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
        print(json.dumps({'state': 'preserved', 'step': step, 'cursor': cursor,
            'checkpoint_sha256': checkpoint_sha, 'receipt': str(receipt_path), 'snapshot': str(snapshot),
            'free_bytes': shutil.disk_usage(destination).free, 'source_run_modified': False,
            'optimizer_updates': 0, 'files_deleted': 0}), flush=True)


if __name__ == '__main__':
    main()
