"""An immutable source-plan extension and bounded, checkpointed target cache."""
from __future__ import annotations
import json
import fcntl
from contextlib import contextmanager
from pathlib import Path
import torch
from fresh_training_data import FreshTrainingData, digest, sha, stat_identity

VERSION = 'audiovae2_group_continuation_extension_v2'
PARENT_ID = '3c65151fd2b37900257e179fee62218c3d33055c94d47dbb7c213593777e7c7d'
START, PARENT_STOP, STOP = 24000, 27000, 76500
KEYS = ('source_id', 'audio_sha256', 'parent_recording_id')
SHARD_SIZE, MAX_CACHED_SOURCES, RETAIN_CONSUMED = 300, 6000, 0


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_extension(parent, extension):
    if (extension.get('version') != VERSION or parent.identity != PARENT_ID
            or extension.get('parent_plan_identity_sha256') != parent.identity
            or extension.get('authorized_source_interval') != [START, STOP]
            or extension.get('appended_start_index') != PARENT_STOP
            or extension.get('total_fresh_sources') != STOP
            or extension.get('teacher_source_sha256') != parent.plan['teacher_source_sha256']
            or extension.get('teacher_checkpoint_sha256') != parent.plan['teacher_checkpoint_sha256']):
        raise ValueError('Continuation extension contract differs')
    rows = extension['appended_rows']
    if len(rows) != STOP-PARENT_STOP:
        raise ValueError('Extension must contain exactly49500 appended sources')
    blocked = {key:set(parent.plan['blocked_identities'][key]) for key in KEYS}
    for row in parent.plan['rows']:
        for key in KEYS: blocked[key].add(row['manifest_row'][key])
    for row in rows:
        source = row['manifest_row']
        if row['source_id'] != source['source_id'] or source['split'] != 'train':
            raise ValueError('Invalid extension training identity')
        FreshTrainingData._geometry(row)
        for key in KEYS:
            value = source[key]
            if not value or value in blocked[key]:
                raise ValueError('Repeated, previously planned, or reserved identity: '+key)
            blocked[key].add(value)
    ids = list(parent.source_ids) + [row['source_id'] for row in rows]
    if extension['source_ids_sha256'] != digest(ids):
        raise ValueError('Extended source order differs')
    return list(parent.plan['rows']) + rows, ids


def validate_committed_checkpoint(payload, cursor, source_ids, original_ids, identity):
    if (type(cursor) is not int or not START <= cursor <= STOP or (cursor-START) % 12
            or payload.get('fresh_cursor') != cursor
            or payload.get('identity',{}).get('source_plan_identity_sha256') != identity):
        raise ValueError('Checkpoint does not bind the continuation cache cursor')
    expected = list(original_ids) + list(source_ids[:cursor])
    observed = list(payload.get('historical_sources_seen',[]))+list(payload.get('additional_sources_seen',[]))
    if observed != expected or payload.get('total_source_count') != len(expected) or len(set(expected)) != len(expected):
        raise ValueError('Checkpoint source history differs from the immutable plan')
    expected_step = 5625+(cursor-START)//12
    if payload.get('optimizer_step') != expected_step or payload.get('accumulation') != 12:
        raise ValueError('Checkpoint optimizer and exposure counters differ')
    return max(START, ((cursor-RETAIN_CONSUMED)//SHARD_SIZE)*SHARD_SIZE)


class ContinuationData(FreshTrainingData):
    @contextmanager
    def _cache_lock(self, exclusive=False):
        with (self.shards_dir/'.cache.lock').open('a+b') as handle:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try: yield
            finally: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

    def __init__(self, plan_path, original_manifest_path, original_pools, shards_dir):
        plan_path = Path(plan_path)
        extension = json.loads(plan_path.read_text())
        identity = extension.get('identity_sha256')
        if digest({k:v for k,v in extension.items() if k != 'identity_sha256'}) != identity:
            raise ValueError('Extension plan identity differs')
        parent_path = Path(extension['parent_plan_path'])
        if sha(parent_path) != extension['parent_plan_sha256']:
            raise ValueError('Original source-plan bytes changed')
        parent = FreshTrainingData(parent_path, original_manifest_path, original_pools, shards_dir)
        rows, ids = validate_extension(parent, extension)
        self.plan_path, self.shards_dir = plan_path, Path(shards_dir)
        self.identity, self.extension = identity, extension
        self.plan = dict(parent.plan, rows=rows, source_ids=ids, identity_sha256=identity)
        self.source_ids = tuple(ids)
        self.original_ids = [r['source_id'] for r in original_pools['fit']]
        self._hashes = dict(parent._hashes)
        self._hashes[str(plan_path)] = sha(plan_path)
        self._stats = {path:stat_identity(path) for path in self._hashes}
        self._cached_index, self._cached_crops = None, None
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.retired_before = START
        self.committed_cursor = START
        marker = self.shards_dir/'committed-cursor.json'
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved['plan_identity_sha256'] != self.identity:
                raise ValueError('Cache commitment belongs to another plan')
            checkpoint = Path(saved['checkpoint_path'])
            if sha(checkpoint) != saved['checkpoint_sha256']:
                raise ValueError('Committed checkpoint bytes differ')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            before = validate_committed_checkpoint(payload, saved['fresh_cursor'], self.source_ids, self.original_ids, self.identity)
            if saved['retired_before'] != before:
                raise ValueError('Cache retirement bound differs from saved checkpoint')
            self.retired_before, self.committed_cursor = before, saved['fresh_cursor']

    def take(self, start, count):
        with self._cache_lock():
            self._refresh_committed_unlocked()
            if type(start) is not int or start < max(START, self.retired_before):
                raise ValueError('Source interval is historical or has been retired')
            return super().take(start, count)

    def checkpoint_committed(self, checkpoint_path, checkpoint_sha256, fresh_cursor):
        with self._cache_lock(exclusive=True):
            return self._checkpoint_committed_unlocked(checkpoint_path,checkpoint_sha256,fresh_cursor)

    def _checkpoint_committed_unlocked(self, checkpoint_path, checkpoint_sha256, fresh_cursor):
        """Retire only new cached tensors covered by an authenticated checkpoint.

        Receipts and source provenance remain. No file in the original cache
        is touched. Zero retention permits the next6000-source review interval.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if sha(checkpoint_path) != checkpoint_sha256:
            raise ValueError('Cannot release cache for unverified checkpoint bytes')
        receipt = json.loads(checkpoint_path.with_suffix('.json').read_text())
        if (receipt.get('checkpoint_sha256') != checkpoint_sha256 or receipt.get('fresh_cursor') != fresh_cursor
                or receipt.get('optimizer_and_rng_saved') is not True or receipt.get('frozen_state_preserved') is not True):
            raise ValueError('Cache retirement requires its durable validated checkpoint receipt')
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        before = validate_committed_checkpoint(payload, fresh_cursor, self.source_ids, self.original_ids, self.identity)
        if fresh_cursor < self.committed_cursor or before < self.retired_before:
            raise ValueError('Cache commitment cannot move backwards')
        marker = {'plan_identity_sha256':self.identity,'checkpoint_path':str(checkpoint_path),
                  'checkpoint_sha256':checkpoint_sha256,'fresh_cursor':fresh_cursor,'retired_before':before,
                  'source_prefix_sha256':digest(list(self.source_ids[:fresh_cursor]))}
        for index in range(START, before, SHARD_SIZE):
            stop = index+SHARD_SIZE
            if stop > before: break
            directory = self.shards_dir/f'{index:06d}-{stop:06d}'
            path, receipt_path = directory/'pairs.pt', directory/'receipt.json'
            if path.is_symlink() or directory.is_symlink():
                raise ValueError('Refusing cache retirement through a symbolic link')
            if not path.exists():
                if not (directory/'retired.json').is_file():
                    raise FileNotFoundError('Missing consumed shard without retirement receipt')
                continue
            receipt = json.loads(receipt_path.read_text())
            if (receipt['plan_identity_sha256'] != self.identity or receipt['start_index'] != index
                    or receipt['stop_index'] != stop or receipt['source_ids'] != list(self.source_ids[index:stop])
                    or sha(path) != receipt['pairs_sha256']):
                raise ValueError('Refusing retirement of changed source targets')
            atomic_json(directory/'retired.json',dict(marker,start_index=index,stop_index=stop,
                        pairs_sha256=receipt['pairs_sha256'],reason='Regenerable targets covered by committed source ledger'))
            path.unlink()
            self._hashes.pop(str(path), None); self._stats.pop(str(path), None)
            if self._cached_index == index: self._cached_index, self._cached_crops = None, None
        atomic_json(self.shards_dir/'committed-cursor.json', marker)
        self.retired_before, self.committed_cursor = before, fresh_cursor
        return marker

    def refresh_committed(self):
        with self._cache_lock():
            self._refresh_committed_unlocked()

    def _refresh_committed_unlocked(self):
        marker = self.shards_dir/'committed-cursor.json'
        if not marker.is_file(): return
        current = stat_identity(marker)
        if getattr(self,'_commit_stat',None) == current: return
        saved = json.loads(marker.read_text())
        if saved['plan_identity_sha256'] != self.identity:
            raise ValueError('Foreign rolling-cache commitment')
        checkpoint = Path(saved['checkpoint_path'])
        if sha(checkpoint) != saved['checkpoint_sha256']:
            raise ValueError('Committed checkpoint changed')
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        before = validate_committed_checkpoint(payload,saved['fresh_cursor'],self.source_ids,self.original_ids,self.identity)
        if saved['retired_before'] != before or before < self.retired_before:
            raise ValueError('Invalid or backwards cache commitment')
        for path, expected in list(self._hashes.items()):
            p = Path(path)
            if p.name != 'pairs.pt' or p.parent.parent.resolve() != self.shards_dir.resolve() or p.exists(): continue
            receipt = json.loads((p.parent/'retired.json').read_text())
            if (receipt['plan_identity_sha256']!=self.identity or receipt['stop_index']>before
                    or receipt['pairs_sha256']!=expected):
                raise ValueError('Consumed target disappeared without authenticated retirement')
            self._hashes.pop(path);self._stats.pop(path,None)
        self.retired_before,self.committed_cursor = before,saved['fresh_cursor']
        self._commit_stat = current
        if self._cached_index is not None and self._cached_index < before:
            self._cached_index,self._cached_crops = None,None

    def can_produce_through(self, stop):
        self.refresh_committed()
        return (type(stop) is int and START < stop <= STOP and stop%SHARD_SIZE==0
                and stop <= self.retired_before+MAX_CACHED_SOURCES)

    def assert_unchanged(self):
        with self._cache_lock():
            self._refresh_committed_unlocked()
            super().assert_unchanged()
