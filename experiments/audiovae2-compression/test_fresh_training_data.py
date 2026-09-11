"""CPU checks for immutable source plans and sealed continuation shards."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_training_data as fresh


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')))


@pytest.fixture
def plan_files(tmp_path):
    keys = ('source_id', 'audio_sha256', 'parent_recording_id')
    original = {'splits': {}}
    pools = {}
    for name in ('fit', 'calibration', 'development'):
        row = {'source_id': 'old-' + name, 'manifest_row': {key: 'old-' + name for key in keys}}
        original['splits'][name] = {'rows': [row]}
        pools[name] = [row]
    original['identity_sha256'] = fresh.digest(original)
    original_path = tmp_path / 'original.json'; write_json(original_path, original)
    rows = []
    for index in range(27000):
        identity = f'new-{index}'
        row = {'source_id': identity, 'manifest_row': {**{key: identity for key in keys}, 'split': 'train'},
               'start_frame': 0, 'context_start_frame': 0, 'context_frames': 0,
               'scored_frames': 3, 'valid_scored_samples': 4098, 'input_samples16k': 1366,
               'phase': 'approved' if index < 12000 else 'conditional_reserve'}
        rows.append(row)
    plan = {'version': fresh.PLAN_VERSION, 'teacher_source_sha256': fresh.SOURCE_SHA256,
            'teacher_checkpoint_sha256': fresh.CHECKPOINT_SHA256,
            'original_manifest_sha256': fresh.sha(original_path), 'source_ids': [r['source_id'] for r in rows],
            'rows': rows, 'shard_size': 300, 'approved_range': [0, 12000],
            'conditional_reserve_range': [12000, 27000],
            'blocked_identities': {key: ['old-' + name for name in pools] for key in keys}}
    plan['source_ids_sha256'] = fresh.digest(plan['source_ids'])
    plan['identity_sha256'] = fresh.digest(plan)
    plan_path = tmp_path / 'plan.json'; write_json(plan_path, plan)
    shards = tmp_path / 'shards'; shards.mkdir()
    return plan_path, original_path, pools, shards


def seal_first(data):
    directory = data.shards_dir / '000000-000300'; directory.mkdir()
    latents = torch.zeros(1, 64, 3)
    audio = torch.zeros(1, 1, 3 * 1920)
    reference = torch.zeros(1, 1, 3 * 640)
    crops = [{**{key: row[key] for key in fresh.GEOMETRY}, 'latents': latents,
              'teacher_audio': audio, 'reference16k': reference, 'cache_key': 'a' * 64}
             for row in data.plan['rows'][:300]]
    payload = {'format': fresh.PAIR_VERSION, 'plan_identity_sha256': data.identity,
               'start_index': 0, 'stop_index': 300, 'crops': crops}
    path = directory / 'pairs.pt'; torch.save(payload, path)
    receipt = {'format': fresh.PAIR_VERSION, 'complete': True,
               'plan_identity_sha256': data.identity, 'start_index': 0, 'stop_index': 300,
               'source_ids': list(data.source_ids[:300]), 'source_ids_sha256': fresh.digest(list(data.source_ids[:300])),
               'teacher_source_sha256': fresh.SOURCE_SHA256, 'teacher_checkpoint_sha256': fresh.CHECKPOINT_SHA256,
               'parameter_updates': 0, 'pairs_path': str(path), 'pairs_sha256': fresh.sha(path)}
    write_json(directory / 'receipt.json', receipt)
    return path, receipt, payload


def test_sealed_source_order_missing_shard_and_immutable_bytes(plan_files):
    data = fresh.FreshTrainingData(*plan_files)
    path, receipt, _ = seal_first(data)
    assert [row['source_id'] for row in data.take(297, 3)] == ['new-297', 'new-298', 'new-299']
    with pytest.raises(FileNotFoundError): data.take(300, 3)
    data.assert_unchanged()
    with path.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(RuntimeError, match='changed'): data.assert_unchanged()


@pytest.mark.parametrize('key', ['source_id', 'audio_sha256', 'parent_recording_id'])
def test_resealed_plan_still_rejects_reserved_or_repeated_identities(plan_files, key):
    path, *_ = plan_files
    plan = json.loads(path.read_text())
    plan['rows'][-1]['manifest_row'][key] = plan['rows'][0]['manifest_row'][key]
    if key == 'source_id':
        plan['rows'][-1]['source_id'] = plan['source_ids'][-1] = plan['source_ids'][0]
        plan['source_ids_sha256'] = fresh.digest(plan['source_ids'])
    plan['identity_sha256'] = fresh.digest({k: v for k, v in plan.items() if k != 'identity_sha256'})
    write_json(path, plan)
    with pytest.raises(ValueError, match='Repeated or reserved'): fresh.FreshTrainingData(*plan_files)


def test_sealed_digest_does_not_allow_nonfinite_target_or_geometry_change(plan_files):
    data = fresh.FreshTrainingData(*plan_files)
    path, receipt, payload = seal_first(data)
    payload['crops'][-1]['latents'] = torch.full((1, 64, 3), float('nan'))
    torch.save(payload, path)
    receipt['pairs_sha256'] = fresh.sha(path); write_json(path.parent / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='Invalid frozen target'): data.take(0, 3)


def test_unchanged_consumed_shards_do_not_rehash_large_targets(plan_files, monkeypatch):
    data = fresh.FreshTrainingData(*plan_files)
    seal_first(data); data.take(0, 3)
    monkeypatch.setattr(fresh, 'sha', lambda path: pytest.fail('Unchanged files must not be rehashed'))
    data.assert_unchanged()
    assert len(data.take(3, 3)) == 3


def test_geometry_rejects_noncausal_context_and_invalid_tail():
    row = {'start_frame': 30, 'context_start_frame': 0, 'context_frames': 30,
           'scored_frames': 3, 'valid_scored_samples': 4098, 'input_samples16k': 30 * 640 + 1366}
    fresh.FreshTrainingData._geometry(row)
    for changed in ({'context_frames': 29}, {'valid_scored_samples': 4095}, {'input_samples16k': 1}, {'start_frame': 30.0}):
        with pytest.raises(ValueError): fresh.FreshTrainingData._geometry({**row, **changed})
