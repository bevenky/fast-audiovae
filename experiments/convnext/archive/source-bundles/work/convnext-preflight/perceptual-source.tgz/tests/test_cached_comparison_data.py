"""Cache reuse must preserve every parent target and perform no teacher work."""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json

import pytest
import torch

from audiovae_student.cached_comparison_data import load_parent_crops
from audiovae_student.preflight_distillation import PreflightConfig, _crop_identity, fixed_crops
from audiovae_student.teacher import CHECKPOINT_SHA256
from test_source_corpus import make_rows, open_corpus


def fixture(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(640 * 36 + 5, 640 * 40), dev=True)
    with open_corpus(tmp_path, rows, counts) as corpus:
        groups = {role: fixed_crops(corpus.get(row.source_id), role=role)
                  for row, role in zip(rows, ("diagnostic", "sentinel"))}
        identity = deepcopy(corpus.identity)
    cache = tmp_path / "cache"
    qualification = cache / "initial-prefill.json"
    qualification.write_text('{"fixture":true}\n')
    return {"identity": {"config": asdict(PreflightConfig()),
        "data": {"teacher_checkpoint_sha256": CHECKPOINT_SHA256, "source_corpus": identity,
                 "initial_teacher_prefill_sha256": hashlib.sha256(qualification.read_bytes()).hexdigest()},
        **{role: _crop_identity(crops) for role, crops in groups.items()}}}, cache, groups


def test_loads_parent_exactly_without_writes_or_teacher(tmp_path):
    parent, cache, expected = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    diagnostic, sentinel, evidence = load_parent_crops(parent, cache)
    for role, actual in (("diagnostic", diagnostic), ("sentinel", sentinel)):
        for wanted, got in zip(expected[role], actual):
            torch.testing.assert_close(wanted.latents, got.latents, rtol=0, atol=0)
            torch.testing.assert_close(wanted.teacher_audio, got.teacher_audio, rtol=0, atol=0)
    assert {str(p): p.read_bytes() for p in cache.rglob("*") if p.is_file()} == before
    assert evidence["read_only_parent_cache"]["teacher_inference_calls"] == 0


@pytest.mark.parametrize("field", ["latents_sha256", "teacher_sha256", "context_frames", "valid_scored_samples"])
def test_rejects_changed_parent_crop(tmp_path, field):
    parent, cache, _ = fixture(tmp_path)
    parent["identity"]["diagnostic"][0][field] = "0" * 64 if field.endswith("sha256") else 123
    with pytest.raises(ValueError, match="differs from the parent"):
        load_parent_crops(parent, cache)


def test_rejects_damaged_cache_and_active_writer(tmp_path):
    parent, cache, _ = fixture(tmp_path)
    with (cache / ".writer.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            load_parent_crops(parent, cache)
    target = next((cache / "targets").glob("*.pt"))
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_parent_crops(parent, cache)
