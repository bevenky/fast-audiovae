"""CPU-only fixtures for exact startup stabilization and cache preservation."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.objective_comparison import state_fingerprint
from test_full_source_teacher_head import fixture
from teacher_reference_warmup import warmup_authenticated_teacher


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(80527)


def test_stable_teacher_reproduces_entire_cache_without_changes(tmp_path):
    teacher, crop, data, before, _ = fixture(tmp_path)
    receipt = warmup_authenticated_teacher(teacher, crop, data, expected_teacher_state_sha256=before)
    assert teacher.calls == [(1, 64, 5)] * 3
    assert all(r["equal_to_final"] for r in receipt["encoder_calls"] + receipt["decoder_calls"])
    assert receipt["cache_key_recreated"] == crop.cache_key
    assert receipt["entire_cached_latents_exact"] and receipt["entire_cached_post_target_exact"]
    assert receipt["teacher_state_preserved"] and not receipt["target_changed"]
    assert not receipt["tolerance_relaxed"] and receipt["optimizer_updates"] == 0
    assert state_fingerprint(teacher.model.state_dict()) == before


def test_cold_first_encoder_and_decoder_differences_are_recorded_then_exact(tmp_path):
    teacher, crop, data, before, _ = fixture(tmp_path)
    original_encode, original_decode = teacher.model.encode, teacher.decode
    count = {"encoder": 0, "decoder": 0}

    def cold_encode(audio, rate):
        count["encoder"] += 1
        output = original_encode(audio, rate)
        return output + 1e-6 if count["encoder"] == 1 else output

    def cold_decode(z):
        count["decoder"] += 1
        output = original_decode(z)
        return output + 2e-6 if count["decoder"] == 1 else output

    teacher.model.encode, teacher.decode = cold_encode, cold_decode
    receipt = warmup_authenticated_teacher(teacher, crop, data, expected_teacher_state_sha256=before)
    assert count == {"encoder": 3, "decoder": 3}
    for name in ("encoder_calls", "decoder_calls"):
        assert not receipt[name][0]["equal_to_final"]
        assert receipt[name][0]["max_abs_difference_to_final"] > 0
        assert receipt[name][1]["equal_to_final"] and receipt[name][2]["equal_to_final"]
    assert receipt["original_full_source_cache_key_exact"]
    assert state_fingerprint(teacher.model.state_dict()) == before


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_persistently_unstable_teacher_is_rejected(tmp_path, component):
    teacher, crop, data, before, _ = fixture(tmp_path)
    count = 0
    if component == "encoder":
        original = teacher.model.encode
        def unstable(audio, rate):
            nonlocal count
            count += 1
            return original(audio, rate) + count * 1e-5
        teacher.model.encode = unstable
    else:
        original = teacher.decode
        def unstable(z):
            nonlocal count
            count += 1
            return original(z) + count * 1e-5
        teacher.decode = unstable
    with pytest.raises(RuntimeError, match=component + " warmup last two"):
        warmup_authenticated_teacher(teacher, crop, data, expected_teacher_state_sha256=before)
    assert state_fingerprint(teacher.model.state_dict()) == before


@pytest.mark.parametrize("corruption", ["latents", "waveform", "key"])
def test_stabilization_never_overrides_corrupt_cached_targets(tmp_path, corruption):
    teacher, crop, data, before, _ = fixture(tmp_path)
    if corruption == "latents":
        crop = replace(crop, latents=crop.latents + .01)
        message = "cropped latents versus cache"
    elif corruption == "waveform":
        changed = crop.teacher_audio.clone(); changed[..., 0] += .01
        crop = replace(crop, teacher_audio=changed)
        message = "context/padding target versus cache"
    else:
        crop = replace(crop, cache_key="0" * 64)
        message = "cache key"
    with pytest.raises(RuntimeError, match=message):
        warmup_authenticated_teacher(teacher, crop, data, expected_teacher_state_sha256=before)
    assert state_fingerprint(teacher.model.state_dict()) == before


def test_source_authentication_fails_before_teacher_calls(tmp_path):
    teacher, crop, data, before, path = fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash"):
        warmup_authenticated_teacher(teacher, crop, data, expected_teacher_state_sha256=before)
    assert teacher.calls == []
