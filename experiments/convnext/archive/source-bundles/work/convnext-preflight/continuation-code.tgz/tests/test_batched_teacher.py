"""Synthetic CPU parity and ownership tests, not qualification of real weights."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.batched_teacher import (
    TeacherBatchConfig, _batch_outputs, length_buckets, qualify_batch, record_from_outputs,
)
from audiovae_student.cache import prepare_utterance_cache
from audiovae_student.teacher import FrozenAudioVAE2
from test_source_corpus import FixtureReader, make_rows, make_teacher, open_corpus


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class CausalTeacher(nn.Module):
    def __init__(self, *, coupled=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []
        self.coupled = coupled

    def encode(self, audio, sample_rate):
        assert sample_rate == 16000 and not self.training and not torch.is_grad_enabled()
        assert not self.scale.requires_grad
        self.calls.append((audio.shape[0], audio.shape[-1]))
        audio = F.pad(audio, (0, (-audio.shape[-1]) % 640))
        # Stateful-in-time but causal, with padding before encoding just like
        # the real wrapper contract. No crop resets or cross-example coupling.
        value = audio.cumsum(-1).reshape(audio.shape[0], 1, -1, 640).mean(-1)
        if self.coupled:
            value = value + audio.mean()
        return value.expand(-1, 64, -1).contiguous()

    def decode(self, latents, sr_cond):
        assert (sr_cond == 48000).all() and sr_cond.shape == (latents.shape[0],)
        return latents[:, :1].cumsum(-1).repeat_interleave(1920, -1)


def teacher(coupled=False):
    provenance = make_teacher().provenance
    provenance["config"] = {"synthetic_causal_batch_fixture": True}
    return FrozenAudioVAE2(CausalTeacher(coupled=coupled), provenance)


def read_audio(path):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).reshape(1, 1, -1)


def test_length_buckets_cap_padded_total_and_keep_oversized_utterance_whole():
    config = TeacherBatchConfig(max_batch_size=3, max_total_input_samples=640 * 8)
    lengths = [641, 1, 1280, 640 * 9, 640 * 2 + 1, 700]
    buckets = length_buckets(lengths, config)
    assert sorted(i for batch in buckets for i in batch) == list(range(len(lengths)))
    assert buckets == [(1, 0, 5), (2, 4), (3,)]
    for batch in buckets:
        assert len(batch) <= 3
        longest = max((lengths[i] + 639) // 640 * 640 for i in batch)
        assert len(batch) == 1 or len(batch) * longest <= config.max_total_input_samples


def test_batch_trim_and_complete_cache_metadata_match_original_serial(tmp_path):
    rows, _ = make_rows(tmp_path, lengths=(640 * 3 + 1, 640 * 5, 640 * 4 + 19), dev=True)
    audios = [read_audio(tmp_path / row.audio_path) for row in rows]
    original = teacher()
    values = _batch_outputs(original, audios, TeacherBatchConfig())
    for row, audio, (latent, waveform) in zip(rows, audios, values):
        actual = record_from_outputs(audio, row, original, latent, waveform)
        expected = prepare_utterance_cache(audio, row, original)
        assert actual.cache_key == expected.cache_key
        assert actual.metadata == expected.metadata
        assert actual.valid_output_samples == 3 * audio.shape[-1]
        assert actual.teacher_audio.shape[-1] == 1920 * ((audio.shape[-1] + 639) // 640)
        torch.testing.assert_close(actual.latents, expected.latents, atol=0, rtol=0)
        torch.testing.assert_close(actual.teacher_audio, expected.teacher_audio, atol=0, rtol=0)
        torch.testing.assert_close(actual.reference16k, audio, atol=0, rtol=0)
    assert all(not p.requires_grad for p in original.parameters())


def test_real_gate_contract_rejects_batch_coupling_and_checks_all_valid_samples():
    audios = [torch.linspace(-0.1, 0.2, 1234).reshape(1, 1, -1),
              torch.linspace(0.3, 0.1, 2561).reshape(1, 1, -1)]
    success = qualify_batch(teacher(), audios, TeacherBatchConfig(), source_ids=["a", "b"])
    assert success["passed"]
    assert all(clip["latents"]["exact"] and clip["valid_waveform"]["exact"] for clip in success["clips"])
    failure = qualify_batch(teacher(coupled=True), audios, TeacherBatchConfig(), source_ids=["a", "b"])
    assert not failure["passed"] and failure["reason"] == "numerical_parity_threshold_exceeded"
    assert any(not clip["valid_waveform"]["passed"] for clip in failure["clips"])
    json.dumps(success, allow_nan=False)
    json.dumps(failure, allow_nan=False)


def test_prefetch_caches_misses_preserves_requested_order_and_reuses_verified_hits(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1281, 1920, 2569, 3211), dev=True)
    original = teacher()
    reader = FixtureReader()
    with open_corpus(tmp_path, rows, counts, teacher=original, reader=reader) as corpus:
        first = corpus.get("row-1")
        report = corpus.prefetch(["row-3", "row-1", "row-0", "row-2", "row-3"],
                                  max_batch_size=3, max_total_input_samples=640 * 18)
        assert report["requested_row_ids"] == ["row-3", "row-1", "row-0", "row-2"]
        assert report["cache_hits"] == 1 and report["cache_misses"] == 3
        assert report["qualification"]["passed"] and report["batched_utterances"] == 3
        assert report["serial_utterances"] == 0
        assert corpus.metrics()["prepared_utterances"] == 4
        calls_before_hits = list(original.model.calls)
        again = corpus.prefetch(["row-0", "row-1", "row-2", "row-3"])
        assert again["cache_hits"] == 4 and again["cache_misses"] == 0
        assert original.model.calls == calls_before_hits
        torch.testing.assert_close(corpus.get("row-1").teacher_audio, first.teacher_audio, atol=0, rtol=0)
        for row in rows:
            record = corpus.get(row.source_id)
            audio = read_audio(tmp_path / "audio" / row.audio_path)
            expected = prepare_utterance_cache(audio, row, original)
            assert record.cache_key == expected.cache_key
            assert record.metadata["identity"] == expected.metadata["identity"]
            assert record.metadata["tensor_sha256"] == expected.metadata["tensor_sha256"]
            assert record.metadata["source_preparation"]["source_file_sha256"] == row.audio_sha256
        json.dumps(report, allow_nan=False)
    # Opening the same code/reader/config cache performs no teacher execution.
    resumed_teacher = teacher()
    with open_corpus(tmp_path, rows, counts, teacher=resumed_teacher) as corpus:
        report = corpus.prefetch([r.source_id for r in rows])
        assert report["cache_hits"] == 4 and not resumed_teacher.model.calls


def test_failed_qualification_uses_only_original_serial_targets(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1301, 3221))
    original = teacher(coupled=True)
    with open_corpus(tmp_path, rows, counts, teacher=original) as corpus:
        report = corpus.prefetch([r.source_id for r in rows])
        assert not report["qualification"]["passed"]
        assert report["batched_utterances"] == 0 and report["serial_utterances"] == 2
        for row in rows:
            expected = prepare_utterance_cache(read_audio(tmp_path / "audio" / row.audio_path), row, original)
            actual = corpus.get(row.source_id)
            torch.testing.assert_close(actual.teacher_audio, expected.teacher_audio, atol=0, rtol=0)


def test_prefetch_source_hash_and_request_bounds_fail_before_caching(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1301, 3221))
    original = teacher()
    with open_corpus(tmp_path, rows, counts, teacher=original) as corpus:
        with pytest.raises(KeyError):
            corpus.prefetch(["row-0", "unknown"])
        with pytest.raises(ValueError, match="256"):
            corpus.prefetch([str(i) for i in range(257)])
        assert not original.model.calls
        (tmp_path / "audio" / rows[0].audio_path).write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA-256"):
            corpus.prefetch([r.source_id for r in rows])
        assert corpus.metrics()["prepared_utterances"] == 0


def test_equal_length_only_set_and_singletons_fall_back_explicitly(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280, 1280))
    original = teacher()
    with open_corpus(tmp_path, rows, counts, teacher=original) as corpus:
        report = corpus.prefetch([r.source_id for r in rows], max_total_input_samples=1280)
        assert not report["qualification"]["passed"]
        assert report["serial_utterances"] == 2
        assert original.model.calls == [(1, 1280), (1, 1280)]


def test_cache_output_and_configuration_contracts_fail_closed(tmp_path):
    rows, _ = make_rows(tmp_path, lengths=(1301,))
    audio, original = read_audio(tmp_path / rows[0].audio_path), teacher()
    values = _batch_outputs(original, [audio], TeacherBatchConfig())[0]
    with pytest.raises(ValueError, match="teacher_cache_key"):
        record_from_outputs(audio, replace(rows[0], teacher_cache_key="f" * 64), original, *values)
    for settings in ({"max_batch_size": 9}, {"max_total_input_samples": 0}, {"minimum_snr_db": float("nan")}):
        with pytest.raises(ValueError):
            TeacherBatchConfig(**settings)
    with pytest.raises(ValueError, match="sample budget"):
        _batch_outputs(original, [audio, audio], TeacherBatchConfig(max_total_input_samples=640))


@pytest.mark.parametrize("original_rate,policy", [
    (16000, "prepared-native-16000-float-v1"), (48000, "prepared-soxr-vhq-to-16000-v1")])
def test_prepared_source_prefetch_preserves_original_reader_and_cache_provenance(tmp_path, original_rate, policy):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1301, 2609))
    rows = [replace(row, original_sample_rate_hz=original_rate, resampler_policy=policy,
                    gain_policy="no-additional-gain-normalization") for row in rows]
    original = teacher()
    with open_corpus(tmp_path, rows, counts, teacher=original, allow_prepared_source=True) as corpus:
        report = corpus.prefetch([r.source_id for r in reversed(rows)])
        assert report["qualification"]["passed"] and report["batched_utterances"] == 2
        for row in rows:
            audio = read_audio(tmp_path / "audio" / row.audio_path)
            expected = prepare_utterance_cache(audio, row, original)
            actual = corpus.get(row.source_id)
            assert actual.metadata["identity"] == expected.metadata["identity"]
            assert actual.metadata["tensor_sha256"] == expected.metadata["tensor_sha256"]
            assert actual.metadata["source_preparation"]["source_file_sha256"] == row.audio_sha256
            assert actual.metadata["source_preparation"]["reader"]["prepared_source_policy"]["reader_resampling"] is False
            torch.testing.assert_close(actual.reference16k, audio, atol=0, rtol=0)


def test_failed_post_qualification_batch_publishes_only_serial_targets(tmp_path, monkeypatch):
    import audiovae_student.batched_teacher as module
    rows, counts = make_rows(tmp_path / "audio", lengths=(1301, 2609))
    original = teacher()
    real_batch = module._batch_outputs
    calls = 0
    def fourth_call_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic batch allocation failure")
        return real_batch(*args, **kwargs)
    monkeypatch.setattr(module, "_batch_outputs", fourth_call_fails)
    with open_corpus(tmp_path, rows, counts, teacher=original) as corpus:
        report = corpus.prefetch([r.source_id for r in rows])
        assert report["qualification"]["passed"]
        assert report["runtime_fallback"]["reason"] == "qualified_batch_execution_failed"
        assert report["batched_utterances"] == 0 and report["serial_utterances"] == 2
        assert report["batches"][0]["mode"] == "serial_after_batch_failure"
        for row in rows:
            expected = prepare_utterance_cache(read_audio(tmp_path / "audio" / row.audio_path), row, original)
            actual = corpus.get(row.source_id)
            assert actual.metadata["tensor_sha256"] == expected.metadata["tensor_sha256"]
