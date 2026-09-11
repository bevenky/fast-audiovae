"""Synthetic CPU fixtures qualify cache mechanics, not teacher asset quality."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.cache import sample_training_crop
from audiovae_student.data import ManifestRow, ManifestValidationError
from audiovae_student.source_corpus import SourceCorpus, read_native_16k
from audiovae_student.teacher import CHECKPOINT_SHA256, SOURCE_SHA256, FrozenAudioVAE2


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixtureTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.encoded_lengths = []
        self.decoded_frames = []

    def encode(self, audio, sample_rate):
        assert not torch.is_grad_enabled() and not self.training and not self.scale.requires_grad
        self.encoded_lengths.append(audio.shape[-1])
        value = F.pad(audio.cumsum(-1), (0, (-audio.shape[-1]) % 640))
        return value.reshape(1, 1, -1, 640).mean(-1).expand(1, 64, -1).contiguous()

    def decode(self, latents, sr_cond):
        assert not torch.is_grad_enabled() and not self.training
        self.decoded_frames.append(latents.shape[-1])
        return latents[:, :1].repeat_interleave(1920, dim=-1)


def make_teacher():
    # These identity fields exercise production pin enforcement. The module is
    # explicitly synthetic; from_files performs real asset verification in use.
    provenance = {"config": {"synthetic_fixture": True}, "source_sha256": SOURCE_SHA256,
                  "checkpoint_sha256": CHECKPOINT_SHA256, "posterior": "raw_mu", "sample_rate_in": 16000,
                  "sample_rate_out": 48000, "latent_channels": 64, "encoder_hop": 640,
                  "decoder_hop": 1920, "sr_cond": 48000, "dtype": "float32",
                  "target_preparation_policy": "fixture_whole_utterance"}
    return FrozenAudioVAE2(FixtureTeacher(), provenance)


class FixtureReader:
    identity = {"name": "synthetic_f32_fixture", "code_sha256": hashlib.sha256(b"fixture-reader-v1").hexdigest(),
                "config": {"format": "raw_float32", "sample_rate": 16000, "channels": 1}}

    def __init__(self):
        self.calls = []

    def __call__(self, payload, row):
        assert hashlib.sha256(payload).hexdigest() == row.audio_sha256
        self.calls.append(row.source_id)
        return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(1, 1, -1)


def make_rows(directory, lengths=(1280, 1280, 1280), dev=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {}
    for index, length in enumerate(lengths):
        identifier = f"row-{index}"
        path = directory / f"{identifier}.f32"
        samples = (torch.sin(torch.arange(length) * (0.01 + index * 0.001)) * 0.01).numpy().tobytes()
        path.write_bytes(samples)
        split = "dev" if dev and index == len(lengths) - 1 else "train"
        row = ManifestRow(
            dataset="fleurs", source_revision="synthetic-unit-test", source_id=identifier,
            source_url="https://example.test/synthetic", audio_path=path.name,
            audio_sha256=hashlib.sha256(samples).hexdigest(), parent_recording_id=f"parent:{identifier}",
            parent_start_seconds=0, speaker_id=f"speaker:{identifier}", session_id=f"session:{identifier}",
            language="hi", sample_rate_hz=16000, original_sample_rate_hz=16000, bandwidth_hz=7600,
            bandwidth_class="speech_band", bandwidth_evidence="synthetic fixture", native_recording=True,
            enhanced=False, duration_seconds=length / 16000, split=split,
            source_split="validation" if split == "dev" else "train", license="CC-BY-4.0",
            license_url="https://example.test/license", attribution="Synthetic fixture",
            access_record="synthetic fixture only", gain_policy="unchanged", resampler_policy="none",
            teacher_cache_key=None)
        rows.append(row)
        counts[identifier] = length
    return rows, counts


def open_corpus(directory, rows, counts, *, teacher=None, reader=None, **kwargs):
    teacher = make_teacher() if teacher is None else teacher
    reader = FixtureReader() if reader is None else reader
    config = {"max_disk_bytes": 1024**3, "min_free_bytes": 0, "max_memory_utterances": 1}
    config.update(kwargs)
    return SourceCorpus(rows, teacher, cache_dir=directory / "cache", input_sample_counts=counts,
                        source_root=directory / "audio", audio_reader=reader,
                        reader_identity=reader.identity, **config)


def test_lazy_whole_utterance_targets_memory_disk_and_resume(tmp_path, monkeypatch):
    rows, counts = make_rows(tmp_path / "audio", lengths=(640 * 35 + 37, 1280, 1600), dev=True)
    reader, teacher = FixtureReader(), make_teacher()
    python_rng, torch_rng, numpy_rng = random.getstate(), torch.get_rng_state(), np.random.get_state()
    with open_corpus(tmp_path, rows, counts, teacher=teacher, reader=reader) as corpus:
        assert not reader.calls and not teacher.model.encoded_lengths
        assert corpus.rows == tuple(rows) and corpus.rows_by_id["row-0"] == rows[0]
        assert corpus.input_sample_counts == counts
        record = corpus.get("row-0")
        assert teacher.model.encoded_lengths == [counts["row-0"]]
        assert teacher.model.decoded_frames == [36]
        assert record.valid_output_samples == counts["row-0"] * 3
        assert record.latents.shape == (1, 64, 36)
        crop = sample_training_crop(record, start_frame=30, scored_frames=3)
        assert crop.context_frames == 29
        assert corpus.get("row-0") is record
        corpus.get("row-1")  # Evicts the first CPU record, retaining its disk file.
        reloaded = corpus.get("row-0")
        assert torch.equal(record.latents, reloaded.latents)
        assert torch.equal(record.teacher_audio, reloaded.teacher_audio)
        assert len(teacher.model.encoded_lengths) == 2
        dev = corpus.get("row-2")
        assert dev.metadata["identity"]["source"]["split"] == "dev"
        metrics = corpus.metrics()
        assert metrics["memory_hits"] == metrics["disk_hits"] == 1
        assert metrics["cache_misses"] == metrics["prepared_utterances"] == 3
        assert metrics["source_bytes_read"] == sum(counts.values()) * 4
        assert metrics["teacher_seconds"] > 0 and metrics["cache_bytes_written"] > 0
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    actual_numpy = np.random.get_state()
    assert np.array_equal(actual_numpy[1], numpy_rng[1]) and actual_numpy[2:] == numpy_rng[2:]
    # Existing disk entries are inspected using metadata/stat only on startup.
    original_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Startup loaded target tensors"))
    resumed = open_corpus(tmp_path, rows, counts)
    monkeypatch.setattr(torch, "load", original_load)
    with resumed:
        restored = resumed.get("row-0")
        assert torch.equal(record.teacher_audio, restored.teacher_audio)
        assert resumed.metrics()["prepared_utterances"] == 3
        assert resumed.metrics()["disk_hits"] == 2


def test_disk_lru_keeps_recent_records_with_exact_byte_budget(tmp_path):
    rows, counts = make_rows(tmp_path / "audio")
    # Measure fixture file format size, not inference performance.
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    with SourceCorpus(rows, make_teacher(), cache_dir=probe_root / "cache", input_sample_counts=counts,
                       source_root=tmp_path / "audio", max_disk_bytes=1024**3, min_free_bytes=0,
                       audio_reader=FixtureReader(), reader_identity=FixtureReader.identity) as probe:
        probe.get("row-0")
        one_record_bytes = probe.metrics()["disk_bytes"]
    budget = one_record_bytes * 2
    with open_corpus(tmp_path, rows, counts, max_disk_bytes=budget, max_memory_utterances=0) as corpus:
        first = corpus.get("row-0")
        corpus.get("row-1")
        corpus.get("row-0")  # Touch the oldest record, making row-1 the victim.
        corpus.get("row-2")
        metrics = corpus.metrics()
        assert metrics["disk_bytes"] <= budget and metrics["disk_records"] == 2
        assert metrics["evictions"] == 1
        remaining = [json.loads(info)["row_id"] for (info,) in corpus._database.execute("SELECT info FROM entries")]
        assert set(remaining) == {"row-0", "row-2"}
        assert torch.equal(corpus.get("row-0").teacher_audio, first.teacher_audio)
        assert corpus.metrics()["prepared_utterances"] == 3


def test_free_space_reserve_and_oversized_target_never_delete_unrelated_files(tmp_path, monkeypatch):
    import audiovae_student.source_corpus as module

    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    with open_corpus(tmp_path, rows, counts, max_disk_bytes=1) as corpus:
        unrelated = tmp_path / "unrelated.bin"
        unrelated.write_bytes(b"keep this")
        with pytest.raises(OSError, match="exceeds the disk-cache"):
            corpus.get("row-0")
        assert unrelated.read_bytes() == b"keep this"
        assert not list((tmp_path / "cache/targets").glob("*.pt"))
    reserve_root = tmp_path / "reserve"
    reserve_root.mkdir()
    disk = module.shutil.disk_usage(tmp_path)
    with SourceCorpus(rows, make_teacher(), cache_dir=reserve_root, input_sample_counts=counts,
                       source_root=tmp_path / "audio", min_free_bytes=100,
                       audio_reader=FixtureReader(), reader_identity=FixtureReader.identity) as corpus:
        monkeypatch.setattr(module.shutil, "disk_usage", lambda path: type(disk)(total=disk.total, used=disk.used, free=100))
        with pytest.raises(OSError, match="free-space reserve"):
            corpus.get("row-0")
        assert corpus.metrics()["disk_bytes"] == 0


@pytest.mark.parametrize("failure", ["source_hash", "reader_count", "reader_dtype", "reader_finite"])
def test_unverified_or_incorrect_source_never_reaches_teacher(tmp_path, failure):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    teacher, reader = make_teacher(), FixtureReader()
    if failure == "source_hash":
        (tmp_path / "audio" / rows[0].audio_path).write_bytes(b"changed file")
    else:
        normal = reader

        def changed_reader(payload, row):
            audio = normal(payload, row)
            if failure == "reader_count":
                return audio[..., :-1]
            if failure == "reader_dtype":
                return audio.double()
            return audio.fill_(float("nan"))

        changed_reader.identity = reader.identity
        reader = changed_reader
    with open_corpus(tmp_path, rows, counts, teacher=teacher, reader=reader) as corpus:
        with pytest.raises(ValueError, match="SHA-256 mismatch|prepared-input contract"):
            corpus.get("row-0")
        assert not teacher.model.encoded_lengths


@pytest.mark.parametrize("change", ["manifest", "teacher", "reader", "config"])
def test_resume_rejects_identity_changes(tmp_path, change):
    rows, counts = make_rows(tmp_path / "audio")
    with open_corpus(tmp_path, rows, counts) as corpus:
        corpus.get("row-0")
    teacher, reader, kwargs = make_teacher(), FixtureReader(), {}
    if change == "manifest":
        rows[0] = replace(rows[0], source_revision="different-pinned-revision")
    elif change == "teacher":
        teacher._provenance["target_preparation_policy"] = "different"
    elif change == "reader":
        reader.identity = deepcopy(reader.identity)
        reader.identity["code_sha256"] = "d" * 64
    else:
        kwargs["max_memory_utterances"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        open_corpus(tmp_path, rows, counts, teacher=teacher, reader=reader, **kwargs)


def test_existing_cache_corruption_and_symlink_are_rejected(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    with open_corpus(tmp_path, rows, counts, max_memory_utterances=0) as corpus:
        corpus.get("row-0")
        path = next((tmp_path / "cache/targets").glob("*.pt"))
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="file SHA-256 mismatch"):
            corpus.get("row-0")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"do not touch")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        open_corpus(tmp_path, rows, counts, max_memory_utterances=0)
    assert outside.read_bytes() == b"do not touch"


def test_orphan_publication_recovery_verifies_cache_source_and_reader_provenance(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    with open_corpus(tmp_path, rows, counts) as corpus:
        expected = corpus.get("row-0")
    with sqlite3.connect(tmp_path / "cache/index.sqlite3") as database:
        database.execute("DELETE FROM entries")
    with open_corpus(tmp_path, rows, counts) as corpus:
        restored = corpus.get("row-0")
        assert restored.cache_key == expected.cache_key
        assert corpus.metrics()["prepared_utterances"] == 1
    # An orphan still has to match the expected source/preparation policy.
    path = next((tmp_path / "cache/targets").glob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["source_preparation"]["reader"]["code_sha256"] = "e" * 64
    torch.save(payload, path)
    with sqlite3.connect(tmp_path / "cache/index.sqlite3") as database:
        database.execute("DELETE FROM entries")
    with open_corpus(tmp_path, rows, counts) as corpus:
        with pytest.raises(ValueError, match="provenance mismatch"):
            corpus.get("row-0")


def test_reserved_audio_bad_teacher_and_concurrent_writer_are_rejected(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    with pytest.raises(ManifestValidationError, match="leakage"):
        open_corpus(tmp_path, rows, counts, reserved_rows=rows)
    wrong_teacher = make_teacher()
    wrong_teacher._provenance["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="pinned original"):
        open_corpus(tmp_path, rows, counts, teacher=wrong_teacher)
    with open_corpus(tmp_path, rows, counts) as corpus:
        with pytest.raises(RuntimeError, match="active writer"):
            open_corpus(tmp_path, rows, counts)
        with ThreadPoolExecutor(max_workers=1) as worker:
            record = worker.submit(corpus.get, "row-0").result()
            assert record.input_samples == 1280


def test_default_reader_rejects_non16k_or_stereo_without_conversion(tmp_path):
    sf = pytest.importorskip("soundfile")
    import io

    rows, _ = make_rows(tmp_path, lengths=(1280,))
    for rate, channels in ((48000, 1), (16000, 2)):
        stream = io.BytesIO()
        sf.write(stream, np.zeros((1280, channels), dtype=np.float32), rate, format="WAV", subtype="FLOAT")
        with pytest.raises(ValueError, match="no implicit conversion"):
            read_native_16k(stream.getvalue(), rows[0])


@pytest.mark.parametrize("original_rate,policy", [
    (16000, "prepared-native-16000-float-v1"), (48000, "prepared-soxr-vhq-to-16000-v1"),
    (22050, "prepared-soxr-vhq-to-16000-v1"),
])
def test_prepared_source_requires_opt_in_and_is_read_without_reconversion(tmp_path, original_rate, policy):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    rows[0] = replace(rows[0], original_sample_rate_hz=original_rate, resampler_policy=policy,
                      gain_policy="no-additional-gain-normalization")
    with pytest.raises(ValueError, match="allow_prepared_source"):
        open_corpus(tmp_path, rows, counts)
    reader = FixtureReader()
    with open_corpus(tmp_path, rows, counts, reader=reader, allow_prepared_source=True) as corpus:
        record = corpus.get("row-0")
        actual = np.frombuffer((tmp_path / "audio/row-0.f32").read_bytes(), dtype=np.float32)
        assert np.array_equal(record.reference16k.numpy().reshape(-1), actual)
        assert record.input_samples == 1280 and reader.calls == ["row-0"]
        assert record.metadata["identity"]["source"]["resampler_policy"] == policy
        assert record.metadata["identity"]["source"]["original_sample_rate_hz"] == original_rate
        assert corpus.identity["config"]["allow_prepared_source"] is True
        assert record.metadata["source_preparation"]["reader"]["prepared_source_policy"]["reader_resampling"] is False


@pytest.mark.parametrize("field,value", [
    ("resampler_policy", "unknown-resampler-v1"), ("gain_policy", "peak-normalized"),
    ("resampler_policy", "prepared-native-16000-float-v1"),
])
def test_prepared_source_opt_in_does_not_allow_unknown_gain_or_rate_policies(tmp_path, field, value):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    prepared = replace(rows[0], original_sample_rate_hz=48000, resampler_policy="prepared-soxr-vhq-to-16000-v1",
                        gain_policy="no-additional-gain-normalization")
    rows[0] = replace(prepared, **{field: value})
    with pytest.raises(ValueError, match="supported prepared policy|contradicts"):
        open_corpus(tmp_path, rows, counts, allow_prepared_source=True)


def test_prepared_native_none_policy_keeps_existing_acquisition_metadata(tmp_path):
    rows, counts = make_rows(tmp_path / "audio", lengths=(1280,))
    rows[0] = replace(rows[0], resampler_policy="none: original mono 16000 Hz",
                      gain_policy="none: preserve original amplitude")
    with open_corpus(tmp_path, rows, counts) as corpus:
        record = corpus.get("row-0")
        assert record.metadata["identity"]["source"]["gain_policy"] == rows[0].gain_policy
