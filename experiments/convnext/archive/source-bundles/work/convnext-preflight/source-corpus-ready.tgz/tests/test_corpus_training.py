"""Synthetic cache fixtures qualify real corpus mechanics, not speech quality."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.cache import prepare_utterance_cache, sample_training_crop, save_cache
from audiovae_student.corpus_training import (
    CachedCorpus, CorpusTrainingConfig, _validate_crop_lengths, draw_training_crop,
    evaluate_reconstruction, scored_crop_loss, train_cached_corpus,
)
from audiovae_student.data import ManifestRow
from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.optimizers import native_muon_available
from audiovae_student.teacher import FrozenAudioVAE2


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class FixtureTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode(self, audio, sample_rate):
        value = F.pad(audio, (0, (-audio.shape[-1]) % 640))
        return value.reshape(1, 1, -1, 640).mean(-1).expand(1, 64, -1).contiguous()

    def decode(self, latents, sr_cond):
        return latents[:, :1].repeat_interleave(1920, dim=-1)


def make_cache(directory, identifier, split, *, length=3200, language="hi", reference=True, teacher_version=1):
    source = ManifestRow(
        dataset="fleurs", source_revision="synthetic-unit-test", source_id=identifier,
        source_url="https://example.test/synthetic", audio_path=f"{identifier}.wav",
        audio_sha256=hashlib.sha256(identifier.encode()).hexdigest(), parent_recording_id=f"parent:{identifier}",
        parent_start_seconds=0, speaker_id=f"speaker:{identifier}", session_id=f"session:{identifier}",
        language=language, sample_rate_hz=16000, original_sample_rate_hz=16000, bandwidth_hz=7600,
        bandwidth_class="speech_band", bandwidth_evidence="synthetic fixture band contract",
        native_recording=True, enhanced=False, duration_seconds=length / 16000,
        split=split, source_split="train" if split == "train" else "validation",
        license="CC-BY-4.0", license_url="https://example.test/license", attribution="Synthetic fixture",
        access_record="synthetic fixture; no downloaded audio", gain_policy="unchanged",
        resampler_policy="none", teacher_cache_key=None)
    provenance = {"config": {"synthetic_teacher_version": teacher_version},
                  "source_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
                  "posterior": "raw_mu", "sample_rate_in": 16000, "sample_rate_out": 48000,
                  "latent_channels": 64, "encoder_hop": 640, "decoder_hop": 1920,
                  "sr_cond": 48000, "dtype": "float32"}
    frequency = 0.01 + int(hashlib.sha256(identifier.encode()).hexdigest()[:4], 16) / 1e7
    audio = (0.12 * torch.sin(torch.arange(length) * frequency))[None, None]
    teacher = FrozenAudioVAE2(FixtureTeacher(), provenance)
    record = prepare_utterance_cache(audio, source, teacher, include_reference=reference)
    path = directory / f"{identifier}.pt"
    save_cache(record, path)
    return {"path": path.name, "cache_key": record.cache_key}, record


def make_index(directory, *, include_references=True):
    directory.mkdir(parents=True, exist_ok=True)
    train = [make_cache(directory, f"train-{i}", "train", language=("hi" if i == 0 else "en"),
                        reference=include_references)[0] for i in range(2)]
    dev = [make_cache(directory, "dev-0", "dev", reference=include_references)[0]]
    path = directory / "index.json"
    path.write_text(json.dumps({"format_version": 1, "train": train, "dev": dev}))
    return path


def small_model():
    return StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8))


def small_losses():
    return WarmupLossConfig(teacher_fft_sizes=(32, 64), reference_fft_sizes_16k=(16,),
                            teacher_spectral_weight=1, teacher_waveform_weight=1,
                            reference_spectral_weight=1)


def small_config(**changes):
    values = dict(total_steps=4, accumulation_steps=2, scored_frames=2, learning_rate=1e-3,
                  warmup_steps=3, optimizer="adamw", validation_interval=3, checkpoint_interval=2)
    values.update(changes)
    return CorpusTrainingConfig(**values)


def test_scored_loss_excludes_context_and_partial_padded_tail(tmp_path):
    _, record = make_cache(tmp_path, "tail", "train", length=31 * 640 + 37)
    crop = sample_training_crop(record, 30, scored_frames=3)
    model = small_model()
    criterion = WarmupReconstructionLoss(small_losses())
    expected = scored_crop_loss(model, crop, criterion, "cpu")
    changed_teacher = crop.teacher_audio.clone()
    changed_teacher[..., :crop.scored_slice.start] = 500
    changed_teacher[..., crop.scored_slice.stop:] = -500
    changed_reference = crop.reference16k.clone()
    changed_reference[..., :crop.reference_scored_slice.start] = -200
    changed_reference[..., crop.reference_scored_slice.stop:] = 200
    changed_latents = crop.latents.clone()
    real_frames = record.latent_frames - crop.context_start_frame
    changed_latents[..., real_frames:] = 300
    changed = replace(crop, teacher_audio=changed_teacher, reference16k=changed_reference,
                      latents=changed_latents)
    actual = scored_crop_loss(model, changed, criterion, "cpu")
    assert crop.context_frames == 29 and crop.valid_scored_samples == 2031
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    expected["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampler_never_selects_tails_shorter_than_loss_windows(tmp_path):
    train, _ = make_cache(tmp_path, "short-train", "train", length=680)
    dev, _ = make_cache(tmp_path, "short-dev", "dev", length=680)
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"format_version": 1, "train": [train], "dev": [dev]}))
    corpus = CachedCorpus(path)
    config, losses, sampler = small_config(), small_losses(), random.Random(5)
    _validate_crop_lengths(corpus, config, losses)
    crops = [draw_training_crop(corpus, config, losses, sampler) for _ in range(50)]
    assert {crop.start_frame for crop in crops} == {0, 1}
    assert all(crop.valid_scored_samples >= 64 for crop in crops)
    with pytest.raises(ValueError, match="insufficient valid scored audio"):
        _validate_crop_lengths(corpus, config, WarmupLossConfig())


def test_sampler_balances_languages_before_utterance_count(tmp_path):
    train = [make_cache(tmp_path, f"en-{i}", "train", language="en")[0] for i in range(3)]
    train.append(make_cache(tmp_path, "hi-0", "train", language="hi")[0])
    dev = [make_cache(tmp_path, "dev", "dev")[0]]
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"format_version": 1, "train": train, "dev": dev}))
    corpus, sampler = CachedCorpus(path), random.Random(42)
    counts = {"en": 0, "hi": 0}
    for _ in range(400):
        crop = draw_training_crop(corpus, small_config(), small_losses(), sampler)
        counts[corpus.by_key[crop.cache_key].language] += 1
    assert 160 <= counts["hi"] <= 240


def test_dev_evaluation_never_updates_model_gradients_or_rng(tmp_path):
    corpus = CachedCorpus(make_index(tmp_path, include_references=False))
    model = small_model().train()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.2)
    before = deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    values = evaluate_reconstruction(model, corpus, small_config(), small_losses(), "cpu")
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert values["examples"] == 1 and values["reference_examples"] == 0
    assert "reference_spectral" not in values
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.equal(p.grad, gradient) for p, gradient in zip(model.parameters(), gradients))


@pytest.mark.parametrize("failure", ["split", "teacher", "empty_dev"])
def test_index_rejects_invalid_partition_or_teacher(tmp_path, failure):
    path = make_index(tmp_path)
    payload = json.loads(path.read_text())
    if failure == "split":
        payload["dev"] = [payload["train"][0]]
    elif failure == "teacher":
        payload["dev"] = [make_cache(tmp_path, "dev-new", "dev", teacher_version=2)[0]]
    else:
        payload["dev"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        CachedCorpus(path)


@pytest.mark.parametrize("optimizer", ["adamw", pytest.param("muon_adamw",
    marks=pytest.mark.skipif(not native_muon_available(), reason="Native torch.optim.Muon is unavailable"))])
def test_corpus_checkpoint_resume_matches_continuous_updates(tmp_path, optimizer):
    index = make_index(tmp_path / "data")
    torch.manual_seed(19)
    initial = small_model()
    continuous, partial = deepcopy(initial), deepcopy(initial)
    config = small_config(optimizer=optimizer)
    expected = train_cached_corpus(continuous, index, output_dir=tmp_path / "continuous",
                                   config=config, loss_config=small_losses())
    first = train_cached_corpus(partial, index, output_dir=tmp_path / "resumed", config=config,
                                loss_config=small_losses(), stop_after_updates=2)
    assert first["state"] == "paused" and first["step"] == first["validation_step"] == 2
    resumed = small_model()
    actual = train_cached_corpus(resumed, index, output_dir=tmp_path / "resumed", config=config,
                                 loss_config=small_losses(), resume_from=tmp_path / "resumed/latest.pt")
    assert actual["state"] == expected["state"] == "completed"
    assert actual["step"] == actual["checkpoint_step"] == actual["validation_step"] == 4
    assert actual["scored_samples_seen"] == expected["scored_samples_seen"]
    assert actual["language_exposure"] == expected["language_exposure"]
    assert actual["last_training_sources"] == expected["last_training_sources"]
    assert all(source["source_id"].startswith("train-") for source in actual["last_training_sources"])
    assert actual["train_examples_seen"] == 8
    assert actual["train"]["learning_rate"] == config.learning_rate
    for name, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
    for name in ("total", "teacher_spectral", "teacher_waveform", "reference_spectral", "gradient_norm"):
        assert actual["train"][name] == expected["train"][name]
    assert actual["validation"] == expected["validation"]
    saved_expected = torch.load(tmp_path / "continuous/latest.pt", weights_only=True)
    saved_actual = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert saved_actual["sampler_rng"] == saved_expected["sampler_rng"]
    assert torch.equal(saved_actual["rng"]["torch_cpu"], saved_expected["rng"]["torch_cpu"])
    assert set(saved_actual["optimizer"]["optimizers"]) == ({"adamw"} if optimizer == "adamw" else {"adamw", "muon"})


def test_resume_rejects_changed_data_identity(tmp_path):
    index = make_index(tmp_path / "data")
    config = small_config()
    train_cached_corpus(small_model(), index, output_dir=tmp_path / "run", config=config,
                         loss_config=small_losses(), stop_after_updates=1)
    payload = json.loads(index.read_text())
    payload["train"].reverse()
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        train_cached_corpus(small_model(), index, output_dir=tmp_path / "run", config=config,
                             loss_config=small_losses(), resume_from=tmp_path / "run/latest.pt")


def test_real_training_and_validation_events_and_status(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    index = make_index(tmp_path / "data", include_references=False)
    config = small_config(total_steps=2, accumulation_steps=1, validation_interval=1)
    result = train_cached_corpus(small_model(), index, output_dir=tmp_path / "run", config=config,
                                 loss_config=small_losses(), log_dir=tmp_path / "logs", run_name="warmup-test")
    events = EventAccumulator(str(tmp_path / "logs/warmup-test")).Reload()
    assert [p.step for p in events.Scalars("train/total")] == [1, 2]
    assert [p.step for p in events.Scalars("validation/total")] == [0, 1, 2]
    assert "train/reference_spectral" not in events.Tags()["scalars"]
    assert "validation/reference_spectral" not in events.Tags()["scalars"]
    assert events.Scalars("train/total")[-1].value == pytest.approx(result["train"]["total"])
    assert events.Scalars("validation/total")[-1].value == pytest.approx(result["validation"]["total"])
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["state"] == "completed" and status["stage"] == "reconstruction_warmup"
    assert status["train"]["step_seconds"] > 0
    assert sum(v["examples"] for v in status["language_exposure"].values()) == 2
    assert status["scored_audio_hours_seen"] == status["scored_samples_seen"] / 48000 / 3600
