"""Addon heldout construction must retain valid length and metadata identity."""
from types import SimpleNamespace
import math
import json

import pytest
import torch
from torch import nn

from audiovae_student.batching import _validate_crop
from audiovae_student.fusion_evaluation import evaluate_fusion
from audiovae_student.losses_distillation import DistillationLossConfig, DistillationReconstructionLoss
from audiovae_student.quiet_audio import QuietAudioConfig
from run_corrected_screen import build_addon_crop


@pytest.fixture(autouse=True)
def cpu_only():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(samples):
    generator = torch.Generator().manual_seed(930)
    frames = math.ceil(samples / 640)
    audio = torch.randn(1, 1, samples, generator=generator) * .01
    z = torch.randn(1, 64, frames, generator=generator)
    target = torch.randn(1, 1, frames * 1920, generator=generator) * .02
    row = SimpleNamespace(source_id="heldout_laughter_sample", audio_sha256="a" * 64,
                          sample_rate_hz=16000, duration_seconds=samples / 16000)
    return z, target, audio, row


@pytest.mark.parametrize("samples", [114386, 12800, 5003])
def test_addon_reference_padding_preserves_real_samples_and_source_identity(samples):
    z, target, audio, row = fixture(samples)
    crop = build_addon_crop(z, target, audio, row)
    _validate_crop(crop)
    frames = math.ceil(samples / 640)
    assert crop.source_id == row.source_id
    assert crop.cache_key == row.audio_sha256
    assert crop.valid_scored_samples == samples * 3
    assert crop.scored_frames == frames
    assert crop.start_frame == crop.context_start_frame == crop.context_frames == 0
    assert crop.latents.shape == (1, 64, frames)
    assert crop.teacher_audio.shape == (1, 1, frames * 1920)
    assert crop.reference16k.shape == (1, 1, frames * 640)
    assert torch.equal(crop.latents, z)
    assert torch.equal(crop.teacher_audio, target)
    assert torch.equal(crop.reference16k[..., :samples], audio)
    assert torch.count_nonzero(crop.reference16k[..., samples:]) == 0
    assert crop.loss_mask().sum().item() == samples * 3
    assert not crop.loss_mask()[..., samples * 3:].any()
    assert not crop.latents.requires_grad and not crop.teacher_audio.requires_grad


class StoredDecoder(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", output.clone())

    def forward(self, z):
        return self.output


def test_complete_addon_evaluation_does_not_score_padding_or_lose_event_metadata():
    z, target, audio, row = fixture(114386)
    target[..., audio.shape[-1] * 3:] = 200
    crop = build_addon_crop(z, target, audio, row)
    output = crop.teacher_audio.clone()
    output[..., crop.valid_scored_samples:] = -200
    engine = SimpleNamespace(model=StoredDecoder(output),
        criterion=DistillationReconstructionLoss(DistillationLossConfig()),
        device=torch.device("cpu"), step=8090,
        config=SimpleNamespace(quiet_audio=QuietAudioConfig()))
    result = evaluate_fusion(engine, [crop], {row.source_id: {
        "group": "expressive", "condition": "laughter", "dataset": "addon", "language": "und"}})
    measured = result["rows"][0]
    assert measured["source_id"] == row.source_id
    assert measured["samples"] == 343158
    assert measured["quiet_windows"]["valid_samples"] == 343158
    assert measured["student_peak_abs"] < 1 and measured["teacher_peak_abs"] < 1
    assert measured["student_overshoot_samples"] == measured["teacher_overshoot_samples"] == 0
    assert measured["waveform_raw_mae"] == 0
    assert measured["teacher_mel"] == 0
    assert measured["high_frequency"]["complex_residual_rms"] == 0
    assert result["groups"]["condition/laughter"]["crops"] == 1
    assert result["groups"]["group/expressive"]["crops"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["short_teacher", "too_many_frames", "bad_latent_channels", "stereo_input", "nonfinite_target"])
def test_malformed_addon_rejected_before_target_preparation(kind):
    z, target, audio, row = fixture(5003)
    if kind == "short_teacher":
        target = target[..., :-1]
    elif kind == "too_many_frames":
        z = torch.cat((z, z[..., :1]), dim=-1)
        target = torch.cat((target, target[..., :1920]), dim=-1)
    elif kind == "bad_latent_channels":
        z = z[:, :63]
    elif kind == "stereo_input":
        audio = audio.expand(1, 2, -1)
    else:
        target[..., 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        build_addon_crop(z, target, audio, row)
