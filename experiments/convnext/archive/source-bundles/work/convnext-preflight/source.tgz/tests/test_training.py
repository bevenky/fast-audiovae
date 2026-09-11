"""Behavioral checks for reconstruction, overfit and exact CPU resume."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss
from audiovae_student.training import TrainingBatch, TrainingConfig, train_fixed_batch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_loss(**overrides):
    return WarmupLossConfig(teacher_fft_sizes=(16, 32),
                            reference_fft_sizes_16k=(16,), **overrides)


class TinyDecoder(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.config = {"channels": 2, "upsample": 3, "stochastic": stochastic}
        self.projection = nn.Conv1d(2, 1, 1)
        self.dropout = nn.Dropout(0.15) if stochastic else nn.Identity()
        self.stochastic = stochastic

    def forward(self, latents):
        prediction = self.projection(self.dropout(latents)).repeat_interleave(3, dim=-1)
        if self.stochastic:
            # Exercises Python and NumPy RNG restoration as well as Dropout's
            # torch RNG. This noise belongs only to the test fixture.
            prediction = prediction + (random.random() + np.random.random()) * 0.001
        return prediction


def fixed_batch():
    generator = torch.Generator().manual_seed(25)
    latents = torch.randn(2, 2, 32, generator=generator)
    teacher = (0.3 * latents[:, :1] - 0.15 * latents[:, 1:2]).repeat_interleave(3, dim=-1)
    return TrainingBatch(latents, teacher)


def test_losses_keep_student_gradients_and_freeze_targets():
    torch.manual_seed(3)
    student = torch.randn(2, 1, 96, requires_grad=True)
    teacher = torch.randn(2, 1, 96, requires_grad=True)
    reference = torch.randn(2, 1, 32, requires_grad=True)
    components = WarmupReconstructionLoss(small_loss())(student, teacher, reference)
    components["total"].backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert reference.grad is None
    assert all(torch.isfinite(value) for value in components.values())


def test_identity_teacher_has_zero_loss_even_for_silence():
    criterion = WarmupReconstructionLoss(small_loss())
    for waveform in (torch.zeros(1, 1, 96), torch.randn(1, 1, 96)):
        assert criterion(waveform, waveform)["total"].item() == 0


def test_16khz_reference_does_not_penalize_12khz_student_energy():
    # Integer-bin tones plus a Hann window confine the 12 kHz tone above the
    # reference band. The test would fail for fullband upsampled-reference loss.
    time_16 = torch.arange(256, dtype=torch.float64) / 16000
    time_48 = torch.arange(768, dtype=torch.float64) / 48000
    reference = (0.2 * torch.sin(2 * torch.pi * 1500 * time_16)).float()[None, None]
    low = (0.2 * torch.sin(2 * torch.pi * 1500 * time_48)).float()[None, None]
    high = (0.2 * torch.sin(2 * torch.pi * 12000 * time_48)).float()[None, None]
    criterion = WarmupReconstructionLoss(WarmupLossConfig(
        teacher_fft_sizes=(192,), reference_fft_sizes_16k=(64,)))
    low_loss = criterion(low, low, reference)["reference_spectral"]
    high_loss = criterion(low + high, low + high, reference)["reference_spectral"]
    assert abs(high_loss.item() - low_loss.item()) < 2e-4
    assert criterion(low + high, low)["teacher_spectral"] > 0.1


@pytest.mark.parametrize("failure", ["teacher_length", "reference_length", "nonfinite"])
def test_loss_rejects_misalignment_and_nonfinite_audio(failure):
    student = torch.ones(1, 1, 96)
    teacher = student.clone()
    reference = None
    if failure == "teacher_length":
        teacher = teacher[..., :-1]
    elif failure == "reference_length":
        reference = torch.ones(1, 1, 31)
    else:
        teacher[..., 0] = float("nan")
    with pytest.raises(ValueError):
        WarmupReconstructionLoss(small_loss())(student, teacher, reference)


def test_tiny_set_reconstruction_loss_falls():
    torch.manual_seed(7)
    model = TinyDecoder()
    result = train_fixed_batch(model, fixed_batch(), steps=100,
                               config=TrainingConfig(learning_rate=0.03, weight_decay=0),
                               loss_config=small_loss(teacher_spectral_weight=1,
                                                      teacher_waveform_weight=1))
    early = sum(record["total"] for record in result.metrics[:10]) / 10
    late = sum(record["total"] for record in result.metrics[-10:]) / 10
    assert late < early * 0.3
    assert all(np.isfinite(record["gradient_norm"]) for record in result.metrics)


def test_checkpoint_resumes_identical_updates_and_rng(tmp_path):
    torch.manual_seed(9)
    initial = TinyDecoder(stochastic=True)
    continuous = deepcopy(initial)
    interrupted = deepcopy(initial)
    config = TrainingConfig(learning_rate=0.004, seed=123)
    loss_config = small_loss(teacher_spectral_weight=1)
    expected = train_fixed_batch(continuous, fixed_batch(), steps=8, config=config,
                                 loss_config=loss_config)
    checkpoint = tmp_path / "nested" / "checkpoint.pt"
    first = train_fixed_batch(interrupted, fixed_batch(), steps=3, config=config,
                              loss_config=loss_config, checkpoint_path=checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["step"] == 3
    assert payload["model_spec"]["config"] == initial.config
    assert "optimizer" in payload and "rng" in payload
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = TinyDecoder(stochastic=True)
    rest = train_fixed_batch(resumed, fixed_batch(), steps=5, config=config,
                             loss_config=loss_config, resume_from=checkpoint,
                             checkpoint_path=checkpoint)
    assert rest.step == expected.step == 8
    assert first.metrics + rest.metrics == expected.metrics
    for name, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
    assert not list(checkpoint.parent.glob("*.tmp"))


@pytest.mark.parametrize("mismatch", ["batch", "optimizer", "architecture", "loss"])
def test_checkpoint_rejects_changed_run_contract(tmp_path, mismatch):
    checkpoint = tmp_path / "checkpoint.pt"
    config = TrainingConfig()
    batch = fixed_batch()
    model = TinyDecoder()
    loss_config = small_loss()
    train_fixed_batch(model, batch, steps=1, config=config, loss_config=loss_config,
                       checkpoint_path=checkpoint)
    if mismatch == "batch":
        batch = TrainingBatch(batch.latents + 0.01, batch.teacher_audio)
    elif mismatch == "optimizer":
        config = TrainingConfig(learning_rate=0.01)
    elif mismatch == "architecture":
        model = TinyDecoder(stochastic=True)
    else:
        loss_config = small_loss(teacher_spectral_weight=1)
    with pytest.raises(ValueError, match="Checkpoint"):
        train_fixed_batch(model, batch, steps=1, config=config, loss_config=loss_config,
                           resume_from=checkpoint)


def test_failed_update_preserves_previous_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = TinyDecoder()
    batch = fixed_batch()
    train_fixed_batch(model, batch, steps=1, loss_config=small_loss(), checkpoint_path=checkpoint)
    previous = checkpoint.read_bytes()
    bad = TrainingBatch(batch.latents, torch.full_like(batch.teacher_audio, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        train_fixed_batch(model, bad, steps=1, loss_config=small_loss(), checkpoint_path=checkpoint)
    assert checkpoint.read_bytes() == previous


def test_steps_must_be_explicit_and_positive():
    with pytest.raises(ValueError, match="positive integer"):
        train_fixed_batch(TinyDecoder(), fixed_batch(), steps=0, loss_config=small_loss())
