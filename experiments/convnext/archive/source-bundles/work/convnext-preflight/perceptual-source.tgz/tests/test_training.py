"""Behavioral checks for reconstruction, overfit and exact CPU resume."""

from copy import deepcopy
import os
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

from audiovae_student.losses import WarmupLossConfig, WarmupReconstructionLoss
from audiovae_student.optimizers import native_muon_available
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
    assert payload["format_version"] == 2
    assert payload["optimizer_group_fingerprint"] == payload["optimizer"]["group_fingerprint"]
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


def test_legacy_checkpoint_is_rejected_before_loading(tmp_path):
    checkpoint = tmp_path / "old.pt"
    model = TinyDecoder()
    train_fixed_batch(model, fixed_batch(), steps=1, loss_config=small_loss(),
                       checkpoint_path=checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["format_version"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="format_version"):
        train_fixed_batch(model, fixed_batch(), steps=1, loss_config=small_loss(),
                           resume_from=checkpoint)


@pytest.mark.parametrize("values", [
    {"optimizer": "muon"}, {"muon_ns_steps": 0}, {"muon_ns_steps": 2.5},
    {"muon_momentum": 1}, {"muon_momentum": float("nan")},
    {"muon_adjust_lr_fn": "original"},
])
def test_optimizer_configuration_rejects_unsupported_settings(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


@pytest.mark.skipif(not native_muon_available(), reason="Native torch.optim.Muon is unavailable")
def test_native_hybrid_training_checkpoint_replays_updates(tmp_path):
    from audiovae_student.model import StudentConfig, StudentDecoder

    torch.manual_seed(11)
    architecture = StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8)
    initial = StudentDecoder(architecture)
    continuous = deepcopy(initial)
    partial = deepcopy(initial)
    generator = torch.Generator().manual_seed(12)
    latents = torch.randn(1, 64, 1, generator=generator)
    target = (0.1 * torch.sin(torch.arange(1920) * 0.03))[None, None]
    batch = TrainingBatch(latents, target)
    config = TrainingConfig(optimizer="muon_adamw", learning_rate=1e-3, seed=13)
    loss_config = WarmupLossConfig(teacher_fft_sizes=(32, 64), reference_fft_sizes_16k=(16,))
    expected = train_fixed_batch(continuous, batch, steps=4, config=config, loss_config=loss_config)
    checkpoint = tmp_path / "hybrid.pt"
    first = train_fixed_batch(partial, batch, steps=2, config=config, loss_config=loss_config,
                              checkpoint_path=checkpoint)
    resumed = StudentDecoder(architecture)
    rest = train_fixed_batch(resumed, batch, steps=2, config=config, loss_config=loss_config,
                             resume_from=checkpoint)
    assert first.metrics + rest.metrics == expected.metrics
    assert all(torch.equal(value, resumed.state_dict()[name])
               for name, value in continuous.state_dict().items())
    with pytest.raises(ValueError, match="training_config"):
        train_fixed_batch(resumed, batch, steps=1, loss_config=loss_config,
                           config=TrainingConfig(learning_rate=1e-3, seed=13),
                           resume_from=checkpoint)


@pytest.mark.skipif(os.environ.get("AUDIOVAE_STUDENT_TEST_CUDA") != "1",
                    reason="GPU training smoke requires AUDIOVAE_STUDENT_TEST_CUDA=1")
def test_explicit_cuda_full_architecture_hybrid_step(tmp_path):
    # The environment gate prevents touching CUDA during CPU-only validation.
    # This is a training smoke test, never an inference-performance benchmark.
    from audiovae_student.model import StudentDecoder

    assert native_muon_available(), "Explicit CUDA smoke requires native torch.optim.Muon"
    assert torch.cuda.is_available(), "Explicit CUDA smoke requires an available CUDA device"
    torch.manual_seed(23)
    model = StudentDecoder().to("cuda")
    latents = torch.randn(1, 64, 3, device="cuda")
    target = (0.1 * torch.sin(torch.arange(5760, device="cuda") * 0.03))[None, None]
    before = model.blocks[0].expand.weight.detach().clone()
    checkpoint = tmp_path / "cuda-hybrid.pt"
    result = train_fixed_batch(model, TrainingBatch(latents, target), steps=1,
                               config=TrainingConfig(optimizer="muon_adamw", seed=23),
                               checkpoint_path=checkpoint)
    assert result.step == 1
    assert np.isfinite(result.metrics[0]["total"])
    assert np.isfinite(result.metrics[0]["gradient_norm"])
    assert not torch.equal(before, model.blocks[0].expand.weight)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload["optimizer"]["optimizers"]) == {"muon", "adamw"}
    assert payload["device_type"] == "cuda"


def test_tensorboard_dependency_is_optional_and_fails_clearly_when_requested(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    # A plain preflight still works without importing the optional dependency.
    train_fixed_batch(TinyDecoder(), fixed_batch(), steps=1, loss_config=small_loss())
    with pytest.raises(RuntimeError, match="tensorboard is not installed"):
        train_fixed_batch(TinyDecoder(), fixed_batch(), steps=1, loss_config=small_loss(),
                           log_dir=tmp_path, run_name="preflight-missing-dependency")


def test_tensorboard_logs_actual_losses_and_resume_purges_stale_events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(29)
    initial = TinyDecoder(stochastic=True)
    config = TrainingConfig(learning_rate=0.004, seed=31)
    loss_config = small_loss(teacher_spectral_weight=1)
    expected = train_fixed_batch(deepcopy(initial), fixed_batch(), steps=4,
                                 config=config, loss_config=loss_config)
    checkpoint = tmp_path / "checkpoint.pt"
    log_root = tmp_path / "logs"
    name = "preflight-test-resume"
    first = train_fixed_batch(deepcopy(initial), fixed_batch(), steps=2, config=config,
                              loss_config=loss_config, checkpoint_path=checkpoint,
                              log_dir=log_root, run_name=name)
    # Simulate events written after checkpoint step 2, before a crash. The
    # resumed run must hide these stale points, not show duplicate step 3/4.
    with SummaryWriter(log_dir=str(log_root / name)) as stale:
        stale.add_scalar("loss/total", 999.0, 3)
        stale.add_scalar("loss/total", 999.0, 4)
    rest = train_fixed_batch(TinyDecoder(stochastic=True), fixed_batch(), steps=2,
                             config=config, loss_config=loss_config, resume_from=checkpoint,
                             log_dir=log_root, run_name=name)
    assert first.metrics + rest.metrics == expected.metrics
    assert first.log_path == rest.log_path == log_root / name
    events = EventAccumulator(str(rest.log_path), size_guidance={"scalars": 0}).Reload()
    for component in ("total", "teacher_spectral", "teacher_waveform", "reference_spectral"):
        values = events.Scalars(f"loss/{component}")
        assert [point.step for point in values] == [1, 2, 3, 4]
        assert [point.value for point in values] == pytest.approx(
            [metric[component] for metric in expected.metrics])
    assert [point.value for point in events.Scalars("learning_rate/adamw")] == pytest.approx([0.004] * 4)
    assert [point.value for point in events.Scalars("gradient/global_norm")] == pytest.approx(
        [metric["gradient_norm"] for metric in expected.metrics])
    assert [point.value for point in events.Scalars("timing/step_seconds")] == pytest.approx(
        first.step_elapsed_seconds + rest.step_elapsed_seconds)
    assert all(value > 0 for value in first.step_elapsed_seconds + rest.step_elapsed_seconds)
    assert not any("validation" in tag for tag in events.Tags()["scalars"])
    with pytest.raises(ValueError, match="already has events"):
        train_fixed_batch(TinyDecoder(stochastic=True), fixed_batch(), steps=1, config=config,
                           loss_config=loss_config, log_dir=log_root, run_name=name)


def test_tensorboard_writer_closes_after_training_error(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    import torch.utils.tensorboard as tensorboard

    instances = []
    parent = tensorboard.SummaryWriter

    class TrackedWriter(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_closed = False
            instances.append(self)

        def close(self):
            self.was_closed = True
            super().close()

    monkeypatch.setattr(tensorboard, "SummaryWriter", TrackedWriter)
    batch = fixed_batch()
    batch = TrainingBatch(batch.latents, torch.full_like(batch.teacher_audio, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        train_fixed_batch(TinyDecoder(), batch, steps=1, loss_config=small_loss(),
                           log_dir=tmp_path, run_name="preflight-error")
    assert len(instances) == 1 and instances[0].was_closed


@pytest.mark.skipif(not native_muon_available(), reason="Native torch.optim.Muon is unavailable")
def test_tensorboard_hybrid_logs_both_optimizer_learning_rates(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from audiovae_student.model import StudentConfig, StudentDecoder

    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8))
    batch = TrainingBatch(torch.randn(1, 64, 1), torch.randn(1, 1, 1920) * 0.1)
    result = train_fixed_batch(model, batch, steps=1, loss_config=small_loss(),
                               config=TrainingConfig(optimizer="muon_adamw", learning_rate=1e-3),
                               log_dir=tmp_path, run_name="preflight-native-hybrid")
    events = EventAccumulator(str(result.log_path)).Reload()
    for optimizer in ("muon", "adamw"):
        points = events.Scalars(f"learning_rate/{optimizer}")
        assert [point.step for point in points] == [1]
        assert points[0].value == pytest.approx(1e-3)
