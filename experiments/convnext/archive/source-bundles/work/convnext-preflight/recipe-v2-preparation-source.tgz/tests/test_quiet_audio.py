"""Quiet supervision must retain low-level teacher audio and safe gradients."""
import json

import pytest
import torch

from audiovae_student.quiet_audio import QuietAudioConfig, quiet_residual_loss, quiet_window_metrics


SMALL = QuietAudioConfig(window_samples=4)


def test_exact_silence_has_finite_zero_loss_and_gradients_without_cosine():
    student = torch.zeros(2, 1, 13, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    loss = quiet_residual_loss(student, teacher, config=SMALL)
    assert loss.item() == 0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert not student.grad.count_nonzero()
    assert teacher.grad is None
    metrics = quiet_window_metrics(student, teacher, config=SMALL)
    assert metrics["quiet_passed"] is True
    assert metrics["valid_samples"] == 26
    assert metrics["quiet_window_count"] == 8
    assert metrics["windows"][-1]["valid_samples"] == 1
    json.dumps(metrics, allow_nan=False)


def test_noise_on_exact_silence_fails_absolute_floor():
    teacher = torch.zeros(1, 1, 11)
    student = torch.full_like(teacher, 2e-5, requires_grad=True)
    metrics = quiet_window_metrics(student, teacher, config=SMALL)
    assert metrics["quiet_passed"] is False
    assert metrics["quiet_failed_count"] == 3
    assert all(row["residual_limit"] == pytest.approx(1e-5) for row in metrics["windows"])
    loss = quiet_residual_loss(student, teacher, config=SMALL)
    assert loss.item() == pytest.approx(0.02)
    loss.backward()
    assert student.grad.gt(0).all()


@pytest.mark.parametrize("multiplier", [0.0, -1.0])
def test_muting_or_inverting_low_level_teacher_audio_fails(multiplier):
    teacher = torch.tensor([0.0005, -0.0005, 0.0005, -0.0005]).reshape(1, 1, -1)
    teacher.requires_grad_(True)
    student = (teacher.detach() * multiplier).requires_grad_(True)
    metrics = quiet_window_metrics(student, teacher, config=SMALL)
    assert metrics["quiet_passed"] is False
    row = metrics["windows"][0]
    assert row["student_rms"] <= row["output_rms_limit"]
    assert row["residual_rms"] > row["residual_limit"]
    loss = quiet_residual_loss(student, teacher, config=SMALL)
    assert loss.item() == pytest.approx(abs(1 - multiplier) * 0.5)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert student.grad.count_nonzero() == 4


def test_amplitude_limit_rejects_gain_even_when_residual_limit_passes():
    teacher = torch.full((1, 1, 4), 0.0005)
    student = teacher * 1.13
    row = quiet_window_metrics(student, teacher, config=SMALL)["windows"][0]
    assert row["residual_rms"] < row["residual_limit"]
    assert row["student_rms"] > row["output_rms_limit"]
    assert row["passed"] is False


def test_mask_excludes_poison_context_holes_padding_and_retains_valid_tail():
    teacher = torch.full((1, 1, 11), 0.0005)
    student = teacher.clone() + 1e-5
    mask = torch.tensor([False, False, False, True, True, True, False, True, True, True, True]).reshape(1, 1, -1)
    baseline = quiet_window_metrics(student, teacher, mask, config=SMALL)
    baseline_loss = quiet_residual_loss(student, teacher, mask, config=SMALL)
    teacher = teacher.masked_fill(~mask, float("inf")).requires_grad_(True)
    student = student.masked_fill(~mask, float("nan")).requires_grad_(True)
    metrics = quiet_window_metrics(student, teacher, mask, config=SMALL)
    assert metrics == baseline
    assert metrics["valid_samples"] == 7
    assert [row["valid_samples"] for row in metrics["windows"]] == [1, 3, 3]
    assert metrics["windows"][-1]["stop_sample"] == 11
    loss = quiet_residual_loss(student, teacher, mask, config=SMALL)
    torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
    loss.backward()
    assert teacher.grad is None
    assert not student.grad.masked_select(~mask).count_nonzero()
    assert torch.isfinite(student.grad).all()


def test_loss_reduces_quiet_windows_then_examples_without_loud_dilution():
    teacher = torch.tensor([[0.0] * 8, [0.0] * 4 + [0.1] * 4, [0.1] * 8]).reshape(3, 1, 8)
    student = torch.tensor([[0.001] * 4 + [0.003] * 4,
                            [0.004] * 4 + [0.11] * 4, [0.2] * 8]).reshape(3, 1, 8).requires_grad_(True)
    loss = quiet_residual_loss(student, teacher, config=SMALL)
    # Example zero averages losses 1 and 3, example one contributes 4;
    # example two has no quiet window and does not dilute their mean.
    assert loss.item() == pytest.approx(3.0)
    loss.backward()
    assert not student.grad[1, :, 4:].count_nonzero()
    assert not student.grad[2].count_nonzero()
    metrics = quiet_window_metrics(student, teacher, config=SMALL)
    assert metrics["quiet_window_count"] == 3
    assert metrics["examples"][2]["quiet_passed"] is None


def test_no_quiet_windows_return_connected_zero_and_no_claimed_pass():
    student = torch.zeros(2, 1, 11, requires_grad=True)
    teacher = torch.full_like(student, 0.1, requires_grad=True)
    loss = quiet_residual_loss(student, teacher, config=SMALL)
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert teacher.grad is None
    assert not student.grad.count_nonzero()
    metrics = quiet_window_metrics(student, teacher, config=SMALL)
    assert metrics["quiet_window_count"] == 0
    assert metrics["quiet_passed"] is None


def test_entirely_masked_poison_returns_connected_zero():
    student = torch.full((1, 1, 9), float("nan"), requires_grad=True)
    teacher = torch.full_like(student, float("inf"), requires_grad=True)
    mask = torch.zeros_like(student, dtype=torch.bool)
    loss = quiet_residual_loss(student, teacher, mask, config=SMALL)
    loss.backward()
    assert loss.item() == 0
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and not student.grad.count_nonzero()
    metrics = quiet_window_metrics(student, teacher, mask, config=SMALL)
    assert metrics["window_count"] == metrics["valid_samples"] == 0
    assert metrics["quiet_passed"] is None


def test_teacher_alone_decides_which_windows_are_quiet():
    teacher = torch.full((1, 1, 4), 0.0005)
    loud_student = torch.full_like(teacher, 0.5, requires_grad=True)
    assert quiet_window_metrics(loud_student, teacher, config=SMALL)["quiet_window_count"] == 1
    assert quiet_residual_loss(loud_student, teacher, config=SMALL).item() > 100
    quiet_student = torch.zeros_like(teacher, requires_grad=True)
    loud_teacher = torch.full_like(teacher, 0.1)
    assert quiet_window_metrics(quiet_student, loud_teacher, config=SMALL)["quiet_window_count"] == 0


@pytest.mark.parametrize("kwargs", [{"window_samples": 0}, {"window_samples": 1.5},
    {"quiet_teacher_rms_max": 0}, {"loss_rms_floor": -1},
    {"residual_relative_limit": float("nan")}, {"absolute_rms_floor": float("inf")},
    {"amplitude_db_max": -1}, {"amplitude_db_max": 1e10}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        QuietAudioConfig(**kwargs)


def test_invalid_valid_audio_or_mask_fails_explicitly():
    audio = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError, match="finite"):
        quiet_residual_loss(audio + float("nan"), audio)
    with pytest.raises(ValueError, match="bool"):
        quiet_window_metrics(audio, audio, torch.ones_like(audio))
    with pytest.raises(ValueError, match="shapes"):
        quiet_residual_loss(audio, torch.zeros(1, 4))
