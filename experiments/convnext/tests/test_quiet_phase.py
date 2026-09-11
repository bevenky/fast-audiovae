"""The quiet phase candidate must target repeated residuals without muting teachers."""
import json
import math

import pytest
import torch

from audiovae_student.quiet_phase import QuietPhaseConfig, quiet_phase_loss, quiet_phase_metrics

WINDOW = 960
PERIOD = 480


def pattern():
    return torch.cat((torch.ones(PERIOD // 2), -torch.ones(PERIOD // 2)))


def test_exact_silence_has_connected_zero_and_finite_student_only_gradient():
    student = torch.zeros(2, 1, WINDOW * 8, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    loss = quiet_phase_loss(student, teacher)
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert not student.grad.count_nonzero()
    evidence = quiet_phase_metrics(student, teacher)
    assert evidence['active_examples'] == 2
    assert evidence['eligible_windows'] == 16
    assert evidence['examples'][0]['eligible_head_blocks'] == 16
    json.dumps(evidence, allow_nan=False)


def test_repeated_phase_noise_costs_more_than_equal_energy_canceling_noise():
    teacher = torch.zeros(1, 1, WINDOW * 8)
    repeat = (pattern().repeat(16) * .001).reshape_as(teacher).requires_grad_()
    # Equal sample energy, but residual templates alternate sign across blocks.
    noise = (pattern().repeat(16).reshape(16, PERIOD) *
             torch.tensor([1., -1.] * 8).unsqueeze(-1) * .001).reshape_as(teacher).requires_grad_()
    torch.testing.assert_close(torch.linalg.vector_norm(repeat), torch.linalg.vector_norm(noise))
    coherent = quiet_phase_loss(repeat, teacher)
    canceled = quiet_phase_loss(noise, teacher)
    assert coherent.item() == pytest.approx(1.0)
    assert canceled.item() == 0
    coherent.backward()
    canceled.backward()
    assert torch.isfinite(repeat.grad).all() and repeat.grad.count_nonzero() == repeat.numel()
    assert torch.isfinite(noise.grad).all() and not noise.grad.count_nonzero()
    evidence = quiet_phase_metrics(repeat, teacher)
    assert evidence['examples'][0]['residual_template_rms'] == pytest.approx(.001)


def test_matching_teacher_periodic_tone_and_room_tone_cost_zero():
    grid = torch.arange(WINDOW * 8, dtype=torch.float64)
    waveform = .0004 * torch.sin(2 * math.pi * grid / PERIOD) + .0001
    teacher = waveform.reshape(1, 1, -1).requires_grad_()
    student = teacher.detach().clone().requires_grad_()
    loss = quiet_phase_loss(student, teacher)
    assert loss.item() == 0
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and not student.grad.count_nonzero()
    muted = torch.zeros_like(student, requires_grad=True)
    assert quiet_phase_loss(muted, teacher).item() > .2
    inverted = (-teacher.detach()).requires_grad_()
    assert quiet_phase_loss(inverted, teacher).item() > .4


def test_complete_windows_only_rejects_short_quiet_and_partial_tail():
    student = torch.full((1, 1, WINDOW * 8 - 1), .001, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    loss = quiet_phase_loss(student, teacher)
    loss.backward()
    assert loss.item() == 0 and not student.grad.count_nonzero()
    assert teacher.grad is None
    evidence = quiet_phase_metrics(student, teacher)
    assert evidence['teacher_quiet_complete_windows'] == 7
    assert evidence['eligible_windows'] == evidence['active_examples'] == 0
    assert evidence['examples'][0]['residual_template_rms'] is None


def test_masked_poison_context_holes_and_tail_never_contribute_or_shift_phase():
    teacher = torch.zeros(1, 1, WINDOW * 11 + 123)
    student = (pattern().repeat(23)[:teacher.numel()] * .001).reshape_as(teacher)
    mask = torch.ones_like(student, dtype=torch.bool)
    mask[..., :WINDOW] = False
    mask[..., WINDOW * 5 + 3] = False  # Entire window 5 must be discarded.
    mask[..., WINDOW * 11:] = False
    expected = quiet_phase_loss(student, teacher, mask)
    teacher = teacher.masked_fill(~mask, float('inf')).requires_grad_()
    student = student.masked_fill(~mask, float('nan')).requires_grad_()
    loss = quiet_phase_loss(student, teacher, mask)
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert not student.grad.masked_select(~mask).count_nonzero()
    assert not student.grad[..., WINDOW * 5:WINDOW * 6].count_nonzero()
    assert quiet_phase_metrics(student, teacher, mask)['eligible_windows'] == 9


def test_entirely_masked_poison_returns_finite_connected_zero():
    student = torch.full((2, 1, 137), float('nan'), requires_grad=True)
    teacher = torch.full_like(student, float('inf'), requires_grad=True)
    mask = torch.zeros_like(student, dtype=torch.bool)
    loss = quiet_phase_loss(student, teacher, mask)
    loss.backward()
    assert loss.item() == 0
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and not student.grad.count_nonzero()
    assert quiet_phase_metrics(student, teacher, mask)['complete_valid_windows'] == 0


def test_mean_active_examples_and_no_loud_window_gradient():
    teacher = torch.zeros(3, 1, WINDOW * 16)
    teacher[0, :, WINDOW * 8:] = .1
    teacher[2] = .1
    student = torch.full_like(teacher, .003)
    student[0, :, :WINDOW * 8] = .001
    student.requires_grad_()
    loss = quiet_phase_loss(student, teacher)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert not student.grad[0, :, WINDOW * 8:].count_nonzero()
    assert not student.grad[2].count_nonzero()
    evidence = quiet_phase_metrics(student, teacher)
    assert evidence['active_examples'] == 2 and evidence['eligible_windows'] == 24
    assert evidence['examples'][2]['residual_template_rms'] is None


def test_nonquiet_teacher_cannot_become_eligible_by_muting_student():
    student = torch.zeros(1, 1, WINDOW * 8, requires_grad=True)
    teacher = torch.full_like(student, .1, requires_grad=True)
    loss = quiet_phase_loss(student, teacher)
    assert loss.item() == 0
    loss.backward()
    assert teacher.grad is None and not student.grad.count_nonzero()
    assert quiet_phase_metrics(student, teacher)['eligible_windows'] == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_accumulation_is_at_least_float32(dtype):
    student = torch.zeros(1, 1, WINDOW * 8, dtype=dtype, requires_grad=True)
    teacher = torch.zeros_like(student)
    loss = quiet_phase_loss(student, teacher)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    loss.backward()
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize('kwargs', [dict(period_samples=0), dict(window_samples=480),
    dict(minimum_windows=7), dict(minimum_windows=8.0), dict(quiet_teacher_rms_max=float('nan')),
    dict(loss_rms_floor=0), dict(loss_rms_floor=True)])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        QuietPhaseConfig(**kwargs)


def test_invalid_shapes_masks_and_scored_nan_fail_explicitly():
    student = torch.zeros(1, 1, WINDOW * 8)
    with pytest.raises(ValueError, match='shapes'):
        quiet_phase_loss(student, student.squeeze(1))
    with pytest.raises(ValueError, match='bool'):
        quiet_phase_loss(student, student, torch.ones_like(student))
    with pytest.raises(ValueError, match='finite'):
        quiet_phase_loss(student + float('nan'), student)
    with pytest.raises(ValueError, match='finite'):
        quiet_phase_loss(student, student + float('nan'))
