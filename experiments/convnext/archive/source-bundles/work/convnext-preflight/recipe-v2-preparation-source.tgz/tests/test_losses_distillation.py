from dataclasses import asdict
import json

import pytest
import torch

from audiovae_student.losses_distillation import (
    DistillationLossConfig, DistillationReconstructionLoss, slaney_mel_filterbank,
)

torch.set_num_threads(1)


def small_loss():
    return DistillationReconstructionLoss(DistillationLossConfig(fft_sizes=(64, 128), mel_bands=(4, 8)))


def test_exact_teacher_is_zero_and_teacher_reference_are_detached():
    torch.manual_seed(21)
    teacher = (torch.randn(2, 1, 512) * .07).requires_grad_()
    prediction = teacher.detach().clone().requires_grad_()
    reference = torch.randn(2, 1, 100, requires_grad=True)
    values = small_loss()(prediction, teacher, reference)
    assert all(value.ndim == 0 and value.item() == 0 for value in values.values())
    values['total'].backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) == 0
    assert teacher.grad is None and reference.grad is None


def test_full_recipe_polarity_silence_and_gain_controls():
    time = torch.arange(8192) / 48000
    teacher = (.08 * torch.sin(2 * torch.pi * 223 * time)
               + .03 * torch.sin(2 * torch.pi * 6210 * time)).view(1, 1, -1)
    criterion = DistillationReconstructionLoss()
    inverse = criterion(-teacher, teacher)
    silent = criterion(torch.zeros_like(teacher), teacher)
    attenuated = criterion(.3 * teacher, teacher)
    assert inverse['teacher_mel'].abs() < 1e-7
    torch.testing.assert_close(inverse['teacher_waveform'], 2 * silent['teacher_waveform'])
    assert silent['teacher_mel'] > attenuated['teacher_mel'] > 0
    assert attenuated['teacher_rms_error'] > 0
    assert inverse['teacher_waveform_raw'] > silent['teacher_waveform_raw'] > 0


def test_mel_is_magnitude_not_power_and_reductions_are_per_example():
    torch.manual_seed(6)
    teacher = torch.randn(2, 1, 512) * .08
    criterion = small_loss()
    doubled, tripled = criterion(2 * teacher, teacher), criterion(3 * teacher, teacher)
    torch.testing.assert_close(tripled['teacher_mel_linear'], 2 * doubled['teacher_mel_linear'])
    per_item = [criterion(2 * teacher[i:i+1], teacher[i:i+1]) for i in range(2)]
    for key in doubled:
        torch.testing.assert_close(doubled[key], torch.stack([v[key] for v in per_item]).mean())


def test_gradient_reaches_student_and_reference_changes_do_not_change_loss():
    torch.manual_seed(22)
    teacher = (torch.randn(1, 1, 512) * .02).requires_grad_()
    prediction = (torch.randn_like(teacher) * .02).requires_grad_()
    criterion = small_loss()
    first = criterion(prediction, teacher, torch.ones(1, 1, 16))
    second = criterion(prediction, teacher, torch.zeros(1, 1, 16))
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    first['total'].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert teacher.grad is None


def test_rms_floor_bounds_silence_gradient_and_short_scored_regions_fail():
    prediction = torch.full((1, 1, 256), .01, requires_grad=True)
    teacher = torch.zeros_like(prediction)
    result = small_loss()(prediction, teacher)
    torch.testing.assert_close(result['teacher_waveform'], torch.tensor(10.))
    result['total'].backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match='scored region'):
        small_loss()(prediction[..., :64], teacher[..., :64])


def test_filter_area_and_configuration_identity():
    bank = slaney_mel_filterbank(4096, 64)
    # Slaney area normalization makes each triangular filter integrate to one;
    # this independent numerical quadrature checks frequency-axis scaling.
    area = torch.trapezoid(bank, dx=48000 / 4096, dim=1)
    torch.testing.assert_close(area, torch.ones_like(area), rtol=.035, atol=.005)
    assert bank.shape == (64, 2049) and (bank >= 0).all()
    assert torch.count_nonzero(bank[:, 0]) == 0 and torch.count_nonzero(bank[:, -1]) == 0
    config = DistillationLossConfig()
    assert DistillationLossConfig(**json.loads(json.dumps(asdict(config)))) == config
    with pytest.raises(ValueError, match='48 kHz'):
        DistillationLossConfig(sample_rate=16000)
