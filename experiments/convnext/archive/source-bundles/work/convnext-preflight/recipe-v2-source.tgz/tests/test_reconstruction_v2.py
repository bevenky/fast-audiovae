"""Objective semantics: valid duration, grouping, levels and teacher ownership."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from audiovae_student.losses_distillation import DistillationReconstructionLoss
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(threads)


def criterion():
    return ReconstructionV2(ReconstructionV2Config(fft_sizes=(64, 128), mel_bands=(4, 8)))


def audio(batch, samples, scale=.04):
    return torch.randn(batch, 1, samples) * scale


def test_waveform_gradient_is_identical_for_quiet_and_loud_equal_errors():
    target = torch.cat([torch.full((1, 1, 512), .0002), torch.full((1, 1, 512), .03)])
    target.requires_grad_()
    prediction = (target.detach() + .005).requires_grad_()
    result = criterion()(prediction, target)
    torch.testing.assert_close(result.losses['teacher_waveform'], torch.tensor(.005))
    result.losses['teacher_waveform'].backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1 / 1024), atol=0, rtol=0)
    assert target.grad is None
    assert all(not v.requires_grad for v in result.legacy_diagnostics.values())


def test_unequal_lengths_weight_samples_and_give_each_sample_the_same_gradient():
    short = torch.full((1, 1, 128), .01, requires_grad=True)
    long = torch.full((1, 1, 512), .02, requires_grad=True)
    result = criterion().forward_groups(((short, torch.zeros_like(short)), (long, torch.zeros_like(long))))
    expected = (.01 * 128 + .02 * 512) / 640
    torch.testing.assert_close(result.losses['teacher_waveform'], torch.tensor(expected))
    assert result.counts['valid_samples'] == 640
    # Historical per-clip raw MAE is .015; it remains a separate diagnostic.
    torch.testing.assert_close(result.legacy_diagnostics['teacher_waveform_raw'], torch.tensor(.015))
    assert not torch.isclose(result.losses['teacher_waveform'], result.legacy_diagnostics['teacher_waveform_raw'])
    result.losses['teacher_waveform'].backward()
    for prediction in (short, long):
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1 / 640), atol=0, rtol=0)


def test_batch_partition_preserves_all_losses_and_gradients():
    loss = criterion()
    teacher = audio(3, 512).requires_grad_()
    prediction = audio(3, 512).requires_grad_()
    full = loss(prediction, teacher)
    split = loss.aggregate([loss.group_terms(prediction[:1], teacher[:1]),
                            loss.group_terms(prediction[1:], teacher[1:])])
    for name in full.losses:
        torch.testing.assert_close(full.losses[name], split.losses[name])
    for name in full.legacy_diagnostics:
        torch.testing.assert_close(full.legacy_diagnostics[name], split.legacy_diagnostics[name])
    assert full.counts == split.counts
    full_grad, = torch.autograd.grad(full.losses['total'], prediction, retain_graph=True)
    split_grad, = torch.autograd.grad(split.losses['total'], prediction)
    torch.testing.assert_close(full_grad, split_grad, atol=2e-7, rtol=2e-5)
    assert teacher.grad is None


def test_mel_pools_valid_frames_separately_at_each_resolution():
    loss = criterion()
    pairs = [(audio(2, 130), audio(2, 130)), (audio(1, 515), audio(1, 515))]
    combined = loss.forward_groups(pairs)
    expected = {'teacher_mel_linear': [], 'teacher_mel_log': []}
    element_counts = []
    # Existing single-resolution loss supplies independent mean values; only
    # its spectral terms are used to check the new global count weighting.
    for size, bands in zip(loss.config.fft_sizes, loss.config.mel_bands):
        old = DistillationReconstructionLoss(replace(loss.config.legacy_config(),
                                                     fft_sizes=(size,), mel_bands=(bands,)))
        values = [old(prediction, target) for prediction, target in pairs]
        counts = [prediction.shape[0] * bands * (1 + (prediction.shape[-1] - size) // (size // 4))
                  for prediction, _ in pairs]
        element_counts.append(sum(counts))
        for name in expected:
            expected[name].append(sum(value[name] * count for value, count in zip(values, counts)) / sum(counts))
    for name, values in expected.items():
        torch.testing.assert_close(combined.losses[name], torch.stack(values).mean())
    assert combined.counts['mel_elements_by_resolution'] == tuple(element_counts)
    assert combined.counts['mel_frames_by_resolution'] == tuple(c // b for c, b in zip(element_counts, loss.config.mel_bands))


def test_unequal_length_bucketing_preserves_per_example_and_gradient_behavior():
    loss = criterion()
    long_p, short_p = audio(2, 515).requires_grad_(), audio(1, 130).requires_grad_()
    long_t, short_t = audio(2, 515), audio(1, 130)
    grouped = loss.forward_groups(((long_p, long_t), (short_p, short_t)))
    items = loss.forward_groups(((long_p[:1], long_t[:1]), (short_p, short_t), (long_p[1:], long_t[1:])))
    for name in grouped.losses:
        torch.testing.assert_close(grouped.losses[name], items.losses[name])
    before = torch.autograd.grad(grouped.losses['total'], (long_p, short_p), retain_graph=True)
    after = torch.autograd.grad(items.losses['total'], (long_p, short_p))
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)


def test_legacy_diagnostics_match_unchanged_per_example_criterion():
    loss = criterion()
    pairs = [(audio(2, 130).requires_grad_(), audio(2, 130)),
             (audio(1, 515).requires_grad_(), audio(1, 515))]
    result = loss.forward_groups(pairs)
    old = DistillationReconstructionLoss(loss.config.legacy_config())
    values = [old(p, t) for p, t in pairs]
    for name, metric in result.legacy_diagnostics.items():
        torch.testing.assert_close(metric, (2 * values[0][name] + values[1][name]) / 3)
        assert not metric.requires_grad


def test_context_and_padding_are_excluded_by_valid_scored_slices():
    loss = criterion()
    prediction = audio(2, 900)
    target = audio(2, 900)
    regions = (slice(31, 161), slice(123, 638))
    clean = loss.forward_groups((prediction[i:i + 1, :, region], target[i:i + 1, :, region])
                                for i, region in enumerate(regions))
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    for i, region in enumerate(regions):
        valid[i, :, region] = True
    prediction[~valid], target[~valid] = float('nan'), float('nan')
    prediction.requires_grad_()
    target.requires_grad_()
    poisoned = loss.forward_groups((prediction[i:i + 1, :, region], target[i:i + 1, :, region])
                                   for i, region in enumerate(regions))
    for name in clean.losses:
        torch.testing.assert_close(clean.losses[name], poisoned.losses[name], atol=0, rtol=0)
    assert poisoned.counts['valid_samples'] == 645
    poisoned.losses['total'].backward()
    assert torch.count_nonzero(prediction.grad[~valid]) == 0
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None


def test_exact_teacher_is_zero_for_new_objective_and_old_diagnostics():
    target = audio(1, 8192).requires_grad_()
    prediction = target.detach().clone().requires_grad_()
    result = ReconstructionV2()(prediction, target)
    for value in (*result.losses.values(), *result.legacy_diagnostics.values()):
        assert value.ndim == 0 and value.item() == 0
    result.losses['total'].backward()
    assert torch.count_nonzero(prediction.grad) == 0
    assert target.grad is None


def test_configuration_and_invalid_groups_fail_explicitly():
    loss = criterion()
    assert ReconstructionV2Config(**json.loads(json.dumps(asdict(loss.config)))) == loss.config
    with pytest.raises(ValueError, match='format version'):
        ReconstructionV2Config(format_version=1)
    with pytest.raises(ValueError, match='At least one'):
        loss.forward_groups([])
    with pytest.raises(ValueError, match='largest FFT'):
        loss(audio(1, 127), audio(1, 127))
    with pytest.raises(ValueError, match='aligned shapes'):
        loss(audio(1, 256), audio(2, 256))
    other = ReconstructionV2(replace(loss.config, mel_log_weight=0))
    with pytest.raises(ValueError, match='same reconstruction configuration'):
        loss.aggregate([other.group_terms(audio(1, 256), audio(1, 256))])
