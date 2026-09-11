"""Candidate objective semantics, spectral information and trained-MPD ownership."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from audiovae_student.discriminators import (AudioDiscriminators, DiscriminatorConfig,
    discriminator_loss, generator_losses)
from audiovae_student.fusion_objectives import (ComplexAudioDiscriminators,
    ComplexBandResolutionDiscriminator, ComplexDiscriminatorConfig,
    SHORT_TIME_RECONSTRUCTION_CONFIG, ShortTimeReconstructionV2)
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    rng = torch.random.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(101)
    yield
    torch.random.set_rng_state(rng)
    torch.set_num_threads(threads)


def audio(batch, samples, scale=.03):
    return scale * torch.randn(batch, 1, samples)


def small_original():
    return AudioDiscriminators(DiscriminatorConfig(periods=(2, 3),
        mpd_channels=(2, 4, 8, 8, 8), fft_sizes=(32, 64), mrd_channels=2))


def small_complex_config():
    return ComplexDiscriminatorConfig(periods=(2, 3),
        mpd_channels=(2, 4, 8, 8, 8), fft_sizes=(32, 64), mrd_channels=2)


def test_short_time_candidate_declares_equal_resolution_average_and_same_return_keys():
    criterion = ShortTimeReconstructionV2()
    assert criterion.config.fft_sizes == (256, 512, 1024, 2048, 4096)
    assert criterion.config.mel_bands == (16, 32, 64, 128, 128)
    assert criterion.objective_identity['new_resolution_share'] == 2 / 5
    assert criterion.objective_identity['resolution_reduction'].startswith('equal_mean_')
    assert ReconstructionV2Config(**json.loads(json.dumps(asdict(criterion.config)))) == criterion.config
    prediction, teacher = audio(1, 5000).requires_grad_(), audio(1, 5000).requires_grad_()
    result = criterion(prediction, teacher)
    old = ReconstructionV2()(prediction, teacher)
    assert result.losses.keys() == old.losses.keys()
    assert result.legacy_diagnostics.keys() == old.legacy_diagnostics.keys()
    torch.testing.assert_close(result.losses['teacher_waveform'], old.losses['teacher_waveform'], atol=0, rtol=0)
    result.losses['total'].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert teacher.grad is None


def test_short_time_mel_pools_actual_elements_then_averages_all_five_scales():
    criterion = ShortTimeReconstructionV2()
    pairs = [(audio(2, 5200), audio(2, 5200)), (audio(1, 8000), audio(1, 8000))]
    result = criterion.forward_groups(pairs)
    values, counts = [], []
    for size, bands in zip(criterion.config.fft_sizes, criterion.config.mel_bands):
        single = ReconstructionV2(replace(criterion.config, fft_sizes=(size,), mel_bands=(bands,)))
        individual = [single(prediction, target) for prediction, target in pairs]
        weights = [p.shape[0] * bands * (1 + (p.shape[-1] - size) // (size // 4)) for p, _ in pairs]
        values.append(sum(item.losses['teacher_mel'] * weight for item, weight in zip(individual, weights))
                      / sum(weights))
        counts.append(sum(weights))
    torch.testing.assert_close(result.losses['teacher_mel'], torch.stack(values).mean())
    assert result.counts['mel_elements_by_resolution'] == tuple(counts)
    assert result.counts['valid_samples'] == 18400


def test_short_time_partition_and_scored_slices_keep_padding_teacher_and_counts_safe():
    criterion = ShortTimeReconstructionV2()
    prediction, teacher = audio(3, 5600), audio(3, 5600)
    prediction[..., :200] = teacher[..., :200] = float('nan')
    prediction[..., 5200:] = teacher[..., 5200:] = float('nan')
    prediction.requires_grad_()
    teacher.requires_grad_()
    p, t = prediction[..., 200:5200], teacher[..., 200:5200]
    full = criterion(p, t)
    split = criterion.forward_groups(((p[:1], t[:1]), (p[1:], t[1:])))
    assert full.counts == split.counts
    for name in full.losses:
        torch.testing.assert_close(full.losses[name], split.losses[name])
    full_grad, = torch.autograd.grad(full.losses['total'], prediction, retain_graph=True)
    split_grad, = torch.autograd.grad(split.losses['total'], prediction)
    torch.testing.assert_close(full_grad, split_grad, atol=2e-7, rtol=2e-5)
    assert not torch.count_nonzero(full_grad[..., :200])
    assert not torch.count_nonzero(full_grad[..., 5200:])
    assert torch.isfinite(full_grad).all() and teacher.grad is None


@pytest.mark.parametrize('level', [0., .00001, .03])
def test_exact_teacher_short_time_loss_is_zero_with_finite_zero_gradient(level):
    teacher = audio(1, 5000, level).requires_grad_()
    prediction = teacher.detach().clone().requires_grad_()
    result = ShortTimeReconstructionV2()(prediction, teacher)
    assert all(value.item() == 0 for value in result.losses.values())
    result.losses['total'].backward()
    assert torch.isfinite(prediction.grad).all() and torch.count_nonzero(prediction.grad) == 0
    assert teacher.grad is None


def test_complex_representation_keeps_phase_amplitude_dc_and_valid_stft_length():
    head = ComplexBandResolutionDiscriminator(64, 2)
    samples = torch.arange(257)
    wave = (.1 + .02 * torch.sin(samples * .37))[None, None].requires_grad_()
    bands = head.spectrogram_bands(wave)
    spectrum = torch.cat(bands, dim=2)
    expected = torch.stft(wave[:, 0], n_fft=64, hop_length=16, win_length=64,
        window=torch.hann_window(64), center=False, normalized=False, onesided=True,
        return_complex=True)
    torch.testing.assert_close(spectrum[:, 0], expected.real, atol=0, rtol=0)
    torch.testing.assert_close(spectrum[:, 1], expected.imag, atol=0, rtol=0)
    assert spectrum.shape == (1, 2, 33, 13)
    assert spectrum[:, 0, 0].abs().mean() > 1  # DC was not subtracted.
    assert spectrum[:, 1].abs().sum() > 0
    doubled = torch.cat(head.spectrogram_bands(wave * 2), dim=2)
    torch.testing.assert_close(doubled, 2 * spectrum, atol=0, rtol=0)
    inverted = torch.cat(head.spectrogram_bands(-wave), dim=2)
    torch.testing.assert_close(inverted, -spectrum, atol=0, rtol=0)
    spectrum.square().mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.abs().sum() > 0


def test_complex_bank_outputs_joined_band_heads_and_serializable_configuration():
    config = small_complex_config()
    assert ComplexDiscriminatorConfig(**json.loads(json.dumps(asdict(config)))) == config
    bank = ComplexAudioDiscriminators(config)
    outputs = bank(audio(2, 256))
    assert len(outputs) == 4
    assert [len(head.features) for head in outputs] == [5, 5, 25, 25]
    assert all(head.logits.shape[:2] == (2, 1) for head in outputs)
    assert all(torch.isfinite(value).all() for head in outputs for value in (head.logits, *head.features))
    clone = ComplexAudioDiscriminators(config)
    clone.load_state_dict(bank.state_dict(), strict=True)
    sample = audio(1, 256)
    for a, b in zip(bank(sample), clone(sample)):
        torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)


def test_replacement_copies_trained_mpd_weights_flags_and_outputs_without_aliasing():
    original = small_original().eval()
    original.periods[0].train()
    next(original.periods[0].parameters()).requires_grad_(False)
    original_state = {key: value.clone() for key, value in original.state_dict().items()}
    original_flags = [(module.training, tuple(p.requires_grad for p in module.parameters(recurse=False)))
                      for module in original.modules()]
    sample = audio(2, 256)
    before = [period(sample) for period in original.periods]
    replacement = ComplexAudioDiscriminators.from_existing(original, small_complex_config())
    assert not replacement.training
    assert not any(head.training for head in replacement.resolutions)
    for a, b in zip(before, [period(sample) for period in replacement.periods]):
        torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
        for x, y in zip(a.features, b.features):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    for old, new in zip(original.periods.parameters(), replacement.periods.parameters()):
        assert old.data_ptr() != new.data_ptr() and old.requires_grad == new.requires_grad
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    with torch.no_grad():
        next(replacement.periods.parameters()).add_(1)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, original_state[key], atol=0, rtol=0)
    assert original_flags == [(module.training, tuple(p.requires_grad for p in module.parameters(recurse=False)))
                              for module in original.modules()]


def test_replacement_only_consumes_initialization_randomness_for_new_spectral_heads():
    original, config = small_original(), small_complex_config()
    rng = torch.random.get_rng_state()
    replacement = ComplexAudioDiscriminators.from_existing(original, config)
    after_replacement = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    expected = [ComplexBandResolutionDiscriminator(size, config.mrd_channels, config.bands)
                for size in config.fft_sizes]
    assert torch.equal(torch.random.get_rng_state(), after_replacement)
    for head, reference in zip(replacement.resolutions, expected):
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
    # Serialized optimizer groups depend on parameter order even though model
    # state dict loading itself matches names.
    ordinary = ComplexAudioDiscriminators(config)
    assert list(dict(replacement.named_parameters())) == list(dict(ordinary.named_parameters()))


@pytest.mark.parametrize('level', [0., .00001, .03])
def test_complex_gan_losses_preserve_gradient_ownership_and_finite_quiet_gradients(level):
    bank = ComplexAudioDiscriminators(small_complex_config())
    prediction, teacher = audio(2, 256, level).requires_grad_(), audio(2, 256, level).requires_grad_()
    weights = torch.tensor([1., 3.])
    loss = discriminator_loss(bank, prediction, teacher, example_weights=weights)
    loss.backward()
    assert prediction.grad is None and teacher.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters())
    bank.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in bank.parameters()]
    state = {key: value.clone() for key, value in bank.state_dict().items()}
    losses = generator_losses(bank, prediction, teacher, example_weights=weights)
    assert [p.requires_grad for p in bank.parameters()] == flags
    (losses['adversarial'] + losses['feature_matching']).backward()
    assert torch.isfinite(prediction.grad).all()
    assert teacher.grad is None and all(p.grad is None for p in bank.parameters())
    for key, value in bank.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)


@pytest.mark.parametrize('kwargs', [
    {'bands': ((.1, 1.),)}, {'bands': ((0., .2), (.3, 1.))},
    {'bands': ((0., .3), (.2, 1.))}, {'bands': ((0., float('nan')), (.2, 1.))},
    {'bands': ((False, 1.),)}, {'bands': ((0., 1., 2.),)},
    {'fft_sizes': (8,)}, {'format_version': 2}, {'amplitude_preprocessing': 'peak'},
])
def test_invalid_complex_policies_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        ComplexDiscriminatorConfig(**kwargs)


def test_incompatible_mpd_replacement_and_invalid_audio_fail_explicitly():
    original = small_original()
    with pytest.raises(ValueError, match='preserve'):
        ComplexAudioDiscriminators.from_existing(original, replace(small_complex_config(), periods=(2,)))
    with pytest.raises(TypeError, match='original'):
        ComplexAudioDiscriminators.from_existing(ComplexAudioDiscriminators(small_complex_config()))
    with pytest.raises(TypeError, match='ComplexDiscriminatorConfig'):
        ComplexAudioDiscriminators(DiscriminatorConfig())
    with pytest.raises(ValueError, match='FP32'):
        ComplexAudioDiscriminators.from_existing(original.double(), small_complex_config())
    head = ComplexBandResolutionDiscriminator(64, 2)
    with pytest.raises(ValueError, match='shorter'):
        head(torch.zeros(1, 1, 63))
    with pytest.raises(ValueError, match='finite'):
        head(torch.full((1, 1, 64), float('nan')))
