from copy import deepcopy

import pytest
import torch

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.normalization_calibration import _Moments, calibrate_normalization


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_and_batch(adapter_mode="learned_phase"):
    torch.manual_seed(43)
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
        dilations=(1, 2), normalization_mode="masked_batch_norm", layer_scale_init=.3,
        adapter_mode=adapter_mode)).double()
    latents = torch.randn(4, 64, 10, dtype=torch.float64)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    for index, (start, stop) in enumerate(((2, 10), (3, 8), (1, 5), (0, 3))):
        mask[index, start:stop] = True
    return model, latents, mask


def state(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def assert_state(model, expected):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, equal_nan=True)


def selected_moments(features, mask):
    values = features.transpose(1, 2)[mask.repeat_interleave(4, dim=1)]
    return values.mean(dim=0), values.var(dim=0, unbiased=False)


@pytest.mark.parametrize("adapter_mode", ["learned_phase", "raw_repeat_phase_bias"])
def test_sequential_population_moments_and_exact_weight_gradient_mode_preservation(adapter_mode):
    model, latents, mask = model_and_batch(adapter_mode)
    with torch.no_grad():
        model.stem_norm.running_mean.fill_(7)
        model.stem_norm.running_var.fill_(11)
        model.stem_norm.weight.copy_(torch.tensor([.6, 1, 1.3, 2]))
        model.affine.running_mean.fill_(-9)
        model.affine.num_batches_tracked.fill_(111)
    model.train()
    model.blocks[0].eval()  # Preserve mixed per-module flags, not just root mode.
    modes = [module.training for module in model.modules()]
    weights = {name: parameter.clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .123)
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    latents.requires_grad_()
    with torch.no_grad():
        stem_features = model.stem(model._phase_frames(latents))
        mean, variance = selected_moments(stem_features, mask)
    report = calibrate_normalization(model, [(latents, mask)], provenance={"split": "train", "manifest": "fixture"})
    torch.testing.assert_close(model.stem_norm.running_mean, mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.stem_norm.running_var, variance, rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        features = model.stem_norm.fixed(stem_features)
        for block in model.blocks:
            features = block(features)
        final_mean, final_variance = selected_moments(features, mask)
    torch.testing.assert_close(model.affine.running_mean, final_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.affine.running_var, final_variance, rtol=1e-12, atol=1e-12)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert latents.grad is None
    assert modes == [module.training for module in model.modules()]
    assert report["stem"]["internal_frame_count"] == 4 * int(mask.sum())
    assert report["stem"]["inputs_sha256"] == report["final"]["inputs_sha256"]
    assert report["model_state_before_sha256"] != report["model_state_after_sha256"]
    assert report["parameters_unchanged"]
    assert bool(model.stem_norm.statistics_frozen) and bool(model.affine.statistics_frozen)
    assert model.stem_norm.num_batches_tracked == model.affine.num_batches_tracked == 0


def test_sample_weighted_moments_do_not_depend_on_batch_partition():
    whole, latents, mask = model_and_batch()
    partitioned = deepcopy(whole)
    first = calibrate_normalization(whole, [(latents, mask)], provenance={"split": "train"})
    second = calibrate_normalization(partitioned,
        [(latents[:1], mask[:1]), (latents[1:3], mask[1:3]), (latents[3:], mask[3:])],
        provenance={"split": "train"})
    assert first["stem"]["inputs_sha256"] == second["stem"]["inputs_sha256"]
    for name in ("stem_norm", "affine"):
        a, b = getattr(whole, name), getattr(partitioned, name)
        torch.testing.assert_close(a.running_mean, b.running_mean, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(a.running_var, b.running_var, rtol=1e-10, atol=1e-12)


def test_excluded_nan_right_padding_does_not_poison_moments_and_context_is_retained():
    clean, latents, mask = model_and_batch()
    padded = deepcopy(clean)
    padded_latents = torch.cat((latents, torch.full((4, 64, 5), float("nan"), dtype=latents.dtype)), dim=-1)
    padded_mask = torch.cat((mask, torch.zeros(4, 5, dtype=torch.bool)), dim=-1)
    calibrate_normalization(clean, [(latents, mask)], provenance={"split": "train"})
    calibrate_normalization(padded, [(padded_latents, padded_mask)], provenance={"split": "train"})
    for name in ("stem_norm", "affine"):
        a, b = getattr(clean, name), getattr(padded, name)
        torch.testing.assert_close(a.running_mean, b.running_mean, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(a.running_var, b.running_var, rtol=1e-10, atol=1e-12)
    with torch.no_grad():
        actual = clean.stem(clean._phase_frames(latents))
        removed_context = clean.stem(clean._phase_frames(torch.where(mask[:, None], latents, 0)))
    selected = mask.repeat_interleave(4, dim=1)
    assert not torch.allclose(actual.transpose(1, 2)[selected], removed_context.transpose(1, 2)[selected])


@pytest.mark.parametrize("failure", ["different_replay", "raises_after_stem", "changes_weight", "nonfinite_final_input"])
def test_failure_after_stem_calibration_rolls_back_all_state_and_modes(failure):
    model, latents, mask = model_and_batch()
    initial = state(model)
    model.train()
    model.affine.eval()
    modes = [module.training for module in model.modules()]
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 2:
            assert bool(model.stem_norm.statistics_frozen)
            if failure == "raises_after_stem":
                raise RuntimeError("simulated source read failure")
            if failure == "changes_weight":
                with torch.no_grad():
                    model.output.weight.add_(1)
            if failure == "different_replay":
                return [(latents.flip(0), mask.flip(0))]
            if failure == "nonfinite_final_input":
                broken = latents.clone()
                broken[0, :, 2] = float("nan")
                return [(broken, mask)]
        return [(latents, mask)]

    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        calibrate_normalization(model, factory, provenance={"split": "train"})
    assert_state(model, initial)
    assert modes == [module.training for module in model.modules()]


def test_rejects_insufficient_counts_heldout_provenance_and_one_shot_iterator():
    model, latents, mask = model_and_batch()
    initial = state(model)
    for bad in ({"split": "dev"}, {}, {"split": "test"}):
        with pytest.raises(ValueError, match="training-only"):
            calibrate_normalization(model, [(latents, mask)], provenance=bad)
    with pytest.raises(ValueError, match="at least"):
        calibrate_normalization(model, [(latents, torch.zeros_like(mask))], provenance={"split": "train"})
    with pytest.raises(ValueError, match="repeatable"):
        calibrate_normalization(model, iter([(latents, mask)]), provenance={"split": "train"})
    assert_state(model, initial)


def test_fp64_centered_moments_remain_stable_for_large_channel_offsets():
    values = torch.tensor([1e12 - .25, 1e12, 1e12 + .25, 1e12 + .5], dtype=torch.float64)
    accumulator = _Moments()
    accumulator.add(values[:2].reshape(1, 1, -1), torch.ones(1, 2, dtype=torch.bool))
    accumulator.add(values[2:].reshape(1, 1, -1), torch.ones(1, 2, dtype=torch.bool))
    mean, variance = accumulator.population(2)
    torch.testing.assert_close(mean[0], values.mean(), rtol=0, atol=0)
    torch.testing.assert_close(variance[0], values.var(unbiased=False), rtol=0, atol=0)
