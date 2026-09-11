"""Native stride2/kernel4 reconstruction keeps its phases, bias and causal state."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import native_upsampler_fit_v1 as native
from test_group_model import CausalTransposeConv1d, TinyDecoder, assert_snapshot, gm, selections, snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(712)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def response(x, weight, bias):
    return F.conv_transpose1d(x, weight, bias, stride=2)[..., :2 * x.shape[-1]]


def test_native_design_has_exact_current_previous_phase_order_and_zero_startup():
    x = torch.tensor([[[2., 4., 8.]]], dtype=torch.float64)
    expected = torch.tensor([[[2., 0., 4., 0., 8., 0.],
                              [0., 0., 2., 0., 4., 0.],
                              [0., 2., 0., 4., 0., 8.],
                              [0., 0., 0., 2., 0., 4.]]], dtype=torch.float64)
    torch.testing.assert_close(native.native_design(x), expected, rtol=0, atol=0)
    a = torch.tensor([[1., 10., 2., 20.]], dtype=torch.float64)
    weight = native.native_weight_from_affine(a, 1)
    torch.testing.assert_close(weight, torch.tensor([[[1., 2., 10., 20.]]], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(response(x, weight, torch.tensor([3.], dtype=torch.float64)),
                               torch.tensor([[[5., 7., 27., 51., 51., 99.]]], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("frames", [1, 3, 19])
def test_phase_design_equals_native_convolution_and_input_gradient(frames):
    x = torch.randn(2, 3, frames, dtype=torch.float64, requires_grad=True)
    a = torch.randn(5, 12, dtype=torch.float64)
    c = torch.randn(5, dtype=torch.float64)
    direct = response(x, native.native_weight_from_affine(a, 3), c)
    design = F.conv1d(native.native_design(x), a[..., None], c)
    torch.testing.assert_close(design, direct, rtol=1e-12, atol=1e-12)
    first = torch.autograd.grad(direct.square().sum(), x, retain_graph=True)[0]
    second = torch.autograd.grad(design.square().sum(), x)[0]
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_scored_outputs_keep_previous_input_from_unscored_context_and_partial_tail():
    x = torch.tensor([[[7., 11., 13., 17.]]], dtype=torch.float64)
    weight = torch.tensor([[[0., 0., 2., 3.]]], dtype=torch.float64)
    y = response(x, weight, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(y, torch.tensor([[[0., 0., 14., 21., 22., 33., 26., 39.]]], dtype=torch.float64), rtol=0, atol=0)
    valid = torch.zeros(1, 1, 32, dtype=torch.bool)
    valid[..., 8:29] = True
    stats = native.accumulate_native(None, x, y, valid)
    assert float(stats["n"]) == 21
    # Cell2 is scored, but depends on the preceding unscored frame x0=7.
    assert native.native_design(x)[0, 1, 2] == 7
    expected_target_sum = float((y.repeat_interleave(4, -1) * valid).sum())
    torch.testing.assert_close(stats["d_sum"], torch.tensor([expected_target_sum], dtype=torch.float64))
    assert float(stats["x_sum"][1]) == 4 * 7 + 4 * 11 + 4 * 13
    assert float(stats["x_sum"][3]) == 4 * 7 + 4 * 11 + 1 * 13


def test_closed_form_recovers_known_native_weights_and_one_bias_on_heldout_inputs():
    weight, bias = torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
    x = torch.randn(3, 2, 137, dtype=torch.float64)
    target = response(x, weight, bias)
    valid = torch.ones(3, 1, 137 * 8, dtype=torch.bool)
    valid[0, :, :64] = False
    valid[1, :, -11:] = False
    valid[2, :, :37] = False
    stats = native.accumulate_native(None, x, target, valid)
    fitted_w, fitted_b, report = native.fit_native(stats, 2)
    torch.testing.assert_close(fitted_w, weight, rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(fitted_b, bias, rtol=5e-5, atol=5e-6)
    heldout = torch.randn(1, 2, 47, dtype=torch.float64)
    torch.testing.assert_close(response(heldout, fitted_w, fitted_b), response(heldout, weight, bias), rtol=1e-4, atol=1e-5)
    assert fitted_b.shape == (3,)
    assert report["weighted_sample_count"] == int(valid.sum())
    split = None
    for i in range(3):
        split = native.accumulate_native(split, x[i:i + 1], target[i:i + 1], valid[i:i + 1])
    sw, sb, _ = native.fit_native(split, 2)
    torch.testing.assert_close(sw, fitted_w, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(sb, fitted_b, rtol=1e-10, atol=1e-10)


def test_shared_bias_cannot_cheat_by_learning_two_independent_phase_offsets():
    x = torch.zeros(1, 2, 23, dtype=torch.float64)
    target = torch.tensor([1., 3.], dtype=torch.float64).repeat(23).reshape(1, 1, -1)
    valid = torch.ones(1, 1, 23 * 8, dtype=torch.bool)
    weight, bias, _ = native.fit_native(native.accumulate_native(None, x, target, valid), 2)
    assert bias.shape == (1,)
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    torch.testing.assert_close(bias, torch.tensor([2.], dtype=torch.float64), rtol=0, atol=0)
    predicted = response(x, weight, bias)
    assert float((predicted - target).square().mean()) == 1.
    # A phase-specific intercept would achieve zero error, but is not native.
    assert torch.equal(predicted[..., ::2], predicted[..., 1::2])


def test_fit_fold_uses_existing_weight_norm_and_restores_after_exception():
    module = weight_norm(CausalTransposeConv1d(3, 5, 4, stride=2, padding=1))
    original = snapshot(module)
    objects = {k: id(p) for k, p in module.named_parameters()}
    hooks = dict(module._forward_pre_hooks)
    desired = torch.randn(3, 5, 4) * .13
    bias = torch.randn(5) * .07
    x = torch.randn(1, 3, 13)
    with pytest.raises(RuntimeError, match="sentinel"):
        with native.temporary_native_fit(module, desired, bias):
            torch.testing.assert_close(gm.effective_weight(module), desired, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(module(x), response(x, desired, bias), rtol=1e-5, atol=1e-6)
            assert {k: id(p) for k, p in module.named_parameters()} == objects
            assert dict(module._forward_pre_hooks) == hooks
            raise RuntimeError("sentinel")
    assert_snapshot(module, original)
    assert {k: id(p) for k, p in module.named_parameters()} == objects
    assert dict(module._forward_pre_hooks) == hooks


def test_candidate_native_fit_and_original_residual_control_are_isolated_and_restore():
    teacher = TinyDecoder().eval().requires_grad_(False)
    candidate = gm.build_student(teacher, *selections())
    teacher_before = snapshot(teacher)
    with torch.no_grad():
        for index in (2, 3, 4):
            for p in candidate.decoder.model[5].block[index].parameters():
                p.add_(.023)
    candidate_before = snapshot(candidate)
    module = candidate.decoder.model[5].block[1]
    fitted_w = torch.randn_like(gm.effective_weight(module)) * .1
    fitted_b = torch.randn_like(module.bias) * .05
    native.apply_native_fit(module, fitted_w, fitted_b)
    fitted = snapshot(candidate)
    changed = {key for key in fitted if not torch.equal(fitted[key], candidate_before[key])}
    assert changed and all(key.startswith("decoder.model.5.block.1.") for key in changed)
    assert set(fitted) == set(candidate_before)
    assert_snapshot(teacher, teacher_before)
    with pytest.raises(RuntimeError, match="sentinel"):
        with native.temporary_original_residuals(candidate.decoder, teacher):
            for index in (2, 3, 4):
                assert_snapshot(candidate.decoder.model[5].block[index], snapshot(teacher.model[5].block[index]))
            inside = snapshot(candidate)
            altered = {key for key in inside if not torch.equal(inside[key], fitted[key])}
            assert altered and all(key.startswith(tuple(f"decoder.model.5.block.{i}." for i in (2, 3, 4))) for key in altered)
            torch.testing.assert_close(gm.effective_weight(module), fitted_w, rtol=1e-6, atol=1e-7)
            assert_snapshot(teacher, teacher_before)
            raise RuntimeError("sentinel")
    assert_snapshot(candidate, fitted)
    assert_snapshot(teacher, teacher_before)


def test_native_fit_rejects_wrong_grid_and_kernel_shape():
    x = torch.zeros(1, 2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        native.accumulate_native(None, x, torch.zeros(1, 4, 5), torch.ones(1, 1, 24, dtype=torch.bool))
    with pytest.raises(ValueError):
        native.native_weight_from_affine(torch.zeros(3, 7), 2)
