"""CPU mathematical contracts for the read-only channel attribution audit."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
import diagnose_channel_contributions as audit
from group_model import assign_effective_weight, effective_weight
from test_group_model import CausalTransposeConv1d, CausalResidualUnit


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(386)
    yield
    torch.random.set_rng_state(rng)
    torch.set_num_threads(threads)


def convolution(transpose=False, stride=3, *, inputs=5, outputs=4):
    if transpose:
        result = CausalTransposeConv1d(
            inputs, outputs, 2 * stride, stride=stride,
            padding=(stride + 1) // 2, output_padding=stride % 2,
        )
    else:
        result = nn.Conv1d(inputs, outputs, 1)
    result = weight_norm(result).double()
    with torch.no_grad():
        result.weight_v.copy_(torch.randn_like(result.weight_v))
        result.weight_g.copy_(torch.rand_like(result.weight_g) + .2)
        result.bias.copy_(torch.arange(outputs, dtype=torch.float64) + .7)
    return result


@pytest.mark.parametrize("transpose,stride", [(False, 1), (True, 2), (True, 3), (True, 5)])
def test_linear_response_matches_native_axes_taps_trim_and_single_bias(transpose, stride):
    module = convolution(transpose, stride)
    x = torch.randn(1, 5, 7, dtype=torch.float64)
    # Read an updated legacy-WN weight without relying on a preceding forward.
    w = effective_weight(module).detach().clone()
    actual = audit.linear_response(module, x, include_bias=True)
    expected = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.shape[-1] == x.shape[-1] * (stride if transpose else 1)
    no_bias = audit.linear_response(module, x, weight=w, include_bias=False)
    torch.testing.assert_close(actual - no_bias, module.bias.view(1, -1, 1).expand_as(actual),
                               rtol=1e-13, atol=1e-13)
    input_axis = 0 if transpose else 1
    keep = torch.tensor([0, 3, 4])
    drop = torch.tensor([1, 2])
    wk, wd = torch.zeros_like(w), torch.zeros_like(w)
    wk.index_copy_(input_axis, keep, w.index_select(input_axis, keep))
    wd.index_copy_(input_axis, drop, w.index_select(input_axis, drop))
    reconstructed = audit.linear_response(module, x, weight=wk, include_bias=True)
    reconstructed += audit.linear_response(module, x, weight=wd, include_bias=False)
    torch.testing.assert_close(reconstructed, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("transpose", [False, True])
def test_decomposition_separates_removed_contribution_from_retained_drift(transpose):
    teacher = convolution(transpose, 3)
    student = convolution(transpose, 3, inputs=3, outputs=2)
    keep_in, keep_out = torch.tensor([0, 3, 4]), torch.tensor([1, 3])
    w = effective_weight(teacher).detach()
    selected = w.index_select(0, keep_in).index_select(1, keep_out) if transpose else (
        w.index_select(0, keep_out).index_select(1, keep_in))
    assign_effective_weight(student, selected, teacher.bias.index_select(0, keep_out))
    tx = torch.randn(1, 5, 6, dtype=torch.float64)
    sx = tx.index_select(1, keep_in).clone()
    ty, sy = teacher(tx), student(sx)
    terms = audit.operation_decomposition(teacher, student, tx, sx, ty, sy, keep_in, keep_out)
    torch.testing.assert_close(terms["kept"] + terms["dropped"] + terms["bias"],
                               ty.index_select(1, keep_out), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(terms["retained_drift"], torch.zeros_like(sy), rtol=0, atol=1e-12)
    torch.testing.assert_close(terms["gap"], terms["dropped"], rtol=1e-12, atol=1e-12)
    changed_x = sx + .19
    changed_y = student(changed_x)
    changed = audit.operation_decomposition(
        teacher, student, tx, changed_x, ty, changed_y, keep_in, keep_out)
    torch.testing.assert_close(changed["dropped"], terms["dropped"], rtol=0, atol=0)
    torch.testing.assert_close(changed["gap"], ty.index_select(1, keep_out) - changed_y,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(changed["reconstructed_gap"], changed["gap"], rtol=1e-12, atol=1e-12)
    assert changed["retained_drift"].abs().max() > .01
    if transpose:
        torch.testing.assert_close(changed["dropped_current"] + changed["dropped_previous"],
                                   changed["dropped"], rtol=1e-12, atol=1e-12)
        # No contribution from a preceding input frame exists at startup.
        assert torch.equal(changed["dropped_previous"][..., :3],
                           torch.zeros_like(changed["dropped_previous"][..., :3]))
    with torch.no_grad():
        student.bias.add_(.13)
    with pytest.raises(ValueError, match="original selected bias"):
        audit.operation_decomposition(teacher, student, tx, sx, ty, student(sx), keep_in, keep_out)


def test_transpose_current_previous_taps_have_native_phase_and_startup_meaning():
    teacher = convolution(True, 3, inputs=2, outputs=1)
    student = convolution(True, 3, inputs=1, outputs=1)
    weight = torch.zeros(2, 1, 6, dtype=torch.float64)
    weight[1, 0] = torch.tensor([1., 2., 3., 10., 20., 30.])
    assign_effective_weight(teacher, weight, torch.tensor([.7], dtype=torch.float64))
    assign_effective_weight(student, weight[:1], torch.tensor([.7], dtype=torch.float64))
    tx = torch.tensor([[[0., 0.], [2., 4.]]], dtype=torch.float64)
    sx = tx[:, :1]
    result = audit.operation_decomposition(teacher, student, tx, sx, teacher(tx), student(sx), [0], [0])
    torch.testing.assert_close(result["dropped_current"], torch.tensor([[[2., 4., 6., 4., 8., 12.]]], dtype=torch.float64))
    torch.testing.assert_close(result["dropped_previous"], torch.tensor([[[0., 0., 0., 20., 40., 60.]]], dtype=torch.float64))
    torch.testing.assert_close(result["dropped"], torch.tensor([[[2., 4., 6., 24., 48., 72.]]], dtype=torch.float64))
    assert result["gap"].shape[-1] == 6  # The final pending overlap is trimmed.


def sufficient_stats(x, d):
    """Independent sample-row sufficient-statistic oracle."""
    return {"n": x.shape[0], "x_sum": x.sum(0), "xx_sum": x.T @ x,
            "d_sum": d.sum(0), "xd_sum": x.T @ d}


def test_affine_fit_recovers_linear_map_and_intercept_on_unseen_inputs():
    x = torch.randn(80, 3, dtype=torch.float64)
    matrix = torch.tensor([[.8, -.4, .2], [-.3, .7, 1.1]], dtype=torch.float64)
    bias = torch.tensor([.23, -.42], dtype=torch.float64)
    target = x @ matrix.T + bias
    a, c, receipt = audit.fit_affine(sufficient_stats(x, target), ridge=1e-10)
    unseen = torch.randn(1, 3, 29, dtype=torch.float64)
    expected = torch.einsum("oc,bct->bot", matrix, unseen) + bias.view(1, -1, 1)
    actual = audit.affine_response(a, c, unseen)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
    assert receipt


def test_affine_fit_constant_coordinates_is_finite_and_intercept_unpenalized():
    x = torch.zeros(40, 3, dtype=torch.float64)
    x[:, 1] = 7
    d = torch.tensor([.75, -.125], dtype=torch.float64).expand(40, 2).clone()
    a, c, _ = audit.fit_affine(sufficient_stats(x, d), ridge=1e-6)
    assert torch.isfinite(a).all() and torch.isfinite(c).all()
    prediction = audit.affine_response(a, c, x.T.unsqueeze(0))
    torch.testing.assert_close(prediction, d.T.unsqueeze(0), rtol=0, atol=1e-12)


def test_folded_affine_correction_keeps_native_pointwise_shape_and_one_bias():
    module = convolution(inputs=3, outputs=2)
    x = torch.randn(1, 3, 19, dtype=torch.float64)
    a = torch.randn(2, 3, dtype=torch.float64) * .1
    c = torch.tensor([.8, -.2], dtype=torch.float64)
    before = module(x).detach().clone()
    shapes = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    audit.fold_affine_correction(module, a, c)
    expected = before + torch.einsum("oc,bct->bot", a, x) + c.view(1, -1, 1)
    torch.testing.assert_close(module(x), expected, rtol=1e-12, atol=1e-12)
    assert shapes == {k: tuple(v.shape) for k, v in module.state_dict().items()}


def test_affine_fit_rejects_empty_statistics():
    with pytest.raises((ValueError, RuntimeError)):
        audit.fit_affine(sufficient_stats(torch.empty(0, 3), torch.empty(0, 2)))


def test_fit_rejects_roundoff_negative_covariance_when_ridge_cannot_make_it_positive():
    # A negative eigenvalue can fit the absolute roundoff tolerance but exceed
    # the tiny trace-scaled ridge. Solving an indefinite normal matrix would
    # produce an invalid negative condition estimate and a misleading fit.
    stats = {"n": 1, "x_sum": torch.zeros(2, dtype=torch.float64),
             "xx_sum": torch.diag(torch.tensor([-1e-16, 1e-10], dtype=torch.float64)),
             "d_sum": torch.zeros(1, dtype=torch.float64),
             "xd_sum": torch.ones(2, 1, dtype=torch.float64) * 1e-12}
    with pytest.raises((ValueError, RuntimeError), match="positive|indefinite|regular"):
        audit.fit_affine(stats)


def test_cell_weights_and_affine_statistics_include_partial_tail_exclude_context():
    # A source has two context cells, one complete scored cell, one partial
    # scored tail, then padding. Each stage2 cell covers40 waveform samples.
    valid = torch.zeros(1, 1, 200, dtype=torch.bool)
    valid[..., 80:137] = True
    weights = audit.cell_weights(valid, 5)
    assert weights.tolist() == [[[0, 0, 40, 17, 0]]]
    x = torch.tensor([[[float("nan"), float("nan"), 2., 7., float("nan")],
                       [float("nan"), float("nan"), 3., -1., float("nan")]]], dtype=torch.float64)
    d = torch.tensor([[[float("nan"), float("nan"), 5., 9., float("nan")]]], dtype=torch.float64)
    stats = audit.accumulate_affine(None, x, d, weights)
    observed_x = torch.cat([x[0, :, 2].expand(40, 2), x[0, :, 3].expand(17, 2)])
    observed_d = torch.cat([d[0, :, 2].expand(40, 1), d[0, :, 3].expand(17, 1)])
    oracle = sufficient_stats(observed_x, observed_d)
    for name in oracle:
        torch.testing.assert_close(torch.as_tensor(stats[name]), torch.as_tensor(oracle[name], dtype=torch.float64))


def test_affine_sample_pooling_is_invariant_to_source_partition():
    x = torch.randn(1, 3, 13, dtype=torch.float64)
    d = torch.randn(1, 2, 13, dtype=torch.float64)
    weights = torch.tensor([[[0, 4, 40, 17, 1, 0, 5, 30, 0, 40, 2, 0, 1]]])
    together = audit.accumulate_affine(None, x, d, weights)
    parts = audit.accumulate_affine(None, x[..., :6], d[..., :6], weights[..., :6])
    parts = audit.accumulate_affine(parts, x[..., 6:], d[..., 6:], weights[..., 6:])
    for key in together:
        torch.testing.assert_close(together[key], parts[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("left,right", [([], ["b"]), (["a"], []), (["a", "a"], ["b"]),
                                         (["a"], ["b", "b"]), (["a"], ["a"])])
def test_affine_fit_sources_reject_empty_duplicates_and_holdout_overlap(left, right):
    with pytest.raises(ValueError, match="unique and disjoint"):
        audit.validate_fit_sources([{"source_id": i} for i in left], [{"source_id": i} for i in right])


def test_affine_fit_records_exact_ordered_source_identities():
    fit = [{"source_id": f"fit-{i}"} for i in range(72)]
    development = [{"source_id": f"dev-{i}"} for i in range(96)]
    receipt = audit.validate_fit_sources(fit, development)
    assert receipt["fit_source_ids"] == [v["source_id"] for v in fit]
    assert receipt["development_source_ids"] == [v["source_id"] for v in development]
    assert receipt["fit_sha256"] != receipt["development_sha256"]


def test_invalid_mask_geometry_does_not_drop_a_tail():
    with pytest.raises(ValueError, match="geometry"):
        audit.cell_weights(torch.ones(1, 1, 201, dtype=torch.bool), 5)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_teacher_ablation_preserves_target_unselected_channels_and_removes_hook(raise_inside):
    teacher = convolution(inputs=3, outputs=4)
    x = torch.randn(1, 3, 11, dtype=torch.float64)
    target = teacher(x).detach().clone()
    original_target = target.clone()
    missing = torch.full((1, 2, 11), .17, dtype=torch.float64)
    original_hooks = set(teacher._forward_hooks)
    try:
        with audit.temporary_output_change(teacher, lambda out: audit.ablate_kept_output(out, missing, [1, 3])):
            changed = teacher(x)
            torch.testing.assert_close(changed[:, [0, 2]], target[:, [0, 2]], rtol=0, atol=0)
            torch.testing.assert_close(changed[:, [1, 3]], target[:, [1, 3]] - missing, rtol=0, atol=0)
            if raise_inside:
                raise RuntimeError("diagnostic interrupted")
    except RuntimeError as exc:
        assert raise_inside and str(exc) == "diagnostic interrupted"
    assert set(teacher._forward_hooks) == original_hooks
    torch.testing.assert_close(teacher(x), target, rtol=0, atol=0)
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
    directly_changed = audit.ablate_kept_output(target, missing, [1, 3])
    assert directly_changed.data_ptr() != target.data_ptr()
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)


def test_first_residual_oracle_restores_teacher_coordinates_before_one_skip():
    teacher = CausalResidualUnit(4, 1).double().eval()
    student = CausalResidualUnit(2, 1).double().eval()
    keep = torch.tensor([1, 3])
    for index in [0, 2]:
        audit.group._copy_snake(teacher.block[index], student.block[index], keep)
    for index in [1, 3]:
        audit.group._copy_conv(teacher.block[index], student.block[index], keep, keep)
    x = torch.randn(1, 4, 31, dtype=torch.float64)
    sx = x[:, keep]
    teacher_input = teacher.block[:3](x)
    student_input = student.block[:3](sx)
    terms = audit.operation_decomposition(teacher.block[3], student.block[3],
        teacher_input, student_input, teacher.block[3](teacher_input), student.block[3](student_input), keep, keep)
    original = student(sx).detach().clone()
    with audit.temporary_output_change(student.block[3], lambda output: output + terms["dropped"]):
        restored = student(sx)
    torch.testing.assert_close(restored, teacher(x)[:, keep], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(student(sx), original, rtol=0, atol=0)


def test_diagnostic_masks_do_not_add_phantom_rows_to_validation_observer(monkeypatch):
    calls = []
    original = audit.base.quiet_window_metrics
    def validation_observer(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(audit.base, "quiet_window_metrics", validation_observer)
    target = torch.zeros(1, 1, 3840)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[..., 1920:3827] = True
    crop = {"context_frames": 1, "context_start_frame": 0, "valid_scored_samples": 1907}
    masks = audit.region_masks(target, valid, crop)
    assert calls == []
    assert int(masks["all"].sum()) == 1907
    assert int(masks["quiet"].sum()) == 1907
    assert int(masks["near_silence"].sum()) == 1907
    assert not masks["active"].any()
    assert not masks["startup_40ms"].any()  # Startup was context, not a scored sample.
    for mask in masks.values():
        assert not (mask & ~valid).any()


def test_centered_affine_statistics_preserve_small_variation_on_large_offset():
    values = torch.linspace(-.02, .02, 80, dtype=torch.float64)
    x = (values + 1e6).view(1, 1, -1)
    d = (values * 2.3 + .125).view(1, 1, -1)
    stats = audit.accumulate_affine(None, x[..., :29], d[..., :29], torch.ones(1, 1, 29))
    stats = audit.accumulate_affine(stats, x[..., 29:], d[..., 29:], torch.ones(1, 1, 51))
    a, c, receipt = audit.fit_affine(stats)
    assert receipt["statistics"] == "weighted centered Chan merge"
    actual = audit.affine_response(a, c, x)
    torch.testing.assert_close(actual, d, rtol=0, atol=1e-7)
    assert receipt["effective_rank"] == 1


def test_additive_energy_measures_signed_cancellation_and_counts_bias_once():
    kept = torch.tensor([[[3., 4., float("nan")]]], dtype=torch.float64)
    bias = torch.tensor([[[1., 1., float("nan")]]], dtype=torch.float64)
    dropped = torch.tensor([[[-2., -4., float("nan")]]], dtype=torch.float64)
    terms = {"kept": kept, "bias": bias, "dropped": dropped,
             "teacher_selected": kept + bias + dropped}
    weights = torch.tensor([[[40, 17, 0]]])
    result = audit.additive_energy(terms, weights)
    expected_cross = (-8 * 40 - 20 * 17) / 57
    expected_energy = (4 * 40 + 1 * 17) / 57
    assert result["weighted_elements"] == 57
    assert result["cross_mean_product"] == pytest.approx(expected_cross)
    assert result["cross_mean_product"] < 0
    assert result["teacher_mean_square"] == pytest.approx(expected_energy)
    assert result["reconstructed_mean_square"] == pytest.approx(expected_energy)
    assert result["energy_identity_absolute_error"] < 1e-13
    expanded = sum(result[k] for k in ("kept_mean_square", "bias_mean_square", "dropped_mean_square",
        "twice_kept_dropped_cross", "twice_kept_bias_cross", "twice_dropped_bias_cross"))
    assert expanded == pytest.approx(expected_energy)


def test_narrow_warmup_is_per_model_and_shape_and_has_no_gradients():
    class Counter:
        def __init__(self):
            self.group_calls = 0
            self.suffix_calls = 0
        def group_from_input(self, x):
            assert not torch.is_grad_enabled()
            self.group_calls += 1
            return x
        def suffix_from_group(self, h):
            self.suffix_calls += 1
            return h
    original, fitted = Counter(), Counter()
    x = torch.zeros(1, 4, 11)
    audit.warm_student(original, x)
    audit.warm_student(original, x + 1)
    assert original.group_calls == original.suffix_calls == 3
    audit.warm_student(fitted, x)
    assert fitted.group_calls == fitted.suffix_calls == 3
    audit.warm_student(original, x[..., :-1])
    assert original.group_calls == original.suffix_calls == 6
