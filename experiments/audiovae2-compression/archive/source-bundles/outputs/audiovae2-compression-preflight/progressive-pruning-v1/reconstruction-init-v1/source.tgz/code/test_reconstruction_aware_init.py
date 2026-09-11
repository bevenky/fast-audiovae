"""Native and sequential reconstruction fits, with independent numerical oracles."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import reconstruction_aware_init as fit
import diagnose_progressive_silence as diagnostic
import progressive_model as progressive
from test_group_model import CausalConv1d, CausalTransposeConv1d, TinyDecoder, gm, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(985)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def native(x, weight, bias):
    return F.conv_transpose1d(x, weight, bias, stride=5)[..., :x.shape[-1] * 5]


def upsampler(cin=2, cout=3):
    return weight_norm(CausalTransposeConv1d(cin, cout, 10, stride=5, padding=3, output_padding=1)).double()


@pytest.mark.parametrize('frames', [1, 3])
def test_five_phase_design_matches_native_taps_trim_and_global_slice(frames):
    x = torch.randn(2, 2, frames, dtype=torch.float64)
    weight = torch.randn(2, 3, 10, dtype=torch.float64)
    bias = torch.tensor([.2, -.3, .7], dtype=torch.float64)
    affine = fit.native_affine_from_weight(weight)
    torch.testing.assert_close(fit.native_weight_from_affine(affine, 2), weight, rtol=0, atol=0)
    expected = native(x, weight, bias)
    actual = F.conv1d(fit.native_design(x), affine[..., None], bias)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for start, stop in ((0, 1), (1, frames * 5), (frames * 5 - 1, frames * 5)):
        chunk = F.conv1d(fit.native_design(x, start=start, stop=stop), affine[..., None], bias)
        torch.testing.assert_close(chunk, expected[..., start:stop], rtol=1e-12, atol=1e-12)


def test_startup_previous_frame_is_zero_but_scored_context_retains_real_previous_input():
    x = torch.tensor([[[7., 11., 13.]]], dtype=torch.float64)
    weight = torch.zeros(1, 1, 10, dtype=torch.float64)
    weight[..., 5:] = torch.arange(1, 6, dtype=torch.float64)
    a = fit.native_affine_from_weight(weight)
    expected = torch.tensor([[[0., 0., 0., 0., 0., 7., 14., 21., 28., 35., 11., 22., 33., 44., 55.]]], dtype=torch.float64)
    actual = F.conv1d(fit.native_design(x), a[..., None])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A scored slice beginning at frame1 still reads frame0 from its context.
    torch.testing.assert_close(F.conv1d(fit.native_design(x, start=5, stop=8), a[..., None]),
                               expected[..., 5:8], rtol=0, atol=0)


def test_native_delta_fit_recovers_operator_and_shared_bias_on_heldout_inputs():
    module = upsampler()
    x = torch.randn(2, 2, 103, dtype=torch.float64)
    desired_w = torch.randn(2, 3, 10, dtype=torch.float64) * .1
    desired_b = torch.tensor([.023, -.072, .14], dtype=torch.float64)
    valid = torch.ones(2, 1, 103 * 40, dtype=torch.bool)
    valid[0, :, :47] = False
    valid[1, :, -13:] = False
    weights = valid.reshape(2, 1, -1, 8).sum(-1)
    observation = {'input': x, 'target': native(x, desired_w, desired_b),
                   'current_output': module(x).detach(), 'weights': weights}
    before = snapshot(module)
    dw, db, report = fit.fit_delta(module, [observation], chunk_rows=31)
    assert_snapshot(module, before)
    expected_before = ((observation['current_output'] - observation['target']).square() * weights).sum() / (weights.sum() * 3)
    assert report['weighted_error_before'] == pytest.approx(float(expected_before), rel=1e-12)
    fit.apply_delta(module, dw, db)
    measured_after = ((module(x).detach() - observation['target']).square() * weights).sum() / (weights.sum() * 3)
    assert report['weighted_error_after'] == pytest.approx(float(measured_after), rel=1e-3, abs=1e-12)
    heldout = torch.randn(1, 2, 31, dtype=torch.float64)
    torch.testing.assert_close(module(heldout), native(heldout, desired_w, desired_b), rtol=1e-4, atol=1e-5)
    assert db.shape == (3,)
    assert report['weighted_sample_count'] == int(valid.sum())


def test_shared_bias_cannot_fit_independent_five_phase_offsets():
    module = upsampler(1, 1)
    x = torch.zeros(1, 1, 20, dtype=torch.float64)
    target = torch.arange(1, 6, dtype=torch.float64).repeat(20).reshape(1, 1, -1)
    observation = {'input': x, 'target': target, 'current_output': module(x).detach(),
                   'weights': torch.full((1, 1, 100), 8)}
    dw, db, report = fit.fit_delta(module, [observation])
    assert report['intercept_only'] and db.shape == (1,)
    fit.apply_delta(module, dw, db)
    torch.testing.assert_close(module(x), torch.full_like(target, 3.), rtol=0, atol=1e-12)
    assert float((module(x).detach() - target).square().mean()) == pytest.approx(2.)


def test_residual_target_removes_current_student_skip_not_teacher_skip_or_bias():
    teacher_skip = torch.tensor([[[2., 3.], [7., 9.], [-4., -6.]]])
    teacher_branch = torch.tensor([[[.2, .3], [.7, .9], [-.4, -.6]]])
    student_skip = torch.tensor([[[1., 1.5], [-2., -3.]]])
    full = teacher_skip + teacher_branch
    target = fit.residual_target(full, student_skip, [0, 2])
    torch.testing.assert_close(target + student_skip, full[:, [0, 2]].double(), rtol=0, atol=0)
    assert not torch.allclose(target, teacher_branch[:, [0, 2]].double())


def test_invalid_tail_observations_are_excluded_and_large_offset_fit_is_stable():
    module = weight_norm(CausalConv1d(2, 1, 1)).double()
    x = 1e6 + torch.randn(1, 2, 301, dtype=torch.float64)
    w = torch.tensor([[[.4], [-.2]]], dtype=torch.float64)
    b = torch.tensor([.03], dtype=torch.float64)
    current = module(x).detach(); target = F.conv1d(x, w, b)
    weights = torch.full((1, 1, 301), 40)
    weights[..., 0] = 0; weights[..., -1] = 7
    x[..., 0] = torch.nan; current[..., 0] = torch.nan; target[..., 0] = torch.nan
    observation = {'input': x, 'target': target, 'current_output': current, 'weights': weights}
    dw, db, report = fit.fit_delta(module, [observation], chunk_rows=29)
    assert torch.isfinite(dw).all() and torch.isfinite(db).all()
    assert report['weighted_sample_count'] == 299 * 40 + 7
    assert report['normal_equation_relative_residual'] < 1e-7
    fit.apply_delta(module, dw, db)
    torch.testing.assert_close(module(x[..., 1:]), target[..., 1:], rtol=1e-8, atol=1e-4)


def test_delta_writeback_preserves_native_parameter_objects_and_other_models():
    teacher = TinyDecoder().eval().requires_grad_(False)
    selected = {'stage2_indices': [i for i in range(32) if i % 4 != 1], 'stage3_indices': list(range(16))}
    candidate = progressive.initialize_from_teacher(teacher, selected)
    saved_model = progressive.initialize_from_teacher(teacher, selected)
    before_teacher, before_saved, before = snapshot(teacher), snapshot(saved_model), snapshot(candidate)
    module = candidate.decoder.model[4].block[1]
    objects = {k: id(p) for k, p in candidate.named_parameters()}
    hooks = dict(module._forward_pre_hooks)
    weight = gm.effective_weight(module).detach().clone()
    bias = module.bias.detach().clone()
    dw, db = torch.randn_like(weight) * .01, torch.randn_like(bias) * .01
    fit.apply_delta(module, dw, db)
    torch.testing.assert_close(gm.effective_weight(module), weight + dw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(module.bias, bias + db, rtol=0, atol=0)
    after = snapshot(candidate)
    changed = {k for k in after if not torch.equal(after[k], before[k])}
    assert changed and all(k.startswith('decoder.model.4.block.1.') for k in changed)
    assert set(after) == set(before) and dict(module._forward_pre_hooks) == hooks
    assert objects == {k: id(p) for k, p in candidate.named_parameters()}
    assert_snapshot(teacher, before_teacher); assert_snapshot(saved_model, before_saved)


def test_coupled_fit_recomputes_downstream_inputs_and_current_skip_after_each_writeback():
    teacher = TinyDecoder().double().eval().requires_grad_(False)
    selected = {'stage2_indices': [i for i in range(32) if i % 4 != 1], 'stage3_indices': list(range(16))}
    model = progressive.initialize_from_teacher(teacher, selected).eval()
    sites = diagnostic.four_sites(teacher, selected)
    crops = [{'source_id': str(i), 'z': torch.randn(1, 8, 3, dtype=torch.float64) * .1} for i in range(2)]
    teacher_before = snapshot(teacher); model_before = snapshot(model)
    modes = {k: v.training for k, v in model.named_modules()}
    for p in model.trainable_group_parameters(): p.grad = torch.full_like(p, .25)
    grads = {k: p.grad.clone() if p.grad is not None else None for k, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    def capture(decoder, z):
        with torch.no_grad(), diagnostic.capture_sites(decoder, sites) as result:
            decoder(z)
        return result
    original_inputs = {(c['source_id'], s.name): capture(model.decoder, c['z'])[s.name]['input'].clone()
                       for c in crops for s in sites}
    seen = []
    def observe(student, original, crop, site):
        t, s = capture(original, crop['z'])[site.name], capture(student.decoder, crop['z'])[site.name]
        target = (fit.residual_target(t['ru_output'], s['skip_input'], site.kept_outputs)
                  if site.stride == 1 else t['output'])
        seen.append((crop['source_id'], site.name, s['input'].clone()))
        return {'input': s['input'], 'target': target, 'current_output': s['output'],
                'weights': torch.full((1, 1, target.shape[-1]), 40 if site.stride == 1 else 8)}
    report = fit.sequential_fit(model, teacher, crops, sites, observe)
    assert [r['site'] for r in report] == [s.name for s in sites]
    assert all(r['last_source_native_fold_parity']['allclose_existing'] for r in report)
    assert all(v['allclose_existing'] for r in report for v in r['effective_writeback'].values())
    assert [(source, site) for source, site, _ in seen] == [(c['source_id'], s.name) for s in sites for c in crops]
    assert any(not torch.allclose(value, original_inputs[(source, site)], rtol=1e-10, atol=1e-10)
               for source, site, value in seen if site == 'stage2_ru2')
    after = snapshot(model)
    prefixes = tuple('decoder.' + s.path + '.' for s in sites)
    changed = {k for k in after if not torch.equal(after[k], model_before[k])}
    assert changed and all(k.startswith(prefixes) for k in changed)
    assert_snapshot(teacher, teacher_before)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == {k: v.training for k, v in model.named_modules()}
    for k, p in model.named_parameters():
        assert (p.grad is None) if grads[k] is None else torch.equal(p.grad, grads[k])
    assert all(not module._forward_hooks for module in model.modules())
