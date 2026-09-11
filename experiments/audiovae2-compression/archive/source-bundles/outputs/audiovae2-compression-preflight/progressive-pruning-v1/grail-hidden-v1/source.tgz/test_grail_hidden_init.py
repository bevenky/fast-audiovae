"""Small independent numerical checks for zero-intercept native hidden folds."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import grail_hidden_init as fit
from test_group_model import CausalConv1d, CausalTransposeConv1d, TinyDecoder, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(4905); torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def test_hidden_usage_keeps_previous_context_and_exact_partial_cells():
    q = torch.zeros(1, 1, 20)
    q[..., 10:15] = torch.tensor([8., 8., 3., 0., 0.])
    actual = fit.hidden_usage_weights(q, native=True)
    torch.testing.assert_close(actual, torch.tensor([[[0., 19., 19., 0.]]]), atol=0, rtol=0)
    oracle = torch.zeros_like(actual)
    for cell in range(20):
        frame = cell // 5
        oracle[..., frame] += q[..., cell]
        if frame > 0: oracle[..., frame-1] += q[..., cell]
    assert torch.equal(actual, oracle)
    q0 = torch.zeros_like(q); q0[..., :5] = 8
    assert torch.equal(fit.hidden_usage_weights(q0, native=True), torch.tensor([[[40., 0., 0., 0.]]]))
    assert torch.equal(fit.hidden_usage_weights(torch.tensor([[[0., 7., 40.]]])), torch.tensor([[[0., 7., 40.]]]))


def test_unused_context_and_padding_nan_do_not_enter_fit():
    x = torch.randn(1, 2, 4, dtype=torch.float64)
    matrix = torch.randn(3, 2, dtype=torch.float64)
    h = torch.einsum('hc,bct->bht', matrix, x)
    q = torch.tensor([[[0., 19., 19., 0.]]])
    x[..., 0] = x[..., -1] = torch.nan
    h[..., 0] = h[..., -1] = torch.nan
    stats = fit.accumulate_hidden(None, x, h, q)
    mapping, report = fit.solve_hidden_map(stats)
    assert torch.isfinite(mapping).all() and report['valid_hidden_rows'] == 2
    assert report['weighted_hidden_uses'] == 38
    q[..., 0] = 1
    with pytest.raises(ValueError, match='Nonfinite'):
        fit.accumulate_hidden(None, x, h, q)


def test_uncentered_ridge_matches_direct_weighted_solve_and_chunk_merges():
    x = torch.randn(2, 3, 31, dtype=torch.float64) + 2
    h = torch.randn(2, 5, 31, dtype=torch.float64) + 7
    q = torch.randint(0, 41, (2, 1, 31))
    stats = fit.accumulate_hidden(None, x, h, q)
    split = fit.accumulate_hidden(None, x[..., :11], h[..., :11], q[..., :11])
    split = fit.accumulate_hidden(split, x[..., 11:], h[..., 11:], q[..., 11:])
    mapping, report = fit.solve_hidden_map(stats)
    mapping2, _ = fit.solve_hidden_map(split)
    a = x.movedim(1, -1).reshape(-1, 3); b = h.movedim(1, -1).reshape(-1, 5)
    w = q.reshape(-1).double(); gram = a.T @ (a*w[:, None])
    lam = fit.RIDGE*gram.diag().mean()
    expected = torch.linalg.solve(gram+lam*torch.eye(3), a.T@(b*w[:, None])).T
    torch.testing.assert_close(mapping, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mapping, mapping2, atol=1e-12, rtol=1e-12)
    error = ((a@mapping.T-b).square()*w[:, None]).sum()/(w.sum()*5)
    assert report['weighted_hidden_mse_after'] == pytest.approx(float(error), rel=1e-10)
    assert report['intercept'] is False and report['uncentered'] is True


def test_constant_hidden_offset_is_not_turned_into_an_intercept():
    x = torch.tensor([[[-1., 1., -1., 1.]]], dtype=torch.float64)
    h = 2*x+7
    mapping, report = fit.solve_hidden_map(fit.accumulate_hidden(None, x, h, torch.ones_like(x)))
    assert float(mapping) == pytest.approx(2/(1+fit.RIDGE), rel=1e-12)
    assert report['weighted_hidden_mse_after'] == pytest.approx(49., abs=1e-10)
    assert float((mapping.reshape(1, 1, 1)*x).mean()) == 0
    with pytest.raises(ValueError, match='positive finite trace'):
        fit.solve_hidden_map(fit.accumulate_hidden(None, torch.zeros_like(x), h, torch.ones_like(x)))


@pytest.mark.parametrize('frames', [1, 4])
def test_shared_map_all10_native_taps_matches_explicit_hidden_at_startup(frames):
    teacher = weight_norm(CausalTransposeConv1d(4, 3, 10, stride=5, padding=3, output_padding=1)).double()
    student = weight_norm(CausalTransposeConv1d(2, 3, 10, stride=5, padding=3, output_padding=1)).double()
    with torch.no_grad(): teacher.bias.copy_(torch.tensor([.21, -.17, .04], dtype=torch.float64))
    before_teacher = snapshot(teacher); ids_before = {k: id(p) for k, p in student.named_parameters()}
    r = torch.randn(4, 2, dtype=torch.float64); x = torch.randn(1, 2, frames, dtype=torch.float64)
    desired, bias = fit.folded_coefficients(teacher, r, [0, 1, 2])
    expected_weights = torch.empty_like(desired)
    teacher_w = fit.group.effective_weight(teacher).detach()
    for k in range(2):
        for o in range(3):
            for tap in range(10):
                expected_weights[k, o, tap] = sum(teacher_w[c, o, tap]*r[c, k] for c in range(4))
    torch.testing.assert_close(desired, expected_weights, atol=1e-12, rtol=1e-12)
    hidden = torch.einsum('hc,bct->bht', r, x)
    expected = F.conv_transpose1d(hidden, teacher_w, teacher.bias, stride=5)[..., :frames*5]
    result = fit.install_hidden_map(student, teacher, r, [0, 1, 2])
    torch.testing.assert_close(student(x), expected, atol=1e-12, rtol=1e-12)
    assert result['bias_bitwise_equal'] and torch.equal(student.bias, teacher.bias)
    assert ids_before == {k: id(p) for k, p in student.named_parameters()}
    assert_snapshot(teacher, before_teacher)


def test_pointwise_fold_selects_teacher_output_rows_and_preserves_actual_skip():
    teacher = weight_norm(CausalConv1d(4, 4, 1)).double()
    student = weight_norm(CausalConv1d(2, 2, 1)).double()
    r = torch.randn(4, 2, dtype=torch.float64); x = torch.randn(1, 2, 7, dtype=torch.float64)
    skip = torch.randn_like(x); selected = [0, 2]
    expected = F.conv1d(torch.einsum('hc,bct->bht', r, x),
                        fit.group.effective_weight(teacher)[selected], teacher.bias[selected]) + skip
    fit.install_hidden_map(student, teacher, r, selected)
    torch.testing.assert_close(student(x)+skip, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(student.bias, teacher.bias[selected])


def test_sequential_actual_inputs_change_and_only_four_native_weights_change():
    teacher = TinyDecoder().double().eval().requires_grad_(False)
    selected = {'stage2_indices': [i for i in range(32) if i % 4 != 1], 'stage3_indices': list(range(16))}
    model = fit.progressive.initialize_from_teacher(teacher, selected).eval()
    sites = fit.audit.four_sites(teacher, selected)
    wrapped = SimpleNamespace(model=SimpleNamespace(decoder=teacher))
    crops = [{'source_id': str(i), 'z': torch.randn(1, 8, 3, dtype=torch.float64)*.1} for i in range(2)]
    before_teacher, before_model = snapshot(teacher), snapshot(model)
    def capture(decoder, z):
        with torch.no_grad(), fit.audit.capture_sites(decoder, sites) as values: decoder(z)
        return values
    original = {(c['source_id'], s.name): capture(model.decoder, c['z'])[s.name]['input'].clone()
                for c in crops for s in sites}
    observed = []
    def observe(current, frozen, crop, site):
        t = capture(frozen.model.decoder, crop['z'])[site.name]
        s = capture(current.decoder, crop['z'])[site.name]
        observed.append((crop['source_id'], site.name, s['input'].clone()))
        ow = torch.full((1, 1, s['output'].shape[-1]), 8 if site.stride > 1 else 40)
        return {'source_id': crop['source_id'], 'input': s['input'], 'teacher_input': t['input'],
                'weights': fit.hidden_usage_weights(ow, native=site.stride > 1)}
    rng = torch.get_rng_state().clone()
    reports, maps = fit.sequential_hidden_fit(model, wrapped, crops, sites, observe)
    assert [r['site'] for r in reports] == [s.name for s in sites]
    assert all(r['native_writeback']['bias_bitwise_equal'] and r['last_source_native_fold_parity']['allclose_existing'] for r in reports)
    assert any(not torch.allclose(x, original[(sid, site)], rtol=1e-10, atol=1e-10)
               for sid, site, x in observed if site == 'stage2_ru2')
    after = snapshot(model); prefixes = tuple('decoder.'+s.path+'.' for s in sites)
    changed = {key for key in after if not torch.equal(after[key], before_model[key])}
    assert changed and all(key.startswith(prefixes) for key in changed)
    assert not any(key.endswith('bias') for key in changed)
    assert set(maps) == set(fit.PATHS) and all(x.shape == (32, 24) for x in maps.values())
    assert_snapshot(teacher, before_teacher); assert torch.equal(rng, torch.get_rng_state())


def test_duplicate_calibration_sources_rejected_before_install():
    teacher = TinyDecoder().double().eval().requires_grad_(False)
    selected = {'stage2_indices': list(range(24)), 'stage3_indices': list(range(16))}
    model = fit.progressive.initialize_from_teacher(teacher, selected).eval()
    sites = fit.audit.four_sites(teacher, selected)
    before = snapshot(model)
    def observe(*args):
        return {'source_id': 'same', 'input': torch.randn(1, 24, 30, dtype=torch.float64),
                'teacher_input': torch.randn(1, 32, 30, dtype=torch.float64), 'weights': torch.ones(1, 1, 30)}
    with pytest.raises(ValueError, match='Repeated'):
        fit.sequential_hidden_fit(model, SimpleNamespace(model=SimpleNamespace(decoder=teacher)), [0, 1], sites, observe)
    assert_snapshot(model, before)


def test_portable_artifact_excludes_hidden_maps_and_requires_distinct_fit_ids():
    teacher = TinyDecoder().eval().requires_grad_(False)
    selected = {'stage2_indices': list(range(24)), 'stage3_indices': list(range(16))}
    model = fit.progressive.initialize_from_teacher(teacher, selected).eval()
    ids = [f'calibration-{i}' for i in range(72)]
    payload = fit.export_artifact(model, teacher, ids, base_selected_state_sha256='base', maps_sha256='maps')
    assert payload['format'] == fit.VERSION and payload['variant'] == fit.VARIANT
    assert payload['selection'] == selected and set(payload['operators']) == set(fit.PATHS)
    assert payload['base_selected_state_sha256'] == 'base' and payload['hidden_maps_sha256'] == 'maps'
    assert payload['hidden_maps_exported'] is False and 'maps' not in payload
    assert all(set(state) == {'bias', 'weight_g', 'weight_v'} for state in payload['operators'].values())
    with pytest.raises(ValueError, match='distinct'):
        fit.export_artifact(model, teacher, ['duplicate']*72, base_selected_state_sha256='base', maps_sha256='maps')
