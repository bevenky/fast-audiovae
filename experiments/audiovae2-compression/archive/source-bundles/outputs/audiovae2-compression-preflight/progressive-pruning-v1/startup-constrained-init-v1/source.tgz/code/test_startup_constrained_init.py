"""Independent KKT, native-operator and original-grid constraint controls."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils import weight_norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_constrained_init as constrained
from test_group_model import CausalTransposeConv1d, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(782)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def statistics(features=3, outputs=2):
    x = torch.randn(1, features, 101, dtype=torch.float64)
    y = torch.randn(1, outputs, 101, dtype=torch.float64)
    weights = torch.full((1, 1, 101), 8, dtype=torch.float64)
    weights[..., :3] = 0; weights[..., -1] = 3
    return constrained.common.accumulate_affine(None, x, y, weights)


def test_whitened_projection_matches_independent_full_kkt_solution_and_preserves_statistics():
    stats = statistics()
    before = {k: v.clone() for k, v in stats.items()}
    design = torch.tensor([[1., -2., 3.], [-.2, .4, -.7]], dtype=torch.float64)
    target = torch.tensor([[.7, -.1], [-.3, .5]], dtype=torch.float64)
    a, bias, report = constrained.solve_constrained_delta(stats, design, target, target)
    cov = stats['centered_xx'] / stats['n']
    q = torch.zeros(4, 4, dtype=torch.float64)
    q[:3, :3] = cov + 1e-6 * cov.trace() / 3 * torch.eye(3, dtype=torch.float64)
    q[3, 3] = 1
    e = torch.cat((design - stats['mean_x'], torch.ones(2, 1, dtype=torch.float64)), 1)
    rhs = torch.cat((stats['centered_xd'] / stats['n'], stats['mean_d'][None]), 0)
    # Direct bordered KKT solve is independent of the SVD projection algorithm.
    kkt = torch.cat((torch.cat((q, e.T), 1), torch.cat((e, torch.zeros(2, 2, dtype=torch.float64)), 1)), 0)
    oracle = torch.linalg.solve(kkt, torch.cat((rhs, target), 0))[:4]
    torch.testing.assert_close(a.T, oracle[:3], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(bias, oracle[3] - a @ stats['mean_x'], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(design @ a.T + bias, target, rtol=1e-11, atol=1e-11)
    assert report['feasible'] and report['retained_rank'] == 2
    assert report['rank_relative_cutoff'] == 4 * torch.finfo(torch.float32).eps
    for k, value in stats.items(): torch.testing.assert_close(value, before[k], rtol=0, atol=0)


@pytest.mark.parametrize('perturbation', [0., 1e-9])
def test_incompatible_or_roundoff_separated_constraints_report_infeasible_without_huge_fit(perturbation):
    stats = statistics(2, 1)
    design = torch.tensor([[.3, -.2], [.3 + perturbation, -.2]], dtype=torch.float64)
    target = torch.tensor([[.1], [.2]], dtype=torch.float64)
    a, b, report = constrained.solve_constrained_delta(stats, design, target, target)
    assert a is None and b is None and not report['feasible']
    assert report['status'] == 'no_stable_feasible_solution'
    assert report['retained_rank'] == 1 and report['discarded_modes'] == 1
    assert report['equality_fp64']['max_abs'] > .04
    assert report['change_from_ordinary_frobenius'] < 10


def test_compatibility_uses_complete_teacher_output_scale_not_tiny_delta_scale():
    stats = statistics(2, 1)
    design = torch.tensor([[.1, .2], [.1, .2]], dtype=torch.float64)
    delta = torch.tensor([[.001], [-.001]], dtype=torch.float64)
    full_teacher = torch.full((2, 1), 100., dtype=torch.float64)
    _, _, report = constrained.solve_constrained_delta(stats, design, delta, full_teacher)
    assert report['feasible']
    assert report['equality_fp64']['max_abs'] == pytest.approx(.001)
    _, _, small_scale = constrained.solve_constrained_delta(stats, design, delta, delta)
    assert not small_scale['feasible']


def crop(valid_samples, start=0, context_start=0):
    return {'source_id': 'calibration-only', 'start_frame': start, 'context_start_frame': context_start,
            'context_frames': start - context_start, 'valid_scored_samples': valid_samples}


def test_original_startup_grid_selects_only_whole_eight_sample_cells_and_excludes_partial_tail():
    target = torch.zeros(1, 1, 2400, dtype=torch.float64)
    selected, receipt = constrained.startup_cells(target, torch.ones_like(target, dtype=torch.bool), crop(2400), 300)
    assert int(selected.sum()) == 120 and selected[..., :120].all() and not selected[..., 120:].any()
    assert receipt['observed_near_startup_samples'] == 960
    target = torch.zeros(1, 1, 960, dtype=torch.float64)
    valid = torch.zeros_like(target, dtype=torch.bool); valid[..., :949] = True
    selected, receipt = constrained.startup_cells(target, valid, crop(949), 120)
    assert int(selected.sum()) == 118
    assert receipt['excluded_partial_cell_samples'] == 5


def test_local_crop_start_is_not_source_start_and_quiet_is_not_necessarily_near_silence():
    target = torch.zeros(1, 1, 2880, dtype=torch.float64)
    valid = torch.zeros_like(target, dtype=torch.bool); valid[..., 1920:] = True
    selected, receipt = constrained.startup_cells(target, valid, crop(960, start=1), 360)
    assert not selected.any() and receipt['windows'] == []
    near_limit_exceeded = torch.full((1, 1, 960), 2e-5, dtype=torch.float64)
    selected, _ = constrained.startup_cells(near_limit_exceeded, torch.ones_like(near_limit_exceeded, dtype=torch.bool), crop(960), 120)
    assert not selected.any()
    valid[..., 1920] = False
    with pytest.raises(ValueError, match='contiguous scored validity'):
        constrained.startup_cells(target, valid, crop(960, start=1), 360)


def native(x, w, b):
    return F.conv_transpose1d(x, w, b, stride=5)[..., :x.shape[-1]*5]


def test_native_constrained_fit_has_shared_bias_real_previous_context_and_no_module_mutation():
    module = weight_norm(CausalTransposeConv1d(2, 3, 10, stride=5, padding=3, output_padding=1)).double()
    desired_w = torch.randn(2, 3, 10, dtype=torch.float64) * .1
    desired_b = torch.tensor([.03, -.07, .2], dtype=torch.float64)
    observations = []
    for i in range(2):
        x = torch.randn(1, 2, 43, dtype=torch.float64)
        weights = torch.full((1, 1, 215), 8)
        weights[..., -1] = 3
        constraints = torch.zeros_like(weights, dtype=torch.bool); constraints[..., :15] = True
        observations.append({'source_id': str(i), 'input': x, 'target': native(x, desired_w, desired_b),
                             'current_output': module(x).detach(), 'weights': weights, 'constraint_mask': constraints})
    before = snapshot(module); rng = torch.get_rng_state().clone()
    dw, db, report = constrained.fit_constrained_delta(module, observations, chunk_rows=7)
    assert report['feasible'] and report['constraint_rows'] == 30
    assert report['observations'] == 2 and report['weighted_sample_count'] == 2*(214*8+3)
    assert db.shape == (3,) and dw.shape == (2, 3, 10)
    assert_snapshot(module, before); assert torch.equal(rng, torch.get_rng_state())
    constrained.ordinary.apply_delta(module, dw, db)
    for row in observations:
        mask = row['constraint_mask'].expand_as(row['target'])
        torch.testing.assert_close(module(row['input'])[mask], row['target'][mask], rtol=1e-10, atol=1e-10)
    unseen = torch.randn(1, 2, 11, dtype=torch.float64)
    torch.testing.assert_close(module(unseen), native(unseen, desired_w, desired_b), rtol=1e-9, atol=1e-9)


def test_absent_constraints_are_explicit_and_partial_constraint_cells_rejected():
    module = weight_norm(CausalTransposeConv1d(1, 1, 10, stride=5, padding=3, output_padding=1)).double()
    x = torch.randn(1, 1, 3, dtype=torch.float64); out = module(x).detach()
    row = {'source_id': 'one', 'input': x, 'target': out, 'current_output': out,
           'weights': torch.full((1, 1, 15), 8), 'constraint_mask': torch.zeros(1, 1, 15, dtype=torch.bool)}
    before = snapshot(module)
    dw, db, report = constrained.fit_constrained_delta(module, [row])
    assert dw is None and db is None and report['status'] == 'no_stable_feasible_solution'
    row['weights'][..., 0] = 7; row['constraint_mask'][..., 0] = True
    with pytest.raises(ValueError, match='whole-cell constraint geometry'):
        constrained.fit_constrained_delta(module, [row])
    assert_snapshot(module, before)
