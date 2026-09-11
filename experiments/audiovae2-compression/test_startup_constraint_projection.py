"""Independent small optima and KKT checks for all twelve startup directions."""
import math
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_constraint_projection as full
import quiet_projected_update as original


@pytest.fixture(autouse=True)
def cpu_policy():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def vector(values):
    return [torch.tensor(values, dtype=torch.float64)]


def test_all_twelve_binding_directions_preserve_the_unconstrained_learning_coordinate():
    a = torch.eye(13, dtype=torch.float64)[:12]
    d = [torch.arange(1., 14., dtype=torch.float64)]
    before = d[0].clone()
    actual, report = full.project_displacement([[row] for row in a], d, [0.] * 12)
    expected = torch.zeros(13, dtype=torch.float64)
    expected[-1] = 13
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-12)
    assert torch.equal(d[0], before)
    assert report['kkt_passed'] and report['corrected'] and report['rank'] == 12
    assert report['rows'] == report['nonzero_rows'] == 12
    assert (a @ actual[0]).max() <= 1e-12


def test_duplicate_opposed_and_zero_rows_have_a_known_rank_deficient_optimum():
    rows = [vector([1, 0, 0]), vector([2, 0, 0]), vector([-1, 0, 0]),
            vector([0, 1, 0]), vector([0, 0, 0])]
    actual, report = full.project_displacement(rows, vector([2, 3, 4]), [0, 0, 0, .3, 0])
    torch.testing.assert_close(actual[0], torch.tensor([0., .3, 4.], dtype=torch.float64), rtol=0, atol=1e-12)
    assert report['kkt_passed'] and report['rank'] == 2 and report['nonzero_rows'] == 4


def test_original_six_or_fewer_constraint_results_and_row_rhs_scaling_agree():
    rows = [vector([1, 0, 0]), vector([0, 1, 0]), vector([1, 1, 0]), vector([-1, 0, 0])]
    d, bounds = vector([2, 3, 4]), [.1, .2, .21, .3]
    expected, _ = original.project_displacement(rows, d, bounds)
    actual, report = full.project_displacement(rows, d, bounds)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-11, atol=1e-12)
    scales = (1e-8, 1e5, .25, 17.)
    scaled, other = full.project_displacement([[r[0] * scale] for r, scale in zip(rows, scales)],
                                             d, [b * scale for b, scale in zip(bounds, scales)])
    torch.testing.assert_close(scaled[0], actual[0], rtol=1e-11, atol=1e-12)
    assert report['kkt_passed'] and other['kkt_passed']


def test_independent_primal_dual_complementarity_and_known_optimum():
    a = torch.tensor([[1., 0.], [0., 1.], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1., 0.]], dtype=torch.float64)
    b = torch.tensor([.1, .2, .15, .3], dtype=torch.float64)
    d = torch.tensor([2., 3.], dtype=torch.float64)
    gram, violation = a @ a.T, a @ d - b
    multipliers, report = full.solve_small_qp(gram, violation)
    point = d - a.T @ multipliers
    slack = b - a @ point
    assert multipliers.min() >= -1e-12 and slack.min() >= -1e-12
    assert (multipliers * slack).abs().max() <= 1e-12
    assert (gram @ multipliers - violation).min() >= -1e-12
    torch.testing.assert_close(point, torch.tensor([math.sqrt(2) * .15 - .2, .2], dtype=torch.float64), rtol=0, atol=1e-12)
    assert report['kkt_passed']


def test_zero_gradient_and_no_conflict_preserve_the_exact_proposal():
    d = vector([.123456789123, -2.])
    for rows, bounds in (([vector([0, 0]), vector([0, 0])], [0., 1.]),
                         ([vector([1, 0]), vector([0, 1])], [1., 1.])):
        actual, report = full.project_displacement(rows, d, bounds)
        assert torch.equal(actual[0], d[0]) and actual[0].data_ptr() != d[0].data_ptr()
        assert not report['corrected'] and report['kkt_passed']


@pytest.mark.parametrize('damage', ['negative_bound', 'nan_bound', 'nan_row', 'infinite_proposal', 'shape', 'thirteen'])
def test_invalid_projection_fails_without_changing_the_proposal(damage):
    rows, d, bounds = [vector([1, 0])], vector([1, 2]), [0.]
    if damage == 'negative_bound': bounds[0] = -.1
    elif damage == 'nan_bound': bounds[0] = float('nan')
    elif damage == 'nan_row': rows[0][0][0] = float('nan')
    elif damage == 'infinite_proposal': d[0][0] = float('inf')
    elif damage == 'shape': rows[0] = vector([1, 0, 0])
    else: rows, bounds = [vector([1, 0]) for _ in range(13)], [0.] * 13
    before = d[0].clone()
    with pytest.raises((ValueError, RuntimeError)):
        full.project_displacement(rows, d, bounds)
    assert torch.equal(d[0], before)


@pytest.mark.parametrize('damage', ['asymmetric', 'indefinite', 'nan', 'wrong_precision', 'incompatible'])
def test_invalid_or_incompatible_dual_system_is_not_silently_accepted(damage):
    gram, violation = torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    if damage == 'asymmetric': gram[0, 1] = .1
    elif damage == 'indefinite': gram[0, 0] = -1.
    elif damage == 'nan': gram[0, 0] = float('nan')
    elif damage == 'wrong_precision': gram = gram.float()
    else: gram.zero_()
    with pytest.raises((ValueError, RuntimeError)):
        full.solve_small_qp(gram, violation)
