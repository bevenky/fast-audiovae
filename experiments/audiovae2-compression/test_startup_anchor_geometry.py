"""Distinct zero-step mechanisms in the unchanged projection/backtracking.

These dimensionless CPU examples are mathematical counterexamples, not decoder
measurements or an implementation of a replacement update policy.
"""
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import quiet_projected_update as q


def score(parameter, functions):
    with torch.no_grad():
        values = [float(fn(parameter)) for fn in functions]
    worst = max(range(len(values)), key=values.__getitem__)
    return {'maxima': [(('near_startup', 'residual'), {'value': values[worst], 'ref': (worst, 0)})],
            'signature': [((i, 0), (value,), value <= 0) for i, value in enumerate(values)]}


def run_case(start, proposal_delta, functions, selected):
    p = nn.Parameter(torch.tensor(start, dtype=torch.float32))
    before = [p.detach().clone()]
    baseline = score(p, functions)
    rows = [[torch.autograd.grad(functions[i](p), p)[0]] for i in selected]
    bounds = [-float(functions[i](p).detach()) for i in selected]
    with torch.no_grad():
        p.add_(torch.tensor(proposal_delta, dtype=torch.float32))
    proposal = [p.detach().clone()]
    realized = [proposal[0].double() - before[0].double()]
    projected, projection = q.project_displacement(rows, realized, bounds)
    fraction, after, attempts = q.backtrack([p], before, proposal, projected, baseline,
        lambda: score(p, functions), corrected=projection['corrected'])
    return p, before[0], projected[0], fraction, baseline, after, attempts, rows


def ball(p):
    return p.square().sum() - 1


def test_curvature_alone_makes_every_tested_positive_tangent_step_fail():
    p, start, tangent, fraction, baseline, after, attempts, rows = run_case(
        [1., 0.], [.25, .5], [ball], [0])
    assert baseline['maxima'][0][1]['value'] == 0
    assert torch.equal(tangent, torch.tensor([0., .5], dtype=torch.float64))
    assert q._dot(rows[0], [tangent]) == 0
    # There is only ONE constraint, so this cannot be an omitted/max-switch case.
    assert all(a['maximum_switches'] == 0 for a in attempts)
    for alpha in q.FRACTIONS[:-1]:
        delta = alpha * tangent
        actual = float(ball((start.double() + delta).float()))
        assert actual > 0
        # For this exact boundary ball, the complete finite change is quadratic.
        assert actual == float(delta.square().sum())
    assert fraction == 0 and torch.equal(p, start) and after == baseline
    # A nonzero inward-normal-plus-tangent displacement is nevertheless feasible.
    feasible = torch.tensor([.8, .6], dtype=torch.float32)
    assert float(ball(feasible)) <= 0 and not torch.equal(feasible, start)


def test_omitted_tied_maximum_can_destroy_a_free_learning_direction_without_curvature():
    functions = [lambda p: p[0], lambda p: p[1]]
    p, start, projected, fraction, baseline, _, attempts, rows = run_case(
        [0., 0., 0.], [1., 1., 1.], functions, [0])
    assert torch.equal(projected, torch.tensor([0., 1., 1.], dtype=torch.float64))
    assert q._dot(rows[0], [projected]) == 0
    assert fraction == 0 and torch.equal(p, start)
    assert all(a['maximum_switches'] == 1 for a in attempts[:-1])
    # Both functions are affine: finite-minus-linear change is exactly zero.
    assert float(functions[1](start.double() + projected)) == projected[1].item()
    full = run_case([0., 0., 0.], [1., 1., 1.], functions, [0, 1])
    assert full[3] == 1 and torch.equal(full[0], torch.tensor([0., 0., 1.]))
    assert all(passed for _, _, passed in full[5]['signature'])
    assert full[4] == baseline


def test_coarse_grid_can_return_zero_even_with_strict_margin_and_positive_feasible_fraction():
    p, start, projected, fraction, baseline, _, attempts, rows = run_case(
        [.999, 0.], [0., 2.], [ball], [0])
    assert baseline['maxima'][0][1]['value'] < 0
    assert q._dot(rows[0], [projected]) == 0
    assert fraction == 0 and torch.equal(p, start)
    assert tuple(a['fraction'] for a in attempts) == q.FRACTIONS
    smaller = (start.double() + projected / 64).float()
    assert float(ball(smaller)) < 0 and not torch.equal(smaller, start)
    assert 1 / 64 < min(a for a in q.FRACTIONS if a > 0)
