"""Native finite checks distinguish curvature repair from budget violations."""
from pathlib import Path
import copy
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_anchor_correction_probe as probe
import startup_signed_constraint_projection as signed


def run_toy(before, trial, projected, function):
    p = nn.Parameter(torch.tensor(before, dtype=torch.float32))
    p.grad = torch.full_like(p, 17.)
    slot = p.grad
    old = [p.detach().clone()]
    initial = [torch.tensor(trial, dtype=torch.float32)]
    direction = [torch.tensor(projected, dtype=torch.float64)]
    scores, gradients = [], []
    def score():
        value = float(function(p).detach())
        scores.append((p.detach().clone(), value))
        return {'values': [value], 'summary': {'passed': int(value <= 0)}}
    def gradient(values):
        assert values == [float(function(p).detach())]
        gradients.append(p.detach().clone())
        return [[torch.autograd.grad(function(p), p)[0]]]
    candidate, report = probe.correction_attempts([p], old, initial, direction,
        score_fn=score, gradient_fn=gradient, limit_squares=[1.], project_signed=signed.project_displacement)
    assert torch.equal(p, old[0])
    assert p.grad is slot and torch.equal(p.grad, torch.full_like(p, 17.))
    assert torch.equal(initial[0], torch.tensor(trial, dtype=torch.float32))
    return candidate, report, scores, gradients


def test_circle_needs_inward_scale_two_after_newton_linear_equality():
    def circle(p): return p.double().square().sum() - 1
    candidate, report, scores, gradients = run_toy([1., 0.], [1., .125], [0., .5], circle)
    assert candidate is not None and len(gradients) == 1
    assert len(scores) >= 3 and scores[0][1] > scores[1][1] > 0
    assert scores[2][1] <= 0
    # Exact first-order repair leaves the positive quadratic remainder; scale2
    # moves inward and passes the original nonlinear bound without a margin tweak.
    current = torch.tensor([1., .125], dtype=torch.float64)
    normal = -(circle(current) / (4 * current.square().sum())) * 2 * current
    expected = (current + 2 * normal).float()
    torch.testing.assert_close(candidate[0], expected, rtol=0, atol=0)
    assert float(circle(candidate[0])) <= 0
    assert torch.linalg.vector_norm(candidate[0].double() - current) <= .25 * .5


def test_second_solve_budget_is_total_correction_from_original_trial():
    # Exp's first Newton step and doubled step both improve but remain outside.
    # The next local normal is small; its total distance from the original trial
    # exceeds1.9, so it cannot evade the fixed cumulative budget by restarting it.
    candidate, report, scores, gradients = run_toy([0.], [2.], [7.6], lambda p: p.double().exp().sum() - 1)
    assert candidate is None and len(gradients) == 2
    assert 0 < gradients[1].item() < .3
    current = gradients[1].double()
    second_normal = -(1 - torch.exp(-current))
    assert abs(second_normal.item()) < 1.9
    assert abs((current + second_normal - 2).item()) > 1.9


def test_budget_uses_actual_fp32_writeback_not_smaller_ideal_normal():
    ulp = torch.finfo(torch.float32).eps
    target = 1.5 - .6 * ulp
    candidate, report, scores, gradients = run_toy([1.5 - 2 * ulp], [1.5], [3 * ulp],
                                                 lambda p: p.double().sum() - target)
    assert candidate is None and len(gradients) == 1
    ideal = .6 * ulp
    actual = abs(float(torch.tensor(1.5 - ideal, dtype=torch.float32)) - 1.5)
    assert ideal < .25 * (3 * ulp) < actual


def test_unexpected_candidate_failure_restores_native_parameters_and_grad_slots():
    p = nn.Parameter(torch.tensor([1., 0.], dtype=torch.float32))
    p.grad = torch.full_like(p, 9.)
    slot = p.grad
    before = [torch.tensor([1., 0.])]
    calls = []
    def score():
        calls.append(True)
        if len(calls) > 1: raise RuntimeError('injected native scoring failure')
        return {'values': [float(p.detach().double().square().sum() - 1)], 'summary': {}}
    def gradient(values): return [[2 * p.detach()]]
    with pytest.raises(RuntimeError, match='injected native scoring'):
        probe.correction_attempts([p], before, [torch.tensor([1., .125])], [torch.tensor([0., .5])],
            score_fn=score, gradient_fn=gradient, limit_squares=[1.], project_signed=signed.project_displacement)
    assert torch.equal(p, before[0])
    assert p.grad is slot and torch.equal(p.grad, torch.full_like(p, 9.))


def test_reference_and_exact_scalar_prefix_reject_wrong_policy_first_zero_or_changed_update(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{}')
    records = [{'step': step, 'total': step / 100, 'q_constraints': 12,
                'q_zero_displacement': int(step >= 56), 'q_step_seconds': .1}
               for step in range(1, 65)]
    reference = {'version': probe.pilot.VERSION, 'method': 'complete_anchor',
                 'complete': True, 'all_preservation_checks_passed': True,
                 'updates': 64, 'ordinary_unique_sources': 768,
                 'config_sha256': probe.base.sha(config), 'complete_anchor_policy': {'complete': True},
                 'update_records': records}
    probe.validate_reference(reference, config)
    for old in records[:56]:
        current = {k: v for k, v in old.items() if k != 'step'}
        current['q_step_seconds'] = 100.
        assert probe.compare_record(current, old, old['step'])['passed']
    current['total'] += 1e-12
    assert not probe.compare_record(current, records[55], 56)['passed']
    current['total'] = records[55]['total']
    current['q_unexpected_policy_field'] = 1
    assert not probe.compare_record(current, records[55], 56)['passed']
    for change in ('policy', 'earlier_zero', 'ledger', 'config'):
        bad = copy.deepcopy(reference)
        if change == 'policy': bad['method'] = 'anchor'
        elif change == 'earlier_zero': bad['update_records'][42]['q_zero_displacement'] = 1
        elif change == 'ledger': bad['update_records'][55]['step'] = 55
        else: bad['config_sha256'] = '0' * 64
        with pytest.raises(ValueError):
            probe.validate_reference(bad, config)
