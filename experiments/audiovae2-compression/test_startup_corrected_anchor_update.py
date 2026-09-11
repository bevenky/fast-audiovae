"""Bounded repair keeps one ordinary Adam transaction and truthful counters."""
import copy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_startup_anchor_update import (
    AnchorWave, anchors, assert_equal, cpu_policy, crops, optimizer,
    ordinary_update, teacher,
)
import startup_corrected_anchor_update as corrected

q, grid = corrected.q, corrected.grid


def execute(model, opt, t, coefficients, monkeypatch):
    calls = []
    def ordinary(*args, **kwargs):
        calls.append(True)
        return ordinary_update(*args, **kwargs)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary)
    original = q.backtrack, q.score_entries
    with grid.extended_grid():
        control = corrected.StartupAnchorUpdate(model, t, anchors(), opt)
        cached = copy.deepcopy(control._entries)
        model.seen.clear()
        values, checks = control.perform_update(model, t, crops(), coefficients, opt)
        seen = list(model.seen)
        assert_equal(control._entries, cached)
    assert calls == [True] and len(checks) == 12
    assert (q.backtrack, q.score_entries) == original
    assert values['q_canonical_score_forwards'] == 6 * values['q_canonical_score_calls']
    assert values['startup_anchor_checks'] == values['q_canonical_score_forwards']
    assert sum(not grad for _, grad in seen) == values['q_canonical_score_forwards']
    assert sum(grad for _, grad in seen) == values['q_constraint_gradient_forwards']
    assert all(passed for _, _, passed in q.score_entries(model, anchors())['signature'])
    return values, control


def test_already_feasible_trial_is_exact_ordinary_adam_with_no_normal_work(monkeypatch):
    model = AnchorWave(scale=.1)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    rng = q.screen.rng_state()
    ordinary_update(reference, t, crops(), None, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    values, _ = execute(model, opt, t, None, monkeypatch)
    assert_equal(model.state_dict(), reference.state_dict())
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert all(torch.equal(p.grad, r.grad) for p, r in zip(model.parameters(), reference.parameters()))
    assert values['q_normal_solves'] == values['q_normal_accepted'] == values['q_normal_accepted_norm'] == 0
    assert values['q_full_projected_trial_feasible'] == 1
    assert values['q_extended_grid_fallback'] == values['q_base_fraction_has_normal_correction'] == 0
    assert values['q_accepted_fraction'] == values['q_base_fraction'] == 1


def test_true_nonlinear_repair_preserves_adam_and_reports_actual_normal_work(monkeypatch):
    model = AnchorWave()
    with torch.no_grad():
        for p in model.ps[:6]: p.fill_(9.5e-6)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    coefficients = [1.] + [-1.] * 89
    rng = q.screen.rng_state()
    ordinary_update(reference, t, crops(), coefficients, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    values, _ = execute(model, opt, t, coefficients, monkeypatch)
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert values['q_normal_accepted'] == values['q_base_fraction_has_normal_correction'] == 1
    assert 1 <= values['q_normal_solves'] <= 2
    assert values['q_normal_accepted_scale'] == 2
    assert values['q_normal_accepted_norm'] > 0
    assert values['q_normal_accepted_norm'] <= values['q_normal_budget']
    assert values['q_normal_norm_over_projected'] <= .25
    assert values['q_extended_grid_fallback'] == 0 and values['q_base_fraction'] == 1
    assert values['startup_anchor_after_passed'] == 6
    assert values['q_normal_gradient_sources'] == 6 * values['q_normal_solves']
    assert values['q_normal_score_calls'] == 1 + values['q_normal_nonlinear_candidates']
    assert all(int(s['step']) == 1 for s in opt.state.values())
    assert all(torch.equal(p.grad, torch.full_like(p, c)) for p, c in zip(model.parameters(), coefficients))


def test_budget_rejection_falls_back_to_original_grid_and_zero_keeps_once_advanced_moments(monkeypatch):
    model = AnchorWave(scale=10000.)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    coefficients = [-1.] * 6 + [0.] * 84
    before = copy.deepcopy(model.state_dict())
    rng = q.screen.rng_state()
    ordinary_update(reference, t, crops(), coefficients, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    values, _ = execute(model, opt, t, coefficients, monkeypatch)
    assert_equal(model.state_dict(), before)
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert values['q_normal_solves'] == 1 and values['q_normal_budget_rejections'] == 2
    assert values['q_normal_nonlinear_candidates'] == values['q_normal_accepted'] == 0
    assert values['q_extended_grid_fallback'] == values['q_zero_displacement'] == 1
    assert values['q_extended_grid_score_calls'] == len(grid.EXTENDED_FRACTIONS)
    assert values['q_accepted_fraction'] == values['q_base_fraction'] == 0
    assert all(int(s['step']) == 1 for s in opt.state.values())


def test_correction_error_rolls_back_entire_adam_transaction_and_scoped_globals(monkeypatch):
    model = AnchorWave(scale=.1)
    opt, t = optimizer(model), teacher()
    ordinary_update(model, t, crops(), None, opt)
    for p in model.parameters(): p.grad = torch.full_like(p, 23.)
    slots = [p.grad for p in model.parameters()]
    original = q.backtrack, q.score_entries
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    with grid.extended_grid():
        control = corrected.StartupAnchorUpdate(model, t, anchors(), opt)
        before = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(t.state_dict())
        rng, warmed = q.screen.rng_state(), set(control._warmed)
        def fail(*args, **kwargs):
            assert all(int(s['step']) == 2 for s in opt.state.values())
            raise RuntimeError('injected signed-solver failure')
        monkeypatch.setattr(corrected.correction, 'correction_attempts', fail)
        with pytest.raises(RuntimeError, match='injected signed-solver'):
            control.perform_update(model, t, crops(), None, opt)
    for current, saved in zip((model.state_dict(), opt.state_dict(), t.state_dict()), before):
        assert_equal(current, saved)
    assert_equal(q.screen.rng_state(), rng)
    assert control._updates == 0 and control._warmed == warmed
    assert (q.backtrack, q.score_entries) == original
    assert all(p.grad is slot and torch.equal(p.grad, torch.full_like(p, 23.)) for p, slot in zip(model.parameters(), slots))
