"""Complete startup constraints retain the original one-Adam transaction."""
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
import startup_anchor_grid_pilot as grid
import startup_complete_anchor_update as complete

q = grid.q


def test_no_conflict_keeps_exact_ordinary_adam_rng_gradients_and_anchor_cache(monkeypatch):
    model = AnchorWave(scale=.1)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    rng = q.screen.rng_state()
    expected, checks = ordinary_update(reference, t, crops(), None, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    calls = []
    def ordinary(*args, **kwargs):
        calls.append(True)
        return ordinary_update(*args, **kwargs)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary)
    with grid.extended_grid():
        control = complete.StartupAnchorUpdate(model, t, anchors(), opt)
        cache = copy.deepcopy(control._entries)
        warmed = set(control._warmed)
        values, actual_checks = control.perform_update(model, t, crops(), None, opt)
    assert calls == [True] and checks == actual_checks
    assert values['fixture_loss'] == expected['fixture_loss']
    assert values['q_accepted_fraction'] == 1 and values['q_projected'] == 0
    assert values['q_constraints'] == 12 and values['q_constrained_sources'] == 6
    assert values['q_full_primal_verified'] == values['q_kkt_passed'] == 1
    assert control.receipt['constraints'] == 12
    assert_equal(model.state_dict(), reference.state_dict())
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert_equal(control._entries, cache)
    assert control._warmed == warmed
    assert all(torch.equal(p.grad, r.grad) for p, r in zip(model.parameters(), reference.parameters()))


def test_all_twelve_rows_prevent_omitted_anchor_escape_and_keep_a_real_projected_step(monkeypatch):
    model = AnchorWave()
    with torch.no_grad():
        for p in model.ps[:6]: p.fill_(9.5e-6)
    before = copy.deepcopy(model.state_dict())
    coefficients = [-1.] * 90
    coefficients[0] = 1.
    calls = []
    def ordinary(*args, **kwargs):
        calls.append(True)
        return ordinary_update(*args, **kwargs)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary)
    opt, t = optimizer(model), teacher()
    with grid.extended_grid():
        control = complete.StartupAnchorUpdate(model, t, anchors(), opt)
        values, _ = control.perform_update(model, t, crops(), coefficients, opt)
    assert calls == [True]
    assert values['q_constraints'] == 12 and values['q_constraint_gradient_sources'] == 6
    assert values['q_full_primal_verified'] == values['q_kkt_passed'] == 1
    assert values['q_projected'] == 1 and values['q_accepted_fraction'] == .5
    assert values['startup_anchor_after_passed'] == 6
    assert values['startup_anchor_current_batch_constraints'] == 0
    assert all(passed for _, _, passed in q.score_entries(model, anchors())['signature'])
    assert not q.replay.compare_tree(model.state_dict(), before)['equal']
    assert model.ps[1].item() > before['ps.1'].item()
    assert all(int(s['step']) == 1 for s in opt.state.values())
    assert all(torch.equal(p.grad, torch.full_like(p, c)) for p, c in zip(model.parameters(), coefficients))


@pytest.mark.parametrize('scale, fraction', [(1., .25), (10000., 0.)])
def test_zero_constraint_gradient_uses_finite_checks_and_declares_once_advanced_adam(monkeypatch, scale, fraction):
    model = AnchorWave(scale=scale)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    before = copy.deepcopy(model.state_dict())
    rng = q.screen.rng_state()
    ordinary_update(reference, t, crops(), None, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    with grid.extended_grid():
        control = complete.StartupAnchorUpdate(model, t, anchors(), opt)
        values, _ = control.perform_update(model, t, crops(), None, opt)
    assert values['q_nonzero_rows'] == 0 and values['q_accepted_fraction'] == fraction
    assert values['startup_anchor_after_passed'] == 6
    assert values['q_zero_displacement'] == int(fraction == 0)
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert all(int(s['step']) == 1 for s in opt.state.values())
    if fraction == 0:
        assert_equal(model.state_dict(), before)
        assert 'once' in control.receipt['accepted_zero_policy']
    else:
        torch.testing.assert_close(model.ps[0], torch.tensor([7.5e-6]), rtol=1e-6, atol=1e-12)


def test_nonlinear_error_rolls_back_parameters_moments_rng_grad_objects_and_warm_cache(monkeypatch):
    model = AnchorWave(scale=.1)
    opt, t = optimizer(model), teacher()
    ordinary_update(model, t, crops(), None, opt)
    for p in model.parameters(): p.grad = torch.full_like(p, 19.)
    slots = [p.grad for p in model.parameters()]
    with grid.extended_grid():
        control = complete.StartupAnchorUpdate(model, t, anchors(), opt)
        before = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(t.state_dict())
        cache, warmed, rng = copy.deepcopy(control._entries), set(control._warmed), q.screen.rng_state()
        def fail_after_adam(*args, **kwargs):
            assert all(int(s['step']) == 2 for s in opt.state.values())
            control._warmed.add(('injected transient warm state',))
            raise RuntimeError('injected nonlinear check error')
        monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
        monkeypatch.setattr(q, 'backtrack', fail_after_adam)
        with pytest.raises(RuntimeError, match='injected nonlinear'):
            control.perform_update(model, t, crops(), None, opt)
    for current, saved in zip((model.state_dict(), opt.state_dict(), t.state_dict()), before):
        assert_equal(current, saved)
    assert_equal(q.screen.rng_state(), rng)
    assert_equal(control._entries, cache)
    assert control._warmed == warmed and control._updates == 0
    assert all(p.grad is old and torch.equal(p.grad, torch.full_like(p, 19.)) for p, old in zip(model.parameters(), slots))
