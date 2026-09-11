"""Startup-only projection: native acceptance, real Adam and rollback contracts."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import quiet_projected_update as q
import startup_anchor_update as anchor


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = q.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(602)
    torch.set_num_threads(1)
    yield
    q.screen.restore_rng(rng)
    torch.set_num_threads(threads)


class AnchorWave(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.ps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(90)])
        self.register_buffer('scale', torch.tensor(scale))
        self.seen = []

    def group_named_parameters(self):
        return list(self.named_parameters())

    def group_from_input(self, x):
        return x

    def suffix_from_group(self, x):
        index = int(x[0, 0, 0])
        self.seen.append((index, torch.is_grad_enabled()))
        # Every raw parameter participates in the same graph; six are audible.
        return x * 0 + self.ps[index] * self.scale + sum(p * 0 for p in self.ps)


def anchors():
    result = []
    for index in range(6):
        target = torch.zeros(1, 1, 1920)
        crop = {'source_id': f'calibration-{index}', 'context_frames': 0, 'start_frame': 0,
                'context_start_frame': 0, 'valid_scored_samples': 1920}
        e = q.window_layout(crop, target)
        e['windows'] = [e['windows'][0]]
        e['group_input'] = torch.full_like(target, float(index))
        result.append(e)
    return result


def crops():
    return [{'source_id': f'fit-{index}'} for index in range(12)]


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(.9, .99), eps=1e-8, weight_decay=0)


def teacher():
    return nn.Linear(1, 1).eval().requires_grad_(False)


def ordinary_update(model, teacher, crops, common, optimizer, *, diagnostics=False):
    # A real native Adam step; no manually installed proposal or fabricated state.
    torch.rand(()); random.random(); np.random.random()
    optimizer.zero_grad(set_to_none=True)
    coefficients = common if common is not None else [-1.] * 90
    loss = sum(p.sum() * value for p, value in zip(model.parameters(), coefficients))
    loss.backward()
    optimizer.step()
    return {'fixture_loss': float(loss.detach())}, [{'source_id': c['source_id']} for c in crops]


def assert_equal(actual, expected):
    assert q.replay.compare_tree(actual, expected)['equal']


def apply_update(model, t, fit, common, opt, *, entries=None, diagnostics=False):
    control = anchor.StartupAnchorUpdate(model, t, anchors() if entries is None else entries, opt)
    return control.perform_update(model, t, fit, common, opt, diagnostics=diagnostics)


def test_no_conflict_is_exact_ordinary_adam_including_moments_and_rng(monkeypatch):
    model = AnchorWave(scale=.1)
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    before_rng = q.screen.rng_state()
    expected_values, expected_checks = ordinary_update(reference, t, crops(), None, ref_opt, diagnostics=True)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(before_rng)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    values, checks = apply_update(model, t, crops(), None, opt, diagnostics=True)
    assert_equal(model.state_dict(), reference.state_dict())
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert values['fixture_loss'] == expected_values['fixture_loss'] and checks == expected_checks
    assert values['q_accepted_fraction'] == 1 and values['q_projected'] == 0
    assert values['startup_anchor_after_passed'] == 6
    assert all(torch.equal(p.grad, r.grad) for p, r in zip(model.parameters(), reference.parameters()))
    assert all(r['passed'] for e in anchors() for r in q.window_excesses(q._predict(model, e), e))


def test_selected_maximum_switch_rechecks_all_six_and_keeps_feasible_nonzero_step(monkeypatch):
    model = AnchorWave()
    with torch.no_grad():
        model.ps[0].fill_(8e-6)
        model.ps[1].fill_(8e-6)
    es = anchors()
    baseline = q.score_entries(model, es)
    assert all(value['ref'][0] == 0 for _, value in baseline['maxima'])
    # The selected first anchor improves; the unselected second becomes worst.
    coefficients = [-1.] * 90
    coefficients[0] = 1.
    before = [p.detach().clone() for p in model.parameters()]
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    opt = optimizer(model)
    values, _ = apply_update(model, teacher(), crops(), coefficients, opt, entries=es)
    after = q.score_entries(model, es)
    assert all(value['ref'][0] == 1 for _, value in after['maxima'])
    assert all(passed for _, _, passed in after['signature']) and len(after['signature']) == 6
    # The fixed existing grid accepts 1/16 here, rather than zero or full Adam.
    torch.testing.assert_close(model.ps[1], before[1] + 3e-5 / 16, rtol=1e-6, atol=1e-12)
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    assert all(int(s['step']) == 1 for s in opt.state.values())
    assert {index for index, grad in model.seen if not grad} == set(range(6))
    assert values['q_maximum_switches'] > 0 and values['q_accepted_fraction'] == 1 / 16
    assert values['startup_anchor_checks'] == 6 * (1 + values['q_backtrack_rounds'])


def test_zero_constraint_gradient_still_checks_finite_native_displacement(monkeypatch):
    model, es = AnchorWave(), anchors()
    baseline = q.score_entries(model, es)
    rows = q.constraint_gradients(model, es, baseline, list(model.parameters()))
    assert all(not g.any() and torch.isfinite(g).all() for row in rows for g in row)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    opt = optimizer(model)
    values, _ = apply_update(model, teacher(), crops(), None, opt, entries=es)
    # Full3e-5 violates; quarter-step7.5e-6 satisfies every original limit.
    torch.testing.assert_close(model.ps[0], torch.tensor([7.5e-6]), rtol=1e-6, atol=1e-12)
    assert all(r['passed'] for e in es for r in q.window_excesses(q._predict(model, e), e))
    assert values['q_nonzero_rows'] == 0 and values['q_accepted_fraction'] == .25


def test_accepted_zero_restores_parameters_but_advances_real_adam_once(monkeypatch):
    model, es = AnchorWave(scale=100.), anchors()
    reference = copy.deepcopy(model)
    opt, ref_opt, t = optimizer(model), optimizer(reference), teacher()
    before = copy.deepcopy(model.state_dict())
    rng = q.screen.rng_state()
    ordinary_update(reference, t, crops(), None, ref_opt)
    expected_rng = q.screen.rng_state()
    q.screen.restore_rng(rng)
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    values, _ = apply_update(model, t, crops(), None, opt, entries=es)
    assert_equal(model.state_dict(), before)
    assert_equal(opt.state_dict(), ref_opt.state_dict())
    assert_equal(q.screen.rng_state(), expected_rng)
    assert all(int(s['step']) == 1 for s in opt.state.values())
    assert all(torch.equal(p.grad, torch.full_like(p, -1.)) for p in model.parameters())
    assert values['q_accepted_fraction'] == 0 and values['q_zero_displacement'] == 1
    assert values['q_backtrack_rounds'] == len(q.FRACTIONS)


def test_initial_failed_anchor_rejects_before_ordinary_adam(monkeypatch):
    model, opt = AnchorWave(), None
    opt = optimizer(model)
    with torch.no_grad():
        model.ps[5].fill_(1.01e-5)
    calls = []
    monkeypatch.setattr(q.prior, 'perform_update', lambda *a, **k: calls.append(True))
    with pytest.raises((RuntimeError, ValueError)):
        apply_update(model, teacher(), crops(), None, opt)
    assert not calls and not opt.state


def test_auxiliary_gradients_do_not_pollute_ordinary_gradient_slots(monkeypatch):
    model = AnchorWave(scale=.1)
    for p in model.parameters():
        p.grad = torch.full_like(p, 23.)
    slots = [p.grad for p in model.parameters()]
    called = []
    def checked_ordinary(*args, **kwargs):
        assert all(p.grad is slot and torch.equal(p.grad, torch.full_like(p, 23.))
                   for p, slot in zip(model.parameters(), slots))
        called.append(True)
        return ordinary_update(*args, **kwargs)
    monkeypatch.setattr(q.prior, 'perform_update', checked_ordinary)
    apply_update(model, teacher(), crops(), None, optimizer(model))
    assert called == [True]


def test_hard_failure_after_adam_rolls_back_parameters_moments_rng_and_gradient_objects(monkeypatch):
    model = AnchorWave(scale=.1)
    opt, t = optimizer(model), teacher()
    # Populate genuine nonempty moments before the transaction under test.
    ordinary_update(model, t, crops(), None, opt)
    for p in model.parameters():
        p.grad = torch.full_like(p, 19.)
    slots = [p.grad for p in model.parameters()]
    before = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(t.state_dict())
    rng = q.screen.rng_state()
    def fail_after_adam(*args, **kwargs):
        assert all(int(s['step']) == 2 for s in opt.state.values())
        raise RuntimeError('injected nonlinear verification failure')
    monkeypatch.setattr(q.prior, 'perform_update', ordinary_update)
    monkeypatch.setattr(q, 'backtrack', fail_after_adam)
    with pytest.raises(RuntimeError, match='injected nonlinear'):
        apply_update(model, t, crops(), None, opt)
    for current, saved in zip((model.state_dict(), opt.state_dict(), t.state_dict()), before):
        assert_equal(current, saved)
    assert_equal(q.screen.rng_state(), rng)
    assert all(p.grad is slot and torch.equal(p.grad, torch.full_like(p, 19.))
               for p, slot in zip(model.parameters(), slots))


@pytest.mark.parametrize('invalid', ['duplicate', 'five', 'interior', 'context', 'partial', 'attached'])
def test_anchor_geometry_requires_six_distinct_original_complete_startups(invalid):
    es = anchors()
    if invalid == 'duplicate':
        es[-1]['crop']['source_id'] = es[0]['crop']['source_id']
    elif invalid == 'five':
        es.pop()
    elif invalid == 'interior':
        es[0]['crop']['start_frame'] = 1
        es[0]['crop']['context_start_frame'] = 1
        es[0]['windows'][0]['source_start_sample'] = 1920
        es[0]['windows'][0]['source_stop_sample'] = 2880
    elif invalid == 'context':
        es[0]['span'] = (1920, 3840)
        es[0]['crop']['context_frames'] = 1
    elif invalid == 'partial':
        es[0]['windows'][0]['valid_samples'] = 959
        es[0]['windows'][0]['source_stop_sample'] = 959
    else:
        es[0]['group_input'].requires_grad_()
    with pytest.raises((RuntimeError, ValueError)):
        anchor.validate_anchors(es)
