"""Same-proposal native comparisons and scoped callback/state preservation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_anchor_direction_probe as probe
import startup_anchor_zero_diagnostic as zero
import startup_constraint_projection as full
import test_startup_anchor_update as fixtures

q = probe.q


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = q.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(332)
    torch.set_num_threads(1)
    yield
    q.screen.restore_rng(rng)
    torch.set_num_threads(threads)


def case():
    class FinalWave(fixtures.AnchorWave):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.model = nn.ModuleList([nn.Tanh()])
        def suffix_from_group(self, x):
            return self.decoder.model[-1](super().suffix_from_group(x))
    model, entries = FinalWave(), fixtures.anchors()
    params = list(model.parameters())
    with torch.no_grad():
        for p in params[:6]: p.fill_(9.5e-6)
    before = [p.detach().clone() for p in params]
    baseline = q.score_entries(model, entries)
    selected = q.constraint_gradients(model, entries, baseline, params)
    optimizer = fixtures.optimizer(model)
    for i, p in enumerate(params): p.grad = torch.full_like(p, 1. if i == 0 else -1.)
    optimizer.step()
    proposal = [p.detach().clone() for p in params]
    delta = [a.double() - b.double() for a, b in zip(proposal, before)]
    projected, linear = q.project_displacement(selected, delta, [-v['value'] for _, v in baseline['maxima']])
    fraction, _, _ = q.backtrack(params, before, proposal, projected, baseline,
                                  lambda: q.score_entries(model, entries), corrected=linear['corrected'])
    assert fraction == 0
    capture = {'before': before, 'proposal': proposal, 'projected': projected,
               'baseline': baseline, 'accepted_fraction': fraction}
    return model, entries, params, optimizer, capture, linear


def test_both_directions_share_real_adam_proposal_and_restore_native_state_without_an_optimizer_step(monkeypatch):
    model, entries, params, optimizer, capture, _ = case()
    saved_opt, saved_rng = copy.deepcopy(optimizer.state_dict()), q.screen.rng_state()
    slots = [(p.grad, p.grad.clone()) for p in params]
    original_backtrack = q.backtrack
    calls, measurements = [], []
    def backtrack(*args, **kwargs):
        assert all(torch.equal(p, proposed) for p, proposed in zip(args[0], capture['proposal']))
        assert all(torch.equal(a, b) for a, b in zip(args[1], capture['before']))
        assert all(torch.equal(a, b) for a, b in zip(args[2], capture['proposal']))
        calls.append(q.FRACTIONS)
        return original_backtrack(*args, **kwargs)
    def forbidden(*args, **kwargs): raise AssertionError('Counterfactual entered replay observer or Adam')
    monkeypatch.setattr(q, 'backtrack', forbidden)
    monkeypatch.setattr(optimizer, 'step', forbidden)
    def measure():
        measurements.append([p.detach().clone() for p in params])
        return {'passed': sum(passed for _, _, passed in q.score_entries(model, entries)['signature'])}
    report = probe.compare_directions(zero, model, entries, params, capture,
        measure_candidate=measure, project_all=full.project_displacement, backtrack=backtrack)
    assert calls == [probe.grid.EXTENDED_FRACTIONS] * 2 and len(measurements) == 3
    assert report['arms']['two_max']['accepted_fraction'] == 1 / 64
    assert report['arms']['all12']['accepted_fraction'] > report['arms']['two_max']['accepted_fraction']
    assert all(arm['candidate_metrics']['passed'] == 6 and arm['pre43_parameters_and_scores_restored']
               and not arm['accepted_as_training'] for arm in report['arms'].values())
    assert all(torch.equal(p, old) for p, old in zip(params, capture['before']))
    assert q.replay.compare_tree(optimizer.state_dict(), saved_opt)['equal']
    assert q.replay.compare_tree(q.screen.rng_state(), saved_rng)['equal']
    assert all(p.grad is slot and torch.equal(p.grad, data) for p, (slot, data) in zip(params, slots))
    assert q.FRACTIONS == probe.anchor.FRACTIONS == probe.grid.ORIGINAL_FRACTIONS
    assert report['counterfactual_optimizer_updates'] == report['checkpoints_written'] == 0
    serialized = json.dumps(report, allow_nan=False)
    assert all(e['crop']['source_id'] not in serialized for e in entries)
    assert all('"' + key + '"' not in serialized for key in ('ref', 'gradient', 'target', 'group_input'))


def test_candidate_measurement_exception_restores_written_parameters_and_grid():
    model, entries, params, _, capture, linear = case()
    baseline = capture['baseline']
    rows = zero.all_constraint_rows(model, entries, baseline, params)
    old_q, old_anchor = q.FRACTIONS, probe.anchor.FRACTIONS
    def fail():
        assert any(not torch.equal(p, old) for p, old in zip(params, capture['before']))
        raise RuntimeError('injected candidate measurement failure')
    with pytest.raises(RuntimeError, match='injected candidate measurement'):
        probe.evaluate_direction(zero, model, entries, params, capture, baseline, rows,
            capture['projected'], linear, fail, backtrack=q.backtrack)
    assert all(torch.equal(p, old) for p, old in zip(params, capture['before']))
    assert q.FRACTIONS is old_q and probe.anchor.FRACTIONS is old_anchor
    assert q.score_entries(model, entries) == baseline


def test_runtime_capture_returns_original_objects_and_restores_callbacks_even_after_analysis_failure(monkeypatch):
    model, teacher, common, solver = object(), object(), object(), object()
    sources = [f'fixture-{i}' for i in range(516)]
    takes = []
    def take(start, count):
        takes.append((start, count))
        return [{'source_id': s} for s in sources[start:start + count]]
    data = SimpleNamespace(source_ids=sources, take=take)
    runtime = (None, teacher, model, None, None, data, None)
    panels = {6: (list(range(6)), {'count': 6}), 13: (list(range(13)), {'count': 13})}
    def build(config): return runtime
    def panel(t, crops, expected): return panels[expected]
    def objective(): return common
    def old_analyze(*args): raise AssertionError('Old analysis callback ran')
    local = SimpleNamespace(helper=SimpleNamespace(build=build), base=SimpleNamespace(objective=objective),
        probe=SimpleNamespace(startup_panel=panel, score_panel=lambda m, e: {'windows': len(e)},
            ordinary_objective=lambda m, t, crops, c: {'sources': len(crops)}), analyze_zero=old_analyze)
    old_backtrack = q.backtrack
    results = []
    def compare(z, m, entries, params, capture, **kwargs):
        assert z is local and m is model and kwargs['project_all'] is solver
        assert kwargs['backtrack'] is old_backtrack
        results.append(kwargs['measure_candidate']())
        return {'compared': True}
    monkeypatch.setattr(probe, 'compare_directions', compare)
    with pytest.raises(RuntimeError, match='after successful capture'):
        with probe.analysis_callback(local, solver) as box:
            assert local.helper.build({}) is runtime
            assert local.probe.startup_panel(teacher, [], 6) is panels[6]
            assert local.probe.startup_panel(teacher, [], 13) is panels[13]
            assert local.base.objective() is common
            monkeypatch.setattr(q, 'backtrack', lambda *a, **k: (_ for _ in ()).throw(AssertionError('replay observer used')))
            assert local.analyze_zero(model, list(range(6)), [], {}) == {'compared': True}
            assert box['analysis_calls'] == box['build_calls'] == box['objective_calls'] == 1
            raise RuntimeError('after successful capture')
    assert takes == [(504, 12)]
    assert results == [{'calibration': {'windows': 6}, 'development': {'windows': 13},
                        'ordinary_fitting_batch_objective': {'sources': 12}}]
    assert local.helper.build is build and local.probe.startup_panel is panel
    assert local.base.objective is objective and local.analyze_zero is old_analyze
