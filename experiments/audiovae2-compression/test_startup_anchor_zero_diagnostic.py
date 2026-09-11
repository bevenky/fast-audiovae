"""Transparent first-zero observation and aggregate-only numerical accounting."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import startup_anchor_zero_diagnostic as diagnostic
import test_startup_anchor_update as fixtures

q = diagnostic.q


def simple_score(p):
    value = float(p.detach()) - .3
    return {'maxima': [(('near_startup', 'residual'), {'value': value, 'ref': (0, 0)})],
            'signature': [((0, 0), (value,), value <= 0)]}


def test_observer_preserves_original_return_candidates_baseline_and_cleanup(monkeypatch):
    p = nn.Parameter(torch.ones(1))
    before, proposal, projected = [torch.zeros(1)], [torch.ones(1)], [torch.ones(1, dtype=torch.float64)]
    baseline = simple_score(before[0])
    saved = copy.deepcopy(baseline)
    original = q.backtrack
    expected = original([p], before, proposal, projected, baseline, lambda: simple_score(p), corrected=False)
    with torch.no_grad(): p.copy_(proposal[0])
    def observe_score(model, entries, score_fn):
        score = score_fn()
        return score, {ref: values for ref, values, _ in score['signature']}
    monkeypatch.setattr(diagnostic, 'score_with_rounding', observe_score)
    with diagnostic.observe_backtracking(None, []) as box:
        actual = q.backtrack([p], before, proposal, projected, baseline, lambda: simple_score(p), corrected=False)
    assert q.backtrack is original and actual == expected and baseline == saved
    assert box['calls'] == 1 and box['last']['accepted_fraction'] == .25
    attempts = box['last']['attempts']
    assert [a['fraction'] for a in attempts] == [1., .5, .25]
    assert [a['actual_delta'][0].item() for a in attempts] == [1., .5, .25]
    assert box['last']['before'][0].data_ptr() != before[0].data_ptr()
    def fail(*args, **kwargs): raise RuntimeError('injected score failure')
    monkeypatch.setattr(diagnostic, 'score_with_rounding', fail)
    with torch.no_grad(): p.copy_(proposal[0])
    with pytest.raises(RuntimeError, match='injected score failure'):
        with diagnostic.observe_backtracking(None, []):
            q.backtrack([p], before, proposal, projected, baseline, lambda: simple_score(p), corrected=False)
    assert q.backtrack is original


def test_final_head_observer_uses_only_existing_six_predictions_and_removes_hook():
    class FinalWave(fixtures.AnchorWave):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(model=[nn.Tanh()])
        def suffix_from_group(self, x):
            return self.decoder.model[-1](super().suffix_from_group(x))
    model, entries = FinalWave(), fixtures.anchors()
    with torch.no_grad():
        for i, p in enumerate(model.ps[:6]): p.fill_((i + 1) * 1e-6)
    layer = model.decoder.model[-1]
    hooks = len(layer._forward_hooks)
    with torch.no_grad():
        score, double = diagnostic.score_with_rounding(model, entries, lambda: q.score_entries(model, entries))
    assert len(model.seen) == 6 and len(double) == 6 and len(score['signature']) == 6
    assert len(layer._forward_hooks) == hooks
    for ref, values, _ in score['signature']:
        index = ref[0]
        predicted = torch.tanh(model.ps[index].detach()).double().item()
        limit = entries[index]['windows'][0]['residual_limit']
        expected = predicted**2 - limit**2
        assert double[ref] == pytest.approx((expected, expected), rel=1e-14, abs=1e-25)
    def failing_score():
        q._predict(model, entries[0])
        raise RuntimeError('injected native scoring failure')
    with pytest.raises(RuntimeError, match='injected native scoring'):
        diagnostic.score_with_rounding(model, entries, failing_score)
    assert len(layer._forward_hooks) == hooks


def test_all_twelve_rows_separate_selected_omitted_and_real_fp32_rounding_without_exporting_rows():
    before = torch.ones(2, dtype=torch.float32)
    ideal = [torch.tensor([2.**-25, .25], dtype=torch.float64)]
    actual = [(before.double() + ideal[0]).float().double() - before.double()]
    assert actual[0][0] == 0 and ideal[0][0] > 0
    rows, signature = [], []
    for source in range(6):
        measured = []
        for kind_index, kind in enumerate(q.KINDS):
            gradient = torch.tensor([source + 1., kind_index + 1.], dtype=torch.float64)
            rows.append({'ref': (source, 0), 'kind': kind, 'before': -1., 'selected': source == 0,
                         'gradient': [gradient], 'norm': math.hypot(source + 1, kind_index + 1)})
            measured.append(-1. + .25 * (kind_index + 1) + .125 * (source + 1))
        signature.append(((source, 0), tuple(measured), all(v <= 0 for v in measured)))
    result = diagnostic.aggregate_predictions(rows, actual, {'signature': signature}, ideal_delta=ideal)
    assert {name: result[name]['before']['count'] for name in result} == {
        'all': 12, 'selected': 2, 'omitted': 10, 'residual': 6, 'amplitude': 6}
    assert result['selected']['gradient_dot_actual']['mean'] == .375
    assert result['selected']['parameter_rounding_linear_change']['mean'] == -2.**-25
    assert result['omitted']['parameter_rounding_linear_change']['mean'] == -4 * 2.**-25
    assert result['selected']['remainder']['mean'] == .125
    assert result['omitted']['remainder']['mean'] == .5
    assert result['all']['after']['positive_count'] == 2
    assert result['all']['linear_actual']['positive_count'] == 0
    assert '"ref"' not in json.dumps(result) and '"gradient"' not in json.dumps(result)
    unmeasured = diagnostic.aggregate_predictions(rows, actual, None, ideal_delta=ideal)
    assert all('after' not in row and 'remainder' not in row for row in unmeasured.values())
