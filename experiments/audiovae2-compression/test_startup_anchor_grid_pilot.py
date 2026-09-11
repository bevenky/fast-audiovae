"""Only extend the positive grid; retain fixed pilot scope and cleanup."""
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
import startup_anchor_grid_pilot as grid


def score(p):
    value = float(p.detach().square()) - 1
    return {'maxima': [(('near_startup', 'residual'), {'value': value, 'ref': (0, 0)})],
            'signature': [((0, 0), (value,), value <= 0)]}


def test_real_backtracking_preserves_old_positive_prefix_and_reaches_new_nonzero_fraction():
    q, anchor = grid.q, grid.anchor
    original_q, original_anchor = q.FRACTIONS, anchor.FRACTIONS
    p = nn.Parameter(torch.tensor([20.]))
    before, proposal, delta = [torch.zeros(1)], [p.detach().clone()], [torch.tensor([20.], dtype=torch.float64)]
    baseline = score(before[0])
    old = q.backtrack([p], before, proposal, delta, baseline, lambda: score(p), corrected=False)
    assert old[0] == 0
    with torch.no_grad(): p.copy_(proposal[0])
    with grid.extended_grid():
        expected = original_q[:-1] + tuple(2.**-n for n in range(5, 11)) + (0.,)
        assert q.FRACTIONS == anchor.FRACTIONS == expected
        actual = q.backtrack([p], before, proposal, delta, baseline, lambda: score(p), corrected=False)
    assert actual[0] == 1 / 32 and p.item() == .625
    assert actual[2][:-1] == old[2][:-1]
    assert q.FRACTIONS is original_q and anchor.FRACTIONS is original_anchor
    with pytest.raises(RuntimeError, match='injected pilot failure'):
        with grid.extended_grid():
            raise RuntimeError('injected pilot failure')
    assert q.FRACTIONS is original_q and anchor.FRACTIONS is original_anchor


def report():
    return {'complete': True, 'updates': 64, 'method': 'anchor', 'all_preservation_checks_passed': True,
            'initial_state_sha256': 'a' * 64, 'config_sha256': 'b' * 64,
            'ordinary_unique_sources': 768, 'ordinary_source_prefix_sha256': 'c' * 64,
            'development_before': {'aggregate': {'mae': .01}},
            'fixed_fitting_batch_objective_before': {'total': .125},
            'initialization': {'anchor_identity_sha256': 'd' * 64, 'cached_target_input_sha256': 'e' * 64},
            'update_records': [{'step': i, 'total': float(i), 'q_accepted_fraction': 0. if i >= 43 else 1.,
                                'q_zero_displacement': int(i >= 43), 'q_step_seconds': .1} for i in range(1, 65)]}


def test_prefix_receipt_requires_exact_first42_and_source_identity_but_allows_treatment_after43():
    reference = report()
    actual = copy.deepcopy(reference)
    for row in actual['update_records']:
        row['q_step_seconds'] += row['step']
        if row['step'] >= 43:
            row.update(total=-1., q_accepted_fraction=1 / 32, q_zero_displacement=0)
    receipt = grid.validate_prefix(actual, reference)
    assert receipt['passed'] and receipt['matched_updates'] == 42
    assert receipt['full_tensor_trajectory_claim'] is False and receipt['mismatches'] == []
    for damage in ('loss', 'field', 'source', 'short_run', 'missing_record', 'nonfinite'):
        bad = copy.deepcopy(actual)
        if damage == 'loss': bad['update_records'][41]['total'] += .001
        elif damage == 'field': del bad['update_records'][0]['q_zero_displacement']
        elif damage == 'source': bad['ordinary_source_prefix_sha256'] = 'f' * 64
        elif damage == 'short_run': bad['updates'] = 42
        elif damage == 'missing_record': bad['update_records'].pop()
        else: bad['update_records'][0]['total'] = float('nan')
        assert not grid.validate_prefix(bad, reference)['passed'], damage


def test_explicit_pilot_runs_once_with_only_fixed64_anchor_args_and_restores_globals(tmp_path, monkeypatch):
    # No model/optimizer execution: the tiny imported main records its invocation.
    pilot = tmp_path / 'tiny_pilot.py'
    pilot.write_text('''import argparse, json, sys
from pathlib import Path
import startup_anchor_update as anchor
def main():
    parser=argparse.ArgumentParser()
    for name in ('config','out','updates','method'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    calls=Path(args.out+'.calls')
    calls.write_text(calls.read_text()+'called\\n' if calls.exists() else 'called\\n')
    if Path(args.config).read_text()=='fail': raise RuntimeError('injected imported-main failure')
    Path(args.out).write_text(json.dumps({'argv':sys.argv,'fractions':anchor.q.FRACTIONS}))
''')
    config, out = tmp_path / 'config.json', tmp_path / 'result.json'
    config.write_text('{}')
    old_argv, old_q, old_anchor = sys.argv, grid.q.FRACTIONS, grid.anchor.FRACTIONS
    name = '_startup_anchor_immutable_grid_base'
    sentinel = SimpleNamespace(original=True)
    monkeypatch.setitem(sys.modules, name, sentinel)
    def forbidden(*args, **kwargs): raise AssertionError('Wrapper performed an extra optimizer update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    grid.run_pilot(pilot, config, out)
    result = json.loads(out.read_text())
    assert result['argv'] == [str(pilot), '--config', str(config), '--out', str(out), '--updates', '64', '--method', 'anchor']
    assert result['fractions'] == list(grid.EXTENDED_FRACTIONS)
    assert Path(str(out) + '.calls').read_text() == 'called\n'
    assert sys.argv is old_argv and sys.modules[name] is sentinel
    assert grid.q.FRACTIONS is old_q and grid.anchor.FRACTIONS is old_anchor
    config.write_text('fail')
    with pytest.raises(RuntimeError, match='injected imported-main failure'):
        grid.run_pilot(pilot, config, tmp_path / 'failure.json')
    assert sys.argv is old_argv and sys.modules[name] is sentinel
    assert grid.q.FRACTIONS is old_q and grid.anchor.FRACTIONS is old_anchor
