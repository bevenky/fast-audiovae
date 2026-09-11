"""Fresh64 corrected-policy scope and same-start/source evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_startup_anchor_grid_pilot import report
import startup_corrected_anchor_pilot as pilot


def test_same_start_and_exposure_allow_update_one_policy_change_but_require_valid_counters():
    reference = report()
    reference.update(version=pilot.previous_pilot.VERSION, method='complete_anchor',
                     complete_anchor_policy={'complete': True},
                     reviews=[{'step': 0, 'startup': {'passed': 6}}])
    reference['initialization'].update(initial_model_sha256='a' * 64,
                                      initial={'windows': 6, 'passed': 6})
    actual = copy.deepcopy(reference)
    actual['method'] = 'anchor'
    actual['initialization'].update(version=pilot.corrected.VERSION, constraints=12, nonlinear_correction=True)
    for row in actual['update_records']:
        row.update(total=-float(row['step']), q_constraints=12, q_constrained_sources=6,
                   q_constraint_gradient_sources=6, q_full_primal_verified=1, q_kkt_passed=1, q_caps_passed=1,
                   q_normal_full_primal_verified=1, q_normal_kkt_passed=1, q_normal_budget_passed=1,
                   q_normal_correction_enabled=1, startup_anchor_after_passed=6,
                   q_accepted_fraction=1., q_base_fraction=1., q_normal_solves=0,
                   q_base_fraction_has_normal_correction=0, q_normal_accepted=0,
                   q_normal_accepted_norm=0., q_normal_budget=1., q_zero_displacement=0,
                   q_canonical_score_forwards=18, startup_anchor_checks=18, q_canonical_score_calls=3,
                   q_constraint_gradient_forwards=6, q_normal_gradient_sources=0)
    checked = pilot.validate_baseline_and_exposure(actual, reference)
    assert checked['passed'] and checked['policy_changes_begin_update'] == 1
    assert checked['historical_update_scalar_equality_required'] is False
    for damage in ('source', 'initial', 'short', 'rows', 'certificate', 'budget', 'count'):
        bad = copy.deepcopy(actual)
        if damage == 'source': bad['ordinary_source_prefix_sha256'] = 'f' * 64
        elif damage == 'initial': bad['initialization']['anchor_identity_sha256'] = '0' * 64
        elif damage == 'short': bad['update_records'].pop()
        elif damage == 'rows': bad['update_records'][0]['q_constraints'] = 2
        elif damage == 'certificate': bad['update_records'][0]['q_normal_kkt_passed'] = 0
        elif damage == 'budget': bad['update_records'][0]['q_normal_accepted_norm'] = 2.
        else: bad['update_records'][0]['startup_anchor_checks'] = 6
        assert not pilot.validate_baseline_and_exposure(bad, reference)['passed'], damage


def test_fixed64_driver_restores_class_grid_score_backtrack_argv_and_module_on_failure(tmp_path, monkeypatch):
    source = tmp_path / 'tiny_pilot.py'
    source.write_text('''import argparse, json, sys
from pathlib import Path
import startup_anchor_update as anchor
def main():
    parser=argparse.ArgumentParser()
    for name in ('config','out','updates','method'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    calls=Path(args.out+'.calls')
    calls.write_text(calls.read_text()+'called\\n' if calls.exists() else 'called\\n')
    assert anchor.StartupAnchorUpdate.__module__=='startup_corrected_anchor_update'
    if Path(args.config).read_text()=='fail': raise RuntimeError('injected imported-main error')
    Path(args.out).write_text(json.dumps({'argv':sys.argv,'fractions':anchor.q.FRACTIONS}))
''')
    config, out = tmp_path / 'config.json', tmp_path / 'result.json'
    config.write_text('{}')
    original = (pilot.anchor.StartupAnchorUpdate, pilot.q.FRACTIONS, pilot.anchor.FRACTIONS,
                pilot.q.backtrack, pilot.q.score_entries, sys.argv)
    module_name = '_startup_anchor_immutable_corrected_base'
    sentinel = SimpleNamespace(original=True)
    monkeypatch.setitem(sys.modules, module_name, sentinel)
    def forbidden(*args, **kwargs): raise AssertionError('Wrapper performed an extra Adam update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    pilot.run_pilot(source, config, out)
    result = json.loads(out.read_text())
    assert result['argv'] == [str(source), '--config', str(config), '--out', str(out), '--updates', '64', '--method', 'anchor']
    assert result['fractions'] == list(pilot.grid.EXTENDED_FRACTIONS)
    assert Path(str(out) + '.calls').read_text() == 'called\n'
    assert all(a is b for a, b in zip((pilot.anchor.StartupAnchorUpdate, pilot.q.FRACTIONS,
        pilot.anchor.FRACTIONS, pilot.q.backtrack, pilot.q.score_entries, sys.argv), original))
    assert sys.modules[module_name] is sentinel
    config.write_text('fail')
    with pytest.raises(RuntimeError, match='injected imported-main error'):
        pilot.run_pilot(source, config, tmp_path / 'failure.json')
    assert all(a is b for a, b in zip((pilot.anchor.StartupAnchorUpdate, pilot.q.FRACTIONS,
        pilot.anchor.FRACTIONS, pilot.q.backtrack, pilot.q.score_entries, sys.argv), original))
    assert sys.modules[module_name] is sentinel
