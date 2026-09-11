"""Fresh64 policy changes preserve initial exposure identity and scoped execution."""
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
import startup_complete_anchor_pilot as pilot
import startup_complete_anchor_update as complete
import startup_anchor_grid_pilot as grid


def test_same_initial_state_and_sources_allow_policy_difference_from_first_update():
    reference = report()
    reference['reviews'] = [{'step': 0, 'startup': {'windows': 6, 'passed': 6}}]
    reference['initialization'].update(initial_model_sha256='a' * 64,
                                       initial={'windows': 6, 'passed': 6})
    actual = copy.deepcopy(reference)
    actual['initialization']['constraints'] = 12
    actual['initialization']['version'] = complete.VERSION
    for row in actual['update_records']:
        row.update(total=-float(row['step']), q_constraints=12, q_constrained_sources=6,
                   q_constraint_gradient_sources=6, q_full_primal_verified=1, q_kkt_passed=1,
                   q_caps_passed=1, startup_anchor_after_passed=6,
                   startup_anchor_current_batch_constraints=0,
                   q_accepted_fraction=.5, q_zero_displacement=0)
    checked = pilot.validate_baseline_and_exposure(actual, reference)
    assert checked['passed']
    assert checked['policy_changes_begin_update'] == 1
    assert checked['historical_update_scalar_equality_required'] is False
    for damage in ('source', 'initial', 'short', 'omitted_constraint', 'certificate', 'anchor'):
        bad = copy.deepcopy(actual)
        if damage == 'source': bad['ordinary_source_prefix_sha256'] = 'f' * 64
        elif damage == 'initial': bad['initialization']['anchor_identity_sha256'] = '0' * 64
        elif damage == 'short': bad['update_records'].pop()
        elif damage == 'omitted_constraint': bad['update_records'][0]['q_constraints'] = 2
        elif damage == 'certificate': bad['update_records'][0]['q_full_primal_verified'] = 0
        else: bad['update_records'][0]['startup_anchor_after_passed'] = 5
        assert not pilot.validate_baseline_and_exposure(bad, reference)['passed'], damage


def test_immutable_main_runs_once_with_fixed_scope_and_restores_class_grid_argv(tmp_path, monkeypatch):
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
    assert anchor.StartupAnchorUpdate.__module__=='startup_complete_anchor_update'
    if Path(args.config).read_text()=='fail': raise RuntimeError('injected imported-main error')
    Path(args.out).write_text(json.dumps({'argv':sys.argv,'fractions':anchor.q.FRACTIONS}))
''')
    config, out = tmp_path / 'config.json', tmp_path / 'result.json'
    config.write_text('{}')
    old_argv = sys.argv
    original_class = grid.anchor.StartupAnchorUpdate
    original_q, original_anchor = grid.q.FRACTIONS, grid.anchor.FRACTIONS
    module_name = '_startup_anchor_immutable_complete_base'
    sentinel = SimpleNamespace(original=True)
    monkeypatch.setitem(sys.modules, module_name, sentinel)
    def forbidden(*args, **kwargs): raise AssertionError('Wrapper performed an optimizer update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    pilot.run_pilot(source, config, out)
    result = json.loads(out.read_text())
    assert result['argv'] == [str(source), '--config', str(config), '--out', str(out), '--updates', '64', '--method', 'anchor']
    assert result['fractions'] == list(grid.EXTENDED_FRACTIONS)
    assert Path(str(out) + '.calls').read_text() == 'called\n'
    assert sys.argv is old_argv and grid.anchor.StartupAnchorUpdate is original_class
    assert sys.modules[module_name] is sentinel
    assert grid.q.FRACTIONS is original_q and grid.anchor.FRACTIONS is original_anchor
    config.write_text('fail')
    with pytest.raises(RuntimeError, match='injected imported-main error'):
        pilot.run_pilot(source, config, tmp_path / 'failure.json')
    assert sys.argv is old_argv and grid.anchor.StartupAnchorUpdate is original_class
    assert sys.modules[module_name] is sentinel
    assert grid.q.FRACTIONS is original_q and grid.anchor.FRACTIONS is original_anchor
