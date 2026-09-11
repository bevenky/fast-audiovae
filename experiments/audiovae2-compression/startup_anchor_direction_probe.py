"""Disposable two-max versus all-twelve projection at the original first zero.

The immutable diagnostic replays the original 43 updates. Only its analysis
callback is replaced; both candidate directions then use the same finer grid.
No counterfactual candidate, optimizer state, or checkpoint is retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
import startup_anchor_grid_pilot as grid

anchor, q, base = grid.anchor, grid.q, grid.base
VERSION = 'audiovae2_startup_anchor_direction_probe_v1'
DIAGNOSTIC_SHA256 = '306e4f083b831fdd877b26b12c0b9afd385cc86fda5d6cfaa25a802461ec4e43'
FIRST_ZERO_UPDATE = 43


def scalar_solver_report(report):
    """No row identities or per-window solver arrays in the exported aggregate."""
    result = {key:value for key,value in report.items()
              if value is None or isinstance(value, (bool, int, float, str))}
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite projection diagnostic')
    if 'active_set' in report:
        result['active_constraint_count'] = len(report['active_set'])
    return result


def evaluate_direction(zero, model, entries, params, capture, baseline, rows,
                       projected, linear_report, measure_candidate, *, backtrack):
    """Check one direction on all six anchors, then measure its accepted point."""
    before, proposal = capture['before'], capture['proposal']
    attempts = []
    started = time.monotonic()
    def scored():
        index = len(attempts)
        if index >= len(grid.EXTENDED_FRACTIONS):
            raise RuntimeError('Counterfactual exceeded the fixed extended grid')
        fraction = grid.EXTENDED_FRACTIONS[index]
        score, double = zero.score_with_rounding(model, entries, lambda:q.score_entries(model, entries))
        actual = [p.detach().double()-old.double() for p,old in zip(params, before)]
        ideal = [fraction*d for d in projected]
        attempts.append({'fraction':fraction, 'scores':zero.summarize_score(score, double),
            'displacement':zero.displacement_metrics(actual, ideal),
            'linearization':zero.aggregate_predictions(rows, actual, score, ideal_delta=ideal,
                                                        qp_tolerance=linear_report.get('tolerance', 0.))})
        return score
    try:
        with torch.no_grad():
            for p,saved in zip(params, proposal):p.copy_(saved)
        with grid.extended_grid():
            fraction, score, raw_attempts = backtrack(params, before, proposal, projected,
                baseline, scored, corrected=linear_report['corrected'])
        if len(raw_attempts) != len(attempts) or any(
                a['fraction'] != b['fraction'] for a,b in zip(raw_attempts, attempts)):
            raise RuntimeError('Counterfactual score callback changed backtracking order')
        if len(score['signature']) != 6 or not all(passed for _,_,passed in score['signature']):
            raise RuntimeError('Accepted point does not satisfy all twelve true anchor constraints')
        for observed, raw in zip(attempts, raw_attempts):
            observed.update(accepted=raw['accepted'], maximum_switches=raw['maximum_switches'])
        actual = [p.detach().double()-old.double() for p,old in zip(params, before)]
        ordinary = [p.double()-old.double() for p,old in zip(proposal, before)]
        an, on = math.sqrt(q._dot(actual, actual)), math.sqrt(q._dot(ordinary, ordinary))
        metrics = measure_candidate()
        report = {'solver':scalar_solver_report(linear_report), 'attempts':attempts,
            'accepted_fraction':fraction, 'accepted_as_training':False,
            'accepted_actual_delta_l2':an,
            'accepted_actual_to_ordinary_norm_ratio':an/on if on else None,
            'accepted_actual_ordinary_cosine':q._dot(actual, ordinary)/(an*on) if an*on else None,
            'candidate_metrics':metrics, 'elapsed_seconds':time.monotonic()-started}
    finally:
        with torch.no_grad():
            for p,old in zip(params, before):p.copy_(old)
    if any(not torch.equal(p,old) for p,old in zip(params, before)) or q.score_entries(model, entries) != baseline:
        raise RuntimeError('Direction analysis did not restore the exact pre43 state and scores')
    report['pre43_parameters_and_scores_restored'] = True
    return report


def compare_directions(zero, model, entries, params, capture, *, measure_candidate,
                       project_all, backtrack):
    """Both arms share the same twelve gradients and actual original Adam delta."""
    if (len(params) != 90 or len(entries) != 6 or capture['accepted_fraction'] != 0
            or any(not torch.equal(p,old) for p,old in zip(params, capture['before']))):
        raise RuntimeError('Require the restored original pre43 group and rejected Adam proposal')
    baseline, double = zero.score_with_rounding(model, entries, lambda:q.score_entries(model, entries))
    if baseline != capture['baseline'] or not all(passed for _,_,passed in baseline['signature']):
        raise RuntimeError('Authenticated pre43 anchor baseline changed')
    rows = zero.all_constraint_rows(model, entries, baseline, params)
    if len(rows) != 12 or sum(r['selected'] for r in rows) != 2:
        raise RuntimeError('Require all twelve individual rows and exactly two selected maxima')
    ordinary = [p.double()-old.double() for p,old in zip(capture['proposal'], capture['before'])]
    selected = [next(r['gradient'] for r in rows if r['ref'] == value['ref'] and r['kind'] == key[1])
                for key,value in baseline['maxima']]
    two, two_report = q.project_displacement(selected, ordinary, [-v['value'] for _,v in baseline['maxima']])
    if not all(torch.equal(a,b) for a,b in zip(two, capture['projected'])):
        raise RuntimeError('Original two-max projection did not reproduce bitwise')
    all_rows, all_report = project_all([r['gradient'] for r in rows], ordinary, [-r['before'] for r in rows])
    if (two_report.get('kkt_passed') is not True or all_report.get('kkt_passed') is not True
            or all_report.get('full_primal_verified') is not True
            or all_report.get('rows') != 12 or len(all_rows) != len(params)):
        raise RuntimeError('A complete verified twelve-row projection is required')
    difference = [a-b for a,b in zip(all_rows, two)]
    report = {'baseline':zero.summarize_score(baseline, double),
        'baseline_metrics':measure_candidate(),
        'all_constraint_gradient_rank':zero.gradient_rank(rows),
        'original_two_max_projection_bitwise_reproduced':True,
        'ordinary_delta_l2':math.sqrt(q._dot(ordinary, ordinary)),
        'all12_minus_two_max_direction_l2':math.sqrt(q._dot(difference, difference)),
        'ordinary_delta_linear_predictions':zero.aggregate_predictions(rows, ordinary, None),
        'fractions':list(grid.EXTENDED_FRACTIONS), 'arms':{}}
    for name,direction,linear in (('two_max',two,two_report), ('all12',all_rows,all_report)):
        report['arms'][name] = evaluate_direction(zero, model, entries, params, capture, baseline,
            rows, direction, linear, measure_candidate, backtrack=backtrack)
    report.update(counterfactual_optimizer_updates=0, checkpoints_written=0,
        development_used_for_projection=False, nonlinear_correction=False,
        interpretation='A same-state direction comparison, not a training or convergence result. '
            'All six calibration windows are checked at every fraction; development is observation only. '
            'Linearization remainders contain nonlinear response and arithmetic.')
    return report


@contextmanager
def analysis_callback(zero, project_all):
    """Capture existing runtime returns unchanged; replace only final analysis."""
    old_analyze, old_build = zero.analyze_zero, zero.helper.build
    old_panel, old_objective = zero.probe.startup_panel, zero.base.objective
    original_backtrack = q.backtrack
    box = {'analysis_calls':0, 'build_calls':0, 'objective_calls':0}
    def build(config):
        result = old_build(config)
        box['build_calls'] += 1; box['runtime'] = result
        return result
    def panel(teacher, crops, expected):
        result = old_panel(teacher, crops, expected)
        if expected == 13:
            if 'development' in box:raise RuntimeError('Development panel prepared more than once')
            box['development'] = result[0]
        return result
    def objective():
        result = old_objective()
        box['objective_calls'] += 1; box['common'] = result
        return result
    def analyze(model, entries, params, capture):
        box['analysis_calls'] += 1
        if (box['analysis_calls'] != 1 or box['build_calls'] != 1 or box['objective_calls'] != 1
                or len(box.get('development', [])) != 13):
            raise RuntimeError('Immutable diagnostic runtime capture is incomplete or repeated')
        _,teacher,built_model,_,_,data,_ = box['runtime']
        if built_model is not model:raise RuntimeError('Analysis uses a different candidate')
        start = (FIRST_ZERO_UPDATE-1)*12
        crops = data.take(start, 12)
        if [c['source_id'] for c in crops] != list(data.source_ids[start:start+12]):
            raise RuntimeError('Pre43 objective does not use the exact original ordinary batch')
        def measure():
            return {'calibration':zero.probe.score_panel(model, entries),
                'development':zero.probe.score_panel(model, box['development']),
                'ordinary_fitting_batch_objective':zero.probe.ordinary_objective(model, teacher, crops, box['common'])}
        return compare_directions(zero, model, entries, params, capture, measure_candidate=measure,
                                  project_all=project_all, backtrack=original_backtrack)
    zero.helper.build, zero.probe.startup_panel, zero.base.objective, zero.analyze_zero = build,panel,objective,analyze
    try:
        yield box
    finally:
        zero.helper.build, zero.probe.startup_panel, zero.base.objective, zero.analyze_zero = old_build,old_panel,old_objective,old_analyze


def run_diagnostic(diagnostic_path, config, reference, out, project_all):
    name = '_startup_anchor_immutable_direction_base'
    previous, old_argv = sys.modules.get(name), sys.argv
    spec = importlib.util.spec_from_file_location(name, diagnostic_path)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot load the explicit immutable diagnostic')
    zero = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = zero; spec.loader.exec_module(zero)
        if zero.anchor is not anchor or zero.q is not q or zero.FIRST_ZERO_UPDATE != FIRST_ZERO_UPDATE:
            raise RuntimeError('Diagnostic imports different runtime objects or update bound')
        sys.argv = [str(diagnostic_path), '--config', str(config), '--reference', str(reference), '--out', str(out)]
        with analysis_callback(zero, project_all) as captured:
            zero.main()
            if captured['analysis_calls'] != 1:raise RuntimeError('Expected one pre43 direction analysis')
    finally:
        sys.argv = old_argv
        if previous is None:sys.modules.pop(name, None)
        else:sys.modules[name] = previous


def main():
    import startup_constraint_projection as projection
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic', 'config', 'reference', 'out'):parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():raise FileExistsError('Require a new direction-probe aggregate output')
    reference = json.loads(args.reference.read_text())
    if (base.sha(args.diagnostic) != DIAGNOSTIC_SHA256 or reference.get('complete') is not True
            or reference.get('updates') != 64 or reference.get('method') != 'anchor'
            or reference.get('all_preservation_checks_passed') is not True
            or reference.get('config_sha256') != base.sha(args.config)
            or next((r['step'] for r in reference.get('update_records', []) if r.get('q_zero_displacement')), None) != FIRST_ZERO_UPDATE):
        raise RuntimeError('Original first-zero source/reference/config authentication failed')
    for module in (anchor, q, anchor.warmup):
        if reference.get('source_sha256', {}).get(Path(module.__file__).name) != base.sha(module.__file__):
            raise RuntimeError('Original anchor/Q/warmup source changed')
    if q.FRACTIONS != grid.ORIGINAL_FRACTIONS or anchor.FRACTIONS != grid.ORIGINAL_FRACTIONS:
        raise RuntimeError('Original replay requires its unchanged fraction grid')
    paths = (args.diagnostic, args.config, args.reference, Path(__file__), Path(grid.__file__),
             Path(projection.__file__), Path(anchor.__file__), Path(q.__file__), Path(anchor.warmup.__file__))
    protected = {str(p.resolve()):base.sha(p) for p in paths}
    receipt = {'version':VERSION, 'diagnostic_source_sha256':DIAGNOSTIC_SHA256,
        'wrapper_source_sha256':base.sha(__file__), 'solver_source_sha256':base.sha(projection.__file__),
        'reference_aggregate_sha256':base.sha(args.reference), 'config_sha256':base.sha(args.config),
        'original_replay_fractions':list(grid.ORIGINAL_FRACTIONS),
        'counterfactual_fractions':list(grid.EXTENDED_FRACTIONS), 'failure_category':None,
        'source_sha256':{p.name:base.sha(p) for p in paths if p.suffix == '.py'},
        'two_max_and_all12_same_pre43_model_and_proposal':True, 'no_nonlinear_correction':True,
        'no_counterfactual_optimizer_step':True, 'no_checkpoint_written':True}
    error = None
    try:
        run_diagnostic(args.diagnostic, args.config, args.reference, args.out, projection.project_displacement)
    except BaseException as exc:
        error = exc; receipt['failure_category'] = type(exc).__name__
    finally:
        actual = json.loads(args.out.read_text()) if args.out.exists() else {'complete':False}
        receipt['protected_files_preserved'] = all(base.sha(p) == sha for p,sha in protected.items())
        receipt['fraction_globals_restored'] = q.FRACTIONS == grid.ORIGINAL_FRACTIONS and anchor.FRACTIONS == grid.ORIGINAL_FRACTIONS
        prefix = actual.get('reference_scalar_prefix', {})
        checks = {'immutable_diagnostic_completed':actual.get('complete') is True,
            'original43_scalar_prefix_exact':prefix.get('passed') is True and prefix.get('matched_updates') == FIRST_ZERO_UPDATE,
            'first_zero43':actual.get('first_zero_update') == actual.get('updates') == FIRST_ZERO_UPDATE,
            'original516_sources':actual.get('ordinary_unique_sources') == FIRST_ZERO_UPDATE*12,
            'fresh_state_preserved':actual.get('all_preservation_checks_passed') is True,
            'pre43_optimizer_rng_restored':actual.get('pre_zero_optimizer_and_rng_restored') is True,
            'files_preserved':receipt['protected_files_preserved'], 'globals_restored':receipt['fraction_globals_restored']}
        receipt['checks'] = checks; receipt['complete'] = error is None and all(checks.values())
        if not receipt['complete']:actual.update(complete=False, failure_category=receipt['failure_category'] or 'DirectionProbeVerificationFailure')
        actual['direction_probe'] = receipt
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(actual, indent=2, allow_nan=False)+'\n')
    if error is not None:raise error
    if not receipt['complete']:raise RuntimeError('Direction probe failed authenticated replay/preservation checks')
    print(json.dumps({'direction_probe_complete':True, 'replayed_updates':FIRST_ZERO_UPDATE, 'counterfactual_arms':2}))


if __name__ == '__main__':main()
