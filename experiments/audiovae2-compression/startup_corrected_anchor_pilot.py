"""Independent fresh64 bounded-correction pilot using the immutable driver.

Initialization and 768 ordinary sources match the completed complete-row pilot.
The corrected policy starts at update one, so no historical step equality is
claimed or required. No checkpoints or automatic continuation are produced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys

import startup_corrected_anchor_update as corrected

complete, anchor, q, grid, base = corrected.complete, corrected.anchor, corrected.q, corrected.grid, corrected.base
previous_pilot = corrected.correction.pilot
VERSION = 'audiovae2_corrected_startup_anchor_pilot_v1'
ORIGINAL_CLASS = previous_pilot.ORIGINAL_CLASS
PINNED_SOURCES = {**corrected.correction.SOURCE_PINS,
    'startup_anchor_correction_probe.py':'9ea35174a60169831e7b85ee37bc2f4988977b979273c22d78f1088142eb0062',
    'startup_signed_constraint_projection.py':'fe3b0c8e15569079ea645b0b32539d3b0978e05f2c41e3022e303c5a062b70b5'}


@contextmanager
def corrected_policy():
    old_class = anchor.StartupAnchorUpdate
    old_backtrack,old_score = q.backtrack,q.score_entries
    if old_class is not ORIGINAL_CLASS:
        raise RuntimeError('Original anchor class was replaced before the corrected pilot')
    anchor.StartupAnchorUpdate = corrected.StartupAnchorUpdate
    try:
        with grid.extended_grid():yield
    finally:
        unchanged = (anchor.StartupAnchorUpdate is corrected.StartupAnchorUpdate
                     and q.backtrack is old_backtrack and q.score_entries is old_score)
        anchor.StartupAnchorUpdate = old_class
        q.backtrack,q.score_entries = old_backtrack,old_score
        if not unchanged:raise RuntimeError('A scoped corrected-pilot runtime object was not restored')


def validate_baseline_and_exposure(actual,reference):
    checks = {
        'reference_completed64':reference.get('complete') is True and reference.get('updates') == 64
            and reference.get('version') == previous_pilot.VERSION and reference.get('method') == 'complete_anchor'
            and reference.get('all_preservation_checks_passed') is True
            and reference.get('complete_anchor_policy',{}).get('complete') is True,
        'candidate_completed64':actual.get('complete') is True and actual.get('updates') == 64
            and actual.get('method') == 'anchor' and actual.get('all_preservation_checks_passed') is True,
        'initial_state_exact':isinstance(reference.get('initial_state_sha256'),str)
            and actual.get('initial_state_sha256') == reference['initial_state_sha256'],
        'config_exact':isinstance(reference.get('config_sha256'),str)
            and actual.get('config_sha256') == reference['config_sha256'],
        'original768_source_prefix_exact':actual.get('ordinary_unique_sources') == reference.get('ordinary_unique_sources') == 768
            and isinstance(reference.get('ordinary_source_prefix_sha256'),str)
            and actual.get('ordinary_source_prefix_sha256') == reference['ordinary_source_prefix_sha256'],
        'initial_full96_exact':isinstance(reference.get('development_before'),dict)
            and actual.get('development_before') == reference['development_before'],
        'initial_fitting_objective_exact':isinstance(reference.get('fixed_fitting_batch_objective_before'),dict)
            and actual.get('fixed_fitting_batch_objective_before') == reference['fixed_fitting_batch_objective_before'],
        'initial_startup_review_exact':bool(actual.get('reviews')) and bool(reference.get('reviews'))
            and actual['reviews'][0] == reference['reviews'][0] and actual['reviews'][0].get('step') == 0,
        'corrected_constraint_receipt':actual.get('initialization',{}).get('version') == corrected.VERSION
            and actual.get('initialization',{}).get('constraints') == 12
            and actual.get('initialization',{}).get('nonlinear_correction') is True,
    }
    for key in ('anchor_identity_sha256','cached_target_input_sha256','initial_model_sha256','initial'):
        wanted = reference.get('initialization',{}).get(key)
        checks[key+'_exact'] = wanted is not None and actual.get('initialization',{}).get(key) == wanted
    records = actual.get('update_records',[])
    checks['all64_complete_corrected_updates'] = len(records) == 64 and all(
        row.get('step') == step and row.get('q_constraints') == 12
        and row.get('q_constrained_sources') == row.get('q_constraint_gradient_sources') == 6
        and row.get('q_full_primal_verified') == row.get('q_kkt_passed') == row.get('q_caps_passed') == 1
        and row.get('q_normal_full_primal_verified') == row.get('q_normal_kkt_passed') == row.get('q_normal_budget_passed') == 1
        and row.get('q_normal_correction_enabled') == 1
        and row.get('startup_anchor_after_passed') == 6
        and row.get('q_accepted_fraction') in grid.EXTENDED_FRACTIONS
        and row.get('q_base_fraction') == row.get('q_accepted_fraction')
        and row.get('q_normal_solves') in (0,1,2)
        and row.get('q_base_fraction_has_normal_correction') == row.get('q_normal_accepted')
        and (not row.get('q_normal_accepted') or row.get('q_base_fraction') == 1.)
        and row.get('q_normal_accepted_norm',math.inf) <= row.get('q_normal_budget',-1.)
        and row.get('q_canonical_score_forwards') == row.get('startup_anchor_checks') == 6*row.get('q_canonical_score_calls',-1)
        and row.get('q_constraint_gradient_forwards') == 6+row.get('q_normal_gradient_sources',-1)
        and row.get('q_normal_gradient_sources') == 6*row.get('q_normal_solves',-1)
        and all(not isinstance(v,float) or math.isfinite(v) for v in row.values())
        for step,row in enumerate(records,1))
    return {'passed':all(checks.values()),'checks':checks,'policy_changes_begin_update':1,
            'historical_update_scalar_equality_required':False,'full_tensor_trajectory_claim':False}


def run_pilot(pilot_path,config,out):
    name = '_startup_anchor_immutable_corrected_base'
    previous,old_argv = sys.modules.get(name),sys.argv
    spec = importlib.util.spec_from_file_location(name,pilot_path)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot load explicit immutable pilot source')
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = module; spec.loader.exec_module(module)
        if module.anchor is not anchor or module.anchor.q is not q:
            raise RuntimeError('Pilot imports different anchor/Q runtime objects')
        sys.argv = [str(pilot_path),'--config',str(config),'--out',str(out),'--updates','64','--method','anchor']
        with corrected_policy():module.main()
    finally:
        sys.argv = old_argv
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','config','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new corrected-anchor aggregate output')
    reference = json.loads(args.reference.read_text())
    if (base.sha(args.pilot) != grid.PILOT_SHA256
            or reference.get('version') != previous_pilot.VERSION or reference.get('method') != 'complete_anchor'
            or reference.get('complete') is not True or reference.get('updates') != 64
            or reference.get('complete_anchor_policy',{}).get('complete') is not True
            or reference.get('all_preservation_checks_passed') is not True
            or reference.get('config_sha256') != base.sha(args.config)):
        raise RuntimeError('Immutable pilot/complete64 reference/config authentication failed')
    modules = (anchor,q,anchor.warmup,grid,complete,complete.projection,previous_pilot,
               corrected.correction,corrected.correction.signed,corrected.correction.zero)
    for module in modules:
        path = Path(module.__file__)
        if base.sha(path) != PINNED_SOURCES[path.name]:
            raise RuntimeError('A frozen complete/correction runtime source changed')
    for module in (anchor,q,anchor.warmup,complete,complete.projection,previous_pilot,grid):
        recorded = reference.get('complete_anchor_policy',{}).get('source_sha256',{}).get(Path(module.__file__).name)
        # The prior wrapper's receipt includes itself and every policy dependency.
        if recorded != base.sha(module.__file__):
            raise RuntimeError('Complete64 runtime source differs from the preserved reference')
    paths = (args.pilot,args.config,args.reference,Path(__file__),Path(corrected.__file__),
             *(Path(module.__file__) for module in modules))
    protected = {str(path.resolve()):base.sha(path) for path in paths}
    old_backtrack,old_score = q.backtrack,q.score_entries
    receipt = {'version':VERSION,'constraints':12,'calibration_sources':6,
        'pilot_source_sha256':grid.PILOT_SHA256,'source_sha256':{p.name:base.sha(p) for p in paths if p.suffix == '.py'},
        'config_sha256':base.sha(args.config),'reference_aggregate_sha256':base.sha(args.reference),
        'fractions':list(grid.EXTENDED_FRACTIONS),'maximum_updates':64,'ordinary_unique_sources':768,
        'failure_category':None,'policy_changes_begin_update':1,'nonlinear_correction':True,
        'maximum_normal_solves':2,'normal_scales':[1.,2.],'normal_budget_fraction':.25,
        'normal_budget_definition':'Total actual FP32 correction from original full projected trial',
        'progress_definition':'Maximum positive relative excess using unchanged squared limits',
        'fraction_semantics':'Base projected fraction, plus separately reported accepted normal correction',
        'new_loss_weights':False,'new_inference_operations':0,'checkpoint_written':False,
        'automatic_continuation':False,'development_used_for_update':False}
    error = None
    try:
        run_pilot(args.pilot,args.config,args.out)
    except BaseException as exc:
        error = exc; receipt['failure_category'] = type(exc).__name__
    finally:
        actual = json.loads(args.out.read_text()) if args.out.exists() else {'complete':False}
        receipt['class_restored'] = anchor.StartupAnchorUpdate is ORIGINAL_CLASS
        receipt['score_backtrack_restored'] = q.backtrack is old_backtrack and q.score_entries is old_score
        receipt['fraction_globals_restored'] = q.FRACTIONS == grid.ORIGINAL_FRACTIONS and anchor.FRACTIONS == grid.ORIGINAL_FRACTIONS
        receipt['files_preserved'] = all(base.sha(path) == sha for path,sha in protected.items())
        receipt['baseline_and_exposure'] = validate_baseline_and_exposure(actual,reference)
        receipt['complete'] = error is None and all(receipt[k] for k in
            ('class_restored','score_backtrack_restored','fraction_globals_restored','files_preserved')) and receipt['baseline_and_exposure']['passed']
        if not receipt['complete']:actual.update(complete=False,failure_category=receipt['failure_category'] or 'CorrectedAnchorVerificationFailure')
        actual['pilot_execution_version'] = actual.get('version')
        actual['pilot_execution_method'] = actual.get('method')
        actual.update(version=VERSION,method='corrected_anchor',corrected_anchor_policy=receipt)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
    if error is not None:raise error
    if not receipt['complete']:raise RuntimeError('Corrected pilot failed initialization/source/preservation checks')
    print(json.dumps({'corrected_anchor_pilot_complete':True,'updates':64,'individual_constraints':12}))


if __name__ == '__main__':main()
