"""Fresh64-update all-twelve startup pilot using the immutable original driver.

Only the anchor update class and positive fraction grid change. Original model,
Adam recipe, data order, review schedule and final disposable restoration remain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys

import startup_complete_anchor_update as complete

anchor, q, grid, base = complete.anchor, complete.q, complete.grid, complete.base
VERSION = 'audiovae2_complete_startup_anchor_pilot_v1'
ORIGINAL_CLASS = complete.StartupAnchorUpdate.__bases__[0]
PINNED_SOURCES = {
    'startup_anchor_update.py':'e1c4415bd03e9de05be57c3a72da7efe37c107cfbc31e53b142a5bc4278e00a8',
    'quiet_projected_update.py':'41623773f05dcb274c2fa0f8e0aff88e636559509f618d3c129efa40bf210397',
    'quiet_constraint_warmup.py':'8f0c56a275a1d4135ba4bc188a51f53432228e5971a6a15c925973382f1b1d33',
    'startup_anchor_grid_pilot.py':'d0c5f668a4b65265745b82322820583daa889930fa6e2b61990a5491c6b839c1',
    'startup_constraint_projection.py':'88b894a8653a7989b7de9d4e5ec2a10f7c84e92e8c85397fadbaea5f21bc0042',
}


@contextmanager
def complete_policy():
    old_class = anchor.StartupAnchorUpdate
    if old_class is not ORIGINAL_CLASS:
        raise RuntimeError('Original anchor class was replaced before the complete-row pilot')
    anchor.StartupAnchorUpdate = complete.StartupAnchorUpdate
    try:
        with grid.extended_grid():yield
    finally:
        unchanged = anchor.StartupAnchorUpdate is complete.StartupAnchorUpdate
        anchor.StartupAnchorUpdate = old_class
        if not unchanged:raise RuntimeError('A second anchor class replacement occurred during the pilot')


def validate_baseline_and_exposure(actual, reference):
    """Match initialization and exposure; the new policy starts at update one."""
    checks = {
        'reference_completed64':reference.get('complete') is True and reference.get('updates') == 64
            and reference.get('method') == 'anchor' and reference.get('all_preservation_checks_passed') is True,
        'candidate_completed64':actual.get('complete') is True and actual.get('updates') == 64
            and actual.get('method') == 'anchor' and actual.get('all_preservation_checks_passed') is True,
        'initial_state_exact':isinstance(reference.get('initial_state_sha256'), str)
            and actual.get('initial_state_sha256') == reference['initial_state_sha256'],
        'config_exact':isinstance(reference.get('config_sha256'), str)
            and actual.get('config_sha256') == reference['config_sha256'],
        'original768_source_prefix_exact':actual.get('ordinary_unique_sources') == reference.get('ordinary_unique_sources') == 768
            and isinstance(reference.get('ordinary_source_prefix_sha256'), str)
            and actual.get('ordinary_source_prefix_sha256') == reference['ordinary_source_prefix_sha256'],
        'initial_full96_exact':isinstance(reference.get('development_before'), dict)
            and actual.get('development_before') == reference['development_before'],
        'initial_fitting_objective_exact':isinstance(reference.get('fixed_fitting_batch_objective_before'), dict)
            and actual.get('fixed_fitting_batch_objective_before') == reference['fixed_fitting_batch_objective_before'],
        'initial_startup_review_exact':bool(actual.get('reviews')) and bool(reference.get('reviews'))
            and actual['reviews'][0] == reference['reviews'][0] and actual['reviews'][0].get('step') == 0,
        'complete_constraint_receipt':actual.get('initialization', {}).get('version') == complete.VERSION
            and actual.get('initialization', {}).get('constraints') == 12,
    }
    for key in ('anchor_identity_sha256','cached_target_input_sha256','initial_model_sha256','initial'):
        wanted = reference.get('initialization', {}).get(key)
        checks[key+'_exact'] = wanted is not None and actual.get('initialization', {}).get(key) == wanted
    records = actual.get('update_records', [])
    checks['all64_complete_updates'] = len(records) == 64 and all(
        row.get('step') == step and row.get('q_constraints') == 12
        and row.get('q_constrained_sources') == row.get('q_constraint_gradient_sources') == 6
        and row.get('q_full_primal_verified') == row.get('q_kkt_passed') == row.get('q_caps_passed') == 1
        and row.get('startup_anchor_after_passed') == 6
        and row.get('q_accepted_fraction') in grid.EXTENDED_FRACTIONS
        and all(not isinstance(v,float) or math.isfinite(v) for v in row.values())
        for step,row in enumerate(records,1))
    return {'passed':all(checks.values()), 'checks':checks, 'policy_changes_begin_update':1,
            'historical_update_scalar_equality_required':False, 'full_tensor_trajectory_claim':False}


def run_pilot(pilot_path, config, out):
    name = '_startup_anchor_immutable_complete_base'
    previous, old_argv = sys.modules.get(name), sys.argv
    spec = importlib.util.spec_from_file_location(name, pilot_path)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot load explicit immutable pilot source')
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[name] = module; spec.loader.exec_module(module)
        if module.anchor is not anchor or module.anchor.q is not q:
            raise RuntimeError('Pilot imports different anchor/Q runtime objects')
        sys.argv = [str(pilot_path),'--config',str(config),'--out',str(out),'--updates','64','--method','anchor']
        with complete_policy():module.main()
    finally:
        sys.argv = old_argv
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','config','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new complete-anchor aggregate output')
    reference = json.loads(args.reference.read_text())
    if (base.sha(args.pilot) != grid.PILOT_SHA256
            or reference.get('source_sha256', {}).get('startup_anchor_pilot_v2.py') != grid.PILOT_SHA256
            or reference.get('complete') is not True or reference.get('updates') != 64
            or reference.get('method') != 'anchor' or reference.get('all_preservation_checks_passed') is not True
            or reference.get('config_sha256') != base.sha(args.config)):
        raise RuntimeError('Original pilot/reference/config authentication failed')
    modules = (anchor,q,anchor.warmup,grid,complete.projection)
    for module in modules:
        path = Path(module.__file__)
        if base.sha(path) != PINNED_SOURCES[path.name]:
            raise RuntimeError('A frozen anchor/grid/complete-solver source changed')
    for module in (anchor,q,anchor.warmup):
        if reference.get('source_sha256', {}).get(Path(module.__file__).name) != base.sha(module.__file__):
            raise RuntimeError('Original pilot runtime differs from the preserved reference')
    paths = (args.pilot,args.config,args.reference,Path(__file__),Path(complete.__file__),
             *(Path(module.__file__) for module in modules))
    protected = {str(path.resolve()):base.sha(path) for path in paths}
    receipt = {'version':VERSION, 'constraints':12, 'calibration_sources':6,
        'pilot_source_sha256':grid.PILOT_SHA256, 'source_sha256':{p.name:base.sha(p) for p in paths if p.suffix == '.py'},
        'config_sha256':base.sha(args.config), 'reference_aggregate_sha256':base.sha(args.reference),
        'original_fractions':list(grid.ORIGINAL_FRACTIONS), 'fractions':list(grid.EXTENDED_FRACTIONS),
        'maximum_updates':64, 'ordinary_unique_sources':768, 'failure_category':None,
        'policy_changes_begin_update':1, 'nonlinear_correction':False, 'new_loss_weights':False,
        'new_inference_operations':0, 'checkpoint_written':False, 'automatic_continuation':False,
        'scope':'Fresh C; all twelve individual startup constraints from update1; unchanged ordinary Adam and full-six nonlinear checks'}
    error = None
    try:
        run_pilot(args.pilot,args.config,args.out)
    except BaseException as exc:
        error = exc; receipt['failure_category'] = type(exc).__name__
    finally:
        actual = json.loads(args.out.read_text()) if args.out.exists() else {'complete':False}
        receipt['class_restored'] = anchor.StartupAnchorUpdate is ORIGINAL_CLASS
        receipt['fraction_globals_restored'] = q.FRACTIONS == grid.ORIGINAL_FRACTIONS and anchor.FRACTIONS == grid.ORIGINAL_FRACTIONS
        receipt['files_preserved'] = all(base.sha(path) == sha for path,sha in protected.items())
        receipt['baseline_and_exposure'] = validate_baseline_and_exposure(actual,reference)
        receipt['complete'] = error is None and all(receipt[k] for k in ('class_restored','fraction_globals_restored','files_preserved')) and receipt['baseline_and_exposure']['passed']
        if not receipt['complete']:actual.update(complete=False, failure_category=receipt['failure_category'] or 'CompleteAnchorVerificationFailure')
        actual['pilot_execution_version'] = actual.get('version')
        actual['pilot_execution_method'] = actual.get('method')
        actual.update(version=VERSION, method='complete_anchor', complete_anchor_policy=receipt)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
    if error is not None:raise error
    if not receipt['complete']:raise RuntimeError('Complete-anchor pilot failed initialization/source/preservation checks')
    print(json.dumps({'complete_anchor_pilot_complete':True,'updates':64,'individual_constraints':12}))


if __name__ == '__main__':main()
