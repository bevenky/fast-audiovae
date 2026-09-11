"""One-factor fresh-C pilot: extend only the existing backtracking fraction grid.

Execute the authenticated, unchanged64-step pilot by explicit file path. Neither
the constraint rows, Adam update, losses, data stream nor inference graph change.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import startup_anchor_update as anchor

q=anchor.q
base=anchor.base
VERSION='audiovae2_startup_anchor_extended_grid_pilot_v1'
PILOT_SHA256='681701217441e05717c7b4bb39ba4518af495b40646364a9526c91126415baf0'
ORIGINAL_FRACTIONS=(1.,.5,.25,.125,.0625,0.)
EXTENDED_FRACTIONS=ORIGINAL_FRACTIONS[:-1]+tuple(2.**(-power) for power in range(5,11))+(0.,)
PREFIX_UPDATES=42


@contextmanager
def extended_grid():
    """Restore both globals on success/error; zero remains the final fallback."""
    old_q,old_anchor=q.FRACTIONS,anchor.FRACTIONS
    if old_q!=ORIGINAL_FRACTIONS or old_anchor!=ORIGINAL_FRACTIONS:
        raise RuntimeError('The original fraction globals changed before the one-factor pilot')
    q.FRACTIONS=EXTENDED_FRACTIONS
    anchor.FRACTIONS=EXTENDED_FRACTIONS
    try:
        yield
    finally:
        unchanged=q.FRACTIONS==EXTENDED_FRACTIONS and anchor.FRACTIONS==EXTENDED_FRACTIONS
        q.FRACTIONS=old_q;anchor.FRACTIONS=old_anchor
        if not unchanged:
            raise RuntimeError('A second change to the fraction globals occurred during the pilot')


def validate_prefix(actual,reference):
    """Exact aggregate/scalar consistency, not an independently saved tensor trajectory."""
    checks={
        'reference_completed64':reference.get('complete') is True and reference.get('updates')==64
            and reference.get('method')=='anchor' and reference.get('all_preservation_checks_passed') is True,
        'candidate_completed64':actual.get('complete') is True and actual.get('updates')==64
            and actual.get('method')=='anchor' and actual.get('all_preservation_checks_passed') is True,
        'initial_state_exact':actual.get('initial_state_sha256')==reference.get('initial_state_sha256')
            and isinstance(reference.get('initial_state_sha256'),str),
        'config_exact':actual.get('config_sha256')==reference.get('config_sha256')
            and isinstance(reference.get('config_sha256'),str),
        'ordinary768_source_prefix_exact':actual.get('ordinary_unique_sources')==reference.get('ordinary_unique_sources')==768
            and actual.get('ordinary_source_prefix_sha256')==reference.get('ordinary_source_prefix_sha256')
            and isinstance(reference.get('ordinary_source_prefix_sha256'),str),
        'initial_full96_exact':actual.get('development_before')==reference.get('development_before')
            and isinstance(reference.get('development_before'),dict),
        'initial_fitting_objective_exact':actual.get('fixed_fitting_batch_objective_before')==reference.get('fixed_fitting_batch_objective_before')
            and isinstance(reference.get('fixed_fitting_batch_objective_before'),dict),
    }
    for key in ('anchor_identity_sha256','cached_target_input_sha256'):
        before=reference.get('initialization',{}).get(key)
        checks[key+'_exact']=isinstance(before,str) and actual.get('initialization',{}).get(key)==before
    old_rows,new_rows=reference.get('update_records',[]),actual.get('update_records',[])
    checks['all64_scalar_records_present']=len(old_rows)==len(new_rows)==64
    mismatches=[];matched=0
    if checks['all64_scalar_records_present']:
        for i,(old,new) in enumerate(zip(old_rows[:PREFIX_UPDATES],new_rows[:PREFIX_UPDATES]),start=1):
            keys={k for k in old if not k.endswith('_seconds')}
            new_keys={k for k in new if not k.endswith('_seconds')}
            bad=sorted(keys.symmetric_difference(new_keys)|{k for k in keys&new_keys if old[k]!=new[k]})
            if old.get('step')!=i or new.get('step')!=i:bad=sorted(set(bad)|{'step'})
            if bad:mismatches.append({'step':i,'fields':bad})
            else:matched+=1
    checks['first42_non_timing_scalars_exact']=matched==PREFIX_UPDATES
    return {'passed':all(checks.values()),'checks':checks,'matched_updates':matched,
            'mismatches':mismatches,'full_tensor_trajectory_claim':False,
            'scope':'The unchanged positive fraction prefix covers original updates1..42; divergence from43 is the measured treatment.'}


def run_pilot(pilot_path,config,out):
    """Run that exact source file; no copied main, hidden source fork or subprocess."""
    module_name='_startup_anchor_immutable_grid_base'
    previous=sys.modules.get(module_name)
    old_argv=sys.argv
    spec=importlib.util.spec_from_file_location(module_name,pilot_path)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot load explicit immutable pilot source')
    module=importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name]=module
        spec.loader.exec_module(module)
        if module.anchor is not anchor or module.anchor.q is not q:
            raise RuntimeError('Pilot imports a different anchor/Q runtime')
        sys.argv=[str(pilot_path),'--config',str(config),'--out',str(out),'--updates','64','--method','anchor']
        with extended_grid():module.main()
    finally:
        sys.argv=old_argv
        if previous is None:sys.modules.pop(module_name,None)
        else:sys.modules[module_name]=previous


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','config','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new extended-grid aggregate output')
    reference=json.loads(args.reference.read_text())
    if (base.sha(args.pilot)!=PILOT_SHA256
            or reference.get('source_sha256',{}).get('startup_anchor_pilot_v2.py')!=PILOT_SHA256
            or reference.get('complete') is not True or reference.get('updates')!=64
            or reference.get('all_preservation_checks_passed') is not True or reference.get('method')!='anchor'
            or reference.get('config_sha256')!=base.sha(args.config)
            or next((r['step'] for r in reference.get('update_records',[]) if r.get('q_zero_displacement')),None)!=43):
        raise RuntimeError('Original64-step pilot/reference/config authentication failed')
    for module in (anchor,q,anchor.warmup):
        path=Path(module.__file__)
        if reference.get('source_sha256',{}).get(path.name)!=base.sha(path):
            raise RuntimeError('Anchor/Q/warmup source differs from the preserved pilot')
    protected={str(p.resolve()):base.sha(p) for p in (args.pilot,args.config,args.reference,Path(__file__),
        Path(anchor.__file__),Path(q.__file__),Path(anchor.warmup.__file__))}
    wrapper={'version':VERSION,'original_positive_fraction_prefix_preserved':True,
        'original_fractions':list(ORIGINAL_FRACTIONS),'extended_fractions':list(EXTENDED_FRACTIONS),
        'zero_fallback_last':True,'pilot_source_sha256':PILOT_SHA256,'wrapper_source_sha256':base.sha(__file__),
        'reference_aggregate_sha256':base.sha(args.reference),'config_sha256':base.sha(args.config),
        'only_treatment':'Six additional positive backtracking fractions before the unchanged zero fallback',
        'original_constraint_rows_unchanged':True,'original_optimizer_and_losses_unchanged':True,
        'new_inference_operations':0,'checkpoint_written':False,'prefix_comparison':None,'failure_category':None}
    pilot_error=None
    try:
        run_pilot(args.pilot,args.config,args.out)
    except BaseException as exc:
        pilot_error=exc;wrapper['failure_category']=type(exc).__name__
    finally:
        wrapper['fraction_globals_restored']=q.FRACTIONS==ORIGINAL_FRACTIONS and anchor.FRACTIONS==ORIGINAL_FRACTIONS
        wrapper['protected_files_preserved']=all(base.sha(p)==sha for p,sha in protected.items())
        if args.out.exists():
            actual=json.loads(args.out.read_text())
        else:
            actual={'complete':False,'updates':0,'failure_category':wrapper['failure_category'] or 'MissingPilotAggregate'}
        if pilot_error is None:
            wrapper['prefix_comparison']=validate_prefix(actual,reference)
        passed=(pilot_error is None and wrapper['fraction_globals_restored'] and wrapper['protected_files_preserved']
                and wrapper['prefix_comparison'] is not None and wrapper['prefix_comparison']['passed'])
        wrapper['complete']=passed
        if not passed:
            actual['complete']=False
            actual['failure_category']=wrapper['failure_category'] or 'ExtendedGridVerificationFailure'
        actual['grid_extension']=wrapper
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
    if pilot_error is not None:raise pilot_error
    if not wrapper['complete']:raise RuntimeError('Extended-grid pilot failed source/global/prefix verification')
    print(json.dumps({'extended_grid_complete':True,'updates':64,'first42_scalar_prefix_exact':True}))


if __name__=='__main__':main()
