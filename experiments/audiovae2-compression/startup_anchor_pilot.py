"""Disposable fresh-C anchor pilot; aggregate measurements, no saved weights."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import startup_retention_diagnostic as helper
import startup_retention_probe as probe
import startup_anchor_update as anchor
from unified_monitor import UnifiedMonitor
from joint_recovery_gates_v2 import summarize_regions

base, screen, replay = helper.base, helper.screen, helper.replay
prior, recovery = helper.prior, helper.recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--method', choices=('ordinary','anchor'), required=True)
    args = parser.parse_args()
    if args.out.exists() or args.updates != 64:
        raise ValueError('Use a new aggregate output and the fixed64-update pilot')
    started = time.monotonic()
    inputs, teacher, model, optimizer, pools, data, protected = helper.build(json.loads(args.config.read_text()))
    manifest, _, _ = base.load_data(inputs.manifest)
    metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    evaluator = UnifiedMonitor(None,{},metadata)
    initial = {k:v.clone() for k,v in model.group_state_dict().items()}
    initial_hash = recovery.control.state_hash(model.decoder)
    teacher_hash = recovery.control.state_hash(teacher.model)
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_rng = screen.rng_state()
    frozen = screen.frozen_versions(model)
    protected = dict(protected)
    for module in (helper, probe, anchor, anchor.warmup, anchor.q):
        protected[str(Path(module.__file__).resolve())] = base.sha(module.__file__)
    protected[str(Path(__file__).resolve())] = base.sha(__file__)
    protected[str(args.config.resolve())] = base.sha(args.config)
    report = dict(version='audiovae2_startup_anchor_pilot_v1', complete=False,
                  updates=0, failure_category=None, reviews=[], update_records=[],
                  method=args.method, ordinary_unique_sources=0,
                  repeated_training_calibration_sources=6 if args.method=='anchor' else 0,
                  development_used_for_updates=False, checkpoint_written=False,
                  initial_state_sha256=initial_hash, config_sha256=base.sha(args.config),
                  source_sha256={Path(p).name:s for p,s in protected.items() if p.endswith('.py')})
    seen = []
    try:
        anchors, anchor_cache = probe.startup_panel(teacher, pools['calibration'], 6)
        development, development_cache = probe.startup_panel(teacher, pools['development'], 13)
        updater = anchor.StartupAnchorUpdate(model, teacher, anchors, optimizer)
        report['initialization'] = updater.receipt
        report['startup_cache_checks'] = dict(calibration=anchor_cache, development=development_cache)
        common = base.objective()
        def full_review():
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                full = recovery.joint.evaluate_review(evaluator,model,teacher,pools['development'],common,summarize=summarize_regions)
            return {k:full[k] for k in ('aggregate','quiet_regions')}
        def review(step):
            return dict(step=step, calibration=probe.score_panel(model, anchors),
                        development=probe.score_panel(model, development))
        report['reviews'].append(review(0))
        if report['reviews'][0]['development']['passed'] != 13:
            raise RuntimeError('Fresh startup baseline changed')
        warmed = set()
        prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
        report['development_before'] = full_review()
        first_batch = data.take(0,12)
        prior.warm_student(model,teacher,first_batch,warmed,optimizer)
        report['fixed_fitting_batch_objective_before'] = probe.ordinary_objective(model,teacher,first_batch,common)
        for step in range(1,65):
            crops = data.take((step-1)*12,12)
            ids = [c['source_id'] for c in crops]
            if ids != list(data.source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):
                raise RuntimeError('Original fitting prefix or source uniqueness changed')
            prior.warm_student(model,teacher,crops,warmed,optimizer)
            update = updater.perform_update if args.method=='anchor' else prior.perform_update
            values,checks = update(model,teacher,crops,common,optimizer,diagnostics=True)
            if len(checks) != 12 or not all(c['allclose_original_tolerance'] for c in checks):
                raise RuntimeError('Teacher target cache parity failed')
            if set(optimizer.state) != set(base.parameters(model)) or any(float(s['step']) != step for s in optimizer.state.values()):
                raise RuntimeError('Adam counters or parameter scope changed')
            seen.extend(ids)
            report['updates'] = step
            record = {k:v for k,v in values.items() if (k.startswith(('q_','startup_anchor_')) or k in (*prior.COEFFICIENTS,'total')) and (v is None or isinstance(v,(float,int,bool)))}
            report['update_records'].append(dict(step=step,**record))
            if step <= 8 or step in (16,32,64):
                report['reviews'].append(review(step))
        report['fixed_fitting_batch_objective_after'] = probe.ordinary_objective(model,teacher,first_batch,common)
        report['development_after'] = full_review()
        report['ordinary_unique_sources'] = len(seen)
        report['ordinary_source_prefix_sha256'] = screen.digest(seen)
        report['anchor_update_participations'] = 6*len(report['update_records']) if args.method=='anchor' else 0
        report['complete'] = True
    except BaseException as exc:
        report['failure_category'] = type(exc).__name__
        raise
    finally:
        report['ordinary_unique_sources'] = len(seen)
        report['ordinary_source_prefix_sha256'] = screen.digest(seen)
        report['anchor_update_participations'] = 6*len(report['update_records']) if args.method=='anchor' else 0
        report['ledger_scope'] = 'Completed updates with validated teacher-cache and Adam checks; failed attempts excluded'
        model.load_group_state_dict(initial)
        optimizer.load_state_dict(original_optimizer)
        optimizer.zero_grad(set_to_none=True)
        screen.restore_rng(original_rng)
        data.assert_unchanged()
        report['preserved'] = dict(
            initial_model=recovery.control.state_hash(model.decoder)==initial_hash,
            initial_optimizer=replay.compare_tree(optimizer.state_dict(),original_optimizer)['equal'],
            initial_rng=replay.compare_tree(screen.rng_state(),original_rng)['equal'],
            teacher=recovery.control.state_hash(teacher.model)==teacher_hash,
            frozen_outer=screen.frozen_versions(model)==frozen,
            protected_files=all(base.sha(p)==s for p,s in protected.items()))
        report['all_preservation_checks_passed'] = all(report['preserved'].values())
        if not report['all_preservation_checks_passed']:
            report['complete'] = False
            report['failure_category'] = 'PreservationFailure'
        report['elapsed_seconds'] = time.monotonic()-started
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if not report['all_preservation_checks_passed']:
            raise RuntimeError('Disposable pilot failed preservation')
    print(json.dumps(dict(complete=report['complete'],updates=report['updates'],
                         startup_passed=report['reviews'][-1]['development']['passed'])))


if __name__ == '__main__':
    main()
