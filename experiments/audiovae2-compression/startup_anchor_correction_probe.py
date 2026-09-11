"""Disposable nonlinear-correction comparison at complete-anchor first zero56.

Replay the fixed fresh-C trajectory; only after restoring pre56 compare a
bounded repair with the recorded complete-row extended-grid result.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import startup_anchor_zero_diagnostic as zero
import startup_complete_anchor_update as complete
import startup_complete_anchor_pilot as pilot
import startup_signed_constraint_projection as signed
from unified_monitor import UnifiedMonitor
from joint_recovery_gates_v2 import summarize_regions

anchor, q, grid = complete.anchor, complete.q, complete.grid
base, screen, replay, prior = zero.base, zero.screen, zero.replay, zero.prior
helper, probe = zero.helper, zero.probe
VERSION = 'audiovae2_startup_anchor_correction_probe_v1'
FIRST_ZERO_UPDATE = 56
MAX_CORRECTIONS = 2
CORRECTION_SCALES = (1., 2.)
NORMAL_BUDGET_FRACTION = .25
SOURCE_PINS = {**pilot.PINNED_SOURCES,
    'startup_complete_anchor_update.py':'1f4880c649868e1a9861595c4ab60d4df668cbdbfb4a593a9b81cb51c5e4208d',
    'startup_complete_anchor_pilot.py':'6a82dffeaf61f8d2d8e0381868e8f67ff4b8cb06f311ad85b61667609d734307',
    'startup_anchor_zero_diagnostic.py':'306e4f083b831fdd877b26b12c0b9afd385cc86fda5d6cfaa25a802461ec4e43'}


def _copy(params, values):
    with torch.no_grad():
        for p,value in zip(params,values):p.copy_(value)


def _norm(values):
    return math.sqrt(q._dot(values,values))


def relative_violation(values, limit_squares):
    if (not values or len(values)!=len(limit_squares) or len(values)>12
            or any(not math.isfinite(v) for v in values)
            or any(not math.isfinite(v) or v<=0 for v in limit_squares)):
        raise ValueError('Require finite physical excesses and unchanged positive squared limits')
    return max(max(value,0.)/limit for value,limit in zip(values,limit_squares))


def _solver_summary(report):
    result={k:v for k,v in report.items() if v is None or isinstance(v,(str,int,float,bool))}
    if any(isinstance(v,float) and not math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite signed-solver diagnostic')
    result['active_constraint_count']=len(report.get('active_set',[]))
    return result


def correction_attempts(params, before, initial_trial, projected, *, score_fn,
                        gradient_fn, limit_squares, project_signed):
    """Return a disposable candidate and aggregates; always restore pre-step values.

    score_fn returns internal flat values plus an aggregate-only summary.
    gradient_fn(values) evaluates every row at the currently installed trial.
    The budget is total realized correction from initial_trial, not path length.
    """
    if (not params or len(params)!=len(before) or len(params)!=len(initial_trial)
            or len(params)!=len(projected) or any(not torch.equal(p,b) for p,b in zip(params,before))
            or any(p.shape!=b.shape or p.shape!=v.shape or p.shape!=d.shape
                   for p,b,v,d in zip(params,before,initial_trial,projected))
            or any(not torch.isfinite(t).all() for row in (before,initial_trial,projected) for t in row)):
        raise ValueError('Require the unchanged pre-step state and finite matching trial vectors')
    direction_norm=_norm(projected); budget=NORMAL_BUDGET_FRACTION*direction_norm
    report={'accepted':False,'correction_solves':0,'normal_budget_fraction':NORMAL_BUDGET_FRACTION,
        'original_projected_norm':direction_norm,'normal_budget':budget,'rounds':[],
        'budget_definition':'L2 norm of total actual FP32 candidate minus fixed original full-trial point',
        'progress_definition':'max_i(max(F_i,0)/original_limit_i_squared)',
        'maximum_corrections':MAX_CORRECTIONS,'correction_scales':list(CORRECTION_SCALES)}
    candidate=None
    try:
        _copy(params,initial_trial)
        current=[p.detach().clone() for p in params]
        packet=score_fn(); values=list(packet['values']); progress=relative_violation(values,limit_squares)
        report['initial_trial']={'relative_violation':progress,'scores':packet['summary']}
        if progress==0:
            report.update(accepted=True,status='initial_trial_already_feasible',accepted_normal_norm=0.)
            return [p.detach().clone() for p in params],report
        if direction_norm==0:
            report['status']='zero_projected_direction'
            return None,report
        for number in range(1,MAX_CORRECTIONS+1):
            _copy(params,current)
            gradients=gradient_fn(values)
            if len(gradients)!=len(values):raise RuntimeError('Current-trial Jacobian omitted an inequality')
            normal,certificate=project_signed(gradients,[torch.zeros_like(p,dtype=torch.float64) for p in params],[-v for v in values])
            if (certificate.get('rows')!=len(values) or certificate.get('kkt_passed') is not True
                    or certificate.get('full_primal_verified') is not True):
                raise RuntimeError('Signed repair lacks complete KKT/reconstructed-primal verification')
            report['correction_solves']+=1
            row={'number':number,'before_relative_violation':progress,
                 'solver':_solver_summary(certificate),'normal_l2':_norm(normal),'candidates':[]}
            report['rounds'].append(row); best=None
            for scale in CORRECTION_SCALES:
                _copy(params,[(v.double()+scale*r).to(p) for p,v,r in zip(params,current,normal)])
                realized=[p.detach().double()-v.double() for p,v in zip(params,initial_trial)]
                norm=_norm(realized)
                attempt={'scale':scale,'cumulative_actual_normal_norm':norm,
                         'within_normal_budget':norm<=budget,'evaluated':False,'accepted':False}
                row['candidates'].append(attempt)
                if norm>budget:continue
                after=score_fn(); next_values=list(after['values'])
                next_progress=relative_violation(next_values,limit_squares)
                increment=[p.detach().double()-v.double() for p,v in zip(params,current)]
                dots=[q._dot(g,increment) for g in gradients]
                ideal_dots=[scale*q._dot(g,normal) for g in gradients]
                attempt.update(evaluated=True,relative_violation=next_progress,scores=after['summary'],
                    strict_progress=next_progress<progress,
                    linear_actual=zero._statistics([v+d for v,d in zip(values,dots)]),
                    linear_ideal=zero._statistics([v+d for v,d in zip(values,ideal_dots)]),
                    actual_after=zero._statistics(next_values),
                    remainder=zero._statistics([a-v-d for a,v,d in zip(next_values,values,dots)]))
                if next_progress==0:
                    attempt['accepted']=True
                    candidate=[p.detach().clone() for p in params]
                    report.update(accepted=True,status='true_constraints_passed',accepted_round=number,
                                  accepted_scale=scale,accepted_normal_norm=norm)
                    return candidate,report
                if next_progress<progress and (best is None or next_progress<best[0]):
                    best=(next_progress,[p.detach().clone() for p in params],next_values)
            if best is None:
                report['status']='no_within_budget_strict_progress';break
            progress,current,values=best
            row['next_trial_relative_violation']=progress
        else:report['status']='bounded_corrections_exhausted'
        return None,report
    finally:
        _copy(params,before)
        if not all(torch.equal(p,b) for p,b in zip(params,before)):
            raise RuntimeError('Normal correction search failed exact pre-step restoration')


def analyze(model,teacher,entries,development,params,capture,crops,common,*,backtrack):
    baseline,double=zero.score_with_rounding(model,entries,lambda:q.score_entries(model,entries))
    if baseline!=capture['baseline'] or capture['accepted_fraction']!=0:
        raise RuntimeError('Pre56 baseline differs from the recorded zero update')
    rows=zero.all_constraint_rows(model,entries,baseline,params)
    ordinary=[new.double()-old.double() for new,old in zip(capture['proposal'],capture['before'])]
    reconstructed,linear=complete.projection.project_displacement([r['gradient'] for r in rows],ordinary,[-r['before'] for r in rows])
    if not all(torch.equal(a,b) for a,b in zip(reconstructed,capture['projected'])):
        raise RuntimeError('Complete pre56 projection did not reproduce bitwise')
    refs=[ref for ref,_,_ in baseline['signature']]
    limits=[]
    for ref in refs:
        row=entries[ref[0]]['windows'][0]
        if ref[1]!=row['window_index']:raise RuntimeError('Anchor ordering changed')
        limits.extend((row['residual_limit']**2,row['output_rms_limit']**2))
    def measure():
        return {'calibration':probe.score_panel(model,entries),'development':probe.score_panel(model,development),
                'ordinary_fitting_batch_objective':probe.ordinary_objective(model,teacher,crops,common)}
    report={'baseline':zero.summarize_score(baseline,double),'baseline_metrics':measure(),
        'complete_projection_bitwise_reproduced':True,'base_solver':_solver_summary(linear),
        'baseline_extended_grid':[],'counterfactual_optimizer_updates':0,'checkpoint_written':False,
        'development_used_for_correction':False,'projected_delta_l2':_norm(capture['projected'])}
    for attempt in capture['attempts']:
        ideal=[attempt['fraction']*d for d in capture['projected']]
        report['baseline_extended_grid'].append({'fraction':attempt['fraction'],
            'scores':zero.summarize_score(attempt['score'],attempt['fp64']),
            'displacement':zero.displacement_metrics(attempt['actual_delta'],ideal),
            'linearization':zero.aggregate_predictions(rows,attempt['actual_delta'],attempt['score'],
                                                       ideal_delta=ideal,qp_tolerance=linear.get('tolerance',0.))})
    calls=0
    def scored():
        nonlocal calls
        score,fp64=zero.score_with_rounding(model,entries,lambda:q.score_entries(model,entries))
        if [ref for ref,_,_ in score['signature']]!=refs:raise RuntimeError('Correction scoring changed anchor identities')
        if calls==0 and (score!=capture['attempts'][0]['score'] or fp64!=capture['attempts'][0]['fp64']):
            raise RuntimeError('Original full projected trial did not reproduce')
        calls+=1
        return {'values':[v for _,values,_ in score['signature'] for v in values],
                'summary':zero.summarize_score(score,fp64)}
    def jacobian(values):
        expanded=[(('near_startup',kind),{'ref':ref,'value':value})
                  for ref,pair in zip(refs,zip(values[::2],values[1::2])) for kind,value in zip(q.KINDS,pair)]
        gradients=q.constraint_gradients(model,entries,{'maxima':expanded},params)
        if len(gradients)!=12:raise RuntimeError('Current-trial Jacobian must contain all12 rows')
        return gradients
    before=capture['before']; proposal=capture['proposal']; projected=capture['projected']
    trial=proposal if not linear['corrected'] else [(old.double()+d).to(old) for old,d in zip(before,projected)]
    try:
        candidate,repair=correction_attempts(params,before,trial,projected,score_fn=scored,gradient_fn=jacobian,
                                            limit_squares=limits,project_signed=signed.project_displacement)
        report['normal_correction']=repair;report['normal_score_calls']=calls
        if candidate is None:
            _copy(params,proposal)
            fraction,after,attempts=backtrack(params,before,proposal,projected,baseline,
                lambda:q.score_entries(model,entries),corrected=linear['corrected'])
            if fraction!=capture['accepted_fraction'] or after!=capture['attempts'][-1]['score']:
                raise RuntimeError('Unchanged grid fallback differs from the original pre56 proposal')
            report.update(fallback_used=True,fallback_fraction=fraction,fallback_matches_recorded=True,
                          fallback_attempts=len(attempts))
        else:
            _copy(params,candidate);after=q.score_entries(model,entries)
            if not all(passed for _,_,passed in after['signature']):
                raise RuntimeError('Corrected candidate lost true feasibility on repeat')
            report.update(fallback_used=False,fallback_fraction=None)
        actual=[p.detach().double()-old.double() for p,old in zip(params,before)]
        an,on=_norm(actual),_norm(ordinary)
        report.update(candidate_metrics=measure(),accepted_as_training=False,
            actual_displacement_l2=an,actual_displacement_nonzero=any(bool((d!=0).any()) for d in actual),
            actual_to_ordinary_norm_ratio=an/on if on else None,
            actual_ordinary_cosine=q._dot(actual,ordinary)/(an*on) if an*on else None,
            interpretation='Bounded same-state counterfactual only. Remainders include nonlinear response and arithmetic. '
                           'Canonical waveform checks alone decide feasibility; development and ordinary quality are observations.')
    finally:
        _copy(params,before)
    if q.score_entries(model,entries)!=baseline:raise RuntimeError('Correction analysis did not restore pre56 scores')
    report['pre56_parameters_and_scores_restored']=True
    return report


def validate_reference(reference,config):
    if (reference.get('version')!=pilot.VERSION or reference.get('method')!='complete_anchor'
            or reference.get('complete') is not True or reference.get('all_preservation_checks_passed') is not True
            or reference.get('updates')!=64 or reference.get('ordinary_unique_sources')!=768
            or reference.get('config_sha256')!=base.sha(config)
            or reference.get('complete_anchor_policy',{}).get('complete') is not True
            or len(reference.get('update_records',[]))!=64
            or next((r['step'] for r in reference['update_records'] if r.get('q_zero_displacement')),None)!=FIRST_ZERO_UPDATE):
        raise ValueError('Require the preserved complete64 reference with first zero56 and exact config')
    if any(r.get('step')!=i or r.get('q_constraints')!=12 for i,r in enumerate(reference['update_records'],1)):
        raise ValueError('Reference complete-row update ledger differs')


def compare_record(values,old,step):
    keys={k for k in old if k!='step' and not k.endswith('_seconds')}
    current={k for k,v in values.items() if (k.startswith(('q_','startup_anchor_')) or k in (*prior.COEFFICIENTS,'total'))
             and (v is None or isinstance(v,(float,int,bool))) and not k.endswith('_seconds')}
    bad=sorted(keys.symmetric_difference(current)|{k for k in keys&current if values[k]!=old[k]})
    if old.get('step')!=step:bad=sorted(set(bad)|{'step'})
    return {'passed':not bad,'mismatched_fields':bad}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Require a new correction-probe aggregate')
    reference=json.loads(args.reference.read_text());validate_reference(reference,args.config)
    modules=(anchor,q,anchor.warmup,grid,complete.projection,complete,pilot,zero)
    for module in modules:
        path=Path(module.__file__)
        if base.sha(path)!=SOURCE_PINS[path.name]:raise ValueError('A frozen replay/projection source changed')
        if path.name!='startup_anchor_zero_diagnostic.py' and reference['complete_anchor_policy']['source_sha256'].get(path.name)!=base.sha(path):
            raise ValueError('Reference used different complete-policy sources')
    if base.sha(signed.__file__)!='fe3b0c8e15569079ea645b0b32539d3b0978e05f2c41e3022e303c5a062b70b5':
        raise ValueError('Signed solver source differs from its qualified freeze')
    started=time.monotonic()
    inputs,teacher,model,optimizer,pools,data,protected=helper.build(json.loads(args.config.read_text()))
    initial={k:v.clone() for k,v in model.group_state_dict().items()};params=base.parameters(model)
    initial_hash=helper.recovery.control.state_hash(model.decoder);teacher_hash=helper.recovery.control.state_hash(teacher.model)
    initial_optimizer=copy.deepcopy(optimizer.state_dict());initial_rng=screen.rng_state();frozen=screen.frozen_versions(model)
    if initial_hash!=reference['initial_state_sha256']:raise RuntimeError('Fresh C state differs from complete64 reference')
    if (initial_optimizer['state'] or len(data.source_ids)<768
            or screen.digest(list(data.source_ids[:768]))!=reference.get('ordinary_source_prefix_sha256')):
        raise RuntimeError('Fresh optimizer or original768-source reference prefix differs')
    protected=dict(protected)
    for module in (*modules,helper,probe,signed):protected[str(Path(module.__file__).resolve())]=base.sha(module.__file__)
    for path in (args.config,args.reference,Path(__file__)):protected[str(path.resolve())]=base.sha(path)
    report={'version':VERSION,'complete':False,'updates':0,'first_zero_update':None,'maximum_updates':FIRST_ZERO_UPDATE,
        'failure_category':None,'initial_state_sha256':initial_hash,'config_sha256':base.sha(args.config),
        'reference_aggregate_sha256':base.sha(args.reference),'source_sha256':{Path(p).name:s for p,s in protected.items() if p.endswith('.py')},
        'reference_scalar_prefix':{'passed':True,'matched_updates':0,'full_tensor_trajectory_claim':False},
        'update_records':[],'startup_reviews':[],'ordinary_unique_sources':0,'repeated_calibration_sources':6,
        'no_checkpoint_written':True,'counterfactual_optimizer_updates':0,
        'original768_source_prefix_exact':True,'fresh_optimizer_empty':True,
        'metric_arithmetic':'Canonical evaluator FP32 RMS, squared in FP64; same-prediction FP64 energy is explanatory only'}
    seen=[];original_backtrack=q.backtrack;original_class=anchor.StartupAnchorUpdate
    try:
        with grid.extended_grid():
            anchors,cache=probe.startup_panel(teacher,pools['calibration'],6)
            development,dev_cache=probe.startup_panel(teacher,pools['development'],13)
            updater=complete.StartupAnchorUpdate(model,teacher,anchors,optimizer);entries=updater._entries
            report['initialization']=updater.receipt;report['startup_cache_checks']={'calibration':cache,'development':dev_cache}
            for key in ('initial','initial_model_sha256','anchor_identity_sha256','cached_target_input_sha256','constraints'):
                if updater.receipt[key]!=reference['initialization'][key]:raise RuntimeError('Original complete-anchor initialization differs')
            common=base.objective();warmed=set()
            def review(step):return {'step':step,'calibration':probe.score_panel(model,anchors),'development':probe.score_panel(model,development)}
            first_review=review(0);report['startup_reviews'].append(first_review)
            if first_review!=reference['reviews'][0]:raise RuntimeError('Fresh startup panel differs from the reference')
            manifest,_,_=base.load_data(inputs.manifest);metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
            evaluator=UnifiedMonitor(None,{},metadata)
            prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                full=helper.recovery.joint.evaluate_review(evaluator,model,teacher,pools['development'],common,summarize=summarize_regions)
            report['development_before']={k:full[k] for k in ('aggregate','quiet_regions')}
            if report['development_before']!=reference['development_before']:raise RuntimeError('Initial full96 baseline differs')
            first=data.take(0,12);prior.warm_student(model,teacher,first,warmed,optimizer)
            report['fixed_fitting_batch_objective_before']=probe.ordinary_objective(model,teacher,first,common)
            if report['fixed_fitting_batch_objective_before']!=reference['fixed_fitting_batch_objective_before']:
                raise RuntimeError('Initial ordinary fitting objective differs')
            with zero.observe_backtracking(model,entries) as observer:
                for step in range(1,FIRST_ZERO_UPDATE+1):
                    crops=data.take((step-1)*12,12);ids=[c['source_id'] for c in crops]
                    if ids!=list(data.source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):raise RuntimeError('Original source prefix changed')
                    prior.warm_student(model,teacher,crops,warmed,optimizer)
                    before_optimizer=copy.deepcopy(optimizer.state_dict());before_rng=screen.rng_state()
                    old_grads=[(p.grad,None if p.grad is None else p.grad.detach().clone()) for p in params]
                    values,checks=updater.perform_update(model,teacher,crops,common,optimizer,diagnostics=True)
                    if len(checks)!=12 or not all(c['allclose_original_tolerance'] for c in checks):raise RuntimeError('Teacher cache parity failed')
                    if set(optimizer.state)!=set(params) or any(float(s['step'])!=step for s in optimizer.state.values()):
                        raise RuntimeError('Original all90 Adam counters differ from the replay step')
                    parity=compare_record(values,reference['update_records'][step-1],step)
                    if not parity['passed']:
                        report['reference_scalar_prefix'].update({**parity,'passed':False,'first_mismatch_step':step})
                        raise RuntimeError('Complete-policy non-timing scalar trajectory changed')
                    report['reference_scalar_prefix']['matched_updates']=step
                    seen.extend(ids);report['updates']=step
                    report['update_records'].append({'step':step,**{k:v for k,v in values.items()
                        if (k.startswith(('q_','startup_anchor_')) or k in (*prior.COEFFICIENTS,'total')) and (v is None or isinstance(v,(float,int,bool)))}})
                    if step<=8 or step in (16,32):report['startup_reviews'].append(review(step))
                    if step%8==0:base.event('correction_probe_replay_progress',updates=step,maximum_updates=FIRST_ZERO_UPDATE)
                    if values['q_zero_displacement']:
                        if step!=FIRST_ZERO_UPDATE:raise RuntimeError('Unexpected earlier zero update')
                        capture=observer['last'];report['first_zero_update']=step
                        if capture is None or capture['accepted_fraction']!=0:raise RuntimeError('Missing original zero-displacement capture')
                        _copy(params,capture['before']);optimizer.load_state_dict(before_optimizer);screen.restore_rng(before_rng)
                        with torch.no_grad():
                            for p,(original,saved) in zip(params,old_grads):
                                p.grad=original
                                if saved is not None:original.copy_(saved)
                        with replay.diagnostic_state_guard(model,teacher,optimizer):
                            report['correction_diagnosis']=analyze(model,teacher,entries,development,params,capture,crops,common,backtrack=original_backtrack)
                        report['pre56_optimizer_and_rng_restored']=replay.compare_tree(optimizer.state_dict(),before_optimizer)['equal'] and replay.compare_tree(screen.rng_state(),before_rng)['equal']
                        report['pre56_gradient_slots_restored']=all(p.grad is original and
                            (saved is None or torch.equal(p.grad,saved)) for p,(original,saved) in zip(params,old_grads))
                        if not report['pre56_optimizer_and_rng_restored'] or not report['pre56_gradient_slots_restored']:
                            raise RuntimeError('Pre56 Adam/RNG/gradient preservation failed')
                        break
            if report['first_zero_update']!=FIRST_ZERO_UPDATE:raise RuntimeError('Expected first zero56 was not reproduced')
        report['complete']=True
    except BaseException as exc:
        report['failure_category']=type(exc).__name__;raise
    finally:
        model.load_group_state_dict(initial);optimizer.load_state_dict(initial_optimizer);optimizer.zero_grad(set_to_none=True)
        screen.restore_rng(initial_rng);data.assert_unchanged()
        report['ordinary_unique_sources']=len(seen);report['ordinary_source_prefix_sha256']=screen.digest(seen)
        report['preserved']={'initial_model':helper.recovery.control.state_hash(model.decoder)==initial_hash,
            'initial_optimizer':replay.compare_tree(optimizer.state_dict(),initial_optimizer)['equal'],
            'initial_rng':replay.compare_tree(screen.rng_state(),initial_rng)['equal'],
            'teacher':helper.recovery.control.state_hash(teacher.model)==teacher_hash,'frozen_outer':screen.frozen_versions(model)==frozen,
            'protected_files':all(base.sha(p)==s for p,s in protected.items()),'backtrack_function':q.backtrack is original_backtrack,
            'anchor_class':anchor.StartupAnchorUpdate is original_class,
            'fraction_globals':q.FRACTIONS==grid.ORIGINAL_FRACTIONS and anchor.FRACTIONS==grid.ORIGINAL_FRACTIONS}
        report['all_preservation_checks_passed']=all(report['preserved'].values())
        if not report['all_preservation_checks_passed']:report.update(complete=False,failure_category='PreservationFailure')
        report['elapsed_seconds']=time.monotonic()-started
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if not report['all_preservation_checks_passed']:raise RuntimeError('Correction diagnostic preservation failed')
    print(json.dumps({'complete':report['complete'],'updates':report['updates'],'first_zero_update':report['first_zero_update']}))


if __name__=='__main__':main()
