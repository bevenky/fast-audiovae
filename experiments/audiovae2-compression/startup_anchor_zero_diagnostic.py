"""Observe the first zero startup-anchor displacement without changing the method.

All candidate tensors stay in memory. Only pooled/count/min/max diagnostics are
serialized. Smaller fractions are counterfactual evaluations, never updates.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import torch
import startup_retention_diagnostic as helper
import startup_retention_probe as probe
import startup_anchor_update as anchor
from unified_monitor import UnifiedMonitor
from joint_recovery_gates_v2 import summarize_regions

q = anchor.q
base, screen, replay, prior = q.base, q.screen, q.replay, q.prior
VERSION = 'audiovae2_first_zero_anchor_diagnostic_v1'
SMALL_FRACTIONS = tuple(2.**(-power) for power in range(5,11))
FIRST_ZERO_UPDATE = 43


def score_with_rounding(model, entries, score_fn):
    """Read the same six final predictions; do not perform another forward."""
    double_values=[]
    def observe(module, inputs, output):
        index=len(double_values)
        if index>=6 or output.shape!=entries[index]['target'].shape:
            raise RuntimeError('Final-head observation count/shape changed')
        entry=entries[index]; row=entry['windows'][0]; a,_=entry['span']
        lo,hi=a+row['start_sample'],a+row['stop_sample']
        p=output[...,lo:hi].detach().double(); t=entry['target'][...,lo:hi].double()
        values=(float((p-t).square().mean())-row['residual_limit']**2,
                float(p.square().mean())-row['output_rms_limit']**2)
        if p.numel()!=960 or not all(math.isfinite(v) for v in values):
            raise RuntimeError('Invalid same-prediction FP64 window measurement')
        double_values.append(values)
    handle=model.decoder.model[-1].register_forward_hook(observe)
    try:
        score=score_fn()
    finally:
        handle.remove()
    if len(double_values)!=6 or len(score['signature'])!=6:
        raise RuntimeError('Every native scoring call must produce six anchor predictions')
    return score,{ref:double_values[ref[0]] for ref,_,_ in score['signature']}


@contextmanager
def observe_backtracking(model, entries):
    """Transparent scoped replacement of Q's backtrack call and its score callback."""
    original=q.backtrack; box={'calls':0,'last':None}
    def wrapped(params,before,proposal,projected,baseline,score_fn,*,corrected):
        box['calls']+=1
        capture={'before':[x.detach().clone() for x in before],
                 'proposal':[x.detach().clone() for x in proposal],
                 'projected':[x.detach().clone() for x in projected],
                 'baseline':copy.deepcopy(baseline),'corrected':corrected,'attempts':[]}
        def scored():
            index=len(capture['attempts'])
            if index>=len(q.FRACTIONS):raise RuntimeError('Backtracking exceeded its original fixed grid')
            actual=[p.detach().double()-old.double() for p,old in zip(params,before)]
            score,fp64=score_with_rounding(model,entries,score_fn)
            capture['attempts'].append({'fraction':q.FRACTIONS[index], 'score':copy.deepcopy(score),
                                        'fp64':fp64, 'actual_delta':actual})
            return score
        result=original(params,before,proposal,projected,baseline,scored,corrected=corrected)
        if len(result[2])!=len(capture['attempts']):raise RuntimeError('Observer changed backtracking accounting')
        capture['accepted_fraction']=result[0];box['last']=capture
        return result
    q.backtrack=wrapped
    try:
        yield box
    finally:
        q.backtrack=original


def _statistics(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise RuntimeError('Nonfinite or empty scalar diagnostic')
    return {'count':len(values),'min':min(values),'max':max(values),
            'mean':sum(values)/len(values),'positive_count':sum(v>0 for v in values),
            'exact_zero_count':sum(v==0 for v in values)}


def summarize_score(score, fp64):
    result={}
    for k,kind in enumerate(q.KINDS):
        exact=[v[k] for _,v,_ in score['signature']]
        double=[fp64[ref][k] for ref,_,_ in score['signature']]
        result[kind]={'canonical_excess':_statistics(exact),'slack':_statistics([-v for v in exact]),
            'same_prediction_fp64_excess':_statistics(double),
            'canonical_minus_fp64':_statistics([a-b for a,b in zip(exact,double)]),
            'metric_sign_disagreements':sum((a>0)!=(b>0) for a,b in zip(exact,double))}
    result['passed']=sum(passed for _,_,passed in score['signature'])
    result['fp64_passed']=sum(all(v<=0 for v in fp64[ref]) for ref,_,_ in score['signature'])
    return result


def all_constraint_rows(model, entries, baseline, params):
    selected={(value['ref'],key[1]) for key,value in baseline['maxima']}
    expanded=[]; descriptors=[]
    for ref,values,_ in baseline['signature']:
        for kind,value in zip(q.KINDS,values):
            expanded.append((('near_startup',kind),{'value':value,'ref':ref}))
            descriptors.append({'ref':ref,'kind':kind,'before':value,'selected':(ref,kind) in selected})
    if len(expanded)!=12 or len(selected)!=2:raise RuntimeError('Expected twelve individual and two selected constraints')
    gradients=q.constraint_gradients(model,entries,{'maxima':expanded},params)
    for row,gradient in zip(descriptors,gradients):
        row['gradient']=gradient;row['norm']=math.sqrt(q._dot(gradient,gradient))
    return descriptors


def aggregate_predictions(rows, actual_delta, score, *, ideal_delta=None, qp_tolerance=0.):
    """Keep individual arithmetic internal; export selected/omitted/kind aggregates."""
    after=None if score is None else {ref:values for ref,values,_ in score['signature']}; measured=[]
    for row in rows:
        dot=q._dot(row['gradient'],actual_delta)
        ideal=dot if ideal_delta is None else q._dot(row['gradient'],ideal_delta)
        record={**{k:row[k] for k in ('kind','selected','before')},
            'gradient_dot_actual':dot,'gradient_dot_ideal':ideal,
            'linear_actual':row['before']+dot,'linear_ideal':row['before']+ideal,
            'parameter_rounding_linear_change':dot-ideal,
            'normalized_linear_actual':(row['before']+dot)/row['norm'] if row['norm'] else 0.,
            'linear_positive_beyond_qp_arithmetic':row['before']+dot>4*qp_tolerance*row['norm']}
        if after is not None:
            value=after[row['ref']][q.KINDS.index(row['kind'])]
            record.update(after=value,remainder=value-row['before']-dot)
        measured.append(record)
    result={}
    for name,predicate in (('all',lambda r:True),('selected',lambda r:r['selected']),
                            ('omitted',lambda r:not r['selected']),
                            ('residual',lambda r:r['kind']=='residual'),('amplitude',lambda r:r['kind']=='amplitude')):
        chosen=[r for r in measured if predicate(r)]
        fields=('before','gradient_dot_actual','gradient_dot_ideal','linear_actual','linear_ideal',
                'parameter_rounding_linear_change','normalized_linear_actual')
        if after is not None:fields+=('after','remainder')
        result[name]={key:_statistics([r[key] for r in chosen]) for key in fields}
        result[name]['linear_positive_beyond_qp_arithmetic']=sum(r['linear_positive_beyond_qp_arithmetic'] for r in chosen)
    return result


def gradient_rank(rows):
    positive=[r for r in rows if r['norm']>0]
    gram=torch.zeros((len(positive),len(positive)),dtype=torch.float64)
    for i,left in enumerate(positive):
        for j in range(i,len(positive)):
            right=positive[j]
            gram[i,j]=gram[j,i]=q._dot(left['gradient'],right['gradient'])/(left['norm']*right['norm'])
    eigen=torch.linalg.eigvalsh(gram)
    tolerance=q.QP_EPS_FACTOR*torch.finfo(torch.float64).eps*max(1.,float(eigen.abs().max()) if len(eigen) else 0.)
    if len(eigen) and float(eigen.min()) < -tolerance:
        raise RuntimeError('All-row normalized Gram is numerically indefinite')
    return {'rows':len(rows),'nonzero_rows':len(positive),'rank':int((eigen>tolerance).sum()),
            'rank_tolerance':tolerance,'normalized_eigen_min':float(eigen.min()) if len(eigen) else None,
            'normalized_eigen_max':float(eigen.max()) if len(eigen) else None,
            'gradient_norm_min':min(r['norm'] for r in rows),'gradient_norm_max':max(r['norm'] for r in rows)}


def displacement_metrics(actual, ideal):
    error=[a-b for a,b in zip(actual,ideal)]
    return {'actual_l2':math.sqrt(q._dot(actual,actual)), 'ideal_l2':math.sqrt(q._dot(ideal,ideal)),
            'parameter_rounding_l2':math.sqrt(q._dot(error,error)),
            'parameter_rounding_max_abs':max(float(e.abs().max()) for e in error),
            'changed_elements':sum(int((a!=0).sum()) for a in actual)}


def analyze_zero(model, entries, params, capture):
    baseline,fp64=score_with_rounding(model,entries,lambda:q.score_entries(model,entries))
    if baseline!=capture['baseline']:raise RuntimeError('Restored pre-step state differs from recorded baseline')
    rows=all_constraint_rows(model,entries,baseline,params)
    selected=[]
    for key,value in baseline['maxima']:
        selected.append(next(r['gradient'] for r in rows if r['ref']==value['ref'] and r['kind']==key[1]))
    ordinary=[p.double()-old.double() for p,old in zip(capture['proposal'],capture['before'])]
    reconstructed,linear=q.project_displacement(selected,ordinary,[-v['value'] for _,v in baseline['maxima']])
    projected=capture['projected']
    reproduction=displacement_metrics(reconstructed,projected)
    reproduction['bitwise_equal']=all(torch.equal(a,b) for a,b in zip(reconstructed,projected))
    tolerance=linear.get('tolerance',0.)
    report={'baseline':summarize_score(baseline,fp64),'all_constraint_gradient_rank':gradient_rank(rows),
            'selected_projection_reproduction':reproduction,
            'ordinary_delta_linear_predictions':aggregate_predictions(rows,ordinary,None,qp_tolerance=tolerance),
            'projected_delta_linear_predictions':aggregate_predictions(rows,projected,None,qp_tolerance=tolerance),
            'original_grid':[],'smaller_counterfactual_grid':[]}
    for attempt in capture['attempts']:
        fraction=attempt['fraction']
        ideal=[fraction*d for d in projected]
        report['original_grid'].append({'fraction':fraction, 'scores':summarize_score(attempt['score'],attempt['fp64']),
            'displacement':displacement_metrics(attempt['actual_delta'],ideal),
            'linearization':aggregate_predictions(rows,attempt['actual_delta'],attempt['score'],
                                                  ideal_delta=ideal,qp_tolerance=tolerance),
            'maximum_switches':sum(a['ref']!=b['ref'] for (_,a),(_,b) in zip(attempt['score']['maxima'],baseline['maxima']))})
    try:
        for fraction in SMALL_FRACTIONS:
            ideal=[fraction*d for d in projected]
            with torch.no_grad():
                for p,old,d in zip(params,capture['before'],ideal):p.copy_((old.double()+d).to(p))
            actual=[p.detach().double()-old.double() for p,old in zip(params,capture['before'])]
            score,double=score_with_rounding(model,entries,lambda:q.score_entries(model,entries))
            report['smaller_counterfactual_grid'].append({'fraction':fraction,'accepted_as_training':False,
                'scores':summarize_score(score,double),'displacement':displacement_metrics(actual,ideal),
                'linearization':aggregate_predictions(rows,actual,score,ideal_delta=ideal,qp_tolerance=tolerance)})
    finally:
        with torch.no_grad():
            for p,old in zip(params,capture['before']):p.copy_(old)
    if q.score_entries(model,entries)!=baseline:raise RuntimeError('Counterfactual analysis failed exact pre-state restoration')
    smallest=report['original_grid'][-2] if report['original_grid'][-1]['fraction']==0 else report['original_grid'][-1]
    feasible=[r for r in report['smaller_counterfactual_grid'] if r['scores']['passed']==6 and r['displacement']['changed_elements']>0]
    lin=smallest['linearization']; scores=smallest['scores']
    report['mechanism_evidence']={
        'directional_reproduction_exact':reproduction['bitwise_equal'],
        'original_grid_excludes_demonstrated_nonzero_feasible_point':bool(feasible),
        'largest_tested_smaller_feasible_fraction':feasible[0]['fraction'] if feasible else None,
        'smallest_original_fraction':smallest['fraction'],
        'selected_linear_violations_beyond_qp_arithmetic':lin['selected']['linear_positive_beyond_qp_arithmetic'],
        'omitted_linear_violations_beyond_qp_arithmetic':lin['omitted']['linear_positive_beyond_qp_arithmetic'],
        'all_linear_actual_nonpositive_but_native_fails':lin['all']['linear_actual']['positive_count']==0 and scores['passed']<6,
        'all_linear_nonpositive_but_same_prediction_fp64_fails':lin['all']['linear_actual']['positive_count']==0 and scores['fp64_passed']<6,
        'same_prediction_metric_arithmetic_changes_feasibility':scores['fp64_passed']==6 and scores['passed']<6,
        'parameter_rounding_changes_linear_feasibility':lin['all']['linear_ideal']['positive_count']==0 and lin['all']['linear_actual']['positive_count']>0,
        'interpretation':'Remainders include nonlinear response and arithmetic, not pure curvature. A feasible smaller point establishes grid truncation, not useful convergence; finite-anchor generalization remains unproved.'}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True,help='The completed64-step anchor pilot aggregate')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Require a new first-zero aggregate file')
    reference=json.loads(args.reference.read_text())
    if (reference.get('complete') is not True or reference.get('all_preservation_checks_passed') is not True
            or reference.get('method')!='anchor' or reference.get('updates')!=64
            or reference.get('ordinary_unique_sources')!=768 or len(reference.get('update_records',[]))!=64
            or reference.get('config_sha256')!=base.sha(args.config)):
        raise RuntimeError('Require the intact matched64-step anchor pilot reference')
    if next((r['step'] for r in reference['update_records'] if r.get('q_zero_displacement')),None)!=FIRST_ZERO_UPDATE:
        raise RuntimeError('Reference must reproduce the known first zero at update43')
    started=time.monotonic()
    inputs,teacher,model,optimizer,pools,data,protected=helper.build(json.loads(args.config.read_text()))
    initial={k:v.clone() for k,v in model.group_state_dict().items()};params=base.parameters(model)
    initial_hash=helper.recovery.control.state_hash(model.decoder);teacher_hash=helper.recovery.control.state_hash(teacher.model)
    initial_optimizer=copy.deepcopy(optimizer.state_dict());initial_rng=screen.rng_state();frozen=screen.frozen_versions(model)
    protected=dict(protected)
    for module in (helper,probe,anchor,q,anchor.warmup):protected[str(Path(module.__file__).resolve())]=base.sha(module.__file__)
    for path in (args.config,args.reference,Path(__file__)):protected[str(path.resolve())]=base.sha(path)
    if initial_hash!=reference.get('initial_state_sha256'):
        raise RuntimeError('Initial C state differs from the saved pilot')
    report={'version':VERSION,'complete':False,'updates':0,'first_zero_update':None,'failure_category':None,
        'initial_state_sha256':initial_hash,'config_sha256':base.sha(args.config),
        'reference_aggregate_sha256':base.sha(args.reference),
        'reference_scalar_prefix':{'passed':True,'matched_updates':0,'full_tensor_trajectory_claim':False},
        'source_sha256':{Path(p).name:s for p,s in protected.items() if p.endswith('.py')},
        'maximum_updates':FIRST_ZERO_UPDATE,'ordinary_unique_sources':0,'repeated_calibration_sources':6,
        'update_records':[],'startup_reviews':[],'no_checkpoint_written':True,
        'scope':'Unchanged startup-anchor replay through its first zero; twelve-row gradients and smaller fractions are diagnostic only',
        'metric_arithmetic':'Canonical FP32 subtraction/vector-norm RMS squared in FP64; independent FP64 subtraction/mean-square on exactly the same native prediction'}
    seen=[];original_backtrack=q.backtrack
    try:
        anchors,cache=probe.startup_panel(teacher,pools['calibration'],6)
        development,dev_cache=probe.startup_panel(teacher,pools['development'],13)
        updater=anchor.StartupAnchorUpdate(model,teacher,anchors,optimizer)
        entries=updater._entries
        report['initialization']=updater.receipt;report['startup_cache_checks']={'calibration':cache,'development':dev_cache}
        common=base.objective();warmed=set()
        def review(step):return {'step':step,'calibration':probe.score_panel(model,anchors),'development':probe.score_panel(model,development)}
        report['startup_reviews'].append(review(0))
        # Reproduce the saved pilot's initial full96/warmup and observation order.
        manifest,_,_=base.load_data(inputs.manifest)
        metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
        evaluator=UnifiedMonitor(None,{},metadata)
        prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            full=helper.recovery.joint.evaluate_review(evaluator,model,teacher,pools['development'],common,summarize=summarize_regions)
        report['development_before']={k:full[k] for k in ('aggregate','quiet_regions')}
        first=data.take(0,12);prior.warm_student(model,teacher,first,warmed,optimizer)
        report['fixed_fitting_batch_objective_before']=probe.ordinary_objective(model,teacher,first,common)
        with observe_backtracking(model,entries) as observer:
            for step in range(1,FIRST_ZERO_UPDATE+1):
                crops=data.take((step-1)*12,12);ids=[c['source_id'] for c in crops]
                if ids!=list(data.source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):raise RuntimeError('Original source prefix changed')
                prior.warm_student(model,teacher,crops,warmed,optimizer)
                before_optimizer=copy.deepcopy(optimizer.state_dict());before_rng=screen.rng_state()
                before_grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
                values,checks=updater.perform_update(model,teacher,crops,common,optimizer,diagnostics=True)
                if len(checks)!=12 or not all(c['allclose_original_tolerance'] for c in checks):raise RuntimeError('Teacher cache parity failed')
                old=reference['update_records'][step-1]
                keys=[k for k in old if k!='step' and not k.endswith('_seconds')]
                mismatches=[k for k in keys if k not in values or values[k]!=old[k]]
                if old.get('step')!=step or mismatches:
                    report['reference_scalar_prefix'].update(passed=False,first_mismatch_step=step,mismatched_fields=mismatches)
                    raise RuntimeError('Observed non-timing scalar prefix differs from the saved anchor pilot')
                report['reference_scalar_prefix']['matched_updates']=step
                seen.extend(ids);report['updates']=step
                report['update_records'].append({'step':step,**{k:v for k,v in values.items()
                    if k in (*prior.COEFFICIENTS,'total','q_accepted_fraction','q_zero_displacement','q_projected',
                              'q_qp_rank','q_linear_conflicts','q_maximum_switches','q_proposal_norm','q_correction_norm',
                              'q_accepted_displacement_norm','startup_anchor_after_passed')}})
                if step<=8 or step in (16,32,64):report['startup_reviews'].append(review(step))
                if step%8==0:base.event('first_zero_replay_progress',updates=step,maximum_updates=FIRST_ZERO_UPDATE)
                if values['q_zero_displacement']:
                    capture=observer['last']
                    if capture is None or capture['accepted_fraction']!=0:raise RuntimeError('First zero is not the recorded backtracking rejection')
                    report['first_zero_update']=step
                    with torch.no_grad():
                        for p,old in zip(params,capture['before']):p.copy_(old)
                    optimizer.load_state_dict(before_optimizer);screen.restore_rng(before_rng)
                    for p,g in zip(params,before_grads):p.grad=g
                    if not replay.compare_tree(optimizer.state_dict(),before_optimizer)['equal']:raise RuntimeError('Pre-zero Adam restoration failed')
                    with replay.diagnostic_state_guard(model,teacher,optimizer):
                        report['first_zero_diagnosis']=analyze_zero(model,entries,params,capture)
                    report['pre_zero_optimizer_and_rng_restored']=replay.compare_tree(optimizer.state_dict(),before_optimizer)['equal'] and replay.compare_tree(screen.rng_state(),before_rng)['equal']
                    break
        report['backtrack_function_restored']=q.backtrack is original_backtrack
        if report['first_zero_update']!=FIRST_ZERO_UPDATE:raise RuntimeError('The authenticated first-zero event was not reproduced')
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
            'protected_files':all(base.sha(p)==s for p,s in protected.items()),'backtrack_function':q.backtrack is original_backtrack}
        report['all_preservation_checks_passed']=all(report['preserved'].values())
        if not report['all_preservation_checks_passed']:report.update(complete=False,failure_category='PreservationFailure')
        report['elapsed_seconds']=time.monotonic()-started
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if not report['all_preservation_checks_passed']:raise RuntimeError('First-zero diagnostic preservation failed')
    print(json.dumps({'complete':report['complete'],'updates':report['updates'],'first_zero_update':report['first_zero_update']}))


if __name__=='__main__':main()
