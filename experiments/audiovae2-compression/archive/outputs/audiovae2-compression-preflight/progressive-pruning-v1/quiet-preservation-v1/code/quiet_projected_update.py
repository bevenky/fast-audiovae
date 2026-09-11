"""Current-batch quiet constraints on the actual ordinary Adam displacement.

No replay memory, new loss coefficient, inference operation or optimizer reset.
F uses the evaluator's FP32 RMS arithmetic, squared in FP64 minus the exact
existing limit squared. Nonlinear cap comparisons have no added tolerance.
"""
from __future__ import annotations

import itertools
import math
import time

import torch
from audiovae_student.quiet_audio import QuietAudioConfig, quiet_window_metrics, _window_values
import progressive_train as prior

base, screen, replay = prior.base, prior.screen, prior.replay
VERSION='audiovae2_current_batch_quiet_projected_adam_v1'
CONFIG=QuietAudioConfig()
COHORTS=('near_startup','other_near','remaining_quiet')
KINDS=('residual','amplitude')
FRACTIONS=(1.,.5,.25,.125,.0625,0.)
QP_EPS_FACTOR=256


def solve_small_qp(gram, violation):
    """Enumerate <=64 active sets for min .5*l'G*l-l'v, l>=0.

    G is the Gram of unit rows, v=A*d0-b. Singular active sets use the
    symmetric eigensystem; incompatible systems are rejected, never jittered.
    """
    if (gram.device.type!='cpu' or gram.dtype!=torch.float64 or violation.device.type!='cpu'
            or violation.dtype!=torch.float64 or gram.ndim!=2 or gram.shape!=(violation.numel(),)*2
            or violation.ndim!=1 or violation.numel()>6
            or not torch.isfinite(gram).all() or not torch.isfinite(violation).all()):
        raise ValueError('QP requires finite CPU FP64 arrays with at most six rows')
    n=len(violation)
    if n==0:return violation.clone(),{'active_set':[],'rank':0,'sets_checked':1,'kkt_passed':True,'tolerance':0.}
    eps=torch.finfo(torch.float64).eps
    scale=max(1.,float(gram.abs().max()),float(violation.abs().max()))
    tol=QP_EPS_FACTOR*eps*scale
    if float((gram-gram.T).abs().max())>tol:raise ValueError('QP Gram is not symmetric')
    g=(gram+gram.T)/2
    eigen=torch.linalg.eigvalsh(g)
    if float(eigen.min()) < -tol:raise ValueError('QP Gram is not positive semidefinite')
    best=None;sets=0
    for count in range(n+1):
        for active in itertools.combinations(range(n),count):
            sets+=1; lam=torch.zeros(n,dtype=torch.float64);rank=0
            if active:
                ix=list(active); block=g[ix][:,ix];v=violation[ix]
                values,vectors=torch.linalg.eigh(block)
                cutoff=QP_EPS_FACTOR*eps*max(1.,float(values.abs().max()))
                positive=values>cutoff;rank=int(positive.sum())
                coeff=vectors[:,positive].T@v
                solution=vectors[:,positive]@(coeff/values[positive])
                if float((block@solution-v).abs().max())>tol:continue
                if float(solution.min()) < -tol:continue
                lam[ix]=solution.clamp_min(0)
            slack=violation-g@lam
            complement=lam*slack
            if (float(slack.max())>tol or float(complement.abs().max())>tol*max(1.,float(lam.abs().max()))):continue
            objective=float(.5*lam@(g@lam)-lam@violation)
            if best is None or objective<best[0]:best=(objective,lam,active,rank,slack,complement)
    if best is None:raise RuntimeError('No numerically verified QP active set; unconstrained/zero fallback forbidden')
    _,lam,active,rank,slack,complement=best
    return lam,{'active_set':list(active),'rank':rank,'sets_checked':sets,'kkt_passed':True,
                'tolerance':tol,'primal_violation_max':max(0.,float(slack.max())),
                'complementarity_max':float(complement.abs().max()),'dual_min':float(lam.min())}


def _dot(left,right):
    if len(left)!=len(right):raise ValueError('Parameter vector topology differs')
    total=torch.zeros((),dtype=torch.float64,device=left[0].device)
    for a,b in zip(left,right):
        if a.shape!=b.shape:raise ValueError('Parameter vector shape differs')
        total+=(a.detach().double()*b.detach().double()).sum()
    return float(total)


def project_displacement(rows, displacement, bounds):
    """Normalize row and RHS together, then project the realized Adam delta."""
    if len(rows)!=len(bounds) or len(rows)>6 or not displacement:raise ValueError('Invalid bounded constraint system')
    if any(not math.isfinite(b) or b<0 for b in bounds):raise ValueError('Zero displacement must be feasible')
    if any(not torch.isfinite(t).all() for t in displacement):raise ValueError('Nonfinite actual Adam displacement')
    norms=[]
    for row in rows:
        if any(not torch.isfinite(t).all() for t in row):raise ValueError('Nonfinite constraint gradient')
        norms.append(math.sqrt(_dot(row,row)))
    active=[i for i,norm in enumerate(norms) if norm>0]
    if not active:return [t.clone() for t in displacement],{'rows':len(rows),'nonzero_rows':0,'linear_conflicts':0,
        'corrected':False,'correction_norm':0.,'proposal_norm':math.sqrt(_dot(displacement,displacement)),
        'proposal_projected_cosine':1. if _dot(displacement,displacement)>0 else None,'kkt_passed':True,'rank':0}
    gram=torch.tensor([[_dot(rows[i],rows[j])/(norms[i]*norms[j]) for j in active] for i in active],dtype=torch.float64)
    gram=(gram+gram.T)/2
    violation=torch.tensor([(_dot(rows[i],displacement)-bounds[i])/norms[i] for i in active],dtype=torch.float64)
    lam,report=solve_small_qp(gram,violation)
    # A no-conflict proposal must remain its original FP32 values in backtracking.
    corrected=bool((lam!=0).any())
    projected=[t.detach().double().clone() for t in displacement]
    if corrected:
        for multiplier,index in zip(lam.tolist(),active):
            if multiplier:
                for d,a in zip(projected,rows[index]):d.add_(a.detach().double(),alpha=-multiplier/norms[index])
    for index in active:
        residual=(_dot(rows[index],projected)-bounds[index])/norms[index]
        if residual>report['tolerance']*4:raise RuntimeError('Reconstructed projection fails normalized primal verification')
    correction=[p-d for p,d in zip(projected,displacement)]
    pn,dn=math.sqrt(_dot(projected,projected)),math.sqrt(_dot(displacement,displacement))
    return projected,{**report,'rows':len(rows),'nonzero_rows':len(active),'linear_conflicts':int((violation>0).sum()),
        'corrected':corrected,'correction_norm':math.sqrt(_dot(correction,correction)),'proposal_norm':dn,
        'proposal_projected_cosine':_dot(projected,displacement)/(pn*dn) if pn*dn else None}


def window_layout(crop,target):
    """Same contiguous scored grid/limits as validation; absolute source time."""
    a=crop['context_frames']*1920;b=a+crop['valid_scored_samples']
    if (target.ndim!=3 or target.shape[:2]!=(1,1) or target.dtype!=torch.float32
            or type(crop['start_frame'])is not int or crop['start_frame']<0
            or crop['start_frame']-crop['context_start_frame']!=crop['context_frames']
            or not 0<=a<b<=target.shape[-1]):raise ValueError('Invalid original source/context/scored geometry')
    t=target[...,a:b].detach()
    raw=quiet_window_metrics(t,t)
    rows=[]
    for original in raw['windows']:
        if not original['is_quiet']:continue
        row=dict(original);absolute=crop['start_frame']*1920+row['start_sample']
        near=row['teacher_rms']<=1e-5
        row.update(source_start_sample=absolute,source_stop_sample=crop['start_frame']*1920+row['stop_sample'],
                   cohort='near_startup' if near and absolute<960 else 'other_near' if near else 'remaining_quiet')
        rows.append(row)
    return {'crop':crop,'span':(a,b),'target':target.detach(),'windows':rows}


def window_excesses(prediction,entry):
    """Differentiable values using exactly the evaluator's RMS reduction."""
    a,b=entry['span'];p=prediction[...,a:b];t=entry['target'][...,a:b].detach()
    counts,trms,srms,rrms,quiet=_window_values(p,t,None,CONFIG)
    residual_limit=(trms*CONFIG.residual_relative_limit).clamp_min(CONFIG.absolute_rms_floor)
    output_limit=(trms*10**(CONFIG.amplitude_db_max/20)).clamp_min(CONFIG.absolute_rms_floor)
    out=[]
    for row in entry['windows']:
        i=row['window_index']
        if (not bool(quiet[0,i]) or int(counts[0,i])!=row['valid_samples']
                or float(trms[0,i])!=row['teacher_rms']
                or float(residual_limit[0,i])!=row['residual_limit']
                or float(output_limit[0,i])!=row['output_rms_limit']):
            raise RuntimeError('Teacher window identity or original quiet limits changed')
        values=(rrms[0,i].double().square()-residual_limit[0,i].double().square(),
                srms[0,i].double().square()-output_limit[0,i].double().square())
        passed=bool((rrms[0,i]<=residual_limit[0,i])&(srms[0,i]<=output_limit[0,i]))
        if passed != all(float(v.detach())<=0 for v in values):raise RuntimeError('Physical-excess sign disagrees with original quiet acceptance')
        out.append({'cohort':row['cohort'],'window_index':i,'values':values,'passed':passed})
    return out


def _predict(model,entry):
    return model.suffix_from_group(model.group_from_input(entry['group_input']))


@torch.no_grad()
def score_entries(model,entries):
    maxima={};signature=[]
    for source_index,entry in enumerate(entries):
        records=window_excesses(_predict(model,entry),entry)
        for record in records:
            values=tuple(float(v) for v in record['values'])
            if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite quiet excess')
            ref=(source_index,record['window_index'])
            signature.append((ref,values,record['passed']))
            for kind,value in zip(KINDS,values):
                key=(record['cohort'],kind)
                if key not in maxima or value>maxima[key]['value']:
                    maxima[key]={'value':value,'ref':ref}
    ordered=[(key,maxima[key]) for cohort in COHORTS for kind in KINDS if (key:=(cohort,kind)) in maxima]
    return {'maxima':ordered,'signature':signature}


def constraint_gradients(model,entries,baseline,params):
    selected=baseline['maxima'];rows=[None]*len(selected)
    grouped={}
    for j,(key,value) in enumerate(selected):grouped.setdefault(value['ref'][0],[]).append((j,key,value))
    for source_index,constraints in grouped.items():
        with torch.enable_grad():
            records=window_excesses(_predict(model,entries[source_index]),entries[source_index])
            by_window={r['window_index']:r for r in records}
            for position,(j,key,value) in enumerate(constraints):
                f=by_window[value['ref'][1]]['values'][KINDS.index(key[1])]
                if float(f.detach())!=value['value']:
                    raise RuntimeError('Grad-enabled constraint differs from its no-grad pre-state')
                gradients=torch.autograd.grad(f,params,retain_graph=position<len(constraints)-1,allow_unused=True)
                rows[j]=[torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(params,gradients)]
        del records,by_window,f,gradients
    return rows


def _within_caps(score,baseline):
    if [k for k,_ in score['maxima']]!=[k for k,_ in baseline['maxima']]:raise RuntimeError('Quiet cohort membership changed')
    return all(value['value']<=max(0.,old['value'])
               for (_,value),(_,old) in zip(score['maxima'],baseline['maxima']))


@torch.no_grad()
def backtrack(params,before,proposal,projected,baseline,score_fn,*,corrected):
    """Check every current quiet window; only nonlinear rejection permits zero."""
    attempts=[]
    for fraction in FRACTIONS:
        if fraction==1 and not corrected:
            # Parameters already contain Adam's exact output: no subtraction/addition roundtrip.
            if any(not torch.equal(p,saved) for p,saved in zip(params,proposal)):
                raise RuntimeError('Ordinary proposal changed before its exact acceptance check')
        elif fraction==0:
            for p,old in zip(params,before):p.copy_(old)
        else:
            for p,old,d in zip(params,before,projected):p.copy_((old.double()+fraction*d).to(p))
        score=score_fn()
        if fraction==0 and score!=baseline:raise RuntimeError('Exact zero displacement does not reproduce the complete pre-state')
        accepted=_within_caps(score,baseline)
        switches=sum(v['ref']!=old['ref'] for (_,v),(_,old) in zip(score['maxima'],baseline['maxima']))
        attempts.append({'fraction':fraction,'accepted':accepted,'maximum_switches':switches})
        if accepted:return fraction,score,attempts
    raise RuntimeError('Exact original state failed its own nonlinear caps')


def perform_update(model,teacher,crops,common,optimizer,*,diagnostics=False):
    """Advance original Adam moments once; optionally change only its displacement."""
    started=time.monotonic();params=base.parameters(model)
    if (len(crops)!=12 or len({c['source_id'] for c in crops})!=12 or len(params)!=90
            or any(not p.requires_grad or p.dtype!=torch.float32 for p in params)
            or [id(p) for row in optimizer.param_groups for p in row['params']]!=[id(p) for p in params]
            or any(p.requires_grad for p in teacher.parameters())):
        raise ValueError('Expected twelve unique sources, frozen teacher and ordered90 FP32 group parameters')
    rng=screen.rng_state();entries=[];aux_checks=[]
    # Teacher-only grid scan requires no student graph and no extra teacher call on active-only crops.
    try:
        for crop in crops:
            z,target,_,_=base.batch([crop]);entry=window_layout(crop,target)
            if entry['windows']:entry['z']=z;entries.append(entry)
        if entries:
            with replay.observe_teacher_cache([e['crop'] for e in entries]) as aux_checks:
                for entry in entries:
                    entry['group_input']=base.teacher_forward(teacher,entry.pop('z'))['group_input'].detach()
            if len(aux_checks)!=len(entries) or not all(r['allclose_original_tolerance'] for r in aux_checks):
                raise RuntimeError('Auxiliary teacher target differs from cache or source accounting')
            baseline=score_entries(model,entries)
            rows=constraint_gradients(model,entries,baseline,params)
    finally:screen.restore_rng(rng)
    if not entries:
        before=[p.detach().clone() for p in params]
        ordinary_started=time.monotonic()
        values,checks=prior.perform_update(model,teacher,crops,common,optimizer,diagnostics=diagnostics)
        ordinary_seconds=time.monotonic()-ordinary_started
        unchanged=int(all(torch.equal(p,old) for p,old in zip(params,before)))
        values.update(q_constraints=0,q_constrained_sources=0,q_linear_conflicts=0,q_projected=0,
                      q_accepted_fraction=1.,q_zero_displacement=unchanged,q_backtrack_rounds=0,
                      q_constraint_gradient_sources=0,q_auxiliary_teacher_checks=0,q_nonzero_rows=0,
                      q_qp_rank=0,q_caps_passed=1,q_maximum_switches=0,
                      q_auxiliary_seconds=time.monotonic()-started-ordinary_seconds,
                      q_ordinary_update_seconds=ordinary_seconds,
                      q_step_seconds=time.monotonic()-started)
        for cohort in COHORTS:values['q_'+cohort+'_present']=0
        return values,checks
    before=[p.detach().clone() for p in params]
    ordinary_started=time.monotonic()
    values,checks=prior.perform_update(model,teacher,crops,common,optimizer,diagnostics=diagnostics)
    ordinary_seconds=time.monotonic()-ordinary_started
    post_rng=screen.rng_state();proposal=[p.detach().clone() for p in params]
    displacement=[new.double()-old.double() for new,old in zip(proposal,before)]
    try:
        bounds=[max(0.,row['value'])-row['value'] for _,row in baseline['maxima']]
        projected,report=project_displacement(rows,displacement,bounds)
        fraction,after,attempts=backtrack(params,before,proposal,projected,baseline,
            lambda:score_entries(model,entries),corrected=report['corrected'])
        accepted=[p.detach().double()-old.double() for p,old in zip(params,before)]
        values.update(q_constraints=len(rows),q_constrained_sources=len(entries),q_constraint_gradient_sources=len({v['ref'][0] for _,v in baseline['maxima']}),
            q_nonzero_rows=report['nonzero_rows'],q_linear_conflicts=report['linear_conflicts'],q_projected=int(report['corrected']),
            q_qp_rank=report['rank'],q_proposal_norm=report['proposal_norm'],q_correction_norm=report['correction_norm'],
            q_projection_cosine=report['proposal_projected_cosine'],q_accepted_fraction=fraction,
            q_zero_displacement=int(all(torch.equal(p,old) for p,old in zip(params,before))),
            q_accepted_displacement_norm=math.sqrt(_dot(accepted,accepted)),q_backtrack_rounds=len(attempts),
            q_maximum_switches=sum(a['maximum_switches'] for a in attempts),q_caps_passed=1,
            q_auxiliary_teacher_checks=len(aux_checks),q_step_seconds=time.monotonic()-started,
            q_ordinary_update_seconds=ordinary_seconds,q_auxiliary_seconds=time.monotonic()-started-ordinary_seconds)
        for cohort in COHORTS:
            values['q_'+cohort+'_present']=int(any(key[0]==cohort for key,_ in baseline['maxima']))
            for kind in KINDS:
                key=(cohort,kind);old=dict(baseline['maxima']).get(key);new=dict(after['maxima']).get(key)
                values['q_'+cohort+'_'+kind+'_before']=None if old is None else old['value']
                values['q_'+cohort+'_'+kind+'_after']=None if new is None else new['value']
        return values,checks
    except BaseException:
        with torch.no_grad():
            for p,old in zip(params,before):p.copy_(old)
        raise
    finally:screen.restore_rng(post_rng)
