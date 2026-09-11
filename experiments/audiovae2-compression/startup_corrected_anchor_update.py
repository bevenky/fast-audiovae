"""Bounded normal correction inside the frozen complete-anchor Adam transaction.

A full projected trial is checked first. At most two current-trial signed
repairs test scales 1 and 2 within the frozen cumulative 25% norm budget. A
bounded rejection restarts the unchanged extended grid from the original
proposal. Numerical failures propagate into the parent's complete rollback.
"""
from __future__ import annotations

import math

import torch
import startup_complete_anchor_update as complete
import startup_anchor_correction_probe as correction

anchor, q, grid = complete.anchor, complete.q, complete.grid
base, screen, replay = complete.base, complete.screen, complete.replay
VERSION = 'audiovae2_corrected_six_startup_anchor_adam_v1'
FRACTIONS = grid.EXTENDED_FRACTIONS


def correction_backtrack(model, entries, params, before, proposal, projected,
                         baseline, score_fn, *, corrected, original_backtrack):
    """Return the old backtrack triple plus scalar-only correction accounting."""
    refs = [ref for ref, _, _ in baseline['signature']]
    if len(refs) != 6 or len(set(refs)) != 6:
        raise RuntimeError('Correction requires the same six fixed startup windows')
    limits = []
    for ref in refs:
        row = entries[ref[0]]['windows'][0]
        if row['window_index'] != ref[1]:
            raise RuntimeError('Correction anchor order changed')
        limits.extend((row['residual_limit']**2, row['output_rms_limit']**2))
    attempts, packets, gradient_calls = [], [], 0
    ordinary_grads = [(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in params]

    def scored():
        score = score_fn()
        if [ref for ref, _, _ in score['signature']] != refs:
            raise RuntimeError('Correction scoring changed the fixed windows')
        accepted = q._within_caps(score, baseline)
        attempts.append({'fraction':1., 'accepted':accepted,
            'maximum_switches':sum(v['ref'] != old['ref'] for (_,v),(_,old)
                                   in zip(score['maxima'],baseline['maxima']))})
        packets.append(score)
        return {'values':[v for _, pair, _ in score['signature'] for v in pair],
                'summary':anchor.anchor_metrics(score,entries)}

    def jacobian(values):
        nonlocal gradient_calls
        expanded = [(('near_startup',kind), {'ref':ref,'value':value})
            for ref,pair in zip(refs,zip(values[::2],values[1::2]))
            for kind,value in zip(q.KINDS,pair)]
        if len(expanded) != 12:
            raise RuntimeError('Correction omitted a physical inequality')
        gradient_calls += 1
        return q.constraint_gradients(model,entries,{'maxima':expanded},params)

    # Preserve Adam's exact FP32 proposal if its complete projection is unchanged.
    initial_trial = ([p.detach().clone() for p in proposal] if not corrected else
        [(old.double()+direction).to(p) for p,old,direction in zip(params,before,projected)])
    correction._copy(params,before)
    candidate, report = correction.correction_attempts(params,before,initial_trial,projected,
        score_fn=scored, gradient_fn=jacobian, limit_squares=limits,
        project_signed=correction.signed.project_displacement)
    normal_checks = len(attempts)
    fallback_checks = 0
    if candidate is None:
        correction._copy(params,proposal)
        fraction,after,fallback = original_backtrack(params,before,proposal,projected,
            baseline,score_fn,corrected=corrected)
        fallback_checks = len(fallback)
        attempts.extend(fallback)
    else:
        correction._copy(params,candidate)
        fraction = 1.
        after = score_fn()
        if after != packets[-1] or not q._within_caps(after,baseline):
            raise RuntimeError('Installed normal candidate did not reproduce its complete true scores')
    if any(p.grad is not original or (saved is not None and not torch.equal(p.grad,saved))
           for p,(original,saved) in zip(params,ordinary_grads)):
        raise RuntimeError('Normal correction polluted ordinary gradient slots')
    accepted_normal = (0. if candidate is None else
        correction._norm([p.detach().double()-trial.double() for p,trial in zip(params,initial_trial)]))
    if accepted_normal > report['normal_budget']:
        raise RuntimeError('Installed cumulative native correction exceeded its fixed budget')
    corrected_accept = candidate is not None and report['correction_solves'] > 0
    candidates = [c for row in report['rounds'] for c in row['candidates']]
    diagnostics = {
        'q_normal_correction_enabled':1,
        'q_normal_attempted':int(report['correction_solves'] > 0),
        'q_normal_accepted':int(corrected_accept),
        'q_full_projected_trial_feasible':int(report['status'] == 'initial_trial_already_feasible'),
        'q_normal_solves':report['correction_solves'],
        'q_normal_scale_candidates':len(candidates),
        'q_normal_nonlinear_candidates':sum(c['evaluated'] for c in candidates),
        'q_normal_budget_rejections':sum(not c['within_normal_budget'] for c in candidates),
        'q_normal_strict_progress_candidates':sum(c.get('strict_progress',False) for c in candidates),
        'q_normal_accepted_round':report.get('accepted_round',0) if corrected_accept else 0,
        'q_normal_accepted_scale':report.get('accepted_scale',0.) if corrected_accept else 0.,
        'q_normal_accepted_norm':accepted_normal,
        'q_normal_budget':report['normal_budget'],
        'q_normal_budget_fraction':correction.NORMAL_BUDGET_FRACTION,
        'q_original_projected_norm':report['original_projected_norm'],
        'q_normal_norm_over_projected':accepted_normal/report['original_projected_norm'] if report['original_projected_norm'] else 0.,
        'q_normal_budget_passed':1,
        'q_normal_full_primal_verified':int(all(row['solver']['full_primal_verified'] for row in report['rounds'])),
        'q_normal_kkt_passed':int(all(row['solver']['kkt_passed'] for row in report['rounds'])),
        'q_normal_max_gram_rank':max((row['solver']['gram_rank'] for row in report['rounds']),default=0),
        'q_normal_max_active_rank':max((row['solver']['rank'] for row in report['rounds']),default=0),
        'q_normal_initial_relative_excess':report['initial_trial']['relative_violation'],
        'q_normal_final_relative_excess':correction.relative_violation(
            [v for _,pair,_ in after['signature'] for v in pair],limits),
        'q_normal_score_calls':normal_checks,
        'q_normal_install_verification_calls':int(candidate is not None),
        'q_normal_gradient_sources':6*gradient_calls,
        'q_extended_grid_fallback':int(candidate is None),
        'q_extended_grid_score_calls':fallback_checks,
        'q_base_fraction':fraction,
        'q_base_fraction_has_normal_correction':int(corrected_accept),
    }
    if any(isinstance(v,float) and not math.isfinite(v) for v in diagnostics.values()):
        raise RuntimeError('Nonfinite normal-correction accounting')
    return (fraction,after,attempts),diagnostics


class StartupAnchorUpdate(complete.StartupAnchorUpdate):
    def __init__(self,model,teacher,entries,optimizer):
        super().__init__(model,teacher,entries,optimizer)
        self.receipt.update(version=VERSION, parent_complete_version=complete.VERSION,
            nonlinear_correction=True, maximum_normal_solves=correction.MAX_CORRECTIONS,
            normal_scales=list(correction.CORRECTION_SCALES),
            normal_budget_fraction=correction.NORMAL_BUDGET_FRACTION,
            normal_budget_definition='Total actual FP32 displacement from the original full projected trial',
            normal_progress='Maximum positive physical excess divided by its original squared limit',
            fallback='Original extended grid from unchanged before/proposal/projected vectors',
            fraction_semantics='Base projected fraction; accepted normal motion is reported separately',
            development_used_for_update=False)

    def perform_update(self,model,teacher,crops,common,optimizer,*,diagnostics=False):
        original_backtrack,original_score = q.backtrack,q.score_entries
        accounting = {}; score_calls = 0; intercept_calls = 0
        def counted_score(*args,**kwargs):
            nonlocal score_calls
            score_calls += 1
            return original_score(*args,**kwargs)
        def intercepted(params,before,proposal,projected,baseline,score_fn,*,corrected):
            nonlocal intercept_calls
            intercept_calls += 1
            if intercept_calls != 1:
                raise RuntimeError('A corrected update invoked ordinary backtracking more than once')
            result,extra = correction_backtrack(model,self._entries,params,before,proposal,
                projected,baseline,score_fn,corrected=corrected,original_backtrack=original_backtrack)
            accounting.update(extra)
            return result
        q.backtrack,q.score_entries = intercepted,counted_score
        try:
            values,checks = super().perform_update(model,teacher,crops,common,optimizer,diagnostics=diagnostics)
        finally:
            q.backtrack,q.score_entries = original_backtrack,original_score
        values.update(accounting)
        values.update(q_canonical_score_calls=score_calls,
            q_canonical_score_forwards=6*score_calls,
            q_constraint_gradient_forwards=6+accounting['q_normal_gradient_sources'],
            q_actual_norm_over_adam=values['q_accepted_displacement_norm']/values['q_proposal_norm'] if values['q_proposal_norm'] else 0.,
            startup_anchor_checks=6*score_calls)
        return values,checks

    __call__ = perform_update
