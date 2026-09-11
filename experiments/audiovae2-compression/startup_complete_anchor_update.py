"""All twelve fixed startup inequalities on one realized ordinary Adam step.

Reuse the original six-anchor validation and transaction contract. The only
update-policy changes are complete individual rows and the declared finer grid.
No current-batch quiet loss, nonlinear correction, or inference module is added.
"""
from __future__ import annotations

import copy
import math
import time

import torch
import startup_anchor_update as anchor
import startup_anchor_grid_pilot as grid
import startup_constraint_projection as projection

q, warmup = anchor.q, anchor.warmup
base, screen, replay = anchor.base, anchor.screen, anchor.replay
VERSION = 'audiovae2_complete_six_startup_anchor_adam_v1'
FRACTIONS = grid.EXTENDED_FRACTIONS


def individual_constraints(baseline):
    """Expand both excesses of every window; retain the original score for caps."""
    if ([key for key,_ in baseline['maxima']] != [('near_startup',kind) for kind in q.KINDS]
            or len(baseline['signature']) != 6):
        raise RuntimeError('Expected six startup windows and both original maxima')
    expanded, bounds, seen = [], [], set()
    for ref,values,passed in baseline['signature']:
        if (ref[0] in seen or not 0 <= ref[0] < 6 or len(values) != 2 or not passed
                or not all(math.isfinite(v) and v <= 0 for v in values)):
            raise RuntimeError('Every individual startup inequality must be present and initially feasible')
        seen.add(ref[0])
        for kind,value in zip(q.KINDS,values):
            expanded.append((('near_startup',kind), {'ref':ref,'value':value}))
            bounds.append(-value)
    if len(expanded) != 12 or len(seen) != 6:
        raise RuntimeError('A startup constraint was omitted or duplicated')
    return {'maxima':expanded}, bounds


class StartupAnchorUpdate(anchor.StartupAnchorUpdate):
    def __init__(self, model, teacher, entries, optimizer):
        super().__init__(model, teacher, entries, optimizer)
        self.initialization_receipt.update(version=VERSION, constraints=12,
            constraint_policy='Both original physical excess inequalities for each of six calibration windows',
            projection='Complete rank-aware CPU FP64 projection; every reconstructed row verified',
            fractions=list(FRACTIONS), nonlinear_correction=False,
            parent_anchor_version=anchor.VERSION)
        self.receipt = self.initialization_receipt

    def perform_update(self, model, teacher, crops, common, optimizer, *, diagnostics=False):
        # This transaction follows the unchanged original anchor implementation.
        # Individual row construction and the complete solver replace its two-max QP.
        started=time.monotonic(); params=self._validate_runtime(model, teacher, optimizer)
        if q.FRACTIONS != FRACTIONS or anchor.FRACTIONS != FRACTIONS:
            raise RuntimeError('Complete-anchor pilot requires its explicit extended-grid scope')
        ids=[c['source_id'] for c in crops]
        if len(ids)!=12 or len(set(ids))!=12 or set(ids)&self._anchor_ids:
            raise ValueError('Require12 distinct ordinary sources disjoint from calibration anchors')
        self._assert_cache()
        before=[p.detach().clone() for p in params]
        old_optimizer=copy.deepcopy(optimizer.state_dict()); old_rng=screen.rng_state()
        old_grads=[(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in params]
        old_warmed=set(self._warmed); post_rng=None
        try:
            if anchor._frozen(model)!=self._frozen_model or anchor._frozen(teacher)!=self._frozen_teacher:
                raise RuntimeError('Frozen model or teacher state changed')
            warm=None
            if any(warmup._shape_key(model,e) not in self._warmed for e in self._entries):
                warm=warmup.warm_quiet_entries(model,teacher,self._entries,self._warmed,optimizer=optimizer)
            baseline=q.score_entries(model,self._entries)
            initial=anchor.anchor_metrics(baseline,self._entries)
            if initial['passed']!=6:
                raise RuntimeError('A fixed startup anchor was already invalid before the update')
            expanded,bounds=individual_constraints(baseline)
            rows=q.constraint_gradients(model,self._entries,expanded,params)
            if len(rows)!=12:
                raise RuntimeError('Complete startup control requires all twelve gradient rows')
            if any(p.grad is not original or (saved is not None and not torch.equal(p.grad,saved))
                   for p,(original,saved) in zip(params,old_grads)):
                raise RuntimeError('Constraint gradients polluted ordinary gradient slots')
            screen.restore_rng(old_rng)
            tick=time.monotonic()
            values,checks=q.prior.perform_update(model,teacher,crops,common,optimizer,diagnostics=diagnostics)
            ordinary_seconds=time.monotonic()-tick; post_rng=screen.rng_state()
            old_steps=[float(s['step']) for s in old_optimizer['state'].values()]
            expected=(old_steps[0] if old_steps else 0)+1
            if (old_steps and len(set(old_steps))!=1) or set(optimizer.state)!=set(params):
                raise RuntimeError('Ordinary Adam state topology changed')
            if (any(float(s['step'])!=expected for s in optimizer.state.values())
                    or any(not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all() for p in params)
                    or any(not torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq'))):
                raise RuntimeError('Ordinary Adam counter, gradients, parameters or moments are invalid')
            proposal=[p.detach().clone() for p in params]
            delta=[new.double()-old.double() for new,old in zip(proposal,before)]
            projected,linear=projection.project_displacement(rows,delta,bounds)
            if (linear.get('rows')!=12 or linear.get('kkt_passed') is not True
                    or linear.get('full_primal_verified') is not True):
                raise RuntimeError('Complete projection lacks its full-system certificate')
            fraction,after,attempts=q.backtrack(params,before,proposal,projected,baseline,
                lambda:q.score_entries(model,self._entries),corrected=linear['corrected'])
            final=anchor.anchor_metrics(after,self._entries)
            if final['passed']!=6:
                raise RuntimeError('Accepted displacement violates a fixed startup anchor')
            self._assert_cache()
            if anchor._frozen(model)!=self._frozen_model or anchor._frozen(teacher)!=self._frozen_teacher:
                raise RuntimeError('A frozen model or teacher tensor changed during the update')
            accepted=[p.detach().double()-old.double() for p,old in zip(params,before)]
            total_seconds=time.monotonic()-started
            values.update(q_constraints=12,q_constrained_sources=6,q_linear_conflicts=linear['linear_conflicts'],
                q_projected=int(linear['corrected']),q_qp_rank=linear['rank'],q_nonzero_rows=linear['nonzero_rows'],
                q_proposal_norm=linear['proposal_norm'],q_correction_norm=linear['correction_norm'],
                q_projection_cosine=linear['proposal_projected_cosine'],q_accepted_fraction=fraction,
                q_zero_displacement=int(all(torch.equal(p,old) for p,old in zip(params,before))),
                q_accepted_displacement_norm=math.sqrt(q._dot(accepted,accepted)),q_backtrack_rounds=len(attempts),
                q_maximum_switches=sum(a['maximum_switches'] for a in attempts),q_caps_passed=1,
                q_constraint_gradient_sources=6,q_full_primal_verified=1,q_kkt_passed=1,
                q_qp_sets_checked=linear['sets_checked'],q_active_constraints=len(linear['active_set']),
                q_gram_rank=linear['gram_rank'],q_gram_condition=linear['gram_condition_on_resolved_subspace'],
                q_active_condition=linear['active_condition_on_resolved_subspace'],
                q_normalized_primal_violation=linear['reconstructed_normalized_primal_violation_max'],
                q_auxiliary_teacher_checks=0,q_step_seconds=total_seconds,q_ordinary_update_seconds=ordinary_seconds,
                q_auxiliary_seconds=total_seconds-ordinary_seconds,
                startup_anchor_windows=6,startup_anchor_sources=6,startup_anchor_checks=6*(1+len(attempts)),
                startup_anchor_warmup_grad_forwards=0 if warm is None else warm['warmup_grad_forwards'],
                startup_anchor_warmup_verification_forwards=0 if warm is None else warm['verification_forwards'],
                startup_anchor_updates=self._updates+1,startup_anchor_current_batch_constraints=0)
            for prefix,metrics in (('before',initial),('after',final)):
                for key,value in metrics.items():values['startup_anchor_'+prefix+'_'+key]=value
            for cohort in q.COHORTS:
                values['q_'+cohort+'_present']=int(cohort=='near_startup')
                for kind in q.KINDS:
                    values['q_'+cohort+'_'+kind+'_before']=initial[kind+'_max_excess'] if cohort=='near_startup' else None
                    values['q_'+cohort+'_'+kind+'_after']=final[kind+'_max_excess'] if cohort=='near_startup' else None
            screen.restore_rng(post_rng)
            self._updates+=1
            return values,checks
        except BaseException:
            with torch.no_grad():
                for p,old,(original,saved) in zip(params,before,old_grads):
                    p.copy_(old); p.grad=original
                    if saved is not None: original.copy_(saved)
            optimizer.load_state_dict(old_optimizer)
            self._warmed=old_warmed
            screen.restore_rng(old_rng)
            if (not all(torch.equal(p,old) for p,old in zip(params,before))
                    or not replay.compare_tree(optimizer.state_dict(),old_optimizer)['equal']
                    or not replay.compare_tree(screen.rng_state(),old_rng)['equal']):
                raise RuntimeError('Hard-failure transaction rollback did not reproduce the pre-update state')
            raise

    __call__ = perform_update
