"""The frozen twelve-anchor-inequality policy for a real mixed optimizer proposal.

Only the optimizer topology/recipe/state validator differs from the corrected
AdamW wrapper. The ordinary loss helper advances the supplied optimizer once;
projection and bounded normal correction use its complete native displacement.
No synthetic Adam moments, extra losses, or inference operations are introduced.

The inherited constructor keeps the exact anchor cloning, warmup, waveform
checks and RNG guard. The complete transaction is copied to replace only its
Adam-specific state checks. The frozen correction_backtrack is called directly.
Existing q_* scalar names are retained for paired all-AdamW parity; the legacy
q_actual_norm_over_adam denominator means the selected optimizer's proposal.
"""
from __future__ import annotations

import copy
import math
import time

import torch
import recovery_optimizers as optimizers
import startup_corrected_anchor_update as corrected

complete, anchor, q, grid = corrected.complete, corrected.anchor, corrected.q, corrected.grid
base, screen, replay, warmup = complete.base, complete.screen, complete.replay, complete.warmup
VERSION = 'audiovae2_mixed_optimizer_corrected_startup_v1'
FRACTIONS = corrected.FRACTIONS
correction_backtrack = corrected.correction_backtrack


class StartupAnchorUpdate(corrected.StartupAnchorUpdate):
    def __init__(self, model, teacher, entries, optimizer):
        # The factory binds this config to canonical model order, independent of
        # the optimizer's physical81-Adam/9-matrix parameter-group ordering.
        self._optimizer_config = copy.deepcopy(optimizers.optimizer_config(optimizer))
        super().__init__(model,teacher,entries,optimizer)
        self.receipt.update(version=VERSION,parent_corrected_version=corrected.VERSION,
            optimizer=copy.deepcopy(self._optimizer_config),
            optimizer_proposal='Actual native displacement of the selected optimizer across all90 ordered group tensors',
            accepted_zero_policy='Advance the selected optimizer state and all90 counters once even when accepted parameter displacement is zero',
            legacy_metric_aliases={'q_actual_norm_over_adam':'Actual accepted displacement norm divided by the selected optimizer proposal norm'},
            optimizer_specific_state_validation=True,synthetic_adam_state=False)

    def _validate_runtime(self, model, teacher, optimizer):
        params=base.parameters(model)
        if (model is not self._model or teacher is not self._teacher or optimizer is not self._optimizer
                or len(params)!=90 or len(set(params))!=90
                or any(not p.requires_grad or p.dtype!=torch.float32 for p in params)
                or set(params)!={p for p in model.parameters() if p.requires_grad}
                or any(p.requires_grad for p in teacher.parameters())):
            raise ValueError('Expected the bound model, frozen teacher and all90 ordered FP32 group parameters')
        if hasattr(self,'_parameter_ids') and tuple(id(p) for p in params)!=self._parameter_ids:
            raise RuntimeError('Optimizer parameter objects changed')
        if optimizers.optimizer_config(optimizer)!=self._optimizer_config:
            raise ValueError('The sealed selected optimizer recipe changed')
        optimizers.assert_optimizer_state(optimizer,params,self._updates)
        return params

    def _complete_update(self, model, teacher, crops, common, optimizer, *, diagnostics=False):
        # Frozen complete transaction; only optimizer-specific state checks differ.
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
            expanded,bounds=complete.individual_constraints(baseline)
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
            optimizers.assert_optimizer_state(optimizer,params,self._updates+1)
            if optimizers.optimizer_config(optimizer)!=self._optimizer_config:
                raise RuntimeError('The sealed optimizer recipe changed during the ordinary update')
            if any(not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('Ordinary optimizer parameters or gradients are invalid')
            proposal=[p.detach().clone() for p in params]
            delta=[new.double()-old.double() for new,old in zip(proposal,before)]
            projected,linear=complete.projection.project_displacement(rows,delta,bounds)
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
            values,checks = self._complete_update(model,teacher,crops,common,optimizer,diagnostics=diagnostics)
        finally:
            q.backtrack,q.score_entries = original_backtrack,original_score
        values.update(accounting)
        values.update(q_canonical_score_calls=score_calls,
            q_canonical_score_forwards=6*score_calls,
            q_constraint_gradient_forwards=6+accounting['q_normal_gradient_sources'],
            q_actual_norm_over_adam=values['q_accepted_displacement_norm']/values['q_proposal_norm'] if values['q_proposal_norm'] else 0.,
            startup_anchor_checks=6*score_calls)
        values['q_actual_norm_over_optimizer']=values['q_actual_norm_over_adam']
        return values,checks


    __call__ = perform_update
