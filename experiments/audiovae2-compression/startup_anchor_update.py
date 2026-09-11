"""Protect six fixed calibration starts while advancing ordinary Adam once.

Only two max-excess constraints are used. Every nonlinear candidate is rescored
on all six anchors at their unchanged waveform limits. Calibration constraint
reuse is additional training exposure; it is not a development-set guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import torch
import quiet_projected_update as q
import quiet_constraint_warmup as warmup

VERSION = 'audiovae2_six_startup_anchor_adam_v1'
FRACTIONS = q.FRACTIONS
base, screen, replay = q.base, q.screen, q.replay


def _digest_tensors(named):
    h = hashlib.sha256()
    for name, value in named:
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((tuple(x.shape), str(x.dtype))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def _token(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad)


def _frozen(module):
    return {('parameter', n):_token(p) for n,p in module.named_parameters() if not p.requires_grad} | {
        ('buffer', n):_token(b) for n,b in module.named_buffers()}


def validate_anchors(entries):
    """Clone caller-authenticated calibration inputs; retain the original full shape."""
    if len(entries)!=6:
        raise ValueError('Exactly six fixed calibration startup anchors are required')
    result=[]; ids=[]
    for entry in entries:
        crop = entry['crop']
        metadata = {k:crop[k] for k in ('source_id','start_frame','context_start_frame',
                                       'context_frames','valid_scored_samples')}
        if (metadata['start_frame']!=0 or metadata['context_start_frame']!=0
                or metadata['context_frames']!=0 or not isinstance(metadata['source_id'], str)):
            raise ValueError('Anchors must be original source starts, not substituted crop starts')
        target, x = entry['target'], entry['group_input']
        if (not isinstance(target, torch.Tensor) or not isinstance(x, torch.Tensor)
                or target.dtype!=torch.float32 or x.dtype!=torch.float32
                or target.ndim!=3 or target.shape[:2]!=(1,1) or x.ndim!=3 or x.shape[0]!=1
                or target.requires_grad or x.requires_grad or target.grad_fn is not None or x.grad_fn is not None
                or target.device!=x.device or not torch.isfinite(target).all() or not torch.isfinite(x).all()):
            raise ValueError('Anchor targets/teacher inputs must be finite detached singleton FP32 tensors')
        canonical = q.window_layout(metadata, target)
        rows = [r for r in canonical['windows'] if r['cohort']=='near_startup'
                and r['source_start_sample']==0 and r['source_stop_sample']==960 and r['valid_samples']==960]
        if len(rows)!=1 or entry['span']!=canonical['span'] or entry['windows']!=rows:
            raise ValueError('Anchor window/limit identity differs from the original first960 samples')
        ids.append(metadata['source_id'])
        result.append({'crop':metadata, 'span':tuple(canonical['span']), 'windows':copy.deepcopy(rows),
                       'target':target.detach().clone(), 'group_input':x.detach().clone()})
    if len(set(ids))!=6:
        raise ValueError('Anchor sources must be distinct')
    return result


def anchor_metrics(score, entries):
    """Aggregate canonical per-window RMS values; no source/window identities escape."""
    expected = [('near_startup', kind) for kind in q.KINDS]
    if [k for k,_ in score['maxima']]!=expected or len(score['signature'])!=6:
        raise RuntimeError('The two anchor maxima or six individual windows changed')
    result={'windows':6,'passed':0,'residual_only_failed':0,'amplitude_only_failed':0,'both_failed':0,
            'residual_max_excess':score['maxima'][0][1]['value'],
            'amplitude_max_excess':score['maxima'][1][1]['value']}
    energies={'residual':0.,'student':0.,'teacher':0.}; count=0; seen=set()
    for ref, values, passed in score['signature']:
        source, window = ref
        if source in seen or not 0<=source<6 or len(values)!=2:
            raise RuntimeError('Anchor score has missing/repeated source support')
        seen.add(source); row=entries[source]['windows'][0]
        if window!=row['window_index'] or not all(math.isfinite(v) for v in values):
            raise RuntimeError('Anchor score grid or finite excess changed')
        rf, af = values[0]>0, values[1]>0
        if passed != (not rf and not af):
            raise RuntimeError('Anchor excess signs differ from canonical acceptance')
        result['passed']+=int(passed)
        result['residual_only_failed']+=int(rf and not af)
        result['amplitude_only_failed']+=int(af and not rf)
        result['both_failed']+=int(rf and af)
        n=row['valid_samples']; count+=n
        residual=values[0]+row['residual_limit']**2
        student=values[1]+row['output_rms_limit']**2
        if residual<0 or student<0:
            raise RuntimeError('Canonical squared anchor RMS is negative')
        energies['residual']+=n*residual; energies['student']+=n*student
        energies['teacher']+=n*row['teacher_rms']**2
    result['samples']=count
    for key,value in energies.items(): result[key+'_rms']=math.sqrt(value/count)
    return result


class StartupAnchorUpdate:
    def __init__(self, model, teacher, entries, optimizer):
        self._model, self._teacher, self._optimizer = model, teacher, optimizer
        self._entries = validate_anchors(entries)
        self._anchor_ids = frozenset(e['crop']['source_id'] for e in self._entries)
        self._warmed = set(); self._updates = 0
        self._tensor_tokens = [(_token(e['target']),_token(e['group_input'])) for e in self._entries]
        self._metadata_hash = self._metadata_digest()
        params = self._validate_runtime(model, teacher, optimizer)
        self._parameter_ids = tuple(id(p) for p in params)
        self._frozen_model, self._frozen_teacher = _frozen(model), _frozen(teacher)
        rng = screen.rng_state()
        with replay.diagnostic_state_guard(model, teacher, optimizer):
            receipt = warmup.warm_quiet_entries(model, teacher, self._entries, self._warmed, optimizer=optimizer)
            initial = q.score_entries(model, self._entries)
            metrics = anchor_metrics(initial, self._entries)
            if metrics['passed']!=6:
                raise RuntimeError('All six startup anchors must pass before recovery')
        if not replay.compare_tree(screen.rng_state(), rng)['equal']:
            raise RuntimeError('Anchor initialization consumed the original RNG state')
        self.initialization_receipt = {'version':VERSION, 'anchor_windows':6, 'anchor_samples':5760,
            'constraints':2, 'initial':metrics, 'initial_model_sha256':_digest_tensors(model.state_dict().items()),
            'anchor_identity_sha256':self._metadata_hash,
            'cached_target_input_sha256':_digest_tensors((str(i)+'/'+key,e[key]) for i,e in enumerate(self._entries)
                                                       for key in ('target','group_input')),
            'warmup_grad_forwards':receipt['warmup_grad_forwards'],
            'warmup_verification_forwards':receipt['verification_forwards'],
            'warmup_new_shapes':receipt['new_shapes'],
            'warmup_first_vs_third_nonexact_shapes':sum(not r['first_vs_third']['passed'] for r in receipt['cold_comparisons']),
            'state_and_rng_preserved':True, 'targets_detached':True, 'full_original_shapes_preserved':True,
            'current_batch_quiet_constraints':False, 'new_loss_weights':False, 'new_inference_operations':0,
            'scope':'Repeated six-calibration-source startup constraints; ordinary data stream remains separate; development excluded',
            'rms_reduction':'Sample-pooled squares of original evaluator FP32 window RMS values',
            'accepted_zero_policy':'Advance ordinary Adam moments/counters once even when accepted parameter displacement is zero'}
        self.receipt = self.initialization_receipt

    def _metadata_digest(self):
        return screen.digest([{k:e[k] for k in ('crop','span','windows')} for e in self._entries])

    def _assert_cache(self):
        if (self._metadata_digest()!=self._metadata_hash or
                [(_token(e['target']),_token(e['group_input'])) for e in self._entries]!=self._tensor_tokens):
            raise RuntimeError('Frozen anchor metadata, target or teacher input changed')

    def _validate_runtime(self, model, teacher, optimizer):
        params=base.parameters(model)
        if (model is not self._model or teacher is not self._teacher or optimizer is not self._optimizer
                or len(params)!=90 or len(set(params))!=90
                or any(not p.requires_grad or p.dtype!=torch.float32 for p in params)
                or set(params)!={p for p in model.parameters() if p.requires_grad}
                or any(p.requires_grad for p in teacher.parameters())
                or [id(p) for g in optimizer.param_groups for p in g['params']]!=[id(p) for p in params]
                or not isinstance(optimizer, torch.optim.AdamW)):
            raise ValueError('Expected the bound model/teacher/AdamW and all90 ordered FP32 group parameters')
        if hasattr(self,'_parameter_ids') and tuple(id(p) for p in params)!=self._parameter_ids:
            raise RuntimeError('Optimizer parameter objects changed')
        if any(g['lr']!=3e-5 or g['betas']!=(.9,.99) or g['eps']!=1e-8 or g['weight_decay']!=0
               for g in optimizer.param_groups):
            raise ValueError('Original AdamW recipe changed')
        return params

    def perform_update(self, model, teacher, crops, common, optimizer, *, diagnostics=False):
        started=time.monotonic(); params=self._validate_runtime(model, teacher, optimizer)
        ids=[c['source_id'] for c in crops]
        if len(ids)!=12 or len(set(ids))!=12 or set(ids)&self._anchor_ids:
            raise ValueError('Require12 distinct ordinary sources disjoint from calibration anchors')
        self._assert_cache()
        before=[p.detach().clone() for p in params]
        old_optimizer=copy.deepcopy(optimizer.state_dict()); old_rng=screen.rng_state()
        old_grads=[(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in params]
        old_warmed=set(self._warmed); post_rng=None
        try:
            if _frozen(model)!=self._frozen_model or _frozen(teacher)!=self._frozen_teacher:
                raise RuntimeError('Frozen model or teacher state changed')
            warm=None
            if any(warmup._shape_key(model,e) not in self._warmed for e in self._entries):
                warm=warmup.warm_quiet_entries(model,teacher,self._entries,self._warmed,optimizer=optimizer)
            baseline=q.score_entries(model,self._entries)
            initial=anchor_metrics(baseline,self._entries)
            if initial['passed']!=6:
                raise RuntimeError('A fixed startup anchor was already invalid before the update')
            rows=q.constraint_gradients(model,self._entries,baseline,params)
            if len(rows)!=2:
                raise RuntimeError('Startup-only control must use exactly two max-gradient rows')
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
            bounds=[-v['value'] for _,v in baseline['maxima']]
            projected,projection=q.project_displacement(rows,delta,bounds)
            fraction,after,attempts=q.backtrack(params,before,proposal,projected,baseline,
                lambda:q.score_entries(model,self._entries),corrected=projection['corrected'])
            final=anchor_metrics(after,self._entries)
            if final['passed']!=6:
                raise RuntimeError('Accepted displacement violates a fixed startup anchor')
            self._assert_cache()
            if _frozen(model)!=self._frozen_model or _frozen(teacher)!=self._frozen_teacher:
                raise RuntimeError('A frozen model or teacher tensor changed during the update')
            accepted=[p.detach().double()-old.double() for p,old in zip(params,before)]
            total_seconds=time.monotonic()-started
            values.update(q_constraints=2,q_constrained_sources=6,q_linear_conflicts=projection['linear_conflicts'],
                q_projected=int(projection['corrected']),q_qp_rank=projection['rank'],q_nonzero_rows=projection['nonzero_rows'],
                q_proposal_norm=projection['proposal_norm'],q_correction_norm=projection['correction_norm'],
                q_projection_cosine=projection['proposal_projected_cosine'],q_accepted_fraction=fraction,
                q_zero_displacement=int(all(torch.equal(p,old) for p,old in zip(params,before))),
                q_accepted_displacement_norm=math.sqrt(q._dot(accepted,accepted)),q_backtrack_rounds=len(attempts),
                q_maximum_switches=sum(a['maximum_switches'] for a in attempts),q_caps_passed=1,
                q_constraint_gradient_sources=len({v['ref'][0] for _,v in baseline['maxima']}),
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
