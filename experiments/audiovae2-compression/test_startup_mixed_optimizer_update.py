"""One real optimizer proposal, final waveform acceptance, and full rollback."""
from copy import deepcopy
from pathlib import Path
import random
import sys
import types

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
from test_grail_candidate_recovery import setup as native_setup, cpu_policy
from test_group_model import snapshot, assert_snapshot
from test_startup_anchor_update import anchors, crops, assert_equal
from test_recovery_optimizers import build, set_gradients
import recovery_optimizers as optimizers
import startup_mixed_optimizer_update as mixed
import startup_corrected_anchor_update as original


def group_input(self,x):return x


def suffix(self,x):
    index=int(x[0,0,0]);named=dict(self.group_named_parameters())
    matrix=named[optimizers.MATRIX_NAMES[index]].flatten()[0]
    other=named[self._audible_adam[index]].flatten()[0]
    # Each anchor observes both optimizer components; every native parameter is
    # connected so canonical gradient rows retain all90 positions.
    value=(matrix-self._anchor_origin[index,0]+other-self._anchor_origin[index,1])*self._anchor_scale
    return x*0+value+sum(p.sum()*0 for p in named.values())


def setup(method,scale=.001):
    teacher,model,*_=native_setup();named=dict(model.group_named_parameters())
    model._audible_adam=tuple(n for n in named if n not in optimizers.MATRIX_NAMES)[:6]
    origin=torch.stack([torch.stack((named[n].flatten()[0].detach(),named[a].flatten()[0].detach()))
        for n,a in zip(optimizers.MATRIX_NAMES[:6],model._audible_adam)])
    model.register_buffer('_anchor_origin',origin.clone())
    model.register_buffer('_anchor_scale',torch.tensor(scale))
    model.group_from_input=types.MethodType(group_input,model)
    model.suffix_from_group=types.MethodType(suffix,model)
    return teacher,model,build(model,method)


def ordinary(model,teacher,batch,common,opt,*,diagnostics=False):
    torch.rand(());random.random();np.random.random()
    opt.zero_grad(set_to_none=True)
    params=mixed.base.parameters(model)
    weights=[torch.sin(torch.arange(p.numel(),dtype=p.dtype).reshape_as(p)*.13+i*.17)+1.1
             for i,p in enumerate(params)]
    loss=sum((p*w).sum() for p,w in zip(params,weights))
    loss.backward();opt.step()
    return {'fixture_loss':float(loss.detach())},[{'source_id':c['source_id']} for c in batch]


def non_timing(values):
    return {k:v for k,v in values.items() if 'seconds' not in k and k!='q_actual_norm_over_optimizer'}


def test_all_adam_adapter_matches_frozen_corrected_policy_exactly(monkeypatch):
    teacher,model,opt=setup('adamw');reference=deepcopy(model);ref_opt=build(reference,'adamw')
    monkeypatch.setattr(mixed.q.prior,'perform_update',ordinary)
    with mixed.grid.extended_grid():
        new=mixed.StartupAnchorUpdate(model,teacher,anchors(),opt)
        old=original.StartupAnchorUpdate(reference,teacher,anchors(),ref_opt)
        rng=mixed.screen.rng_state()
        expected,checks=old.perform_update(reference,teacher,crops(),None,ref_opt)
        expected_rng=mixed.screen.rng_state();mixed.screen.restore_rng(rng)
        actual,got=new.perform_update(model,teacher,crops(),None,opt)
    assert_snapshot(model,snapshot(reference));assert_equal(opt.state_dict(),ref_opt.state_dict())
    assert_equal(mixed.screen.rng_state(),expected_rng)
    assert non_timing(actual)==non_timing(expected) and got==checks
    assert all(torch.equal(p.grad,r.grad) for p,r in zip(mixed.base.parameters(model),mixed.base.parameters(reference)))
    assert actual['q_actual_norm_over_optimizer']==actual['q_actual_norm_over_adam']


@pytest.mark.parametrize('method',['muon','normuon','shampoo'])
def test_retention_acceptance_follows_both_real_optimizer_components_and_keeps_state_once(method,monkeypatch):
    teacher,model,opt=setup(method);reference=deepcopy(model);ref_opt=build(reference,method)
    monkeypatch.setattr(mixed.q.prior,'perform_update',ordinary)
    # Prime through this wrapper so its local update ledger and optimizer agree.
    with mixed.grid.extended_grid():
        updater=mixed.StartupAnchorUpdate(model,teacher,anchors(),opt)
        before=deepcopy(updater._entries);rng=mixed.screen.rng_state()
        ordinary(reference,teacher,crops(),None,ref_opt);expected_rng=mixed.screen.rng_state()
        mixed.screen.restore_rng(rng)
        original_backtrack=mixed.correction_backtrack;observed=[]
        def checked(model_,entries,params,old,proposal,projected,baseline,score_fn,**kwargs):
            optimizers.assert_optimizer_state(opt,params,1)
            assert_equal(opt.state_dict(),ref_opt.state_dict())
            assert all(torch.equal(p,r) for p,r in zip(proposal,mixed.base.parameters(reference)))
            result=original_backtrack(model_,entries,params,old,proposal,projected,baseline,score_fn,**kwargs)
            observed.append(snapshot(model_))
            return result
        monkeypatch.setattr(mixed,'correction_backtrack',checked)
        values,_=updater.perform_update(model,teacher,crops(),None,opt)
        assert len(observed)==1 and values['startup_anchor_after_passed']==6
        assert_snapshot(model,observed[0])  # No normalization/write follows acceptance.
        assert values['q_caps_passed']==values['q_normal_budget_passed']==1
        assert_equal(opt.state_dict(),ref_opt.state_dict());assert_equal(mixed.screen.rng_state(),expected_rng)
        assert_equal(updater._entries,before)
        # Accepted parameters are re-read after both optimizer branches and all
        # normal/projection operations. No post-acceptance optimizer mutation.
        final=mixed.q.score_entries(model,anchors())
        assert all(passed for _,_,passed in final['signature'])
        assert updater._updates==1


@pytest.mark.parametrize('method',['muon','normuon','shampoo'])
def test_failed_final_verification_rolls_back_native_parameters_mixed_state_rng_grad_refs_and_globals(method,monkeypatch):
    teacher,model,opt=setup(method)
    monkeypatch.setattr(mixed.q.prior,'perform_update',ordinary)
    globals_before=mixed.q.backtrack,mixed.q.score_entries
    with mixed.grid.extended_grid():
        updater=mixed.StartupAnchorUpdate(model,teacher,anchors(),opt)
        for _ in range(10 if method=='shampoo' else 1):
            updater.perform_update(model,teacher,crops(),None,opt)
        prior_step=updater._updates
        for p in mixed.base.parameters(model):p.grad=torch.full_like(p,19.)
        slots=[p.grad for p in mixed.base.parameters(model)]
        before=snapshot(model),deepcopy(opt.state_dict()),snapshot(teacher)
        rng=mixed.screen.rng_state();warmed=set(updater._warmed)
        original_backtrack=mixed.correction_backtrack
        def fail_after_native_acceptance(*args,**kwargs):
            result=original_backtrack(*args,**kwargs)
            optimizers.assert_optimizer_state(opt,mixed.base.parameters(model),prior_step+1)
            assert all(passed for _,_,passed in result[0][1]['signature'])
            raise RuntimeError('injected error after nonlinear acceptance')
        monkeypatch.setattr(mixed,'correction_backtrack',fail_after_native_acceptance)
        with pytest.raises(RuntimeError,match='after nonlinear acceptance'):
            updater.perform_update(model,teacher,crops(),None,opt)
        assert_snapshot(model,before[0]);assert_equal(opt.state_dict(),before[1]);assert_snapshot(teacher,before[2])
        assert_equal(mixed.screen.rng_state(),rng)
        assert updater._updates==prior_step and updater._warmed==warmed
        assert all(p.grad is slot and torch.equal(slot,torch.full_like(slot,19.)) for p,slot in zip(mixed.base.parameters(model),slots))
        optimizers.assert_optimizer_state(opt,mixed.base.parameters(model),prior_step)
    assert (mixed.q.backtrack,mixed.q.score_entries)==globals_before
