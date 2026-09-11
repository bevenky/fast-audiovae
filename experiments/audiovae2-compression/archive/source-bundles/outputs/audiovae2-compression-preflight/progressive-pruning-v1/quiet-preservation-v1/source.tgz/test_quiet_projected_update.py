"""Actual Adam, native quiet arithmetic and bounded active-set contracts."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import quiet_projected_update as q
import progressive_model as progressive
from test_group_model import TinyDecoder, gm, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_policy():
    rng=q.screen.rng_state();threads=torch.get_num_threads()
    torch.manual_seed(523);torch.set_num_threads(1)
    yield
    q.screen.restore_rng(rng);torch.set_num_threads(threads)


def vector(values):return [torch.tensor(values,dtype=torch.float64)]


def test_projection_known_intersecting_halfspaces():
    d,r=q.project_displacement([vector([1,0]),vector([1,1])],vector([3,2]),[0,1])
    torch.testing.assert_close(d[0],torch.tensor([0.,1.],dtype=torch.float64),atol=1e-13,rtol=0)
    assert r['kkt_passed'] and r['rank']==2 and r['sets_checked']==4


def test_dependent_opposite_rows_and_zero_rows_are_rank_aware():
    rows=[vector([1,0]),vector([2,0]),vector([-1,0]),vector([0,1]),vector([0,0])]
    d,r=q.project_displacement(rows,vector([2,3]),[0]*5)
    torch.testing.assert_close(d[0],torch.zeros(2,dtype=torch.float64),atol=1e-13,rtol=0)
    assert r['nonzero_rows']==4 and r['sets_checked']==16 and r['rank']==2


def test_constraint_scaling_preserves_projection_and_rhs():
    row=[vector([1.,2.]),vector([-1.,1.])];d=vector([4.,3.])
    expected,_=q.project_displacement(row,d,[.3,.7])
    actual,_=q.project_displacement([vector([1e-8,2e-8]),vector([-1000.,1000.])],d,[3e-9,700.])
    torch.testing.assert_close(actual[0],expected[0],atol=1e-12,rtol=0)


@pytest.mark.parametrize('kind',['nan','asymmetric','indefinite','too_many','incompatible'])
def test_invalid_qp_never_silently_returns_zero_or_unconstrained(kind):
    g=torch.eye(2,dtype=torch.float64);v=torch.ones(2,dtype=torch.float64)
    if kind=='nan':g[0,0]=float('nan')
    elif kind=='asymmetric':g[0,1]=.1
    elif kind=='indefinite':g[0,0]=-1
    elif kind=='too_many':g=torch.eye(7,dtype=torch.float64);v=torch.ones(7,dtype=torch.float64)
    else:g.zero_()
    with pytest.raises((ValueError,RuntimeError)):q.solve_small_qp(g,v)


def test_actual_adam_momentum_displacement_is_not_negative_current_gradient():
    p=nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([p],lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    p.grad=torch.ones_like(p);opt.step()
    before=p.detach().clone();p.grad=torch.full_like(p,-.01);opt.step()
    realized=[p.detach().double()-before.double()]
    assert realized[0].item()<0 and -p.grad.item()>0
    projected,r=q.project_displacement([vector([-1])],realized,[0])
    assert r['corrected'] and abs(projected[0].item())<1e-15
    assert opt.state[p]['step']==2


def crop(target,**changes):
    result={'source_id':'example','context_frames':0,'start_frame':0,'context_start_frame':0,
            'valid_scored_samples':target.shape[-1],'teacher_audio':target,
            'latents':torch.zeros(1,64,target.shape[-1]//1920)}
    result.update(changes);return result


def test_window_context_absolute_start_partial_tail_and_disjoint_cohorts():
    t=torch.zeros(1,1,5760);t[...,:1920]=100;t[...,1920:2880]=5e-6;t[...,2880:3840]=5e-4
    t[...,3857:]=float('nan')
    entry=q.window_layout(crop(t,context_frames=1,start_frame=3,context_start_frame=2,valid_scored_samples=1937),t)
    assert [r['valid_samples'] for r in entry['windows']]==[960,960,17]
    assert [r['cohort'] for r in entry['windows']]==['other_near','remaining_quiet','other_near']
    assert [r['source_start_sample'] for r in entry['windows']]==[5760,6720,7680]
    p=t.clone().requires_grad_();values=q.window_excesses(p,entry)
    sum(v for r in values for v in r['values']).backward()
    assert torch.isfinite(p.grad).all() and not p.grad[...,:1920].any() and not p.grad[...,3857:].any()
    zero=torch.zeros(1,1,1920);startup=q.window_layout(crop(zero),zero)
    assert [r['cohort'] for r in startup['windows']]==['near_startup','other_near']


@pytest.mark.parametrize('teacher_level',[0.,5e-6,5e-4])
def test_physical_excess_sign_matches_exact_existing_pass_flags(teacher_level):
    t=torch.full((1,1,1920),teacher_level);entry=q.window_layout(crop(t),t)
    for level in (0.,.999999e-5,1e-5,1.000001e-5,teacher_level,teacher_level*1.2):
        p=torch.full_like(t,level)
        actual=q.window_excesses(p,entry);raw=q.quiet_window_metrics(p,t)['windows']
        assert [r['passed'] for r in actual]==[r['passed'] for r in raw]
        assert [all(float(f)<=0 for f in r['values']) for r in actual]==[r['passed'] for r in raw]


def test_exact_zero_rms_has_finite_zero_gradient():
    t=torch.zeros(1,1,1920);entry=q.window_layout(crop(t),t);p=t.clone().requires_grad_()
    values=q.window_excesses(p,entry)
    sum(v for r in values for v in r['values']).backward()
    assert torch.equal(p.grad,torch.zeros_like(p))


class ScalarWave(nn.Module):
    def __init__(self):
        super().__init__();self.ps=nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(90)])
        self.grad_forwards=0;self.no_grad_forwards=0
    def group_named_parameters(self):return list(self.named_parameters())
    def group_from_input(self,x):
        if torch.is_grad_enabled():self.grad_forwards+=1
        else:self.no_grad_forwards+=1
        return x
    def suffix_from_group(self,x):
        # All90 parameters connected, one effective waveform degree of freedom.
        return x*0+self.ps[0]+sum(p*0 for p in self.ps[1:])


def entries_for(model,values=(0.,0.)):
    result=[]
    for i,value in enumerate(values):
        t=torch.full((1,1,1920),value);entry=q.window_layout(crop(t,source_id=str(i)),t)
        entry['group_input']=t.detach();result.append(entry)
    return result


def test_deterministic_ties_selected_sources_only_and_no_grad_slot_pollution():
    model=ScalarWave();entries=entries_for(model)
    for p in model.parameters():p.grad=torch.full_like(p,17.)
    grad_ids=[id(p.grad) for p in model.parameters()]
    baseline=q.score_entries(model,entries)
    assert all(v['ref'][0]==0 for _,v in baseline['maxima'])
    assert [v['ref'][1] for _,v in baseline['maxima']]==[0,0,1,1]
    rows=q.constraint_gradients(model,entries,baseline,list(model.parameters()))
    assert len(rows)==4 and model.grad_forwards==1
    assert grad_ids==[id(p.grad) for p in model.parameters()]
    assert all(torch.equal(p.grad,torch.full_like(p,17.)) for p in model.parameters())


def test_nonzero_quiet_gradient_retains_physical_squared_units_and_detached_target():
    model=ScalarWave()
    with torch.no_grad():model.ps[0].fill_(2e-5)
    entries=entries_for(model);baseline=q.score_entries(model,entries)
    rows=q.constraint_gradients(model,entries,baseline,list(model.parameters()))
    for row in rows:
        torch.testing.assert_close(row[0],torch.full_like(row[0],4e-5),rtol=1e-6,atol=0)
        assert all(not g.any() for g in row[1:])
    assert all(not e['target'].requires_grad for e in entries)


def stat(values,refs=None):
    refs=refs or [(0,0)]*len(values)
    return {'maxima':[(('near_startup',q.KINDS[i]),{'value':v,'ref':refs[i]}) for i,v in enumerate(values)],
            'signature':[(ref,(value,),value<=0) for ref,value in zip(refs,values)]}


def test_no_conflict_acceptance_never_rewrites_ordinary_fp32_values():
    p=nn.Parameter(torch.tensor([.123456789]));before=[torch.tensor([-.7654321])];proposal=[p.detach().clone()]
    version=p._version;baseline=stat([0.]);d=[p.detach().double()-before[0].double()]
    fraction,_,attempts=q.backtrack([p],before,proposal,d,baseline,lambda:baseline,corrected=False)
    assert fraction==1 and p._version==version and torch.equal(p,proposal[0]) and len(attempts)==1


def test_nonlinear_rescan_catches_changed_maximum_even_without_linear_conflict():
    p=nn.Parameter(torch.tensor([1.]));before=[torch.zeros(1)];proposal=[p.detach().clone()]
    def scores():
        # First window governs pre-state; another becomes positive at the proposal.
        first=0.;second=float(p)**2-.3
        return stat([max(first,second)],[(0,0) if first>=second else (0,1)])
    baseline=stat([0.],[(0,0)])
    fraction,_,attempts=q.backtrack([p],before,proposal,[torch.ones(1,dtype=torch.float64)],baseline,scores,corrected=False)
    assert fraction==.5 and attempts[0]['maximum_switches']==1 and not attempts[0]['accepted']


def test_nonreproducible_zero_displacement_is_error_not_tolerance():
    p=nn.Parameter(torch.ones(1));baseline=stat([-1.])
    with pytest.raises(RuntimeError,match='complete pre-state'):
        q.backtrack([p],[torch.zeros(1)],[p.detach().clone()],[torch.ones(1)],baseline,lambda:stat([1.]),corrected=False)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.decoder=TinyDecoder()
        self.eval().requires_grad_(False)


def test_no_quiet_matches_real_unchanged_adam_params_moments_rng_and_losses(monkeypatch):
    teacher=Teacher();selection={'stage2_indices':list(range(24)),'stage3_indices':list(range(16))}
    model=progressive.initialize_from_teacher(teacher.model.decoder,selection)
    oracle=progressive.initialize_from_teacher(teacher.model.decoder,selection)
    optimizer=progressive.fresh_optimizer(model);expected_opt=progressive.fresh_optimizer(oracle)
    original_batch=q.base.batch
    monkeypatch.setattr(q.base,'batch',lambda rows:original_batch(rows,device='cpu'))
    @torch.no_grad()
    def teacher_forward(instance,z):
        trace=gm.teacher_trace(instance.model.decoder,z[:,:8]);trace['waveform']=trace['waveform']+1
        return trace
    monkeypatch.setattr(q.base,'teacher_forward',teacher_forward)
    crops=[]
    for i in range(12):
        frames=2+i%2;context=i%2;z=torch.randn(1,64,frames)*.01
        t=teacher_forward(teacher,z)['waveform'].clone()
        crops.append(crop(t,source_id=f'crop-{i}',latents=z,context_frames=context,start_frame=context,
                          valid_scored_samples=(frames-context)*1920-i*7))
    objective=q.base.ReconstructionV2(q.base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    teacher_before=snapshot(teacher);rng=q.screen.rng_state()
    expected,expected_checks=q.prior.perform_update(oracle,teacher,crops,objective,expected_opt,diagnostics=True)
    expected_rng=q.screen.rng_state();q.screen.restore_rng(rng)
    actual,checks=q.perform_update(model,teacher,crops,objective,optimizer,diagnostics=True)
    assert {k:v for k,v in actual.items() if not k.startswith('q_')}==expected
    assert actual['q_constraints']==0 and actual['q_auxiliary_teacher_checks']==0
    assert checks==expected_checks and q.replay.compare_tree(optimizer.state_dict(),expected_opt.state_dict())['equal']
    assert q.replay.compare_tree(q.screen.rng_state(),expected_rng)['equal']
    assert_snapshot(model.decoder,snapshot(oracle.decoder));assert_snapshot(teacher,teacher_before)


def test_zero_nonlinear_fallback_restores_weights_but_advances_adam_once(monkeypatch):
    model=ScalarWave();teacher=nn.Linear(1,1).requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1,betas=(.9,.99),eps=1e-8,weight_decay=0)
    t=torch.zeros(1,1,1920);crops=[crop(t,source_id=f'crop-{i}') for i in range(12)]
    original_batch=q.base.batch
    monkeypatch.setattr(q.base,'batch',lambda rows:original_batch(rows,device='cpu'))
    monkeypatch.setattr(q.base,'teacher_forward',lambda teacher,z:{'group_input':t,'waveform':t})
    calls=[]
    def ordinary(model,teacher,crops,common,optimizer,*,diagnostics):
        calls.append((len(crops),diagnostics));optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():p.grad=-torch.ones_like(p)
        optimizer.step()
        return {'waveform':.25},[{'source_id':c['source_id']} for c in crops]
    monkeypatch.setattr(q.prior,'perform_update',ordinary)
    before=snapshot(model);rng=q.screen.rng_state()
    actual,checks=q.perform_update(model,teacher,crops,None,optimizer,diagnostics=True)
    assert calls==[(12,True)] and len(checks)==12 and actual['waveform']==.25
    assert actual['q_nonzero_rows']==0 and actual['q_projected']==0
    assert actual['q_accepted_fraction']==0 and actual['q_backtrack_rounds']==6 and actual['q_zero_displacement']==1
    assert actual['q_constraint_gradient_sources']==1 and actual['q_auxiliary_teacher_checks']==12
    assert_snapshot(model,before)
    assert all(int(s['step'])==1 and torch.equal(s['exp_avg'],torch.full_like(s['exp_avg'],-.1)) for s in optimizer.state.values())
    assert all(torch.equal(p.grad,-torch.ones_like(p)) for p in model.parameters())
    assert q.replay.compare_tree(q.screen.rng_state(),rng)['equal']


def test_wrapper_projects_realized_adam_with_nonzero_quiet_gradient(monkeypatch):
    model=ScalarWave();teacher=nn.Linear(1,1).requires_grad_(False)
    with torch.no_grad():model.ps[0].fill_(2e-5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1,betas=(.9,.99),eps=1e-8,weight_decay=0)
    t=torch.zeros(1,1,1920);crops=[crop(t,source_id=f'crop-{i}') for i in range(12)]
    original_batch=q.base.batch
    monkeypatch.setattr(q.base,'batch',lambda rows:original_batch(rows,device='cpu'))
    monkeypatch.setattr(q.base,'teacher_forward',lambda teacher,z:{'group_input':t,'waveform':t})
    proposed=[]
    def ordinary(model,teacher,crops,common,optimizer,*,diagnostics):
        optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():p.grad=-torch.ones_like(p)
        optimizer.step();proposed.extend(p.detach().clone() for p in model.parameters())
        return {},[{'source_id':c['source_id']} for c in crops]
    monkeypatch.setattr(q.prior,'perform_update',ordinary)
    before=model.ps[0].detach().clone()
    actual,_=q.perform_update(model,teacher,crops,None,optimizer)
    assert actual['q_nonzero_rows']==4 and actual['q_projected']==1 and actual['q_qp_rank']==1
    assert actual['q_accepted_fraction']==1 and torch.equal(model.ps[0],before)
    assert all(torch.equal(p,expected) for p,expected in zip(list(model.parameters())[1:],proposed[1:]))
    assert all(int(s['step'])==1 for s in optimizer.state.values())
    assert actual['q_near_startup_residual_after']<=actual['q_near_startup_residual_before']


def test_rejects_duplicate_sources_before_ordinary_update(monkeypatch):
    model=ScalarWave();teacher=nn.Linear(1,1).requires_grad_(False);t=torch.zeros(1,1,1920)
    opt=torch.optim.AdamW(model.parameters());called=[]
    monkeypatch.setattr(q.prior,'perform_update',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError):q.perform_update(model,teacher,[crop(t)]*12,None,opt)
    assert not called and not opt.state


def test_no_quiet_reports_actual_zero_ordinary_displacement_and_keeps_moments_once(monkeypatch):
    model=ScalarWave();teacher=nn.Linear(1,1).requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    target=torch.ones(1,1,1920);crops=[crop(target,source_id=str(i)) for i in range(12)]
    original_batch=q.base.batch
    monkeypatch.setattr(q.base,'batch',lambda rows:original_batch(rows,device='cpu'))
    def forbidden(*a,**kw):raise AssertionError('No-quiet path performs no auxiliary teacher/student forward')
    monkeypatch.setattr(q.base,'teacher_forward',forbidden)
    monkeypatch.setattr(model,'group_from_input',forbidden)
    calls=[]
    def ordinary(model,teacher,crops,common,optimizer,*,diagnostics):
        calls.append(True);optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():p.grad=torch.zeros_like(p)
        optimizer.step();return {},[]
    monkeypatch.setattr(q.prior,'perform_update',ordinary)
    before=snapshot(model);rng=q.screen.rng_state()
    values,_=q.perform_update(model,teacher,crops,None,optimizer)
    assert calls==[True] and values['q_constraints']==0 and values['q_zero_displacement']==1
    assert values['q_accepted_fraction']==1 and values['q_backtrack_rounds']==0
    assert all(int(s['step'])==1 and not s['exp_avg'].any() and not s['exp_avg_sq'].any() for s in optimizer.state.values())
    assert_snapshot(model,before);assert q.replay.compare_tree(q.screen.rng_state(),rng)['equal']
