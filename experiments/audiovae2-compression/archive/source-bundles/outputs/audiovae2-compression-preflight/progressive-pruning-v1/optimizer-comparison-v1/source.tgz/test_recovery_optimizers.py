"""Independent small-matrix formulas, explicit native scope and state contracts.

References: KellerJordan/Muon muon.py NS5 polynomial; NorMuon Algorithm 1
https://arxiv.org/html/2510.05491v1; Shampoo matrix inverse-fourth-root
preconditioning with the explicitly declared Adam norm graft.
The scalar singular-value oracle deliberately does not call implementation NS.
"""
from copy import deepcopy
from pathlib import Path
import math
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'convnext'))
from test_grail_candidate_recovery import setup as native_setup, cpu_policy
from test_group_model import snapshot, assert_snapshot
import recovery_optimizers as run


def build(model, method, lr=.002):
    return run.build_optimizer(model, method, matrix_lr=None if method=='adamw' else lr,
                               expected_widths=(24,16,8))


def svd_polynomial(matrix):
    """Evaluate NS5 on scalar singular values, independently of matrix iteration."""
    x = matrix.detach().double()
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    singular = singular / (float(torch.linalg.vector_norm(x)) + 1e-7)
    for _ in range(5): singular = 3.4445*singular - 4.7750*singular**3 + 2.0315*singular**5
    return (u*singular.unsqueeze(0)) @ vh


def muon_formula(gradients, *, neuron_scaling=False):
    momentum = torch.zeros_like(gradients[0], dtype=torch.float64)
    rows = torch.zeros(gradients[0].shape[0], 1, dtype=torch.float64)
    results = []
    for gradient in gradients:
        momentum = .95*momentum + .05*gradient.double()
        ortho = svd_polynomial(momentum)
        if neuron_scaling:
            rows = .95*rows + .05*ortho.square().mean(dim=1, keepdim=True)
            ortho = ortho / (rows.sqrt()+1e-8)
        norm = float(torch.linalg.vector_norm(ortho))
        direction = ortho*(.2*math.sqrt(ortho.numel())/norm) if norm else ortho
        results.append((direction.clone(), momentum.clone(), rows.clone()))
    return results


def symmetric_inverse_fourth(matrix):
    values, vectors = torch.linalg.eigh(matrix.double())
    assert torch.all(values > 0)
    return (vectors*values.pow(-.25).unsqueeze(0)) @ vectors.T


def shampoo_formula(gradients):
    """Matrix EMA and independent spectral roots, with roots refreshed at 10,20,..."""
    rows, cols = gradients[0].shape
    first = torch.zeros(rows, cols, dtype=torch.float64)
    diagonal = torch.zeros_like(first)
    left = torch.zeros(rows, rows, dtype=torch.float64)
    right = torch.zeros(cols, cols, dtype=torch.float64)
    roots = None; results = []
    for step, gradient in enumerate(gradients, 1):
        g = gradient.double()
        first = .9*first + .1*g
        diagonal = .99*diagonal + .01*g.square()
        left = .99*left + .01*(g@g.T)
        right = .99*right + .01*(g.T@g)
        mean = first/(1-.9**step)
        graft = mean/(torch.sqrt(diagonal/(1-.99**step))+1e-8)
        if step % 10 == 0:
            roots = (symmetric_inverse_fourth(left/(1-.99**step)+1e-12*torch.eye(rows)),
                     symmetric_inverse_fourth(right/(1-.99**step)+1e-12*torch.eye(cols)))
        if roots is None:
            direction = graft
        else:
            raw = roots[0] @ mean @ roots[1]
            norm = float(torch.linalg.vector_norm(raw))
            direction = raw*(float(torch.linalg.vector_norm(graft))/norm) if norm else raw
        results.append({'direction':direction.clone(), 'first':first.clone(),
            'diagonal':diagonal.clone(), 'left':left.clone(), 'right':right.clone(),
            'roots':None if roots is None else tuple(r.clone() for r in roots)})
    return results


def gradient_sequence(shape, count):
    """Noncommuting, nonsymmetric matrices with different rows and spectra."""
    n = math.prod(shape)
    flat = torch.arange(n, dtype=torch.float32).reshape(shape)
    return [torch.sin(flat*.37 + step*.23) + torch.cos(flat*.11 - step*.41)
            + torch.eye(*shape)*(.3+.02*step) for step in range(count)]


@pytest.mark.parametrize('shape', [(3,5),(5,3),(3,3)])
def test_ns5_matches_independent_singular_value_polynomial_and_does_not_mutate_input(shape):
    x = gradient_sequence(shape,1)[0]; before=x.clone()
    actual=run.newton_schulz(x)
    torch.testing.assert_close(actual.double(),svd_polynomial(x),rtol=3e-5,atol=3e-6)
    assert torch.equal(x,before) and actual.dtype==torch.float32
    assert torch.equal(run.newton_schulz(torch.zeros_like(x)),torch.zeros_like(x))
    with pytest.raises(ValueError):run.newton_schulz(x.unsqueeze(-1))


@pytest.mark.parametrize('method', ['muon','normuon'])
def test_two_steps_match_ema_ns_and_neuron_formula_on_independent_rectangular_matrices(method):
    model=native_setup()[1];config=run.optimizer_config(build(model,method))['matrix']
    batches=[gradient_sequence((3,5),2),[x.T.contiguous()*2.3 for x in gradient_sequence((3,5),2)]]
    states=[{},{}]
    references=[muon_formula(gs,neuron_scaling=method=='normuon') for gs in batches]
    for step in range(2):
        for index,(gs,state) in enumerate(zip(batches,states)):
            original=gs[step].clone()
            actual=run.matrix_direction(gs[step],state,method,config)
            expected,momentum,variance=references[index][step]
            torch.testing.assert_close(actual.double(),expected,rtol=4e-5,atol=3e-6)
            torch.testing.assert_close(state['momentum'].double(),momentum,rtol=2e-6,atol=1e-8)
            assert state['step']==step+1 and torch.equal(original,gs[step])
            assert actual.square().mean().sqrt().item()==pytest.approx(.2,abs=1e-7)
            if method=='normuon':
                assert state['variance_neuron'].shape==(gs[step].shape[0],1)
                torch.testing.assert_close(state['variance_neuron'].double(),variance,rtol=5e-5,atol=1e-8)
    assert states[0]['momentum'].data_ptr()!=states[1]['momentum'].data_ptr()


def test_shampoo_first_second_and_refresh_steps_match_bias_corrected_spectral_graft():
    model=native_setup()[1];config=run.optimizer_config(build(model,'shampoo'))['matrix']
    gradients=gradient_sequence((3,2),21);reference=shampoo_formula(gradients);state={}
    previous_root=None
    for step,g in enumerate(gradients,1):
        actual=run.matrix_direction(g,state,'shampoo',config)
        expected=reference[step-1]
        torch.testing.assert_close(actual,expected['direction'],rtol=2e-8,atol=2e-9)
        for key,ref in [('gradient_ema','first'),('graft_second','diagonal'),('factor_left','left'),('factor_right','right')]:
            assert state[key].dtype==torch.float64
            torch.testing.assert_close(state[key],expected[ref],rtol=2e-13,atol=2e-14)
        assert state['last_refresh']==step//10*10
        if step in (10,20):
            torch.testing.assert_close(state['inverse_left'],expected['roots'][0],rtol=2e-8,atol=1e-8)
            previous_root=state['inverse_left'].clone()
        elif step>10:assert torch.equal(state['inverse_left'],previous_root)
    assert reference[9]['roots'] is not None
    # The matrix preconditioner must actually change direction after its start.
    g=gradients[9];st={};fallback=dict(config,start_preconditioning_step=30)
    for row in gradients[:10]:diagonal=run.matrix_direction(row,st,'shampoo',fallback)
    assert not torch.allclose(reference[9]['direction'],diagonal,rtol=1e-3,atol=1e-4)


def set_gradients(model, step):
    for index,(_,p) in enumerate(model.group_named_parameters()):
        v=torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape)
        p.grad=torch.sin(v*.17+index*.19+step*.31)+torch.cos(v*.07-index*.11)


def test_exact_nine_weight_v_scope_excludes_gains_snakes_embeddings_and_bad_native_geometry(monkeypatch):
    teacher,model,*_=native_setup()
    named=dict(model.group_named_parameters());before=snapshot(model)
    manifest=run.matrix_parameter_manifest(model,expected_widths=(24,16,8))
    assert {row['name'] for row in manifest}==set(run.MATRIX_NAMES) and len(manifest)==9
    opt=build(model,'muon');matrix={id(p) for p in opt.param_groups[1]['params']}
    assert matrix=={id(named[n]) for n in run.MATRIX_NAMES}
    assert len(opt.param_groups[0]['params'])==81
    assert all(id(p) not in matrix for n,p in named.items() if not n.endswith('.block.3.weight_v'))
    assert_snapshot(model,before)
    module=model.decoder.get_submodule(run.MATRIX_NAMES[0].rsplit('.',1)[0])
    monkeypatch.setattr(module,'groups',2)
    with pytest.raises(ValueError):build(model,'muon')
    monkeypatch.setattr(module,'groups',1)
    with pytest.raises(ValueError):run.build_optimizer(model,'muon',matrix_lr=.002,expected_widths=(16,16,8))
    original=model.group_named_parameters
    monkeypatch.setattr(model,'group_named_parameters',lambda:[(n,p) for n,p in original() if n!=run.MATRIX_NAMES[-1]])
    with pytest.raises(ValueError):build(model,'muon')


@pytest.mark.parametrize('method', ['adamw','muon','normuon','shampoo'])
def test_native_updates_match_independent_2d_matrices_and_original_adam_complement(method):
    teacher,model,*_=native_setup();opt=build(model,method)
    oracle=deepcopy(model);matrix_names=set() if method=='adamw' else set(run.MATRIX_NAMES)
    oracle_named=dict(oracle.group_named_parameters())
    adam=torch.optim.AdamW([p for n,p in oracle_named.items() if n not in matrix_names],
        lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    matrix_states={name:{} for name in matrix_names};cfg=run.optimizer_config(opt)['matrix']
    objects=[(id(p),p.data_ptr()) for p in model.parameters()];teacher_before=snapshot(teacher)
    for step in (1,2):
        set_gradients(model,step)
        for name,p in model.group_named_parameters():oracle_named[name].grad=p.grad.clone()
        adam.step()
        with torch.no_grad():
            for name in matrix_names:
                p=oracle_named[name]
                # Separate true2D reference tensors/state for every native filter.
                update=run.matrix_direction(p.grad[:,:,0].clone(),matrix_states[name],method,cfg)
                p.add_(update.float().unsqueeze(-1),alpha=-cfg['lr'])
        opt.step();run.assert_optimizer_state(opt,[p for _,p in model.group_named_parameters()],step)
        assert_snapshot(model,snapshot(oracle))
        for name,p in model.group_named_parameters():
            if name not in matrix_names:
                assert run._step(opt.state[p]['step'])==step
                for key in ('exp_avg','exp_avg_sq'):assert torch.equal(opt.state[p][key],adam.state[oracle_named[name]][key])
    assert objects==[(id(p),p.data_ptr()) for p in model.parameters()]
    assert_snapshot(teacher,teacher_before)


@pytest.mark.parametrize('method', ['muon','normuon','shampoo'])
def test_state_roundtrip_preserves_matrix_precision_and_next_native_update(method):
    _,model,*_=native_setup();opt=build(model,method)
    steps=10 if method=='shampoo' else 2
    for step in range(1,steps+1):set_gradients(model,step);opt.step()
    saved=deepcopy(opt.state_dict());clone=deepcopy(model);other=build(clone,method)
    other.load_state_dict(saved)
    assert run.optimizer_config(other)==run.optimizer_config(opt)
    assert other._adam.state is other.state and other._adam.param_groups[0] is other.param_groups[0]
    if method=='shampoo':
        for p in other.param_groups[1]['params']:
            assert all(other.state[p][key].dtype==torch.float64 for key in run.DOUBLE_STATE)
    from test_startup_anchor_update import assert_equal
    assert_equal(other.state_dict(),saved)
    set_gradients(model,steps+1);set_gradients(clone,steps+1)
    opt.step();other.step();assert_snapshot(model,snapshot(clone));assert_equal(opt.state_dict(),other.state_dict())
    damaged=deepcopy(other.state_dict());sid=damaged['param_groups'][1]['params'][-1]
    key='factor_left' if method=='shampoo' else 'momentum'
    damaged['state'][sid][key].reshape(-1)[0]=float('nan')
    prior=deepcopy(other.state_dict())
    with pytest.raises(ValueError):other.load_state_dict(damaged)
    assert_equal(other.state_dict(),prior)
    reordered=deepcopy(prior)
    ids=reordered['param_groups'][1]['params']
    ids[0],ids[1]=ids[1],ids[0]  # Equal shapes must not allow state/identity swaps.
    with pytest.raises(ValueError):other.load_state_dict(reordered)
    assert_equal(other.state_dict(),prior)


@pytest.mark.parametrize('damage', ['missing','nonfinite'])
def test_missing_or_nonfinite_gradient_fails_before_any_parameter_or_state_update(damage):
    _,model,*_=native_setup();opt=build(model,'muon');set_gradients(model,1)
    last=dict(model.group_named_parameters())[run.MATRIX_NAMES[-1]]
    if damage=='missing':last.grad=None
    else:last.grad.reshape(-1)[0]=float('nan')
    before=snapshot(model)
    with pytest.raises(RuntimeError):opt.step()
    assert not opt.state;assert_snapshot(model,before)


def test_adam_learning_rate_qualification_changes_only_the_nine_matrix_rates():
    _,model,*_=native_setup();clone=deepcopy(model)
    opt=run.build_optimizer(model,'adamw',matrix_lr=1e-4,expected_widths=(24,16,8))
    named=dict(clone.group_named_parameters())
    reference=torch.optim.AdamW([
        {'params':[p for n,p in named.items() if n not in run.MATRIX_NAMES],'lr':3e-5},
        {'params':[p for n,p in named.items() if n in run.MATRIX_NAMES],'lr':1e-4}],
        betas=(.9,.99),eps=1e-8,weight_decay=0)
    set_gradients(model,1);set_gradients(clone,1)
    opt.step();reference.step();assert_snapshot(model,snapshot(clone))
    run.assert_optimizer_state(opt,[p for _,p in model.group_named_parameters()],1)
    assert sorted((len(g['params']),g['lr']) for g in opt.param_groups)==[(9,1e-4),(81,3e-5)]


@pytest.mark.parametrize('method', ['muon','normuon','shampoo'])
def test_exact_zero_matrix_gradient_advances_finite_state_without_fabricated_motion(method):
    _,model,*_=native_setup();cfg=run.optimizer_config(build(model,method))['matrix'];state={}
    for step in range(1,12):
        direction=run.matrix_direction(torch.zeros(3,2),state,method,cfg)
        assert torch.isfinite(direction).all() and not direction.any() and state['step']==step
