"""Independent full-output and causal-phase oracles for shared-support selection."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.utils import weight_norm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import downstream_selector_init as selector
from test_group_model import CausalConv1d,CausalTransposeConv1d,TinyDecoder,snapshot,assert_snapshot


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng,threads=torch.get_rng_state(),torch.get_num_threads()
    torch.manual_seed(932);torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng);torch.set_num_threads(threads)


def module(native=False,cin=3,cout=4):
    raw=(CausalTransposeConv1d(cin,cout,10,stride=5,padding=3,output_padding=1)
         if native else CausalConv1d(cin,cout,1))
    return weight_norm(raw).double()


def explicit_contributions(m,x):
    w=selector.group.effective_weight(m).detach()
    terms=[]
    for c in range(m.in_channels):
        if isinstance(m,torch.nn.ConvTranspose1d):
            value=F.conv_transpose1d(x[:,c:c+1],w[c:c+1],None,stride=5)[...,:x.shape[-1]*5]
        else:value=F.conv1d(x[:,c:c+1],w[:,c:c+1],None)
        terms.append(value)
    return torch.stack(terms,dim=1) # batch, input channel, full output channel, time


def explicit_gram(m,x,q):
    terms=explicit_contributions(m,x)
    gram=torch.einsum('bcot,bdot,bt->cd',terms,terms,q[:,0].double())
    energy=(terms.sum(1).square()*q).sum()
    return gram,energy


def test_pointwise_full_output_uncentered_gram_ignores_bias_once():
    m=module();x=torch.randn(2,3,13,dtype=torch.float64)+3
    q=torch.full((2,1,13),40);q[0,0,:2]=0;q[1,0,-1]=7
    actual=selector.accumulate_consumer(None,m,x,q,chunk_rows=4)
    expected,energy=explicit_gram(m,x,q)
    torch.testing.assert_close(actual['gram'],expected,rtol=2e-14,atol=2e-12)
    torch.testing.assert_close(actual['energy'],energy,rtol=2e-14,atol=2e-12)
    assert actual['output_channels']==4 and actual['weighted_samples']==int(q.sum())
    with torch.no_grad():m.bias.add_(123)
    changed=selector.accumulate_consumer(None,m,x,q)
    torch.testing.assert_close(changed['gram'],actual['gram'],rtol=2e-14,atol=2e-12)
    centered=selector.accumulate_consumer(None,m,x-x.mean(-1,keepdim=True),q)
    assert not torch.allclose(centered['gram'],actual['gram'])


@pytest.mark.parametrize('frames',[1,3,11])
def test_native_five_phase_gram_matches_brute_force_each_channel(frames):
    m=module(True);x=torch.randn(2,3,frames,dtype=torch.float64)
    q=torch.randint(0,9,(2,1,frames*5));q[0,0,0]=8
    actual=selector.accumulate_consumer(None,m,x,q,chunk_rows=2)
    expected,energy=explicit_gram(m,x,q)
    torch.testing.assert_close(actual['gram'],expected,rtol=3e-14,atol=2e-12)
    torch.testing.assert_close(actual['energy'],energy,rtol=3e-14,atol=2e-12)
    assert actual['weighted_samples']==int(q.sum())
    assert actual['valid_rows']==int((q>0).sum())


def test_native_previous_context_is_kept_at_first_scored_frame_and_zero_only_at_start():
    m=module(True,1,1)
    w=torch.zeros(1,1,10,dtype=torch.float64);w[:,:,5:]=torch.arange(1,6,dtype=torch.float64)
    selector.group.assign_effective_weight(m,w,torch.tensor([17.],dtype=torch.float64))
    x=torch.tensor([[[7.,11.,13.]]],dtype=torch.float64)
    q=torch.zeros(1,1,15,dtype=torch.long);q[...,5:10]=8;q[...,9]=3
    s=selector.accumulate_consumer(None,m,x,q,chunk_rows=1)
    expected=sum(7**2*p*p*(3 if p==5 else 8) for p in range(1,6))
    assert float(s['energy'])==pytest.approx(expected,rel=1e-14)
    startup=q.clone();startup.zero_();startup[...,:5]=8
    assert float(selector.accumulate_consumer(None,m,x,startup)['energy'])==0


def test_chunked_accumulation_keeps_phase_and_cross_source_sums():
    m=module(True);x=torch.randn(2,3,17,dtype=torch.float64)
    q=torch.randint(0,9,(2,1,85))
    whole=selector.accumulate_consumer(None,m,x,q,chunk_rows=17)
    split=None
    for i in range(2):split=selector.accumulate_consumer(split,m,x[i:i+1],q[i:i+1],chunk_rows=3)
    torch.testing.assert_close(split['gram'],whole['gram'],rtol=3e-14,atol=2e-12)
    torch.testing.assert_close(split['energy'],whole['energy'],rtol=3e-14,atol=2e-12)
    assert split['weighted_samples']==whole['weighted_samples'] and split['observations']==2


@pytest.mark.parametrize('bad',['negative','too_large','nan_weight','nan_scored_input'])
def test_malformed_scored_weights_and_values_fail_closed(bad):
    m=module();x=torch.ones(1,3,3,dtype=torch.float64);q=torch.full((1,1,3),40.)
    if bad=='negative':q[...,0]=-1
    elif bad=='too_large':q[...,0]=41
    elif bad=='nan_weight':q[...,0]=torch.nan
    else:x[...,0]=torch.nan
    with pytest.raises(ValueError):selector.accumulate_consumer(None,m,x,q)


def test_unscored_nan_excluded_and_changed_teacher_weights_rejected():
    m=module();x=torch.ones(1,3,3,dtype=torch.float64);q=torch.full((1,1,3),40)
    x[...,0]=torch.nan;q[...,0]=0
    s=selector.accumulate_consumer(None,m,x,q)
    assert torch.isfinite(s['gram']).all()
    with torch.no_grad():m.weight_g.add_(.1)
    with pytest.raises(ValueError,match='weights changed'):selector.accumulate_consumer(s,m,x,q)


def four_stats():
    result={}
    for index,name in enumerate(selector.NAMES):
        m=module(index==3,3,4 if index==3 else 3)
        x=torch.randn(1,3,9,dtype=torch.float64)
        q=torch.full((1,1,9*(5 if index==3 else 1)),8 if index==3 else 40)
        result[name]=selector.accumulate_consumer(None,m,x,q)
    return result


def test_normalization_uses_four_fixed_complete_output_energies_equally():
    stats=four_stats();k,report=selector.normalize_grams(stats)
    expected=sum(s['gram']/s['energy'] for s in stats.values())/4
    torch.testing.assert_close(k,expected,rtol=2e-14,atol=2e-14)
    assert float(k.sum())==pytest.approx(1.,abs=1e-13)
    assert k.dtype==torch.float64 and k.device.type=='cpu'
    assert all(r['energy_identity_abs_error']<=r['arithmetic_bound'] for r in report.values())


@pytest.mark.parametrize('bad',['zero','nonfinite','asymmetric','wrong_energy','missing_site'])
def test_normalizer_and_gram_identity_gates(bad):
    stats=four_stats();first=stats[selector.NAMES[0]]
    if bad=='zero':first['energy'].zero_()
    elif bad=='nonfinite':first['gram'][0,0]=torch.nan
    elif bad=='asymmetric':first['gram'][0,1]+=1
    elif bad=='wrong_energy':first['energy'].mul_(1.01)
    else:stats.pop(selector.NAMES[-1])
    with pytest.raises(ValueError):selector.normalize_grams(stats)


def test_greedy_exact_ties_and_signed_cross_terms_are_not_absolute_values():
    k=torch.tensor([[1.,-.9,0.],[-.9,1.,0.],[0.,0.,1.]],dtype=torch.float64)
    result=selector.greedy_remove(k,1)
    assert result['removed_indices']==[0,1] and result['selected_indices']==[2]
    assert result['final_objective']==pytest.approx(.2)
    assert result['trace'][1]['increment']==pytest.approx(-.8)
    assert result['trace'][0]['candidate_scores_before_removal']==[1.,1.,1.]


def test_greedy_choices_equal_brute_force_increment_and_saved_matrix_replays(tmp_path):
    a=torch.randn(7,13,dtype=torch.float64);k=a@a.T
    k=(k+k.T)/2;result=selector.greedy_remove(k,3);removed=[]
    for row in result['trace']:
        scores={c:float(k[c,c]+2*k[c,removed].sum()) for c in range(7) if c not in removed}
        expected=min(scores,key=lambda c:(scores[c],c))
        assert row['removed_channel']==expected
        removed.append(expected)
        assert row['objective']==float(k[removed][:,removed].sum())
        for c,value in scores.items():assert row['candidate_scores_before_removal'][c]==pytest.approx(value,abs=1e-12)
    path=tmp_path/'K.pt';torch.save(k,path)
    rng=torch.get_rng_state().clone()
    assert selector.greedy_remove(torch.load(path,weights_only=True),3)==result
    assert torch.equal(rng,torch.get_rng_state())


def test_teacher_capture_retains_all_outputs_and_one_pass_per_crop(monkeypatch):
    teacher=TinyDecoder().double().eval().requires_grad_(False);before=snapshot(teacher)
    wrapper=SimpleNamespace(model=SimpleNamespace(decoder=teacher));calls=[]
    crops=[{'z':torch.randn(1,8,2,dtype=torch.float64)} for _ in range(2)]
    def batch(crops):
        z=crops[0]['z'];valid=torch.ones(1,1,z.shape[-1]*1920,dtype=torch.bool)
        valid[...,:41]=False;valid[...,-3:]=False
        return z,None,valid,None
    def forward(t,z):calls.append(1);return selector.group.teacher_trace(t.model.decoder,z)
    monkeypatch.setattr(selector.base,'batch',batch);monkeypatch.setattr(selector.base,'teacher_forward',forward)
    k,report=selector.collect_statistics(wrapper,crops)
    assert len(calls)==2 and k.shape==(32,32)
    assert [report[n]['output_channels'] for n in selector.NAMES]==[32,32,32,16]
    assert all(report[n]['observations']==2 for n in selector.NAMES)
    assert_snapshot(teacher,before)
    assert all(not m._forward_hooks for m in teacher.modules())


def test_new_support_B_repair_and_portable_artifact_recreate_full_candidate():
    teacher=TinyDecoder().double().eval().requires_grad_(False);teacher_before=snapshot(teacher)
    old={'stage2_indices':[i for i in range(32) if i%4!=0],'stage3_indices':list(range(16))}
    new={'stage2_indices':[i for i in range(32) if i%4!=1],'stage3_indices':list(range(16))}
    untouched=selector.progressive.initialize_from_teacher(teacher,old).eval();old_state=snapshot(untouched)
    candidate=selector.progressive.initialize_from_teacher(teacher,new).eval()
    pristine=selector.control.state_hash(candidate.decoder);versions=selector.audit._versions(candidate.decoder)
    sites=selector.audit.four_sites(teacher,new)
    crops=[{'z':torch.randn(1,8,2,dtype=torch.float64)*.1} for _ in range(2)]
    inputs=[]
    def capture(decoder,z):
        with torch.no_grad(),selector.audit.capture_sites(decoder,sites) as values:decoder(z)
        return values
    def observe(model,original,crop,site):
        t,s=capture(original,crop['z'])[site.name],capture(model.decoder,crop['z'])[site.name]
        target=(selector.ordinary.residual_target(t['ru_output'],s['skip_input'],site.kept_outputs)
                if site.stride==1 else t['output'])
        inputs.append((site.name,selector.control.state_hash(model.decoder)))
        return {'input':s['input'],'target':target,'current_output':s['output'],
                'weights':torch.full((1,1,target.shape[-1]),40 if site.stride==1 else 8)}
    reports=selector.ordinary.sequential_fit(candidate,teacher,crops,sites,observe)
    assert all(r['last_source_native_fold_parity']['allclose_existing'] for r in reports)
    assert len({h for name,h in inputs})==4
    selector.ordinary._assert_only_sites_changed(versions,candidate,sites)
    artifact=selector.export_artifact(candidate,teacher,[str(i) for i in range(72)],
        base_selected_state_sha256=pristine,statistics_sha256='a'*64,selection_receipt_sha256='b'*64)
    restored=selector.progressive.initialize_from_teacher(teacher,artifact['selection']).eval()
    assert selector.control.state_hash(restored.decoder)==artifact['base_selected_state_sha256']
    assert selector.control.state_hash(untouched.decoder)!=artifact['base_selected_state_sha256']
    for path,state in artifact['operators'].items():restored.decoder.get_submodule(path).load_state_dict(state)
    assert selector.control.state_hash(restored.decoder)==artifact['candidate_state_sha256']
    assert set(artifact['operators'])==set(selector.PATHS) and artifact['automatic_promotion'] is False
    assert_snapshot(teacher,teacher_before);assert_snapshot(untouched,old_state)


def test_cache_gate_rejects_missing_or_outside_tolerance_but_retains_exactness_distinction():
    for records in ([],[{'allclose_original_tolerance':False}]):
        with pytest.raises(RuntimeError):selector._cache_gate(records)
    selector._cache_gate([{'allclose_original_tolerance':True,'bitwise_equal':False}])
