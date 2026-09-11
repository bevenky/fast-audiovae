"""Matched-source accumulation policies preserve the objective and starting state."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import compare_accumulation_v1 as experiment


class TinyGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.tensor(.6))
        self.register_buffer('frozen_suffix_scale',torch.tensor(1.))

    def group_from_input(self,x):return self.gain*x
    def suffix_from_group(self,h):return h.repeat_interleave(4,dim=-1)*self.frozen_suffix_scale
    def group_named_parameters(self):return [('gain',self.gain)]
    def group_state_dict(self):return {'gain':self.gain.detach()}
    @torch.no_grad()
    def load_group_state_dict(self,state):self.gain.copy_(state['gain'])


def test_schedules_align_all26_source_checkpoints_but_preserve_different_adam_steps():
    small=experiment.comparison_schedule(3)
    large=experiment.comparison_schedule(12)
    assert [r['sources_consumed'] for r in small]==list(range(0,1501,60))
    assert [r['sources_consumed'] for r in large]==list(range(0,1501,60))
    assert small[0]==large[0]=={'sources_consumed':0,'optimizer_step':4500,'updates':0}
    assert small[-1]=={'sources_consumed':1500,'optimizer_step':5000,'updates':500}
    assert large[-1]=={'sources_consumed':1500,'optimizer_step':4625,'updates':125}
    assert all(r['updates']*3==r['sources_consumed'] for r in small)
    assert all(r['updates']*12==r['sources_consumed'] for r in large)


@pytest.mark.parametrize('accumulation',[0,1,4,6,24,True])
def test_unapproved_accumulations_are_rejected(accumulation):
    with pytest.raises((ValueError,TypeError)):
        experiment.comparison_schedule(accumulation)


def payload_at4500():
    model=TinyGroup()
    opt=torch.optim.AdamW([model.gain],lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    (model.gain-.7).square().backward();opt.step()
    for state in opt.state.values():state['step'].fill_(4500)
    payload={'group':{k:v.clone() for k,v in model.group_state_dict().items()},'step':4500,
             'optimizer':copy.deepcopy(opt.state_dict()),'rng':experiment.screen.rng_state(),
             'identity':{'learning_rate':3e-5},'fit_cursor':13500,'fresh_cursor':10500}
    return model,payload


def test_each_arm_restarts_from_same_group_moments_counter_and_rng():
    first,payload=payload_at4500()
    frozen=first.frozen_suffix_scale.clone()
    initial_payload=copy.deepcopy(payload)
    opt_a,checks_a=experiment.restart_arm(first,payload)
    sample_a=torch.rand(8)
    (first.gain-.95).square().backward();opt_a.step()
    second=TinyGroup()
    opt_b,checks_b=experiment.restart_arm(second,payload)
    sample_b=torch.rand(8)
    assert all(row['equal'] for row in checks_a.values())
    assert all(row['equal'] for row in checks_b.values())
    torch.testing.assert_close(sample_a,sample_b,rtol=0,atol=0)
    torch.testing.assert_close(second.gain,payload['group']['gain'],rtol=0,atol=0)
    assert experiment.replay.compare_tree(opt_b.state_dict(),payload['optimizer'])['equal']
    assert experiment.replay.compare_tree(payload,initial_payload)['equal']
    torch.testing.assert_close(first.frozen_suffix_scale,frozen,rtol=0,atol=0)
    assert all(float(s['step'])==4500 for s in opt_b.state.values())


def features(z):
    h=z[:,:1].repeat_interleave(480,dim=-1)
    phase=torch.arange(h.shape[-1],dtype=h.dtype,device=h.device)
    return h*(.75+.2*torch.sin(.27*phase))[None,None]


def crops12():
    rows=[]
    for i in range(12):
        frames=2+i%4;context=i%2
        valid=(frames-context)*1920-(i*19+1)
        z=torch.full((1,64,frames),[1e-5,.03,.2][i%3])
        rows.append({'source_id':f'case-{i}','start_frame':context,'context_start_frame':0,
                     'context_frames':context,'valid_scored_samples':valid,'latents':z,
                     'teacher_audio':.9*features(z).repeat_interleave(4,dim=-1)})
    return rows


@pytest.mark.parametrize('accumulation',[3,12])
def test_singleton_accumulation_matches_pooled_variable_length_objective_without_extra_division(monkeypatch,accumulation):
    base=experiment.base
    spectral=base.ReconstructionV2(base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    crops=crops12()[:accumulation]
    real_batch=base.batch
    observed=[]
    monkeypatch.setattr(base,'batch',lambda rows:real_batch(rows,device='cpu'))
    def teacher_forward(teacher,z):
        observed.append(z.shape[0]);h=features(z)
        return {'group_input':h,'group_output':.9*h,'waveform':.9*h.repeat_interleave(4,dim=-1)}
    monkeypatch.setattr(base,'teacher_forward',teacher_forward)
    model=TinyGroup();reference=copy.deepcopy(model)
    coefficients={'waveform':1.,'mel':.0007,'feature':.03}
    z,t,valid,spans=real_batch(crops,device='cpu')
    h=reference.group_from_input(features(z));ht=.9*features(z)
    branches=base.losses(reference.suffix_from_group(h),t,h,ht,valid,spans,spectral)
    total=sum(coefficients[k]*v for k,v in branches.items());total.backward()
    expected_grad=reference.gain.grad.clone()
    # A zero learning-rate SGD object preserves the computed actual gradient.
    optimizer=torch.optim.SGD([model.gain],lr=0.)
    values,direction=experiment.run_update(model,None,crops,spectral,coefficients,optimizer)
    assert observed==[1]*accumulation
    torch.testing.assert_close(model.gain.grad,expected_grad,rtol=2e-5,atol=1e-8)
    for key in coefficients:
        assert values[key]==pytest.approx(float(branches[key].detach()),rel=2e-5,abs=1e-8)
    assert values['total']==pytest.approx(float(total.detach()),rel=2e-5,abs=1e-8)
    # Independent waveform oracle weights real samples, not equal-length examples.
    absolute=sum(float((crop['teacher_audio']*.6/.9-crop['teacher_audio'])[...,crop['context_frames']*1920:crop['context_frames']*1920+crop['valid_scored_samples']].abs().sum()) for crop in crops)
    assert values['waveform']==pytest.approx(absolute/sum(c['valid_scored_samples'] for c in crops),rel=1e-6)
    torch.testing.assert_close(model.frozen_suffix_scale,torch.tensor(1.),rtol=0,atol=0)


def test_both_policies_use_exact_same1500_ids_without_repetitions_or_cursor_shortcuts():
    fresh=[f'fresh-{i}' for i in range(27000)]
    paths=[]
    for accumulation in (3,12):
        seen=[]
        for consumed in range(0,1500,accumulation):
            (start,stop),ids=experiment.source_batch(fresh,consumed,accumulation)
            assert start==10500+consumed and stop-start==accumulation
            assert ids==fresh[start:stop]
            seen.extend(ids)
        assert len(seen)==len(set(seen))==1500
        paths.append(seen)
    assert paths[0]==paths[1]==fresh[10500:12000]
    for consumed,accumulation in ((1500,3),(-3,3),(3,12),(1497,12)):
        with pytest.raises(ValueError):experiment.source_batch(fresh,consumed,accumulation)


def test_batch12_artifact_has_real_optimizer_counter_and_independent_source_ledger(tmp_path):
    model,payload=payload_at4500()
    payload['sources_seen']=[f'historical-{i}' for i in range(13500)]
    optimizer,_=experiment.restart_arm(model,payload)
    for state in optimizer.state.values():state['step'].fill_(4625)
    ids=[f'fresh-{i}' for i in range(10500,12000)]
    path=tmp_path/'final.pt'
    receipt=experiment.save_arm(path,model,optimizer,payload,12,ids,1500*1920,{'policy':'matched exposure'})
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['format']==experiment.VERSION
    assert saved['optimizer_step']==receipt['optimizer_step']==4625
    assert saved['updates']==125 and saved['sources_consumed']==1500
    assert len(saved['historical_sources_seen'])==13500 and saved['additional_sources_seen']==ids
    assert len(set(saved['historical_sources_seen']+saved['additional_sources_seen']))==15000
    assert saved['automatic_promotion'] is False
    with pytest.raises(FileExistsError):experiment.save_arm(path,model,optimizer,payload,12,ids,1,{})
    for state in optimizer.state.values():state['step'].fill_(4624)
    with pytest.raises(RuntimeError,match='counters'):
        experiment.save_arm(tmp_path/'wrong.pt',model,optimizer,payload,12,ids,1,{})
    assert not (tmp_path/'wrong.pt').exists()


def test_reference_metric_tolerance_never_relaxes_integer_quality_counts():
    expected={'aggregate':{'mae':.02,'quiet_windows':2544},'near':{'passed':4},'optional':None}
    assert experiment.numeric_reference_check(expected,expected)['passed']
    assert experiment.numeric_reference_check({**expected,'aggregate':{'mae':.0200001,'quiet_windows':2544}},expected)['passed']
    assert not experiment.numeric_reference_check({**expected,'near':{'passed':3}},expected)['passed']
    assert not experiment.numeric_reference_check({**expected,'near':{'passed':4.}},expected)['passed']
    assert not experiment.numeric_reference_check({**expected,'optional':0.},expected)['passed']
    for invalid in (float('nan'),float('inf')):
        assert not experiment.numeric_reference_check({**expected,'aggregate':{'mae':invalid,'quiet_windows':2544}},expected)['passed']


def test_trajectory_summary_scores_all_matched_points_not_the_best_checkpoint():
    import math
    snapshots=[]
    for i in range(26):
        rows=[{'source_id':sid,'active_rms_gain':.5 if i%2==0 else 2.,'mae':.01*(i+1),'active_cosine':.9}
              for sid in experiment.replay.CASE_IDS]
        snapshots.append({'sources_consumed':i*60,'cases':rows})
    summary=experiment.trajectory_summary(snapshots)
    for row in summary.values():
        assert row['observations']==26
        assert row['mean_absolute_log_gain']==pytest.approx(math.log(2.))
        assert row['gain_population_std']==.75
        assert row['gain_direction_reversals']==24
        assert row['mean_waveform_mae']==pytest.approx(.135)


def test_four_replicas_of_three_sources_preserve_one_update_gradient_scale(monkeypatch):
    base=experiment.base
    spectral=base.ReconstructionV2(base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    original_batch=base.batch
    monkeypatch.setattr(base,'batch',lambda rows:original_batch(rows,device='cpu'))
    monkeypatch.setattr(base,'teacher_forward',lambda teacher,z:{'group_input':features(z),'group_output':.9*features(z)})
    three=crops12()[:3]
    twelve=[{**copy.deepcopy(three[i%3]),'source_id':f'mock-copy-{i}'} for i in range(12)]
    coefficients={'waveform':1.,'mel':.0007,'feature':.03}
    models=[TinyGroup(),TinyGroup()]
    values=[]
    for model,rows in zip(models,(three,twelve)):
        optimizer=torch.optim.SGD([model.gain],lr=0.)
        measured,_=experiment.run_update(model,None,rows,spectral,coefficients,optimizer)
        values.append(measured)
    torch.testing.assert_close(models[0].gain.grad,models[1].gain.grad,rtol=2e-6,atol=1e-8)
    for key in ('waveform','mel','feature','total'):
        assert values[0][key]==pytest.approx(values[1][key],rel=2e-6,abs=1e-8)
