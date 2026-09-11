"""CPU contracts for isolated gradual cuts and the unchanged fitting update."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'convnext'))
import progressive_train as train


@pytest.fixture(autouse=True)
def cpu_state():
    rng,threads = train.screen.rng_state(),torch.get_num_threads()
    torch.manual_seed(301); torch.set_num_threads(1)
    yield
    train.screen.restore_rng(rng); torch.set_num_threads(threads)


class FakeFresh:
    def __init__(self,n=27000):
        self.source_ids = tuple(f'fresh-{i}' for i in range(n)); self.calls=[]
    def take(self,a,n):
        self.calls.append((a,n)); return [{'source_id':s} for s in self.source_ids[a:a+n]]
    def assert_unchanged(self): pass


def pools():
    return {'fit':[{'source_id':f'original-{i}'} for i in range(3000)],
        'calibration':[{'source_id':'calibration'}],'development':[{'source_id':'development'}]}


def test_composite_data_preserves_exact_cache_and_sealed_boundary_order():
    fresh=FakeFresh(); source=train.SourceStream(pools(),fresh)
    assert [c['source_id'] for c in source.take(2994,12)] == list(source.source_ids[2994:3006])
    assert fresh.calls==[(0,6)]
    window=train.source_window(source.source_ids)
    assert (window['start'],window['stop'])==(0,12000)
    assert window['source_ids']==[c['source_id'] for c in pools()['fit']]+list(fresh.source_ids[:9000])
    assert len(set(window['source_ids']))==12000


@pytest.mark.parametrize('damage',['original_duplicate','fresh_duplicate','heldout','missing_fit'])
def test_source_stream_rejects_leakage_or_repetition(damage):
    p=pools();f=FakeFresh()
    if damage=='original_duplicate':p['fit'][1]=p['fit'][0]
    elif damage=='fresh_duplicate':f.source_ids=('original-0',)+f.source_ids[1:]
    elif damage=='heldout':f.source_ids=('development',)+f.source_ids[1:]
    else:p['fit'].pop()
    with pytest.raises(ValueError):train.SourceStream(p,f)


def test_later_cut_never_replays_prior_students_fitting_prefix():
    ids=train.SourceStream(pools(),FakeFresh()).source_ids
    parent={'sources_seen':list(ids[:12000])};before=copy.deepcopy(parent)
    window=train.source_window(ids,parent)
    assert (window['start'],window['stop'])==(12000,24000)
    assert window['historical']==list(ids[:12000]) and window['source_ids']==list(ids[12000:24000])
    assert parent==before
    with pytest.raises(ValueError):train.source_window(ids,{'sources_seen':list(ids[:24000])})
    with pytest.raises(ValueError):train.source_window(ids,{'sources_seen':list(reversed(ids[:12000]))})
    with pytest.raises(ValueError):train.source_window(ids,updates=2000)


def schedule():
    return {'steps':[{'stage2_indices':list(range(a)),'stage3_indices':list(range(b))}
        for a,b in ((512,256),(384,256),(384,192),(256,192),(256,128))]}


def test_fixed_nested_schedule_and_recipe_are_validated_without_mutation():
    s=schedule();before=copy.deepcopy(s)
    assert train.validate_schedule(s)==s['steps'] and s==before
    recipe={'coefficients':train.COEFFICIENTS.copy(),'learning_rate':3e-5,
        'gradient_accumulation':12,'execution_batch_size':1,'trainable_stages':[2,3,4]}
    assert train.validate_recipe(recipe)==train.COEFFICIENTS
    for field,value in [('learning_rate',1e-4),('gradient_accumulation',3),('execution_batch_size',12),('coefficients',{'waveform':1.})]:
        bad=copy.deepcopy(recipe);bad[field]=value
        with pytest.raises(ValueError):train.validate_recipe(bad)
    for damage in ('shape','reintroduced','reorder'):
        bad=schedule()
        if damage=='shape':bad['steps'][1]['stage2_indices'].pop()
        elif damage=='reintroduced':bad['steps'][3]['stage2_indices'][-1]=500
        else:bad['steps'][1]['stage2_indices'].reverse()
        with pytest.raises(ValueError):train.validate_schedule(bad)


@pytest.mark.parametrize('damage',[None,'identity','manifest','plan','teacher','calibration','endpoint'])
def test_schedule_seal_binds_exact_training_only_calibration_and_original_teacher(damage):
    ids=[f'calibration-{i}' for i in range(72)]
    value={**schedule(),'version':'audiovae2_progressive_schedule_v1',
        'teacher_source_sha256':train.base.SOURCE_SHA256,'teacher_checkpoint_sha256':train.base.CHECKPOINT_SHA256,
        'manifest_sha256':'manifest','source_plan_sha256':'plan','calibration_source_ids':ids,
        'original_endpoint_reproduced':True,'teacher_frozen':True}
    if damage=='manifest':value['manifest_sha256']='foreign'
    elif damage=='plan':value['source_plan_sha256']='foreign'
    elif damage=='teacher':value['teacher_checkpoint_sha256']='foreign'
    elif damage=='calibration':value['calibration_source_ids']=list(reversed(ids))
    elif damage=='endpoint':value['original_endpoint_reproduced']=False
    value['identity_sha256']=train.screen.digest(value)
    if damage=='identity':value['steps'][1]['stage2_indices'][-1]=511
    kwargs={'manifest_sha':'manifest','source_plan_sha':'plan','calibration_ids':ids}
    if damage is None:assert train.authenticate_schedule(value,**kwargs)==value['steps']
    else:
        with pytest.raises(ValueError):train.authenticate_schedule(value,**kwargs)


class TinyGroup(nn.Module):
    def __init__(self):
        super().__init__(); self.gain=nn.Parameter(torch.tensor(.6))
        self.register_buffer('frozen_suffix',torch.tensor(1.))
    def group_from_input(self,x):return self.gain*x
    def suffix_from_group(self,h):return h.repeat_interleave(4,-1)*self.frozen_suffix
    def group_named_parameters(self):return [('gain',self.gain)]
    def group_state_dict(self):return {'gain':self.gain.detach()}


def features(z):
    h=z[:,:1].repeat_interleave(480,-1)
    return h*(.75+.2*torch.sin(.27*torch.arange(h.shape[-1])))[None,None]


def crops12():
    rows=[]
    for i in range(12):
        frames=3+i%3;context=i%2;z=torch.full((1,64,frames),[1e-5,.03,.2][i%3])
        rows.append({'source_id':f'case-{i}','start_frame':context,'context_start_frame':0,'context_frames':context,
            'valid_scored_samples':(frames-context)*1920-i*19-1,'latents':z,
            'teacher_audio':.9*features(z).repeat_interleave(4,-1)})
    return rows


def test_actual_update_is_bitwise_same_as_original_accumulation12_and_frozen_teacher(monkeypatch):
    base=train.base;real_batch=base.batch
    monkeypatch.setattr(base,'batch',lambda rows:real_batch(rows,device='cpu'))
    teacher_parameter=nn.Parameter(torch.tensor(.9),requires_grad=False)
    def target(teacher,z):
        h=features(z);return {'group_input':h,'group_output':teacher_parameter*h,
            'waveform':teacher_parameter*h.repeat_interleave(4,-1)}
    monkeypatch.setattr(base,'teacher_forward',target)
    common=base.ReconstructionV2(base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    a,b=TinyGroup(),TinyGroup(); crops=crops12()
    oa=torch.optim.AdamW(a.parameters(),lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    ob=torch.optim.AdamW(b.parameters(),lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    before=train.screen.rng_state()
    expected=base.training_update(a,None,crops,common,train.COEFFICIENTS,oa)
    actual,checks=train.perform_update(b,None,crops,common,ob)
    assert actual==expected
    assert train.replay.compare_tree(a.state_dict(),b.state_dict())['equal']
    assert train.replay.compare_tree(oa.state_dict(),ob.state_dict())['equal']
    assert train.replay.compare_tree(before,train.screen.rng_state())['equal']
    assert len(checks)==12 and all(c['bitwise_equal'] for c in checks)
    assert teacher_parameter.grad is None and float(teacher_parameter)==pytest.approx(.9)
    assert base.teacher_forward is target
    assert float(a.frozen_suffix)==float(b.frozen_suffix)==1.


def checkpoint_fixture(cut=1,updates=500):
    model=TinyGroup();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    if updates:
        model.gain.square().backward();optimizer.step()
        for state in optimizer.state.values():state['step'].fill_(updates)
    history=[f'source-{i}' for i in range((cut-1)*12000)]
    chosen=[f'source-{i}' for i in range(len(history),len(history)+12000)]
    identity={'cut_index':cut,'historical_sources':history,'source_ids':chosen,
        'source_ids_sha256':train.screen.digest(chosen),'schedule_sha256':'schedule'}
    return model,optimizer,identity,history+chosen[:updates*12]


@pytest.mark.parametrize('cut,updates',[(1,0),(1,500),(1,1000),(2,500)])
def test_checkpoint_preserves_fresh_per_cut_adam_counter_rng_and_full_ledger(tmp_path,cut,updates):
    model,opt,identity,seen=checkpoint_fixture(cut,updates)
    before=copy.deepcopy(opt.state_dict());rng=train.screen.rng_state();group=model.gain.detach().clone()
    path=tmp_path/'checkpoint.pt'
    receipt=train.save_state(path,model,opt,cut_index=cut,cut_updates=updates,selection={},identity=identity,
        seen=seen,samples=updates*123)
    saved=torch.load(path,weights_only=True)
    assert saved['cut_updates']==updates and saved['global_updates']==(cut-1)*1000+updates
    assert saved['sources_seen']==seen and saved['optimizer_restart_at_each_cut']
    assert train.replay.compare_tree(saved['optimizer'],before)['equal']
    assert train.replay.compare_tree(saved['rng'],rng)['equal']
    assert train.replay.compare_tree(opt.state_dict(),before)['equal']
    torch.testing.assert_close(model.gain,group,rtol=0,atol=0)
    assert receipt['checkpoint_sha256']==train.base.sha(path)
    with pytest.raises(FileExistsError):train.save_state(path,model,opt,cut_index=cut,cut_updates=updates,
        selection={},identity=identity,seen=seen,samples=updates*123)


@pytest.mark.parametrize('damage',['old_adam','reordered','missing','bad_rate','bad_review','bad_digest'])
def test_bad_counter_or_source_receipt_cannot_save(tmp_path,damage):
    model,opt,identity,seen=checkpoint_fixture(2,500);updates=500
    if damage=='old_adam':next(iter(opt.state.values()))['step'].fill_(1500)
    elif damage=='reordered':seen[:2]=reversed(seen[:2])
    elif damage=='missing':seen.pop()
    elif damage=='bad_rate':opt.param_groups[0]['lr']=1e-4
    elif damage=='bad_review':updates=750
    else:identity['source_ids_sha256']='bad'
    with pytest.raises(ValueError):train.save_state(tmp_path/'bad.pt',model,opt,cut_index=2,cut_updates=updates,
        selection={},identity=identity,seen=seen,samples=100)
    assert not (tmp_path/'bad.pt').exists()


def test_parent_checkpoint_must_be_completed_exact_prior_cut(tmp_path):
    model,opt,identity,seen=checkpoint_fixture(1,1000)
    path=tmp_path/'checkpoint-step1000.pt'
    receipt=train.save_state(path,model,opt,cut_index=1,cut_updates=1000,selection={},identity=identity,seen=seen,samples=100)
    receipt['frozen_state_preserved']=True
    path.with_suffix('.json').write_text(json.dumps(receipt))
    done={'last_checkpoint_sha256':receipt['checkpoint_sha256'],'status':'awaiting_review','cut_updates':1000,'original_files_preserved':True}
    (tmp_path/'completed.json').write_text(json.dumps(done))
    saved,checksum=train.authenticate_parent(path,'schedule',2,seen+[f'new-{i}' for i in range(12000)])
    assert saved['sources_seen']==seen and checksum==receipt['checkpoint_sha256']
    done['status']='failed';(tmp_path/'completed.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):train.authenticate_parent(path,'schedule',2,seen)
