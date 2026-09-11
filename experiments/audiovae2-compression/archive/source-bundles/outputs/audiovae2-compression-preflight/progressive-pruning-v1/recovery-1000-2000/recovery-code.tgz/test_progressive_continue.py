"""Same-cut recovery keeps its AdamW history and consumes only new sources."""
import copy
from pathlib import Path
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import progressive_continue as run
import test_progressive_train as fixture


@pytest.fixture(autouse=True)
def cpu_state():
    rng,threads=run.screen.rng_state(),torch.get_num_threads()
    torch.manual_seed(816);torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng);torch.set_num_threads(threads)


class Model(fixture.TinyGroup):
    @torch.no_grad()
    def load_group_state_dict(self,state):self.gain.copy_(state['gain'])


def parent_state():
    model=Model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)
    model.gain.square().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():state['step'].fill_(1000)
    parent={'group':copy.deepcopy(model.group_state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
        'rng':run.screen.rng_state(),'cut_index':1,'cut_updates':1000,'global_updates':1000,'accumulation':12,
        'sources_seen':[f'source-{i}' for i in range(12000)],'selection':{},'schedule_sha256':'schedule',
        'scored_samples_this_cut':100,'identity':{}}
    return model,optimizer,parent


def test_source_window_is_next12000_with_complete_prior_prefix():
    _,_,p=parent_state();ids=[f'source-{i}' for i in range(30000)];before=copy.deepcopy(p['sources_seen'])
    window=run.source_window(p,ids)
    assert window['start']==12000 and window['stop']==24000
    assert window['source_ids']==ids[12000:24000] and window['historical']==ids[:12000]
    assert p['sources_seen']==before


def test_cumulative_dashboard_time_and_exposure_do_not_reset_at1001():
    parent={'scored_samples_this_cut':48000*3600*8}
    previous={'elapsed_seconds':1800.}
    first=run.progress_accounting(parent,previous,48000*24,2.)
    assert first['audio_hours']==pytest.approx(8+24/3600)
    assert first['elapsed_seconds']==1802.
    assert first['continuation_audio_hours']==pytest.approx(24/3600)
    assert first['continuation_elapsed_seconds']==2.


@pytest.mark.parametrize('damage',['step','width_cut','batch','missing_history','reordered_history','short_data','repeat_data'])
def test_source_window_rejects_wrong_lineage_and_repetition(damage):
    _,_,p=parent_state();ids=[f'source-{i}' for i in range(30000)]
    if damage=='step':p['cut_updates']=0
    elif damage=='width_cut':p['cut_index']=2
    elif damage=='batch':p['accumulation']=3
    elif damage=='missing_history':p['sources_seen'].pop()
    elif damage=='reordered_history':p['sources_seen'][:2]=reversed(p['sources_seen'][:2])
    elif damage=='short_data':ids=ids[:23999]
    else:ids[12000]=ids[0]
    with pytest.raises(ValueError):run.source_window(p,ids)


def test_first_resumed_update_matches_in_memory_continuation_bitwise(monkeypatch):
    base=run.base;real_batch=base.batch
    monkeypatch.setattr(base,'batch',lambda rows:real_batch(rows,device='cpu'))
    target_parameter=torch.nn.Parameter(torch.tensor(.9),requires_grad=False)
    def target(teacher,z):
        h=fixture.features(z);return {'group_input':h,'group_output':target_parameter*h,
            'waveform':target_parameter*h.repeat_interleave(4,-1)}
    monkeypatch.setattr(base,'teacher_forward',target)
    a,oa,p=parent_state();original=copy.deepcopy(p)
    b=Model();b.gain.data.fill_(5.)
    ob,check=run.restore_state(b,p)
    assert all(v['equal'] for v in check.values())
    assert len(ob.state)==1 and float(next(iter(ob.state.values()))['step'])==1000
    spectral=base.ReconstructionV2(base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    expected,_=run.prior.perform_update(a,None,fixture.crops12(),spectral,oa)
    actual,_=run.prior.perform_update(b,None,fixture.crops12(),spectral,ob)
    assert expected==actual
    assert run.replay.compare_tree(a.state_dict(),b.state_dict())['equal']
    assert run.replay.compare_tree(oa.state_dict(),ob.state_dict())['equal']
    assert float(next(iter(ob.state.values()))['step'])==1001
    assert run.replay.compare_tree(p,original)['equal']
    assert target_parameter.grad is None and base.teacher_forward is target


def test_bad_moments_rejected_before_student_mutation():
    _,_,p=parent_state();model=Model();before=model.gain.detach().clone()
    p['optimizer']['state'][0]['step'].fill_(0)
    with pytest.raises(ValueError):run.restore_state(model,p)
    torch.testing.assert_close(before,model.gain,rtol=0,atol=0)


@pytest.mark.parametrize('step',[1500,2000])
def test_snapshot_keeps_continued_counter_complete_ledger_and_rng(tmp_path,step):
    model,opt,parent=parent_state()
    for value in opt.state.values():value['step'].fill_(step)
    chosen=[f'source-{i}' for i in range(12000,24000)]
    identity={'source_ids':chosen,'source_ids_sha256':run.screen.digest(chosen),
        'parent_checkpoint_sha256':run.PARENT_SHA256,'selection':parent['selection']}
    seen=parent['sources_seen']+chosen[:(step-1000)*12]
    before=copy.deepcopy(opt.state_dict());rng=run.screen.rng_state()
    path=tmp_path/'state.pt';receipt=run.save_state(path,model,opt,parent,identity,seen,200,step)
    payload=torch.load(path,weights_only=True)
    assert payload['cut_updates']==step and payload['global_updates']==step
    assert payload['sources_seen']==seen and payload['scored_samples_this_cut']==300
    assert payload['optimizer_reset'] is False and payload['selection']==parent['selection']
    assert run.replay.compare_tree(payload['optimizer'],before)['equal']
    assert run.replay.compare_tree(payload['rng'],rng)['equal']
    assert receipt['checkpoint_sha256']==run.base.sha(path)
    with pytest.raises(FileExistsError):run.save_state(path,model,opt,parent,identity,seen,200,step)


@pytest.mark.parametrize('damage',['reset_moments','reorder','width','parent','digest','off_review'])
def test_bad_continuation_receipt_cannot_save(tmp_path,damage):
    model,opt,parent=parent_state();chosen=[f'source-{i}' for i in range(12000,24000)]
    identity={'source_ids':chosen,'source_ids_sha256':run.screen.digest(chosen),
        'parent_checkpoint_sha256':run.PARENT_SHA256,'selection':parent['selection']}
    seen=parent['sources_seen']+chosen[:6000];step=1500
    for value in opt.state.values():value['step'].fill_(1500)
    if damage=='reset_moments':next(iter(opt.state.values()))['step'].fill_(500)
    elif damage=='reorder':seen[-2:]=reversed(seen[-2:])
    elif damage=='width':identity['selection']={'changed':True}
    elif damage=='parent':identity['parent_checkpoint_sha256']='bad'
    elif damage=='digest':identity['source_ids_sha256']='bad'
    else:step=1600
    with pytest.raises(ValueError):run.save_state(tmp_path/'bad.pt',model,opt,parent,identity,seen,100,step)
    assert not (tmp_path/'bad.pt').exists()
