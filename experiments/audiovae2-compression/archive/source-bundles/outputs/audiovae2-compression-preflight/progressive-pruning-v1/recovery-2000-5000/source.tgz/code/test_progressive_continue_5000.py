"""Step2000 same-cut ancestry, exact optimizer recovery and extended source order."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import progressive_continue_5000 as run
import test_progressive_continue as old_test
import test_progressive_train as fixture


@pytest.fixture(autouse=True)
def cpu_state():
    rng,threads=run.screen.rng_state(),torch.get_num_threads()
    torch.manual_seed(517);torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng);torch.set_num_threads(threads)


def parent_state():
    model,opt,p=old_test.parent_state()
    for value in opt.state.values():value['step'].fill_(2000)
    p.update({'format':run.previous.VERSION,'cut_updates':2000,'global_updates':2000,
        'optimizer':copy.deepcopy(opt.state_dict()),'sources_seen':[f'source-{i}' for i in range(24000)],
        'optimizer_reset':False,'original_cut_identity':{'original':True}})
    return model,opt,p


def test_complete_source_window_crosses_old_stream_end_without_reuse():
    _,_,p=parent_state();ids=[f'source-{i}' for i in range(60000)]
    w=run.source_window(p,ids)
    assert (w['start'],w['stop'])==(24000,60000)
    assert w['source_ids'][:6000]==ids[24000:30000]
    assert w['source_ids'][6000:]==ids[30000:60000]
    assert len(set(w['historical']+w['source_ids']))==60000
    seen=list(w['historical'])
    for step in range(2001,5001):seen.extend(ids[(step-1)*12:step*12])
    assert seen==ids


@pytest.mark.parametrize('damage',['first_cut_format','step','cut','batch','reset','history','short','repeat'])
def test_old_cut_format_or_reused_sources_rejected(damage):
    _,_,p=parent_state();ids=[f'source-{i}' for i in range(60000)]
    if damage=='first_cut_format':p['format']=run.prior.VERSION
    elif damage=='step':p['cut_updates']=1000
    elif damage=='cut':p['cut_index']=2
    elif damage=='batch':p['accumulation']=3
    elif damage=='reset':p['optimizer_reset']=True
    elif damage=='history':p['sources_seen'][:2]=reversed(p['sources_seen'][:2])
    elif damage=='short':ids.pop()
    else:ids[30000]=ids[0]
    with pytest.raises(ValueError):run.source_window(p,ids)


def test_first2001_update_matches_in_memory_parent_and_keeps_moments(monkeypatch):
    base=run.base;real_batch=base.batch
    monkeypatch.setattr(base,'batch',lambda rows:real_batch(rows,device='cpu'))
    teacher=torch.nn.Parameter(torch.tensor(.9),requires_grad=False)
    def target(_,z):
        h=fixture.features(z)
        return {'group_input':h,'group_output':teacher*h,'waveform':teacher*h.repeat_interleave(4,-1)}
    monkeypatch.setattr(base,'teacher_forward',target)
    a,oa,p=parent_state();saved=copy.deepcopy(p)
    b=old_test.Model();b.gain.data.fill_(3.)
    ob,check=run.restore_state(b,p)
    assert all(v['equal'] for v in check.values())
    common=base.ReconstructionV2(base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    expected,_=run.prior.perform_update(a,None,fixture.crops12(),common,oa)
    actual,_=run.prior.perform_update(b,None,fixture.crops12(),common,ob)
    assert actual==expected
    assert run.replay.compare_tree(a.state_dict(),b.state_dict())['equal']
    assert run.replay.compare_tree(oa.state_dict(),ob.state_dict())['equal']
    assert float(next(iter(ob.state.values()))['step'])==2001
    assert run.replay.compare_tree(p,saved)['equal']
    assert teacher.grad is None and base.teacher_forward is target


@pytest.mark.parametrize('step',[2500,3000,3500,4000,4500,5000])
def test_all_six_checkpoints_keep_global_counter_and_original_cut_identity(tmp_path,step):
    model,opt,p=parent_state()
    for v in opt.state.values():v['step'].fill_(step)
    chosen=[f'source-{i}' for i in range(24000,60000)]
    identity={'source_ids':chosen,'source_ids_sha256':run.screen.digest(chosen),
        'parent_checkpoint_sha256':run.PARENT_SHA256,'selection':p['selection']}
    seen=p['sources_seen']+chosen[:(step-2000)*12]
    before=copy.deepcopy(opt.state_dict());rng=run.screen.rng_state()
    path=tmp_path/'state.pt';receipt=run.save_state(path,model,opt,p,identity,seen,200,step)
    q=torch.load(path,weights_only=True)
    assert q['format']==run.VERSION and q['cut_updates']==step and q['global_updates']==step
    assert q['sources_seen']==seen and len(seen)==step*12
    assert q['original_cut_identity']==p['original_cut_identity'] and not q['optimizer_reset']
    assert run.replay.compare_tree(q['optimizer'],before)['equal']
    assert run.replay.compare_tree(q['rng'],rng)['equal']
    assert receipt['checkpoint_sha256']==run.base.sha(path)


def test_partial_segment_counter_cannot_replace_saved_optimizer_history(tmp_path):
    model,opt,p=parent_state();chosen=[f'source-{i}' for i in range(24000,60000)]
    identity={'source_ids':chosen,'source_ids_sha256':run.screen.digest(chosen),
        'parent_checkpoint_sha256':run.PARENT_SHA256,'selection':p['selection']}
    for v in opt.state.values():v['step'].fill_(500)
    with pytest.raises(ValueError):run.save_state(tmp_path/'bad.pt',model,opt,p,identity,p['sources_seen']+chosen[:6000],100,2500)
    assert not (tmp_path/'bad.pt').exists()


def auth_fixture(tmp_path,monkeypatch):
    folder=tmp_path/'parent';folder.mkdir();original=tmp_path/'original';original.mkdir()
    recipe={'coefficients':run.prior.COEFFICIENTS,'learning_rate':3e-5,'gradient_accumulation':12,
        'execution_batch_size':1,'trainable_stages':[2,3,4]}
    recipe_path=tmp_path/'recipe.json';recipe_path.write_text(json.dumps(recipe))
    schedule_path=tmp_path/'schedule.json';schedule_path.write_text('{}')
    ids=[f'source-{i}' for i in range(60000)]
    selection={'stage2_indices':list(range(384)),'stage3_indices':list(range(256))}
    origin={**recipe,'recipe_sha256':run.base.sha(recipe_path),'source_ids':ids[:12000],'cut_index':1}
    (original/'launch.json').write_text(json.dumps(origin))
    (original/'development-step0.json').write_text(json.dumps({'aggregate':{'mae':.1}}))
    protected=tmp_path/'protected.txt';protected.write_text('fixed')
    launch={**recipe,'source_plan_identity_sha256':'old-plan','source_interval':[12000,24000],
        'source_ids':ids[12000:24000],'source_ids_sha256':run.screen.digest(ids[12000:24000]),
        'parent_checkpoint_sha256':run.previous.PARENT_SHA256,'schedule_sha256':run.base.sha(schedule_path),
        'precision':'FP32 TF32 disabled','optimizer_reset':False,'width_changed':False,
        'protected':{str(protected):run.base.sha(protected)}}
    _,_,p=parent_state()
    p.update({'selection':selection,'original_cut_identity':origin,'identity':launch,
        'schedule_sha256':run.base.sha(schedule_path),'coefficients':run.prior.COEFFICIENTS,
        'teacher_source_sha256':run.base.SOURCE_SHA256,'teacher_checkpoint_sha256':run.base.CHECKPOINT_SHA256})
    path=folder/'checkpoint-step2000.pt';torch.save(p,path);checksum=run.base.sha(path)
    monkeypatch.setattr(run,'PARENT_SHA256',checksum)
    report={'aggregate':{'mae':.01}}
    receipt={'checkpoint_sha256':checksum,'cut_updates':2000,'cut_index':1,'global_updates':2000,
        'sources_seen':24000,'optimizer_reset':False,'optimizer_and_rng_saved':True,
        'frozen_state_preserved':True,'quality':report['aggregate']}
    completed={'status':'awaiting_review','step':2000,'cut_updates':2000,'sources_seen':24000,
        'last_checkpoint_sha256':checksum,'original_files_preserved':True,'frozen_state_preserved':True,
        'optimizer_reset':False,'final':report['aggregate']}
    for name,value in [('launch.json',launch),('checkpoint-step2000.json',receipt),('completed.json',completed),
                       ('development-step2000.json',report)]:
        (folder/name).write_text(json.dumps(value))
    (folder/'train.jsonl').write_text(json.dumps({'step':2000,'unique_sources':24000,'source_ids':ids[23988:24000],
        'audio_hours':p['scored_samples_this_cut']/48000/3600,'elapsed_seconds':20.})+'\n')
    args=SimpleNamespace(parent_checkpoint=path,original_dir=original,recipe=recipe_path,schedule=schedule_path)
    data=SimpleNamespace(source_ids=ids,fresh=SimpleNamespace(identity='old-plan'))
    return args,data,[None,selection],protected


@pytest.mark.parametrize('damage',[None,'receipt','completion','original','protected'])
def test_actual_parent_continuation_receipts_and_original_ancestry_are_checked(tmp_path,monkeypatch,damage):
    args,data,schedule,protected=auth_fixture(tmp_path,monkeypatch)
    if damage=='receipt':
        path=args.parent_checkpoint.with_suffix('.json');r=json.loads(path.read_text());r['sources_seen']=12000;path.write_text(json.dumps(r))
    elif damage=='completion':
        path=args.parent_checkpoint.parent/'completed.json';r=json.loads(path.read_text());r['status']='failed';path.write_text(json.dumps(r))
    elif damage=='original':
        path=args.original_dir/'launch.json';r=json.loads(path.read_text());r['learning_rate']=1e-4;path.write_text(json.dumps(r))
    elif damage=='protected':protected.write_text('changed')
    if damage is None:
        p,*_=run.authenticate(args,data,schedule)
        assert p['format']==run.previous.VERSION and p['cut_updates']==2000
    else:
        with pytest.raises(ValueError):run.authenticate(args,data,schedule)


def test_cumulative_exposure_and_elapsed_continue_from2000():
    value=run.progress_accounting({'scored_samples_this_cut':48000*3600*16},{'elapsed_seconds':2500.},48000*24,2.)
    assert value['audio_hours']==pytest.approx(16+24/3600)
    assert value['elapsed_seconds']==2502. and value['continuation_elapsed_seconds']==2.
