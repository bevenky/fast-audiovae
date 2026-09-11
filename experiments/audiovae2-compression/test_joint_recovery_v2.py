"""Focused CPU checks for reviewed, source-unique joint continuation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'convnext'))
import joint_recovery_v2 as joint
import test_joint_recovery_v1 as old_tests
from test_group_model import gm, selections, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_state():
    rng,threads = joint.screen.rng_state(),torch.get_num_threads()
    torch.manual_seed(9162); torch.set_num_threads(1)
    yield
    joint.screen.restore_rng(rng); torch.set_num_threads(threads)


def ledger(step=5625):
    original = [f'original-{i}' for i in range(3000)]
    fresh = [f'fresh-{i}' for i in range(76500)]
    cursor = 24000+(step-5625)*12
    parent = {'format':joint.previous.VERSION if step==5625 else joint.VERSION,
        'optimizer_step':step,'accumulation':12,'fresh_cursor':cursor,'total_source_count':3000+cursor,
        'historical_sources_seen':original+fresh[:12000],
        'additional_sources_seen':fresh[12000:cursor],
        'identity':{'root_checkpoint_sha256':joint.ROOT_SHA256}}
    return parent,original,fresh


@pytest.mark.parametrize('step,target,cursor,stop',[
    (5625,6625,24000,36000),(6625,7625,36000,48000),
    (7625,8625,48000,60000),(8625,9625,60000,72000),(9625,10000,72000,76500)])
def test_exact_segments_preserve_independent_optimizer_and_source_counters(step,target,cursor,stop):
    parent,original,fresh = ledger(step); before = copy.deepcopy(parent)
    w = joint.segment_window(parent,original,fresh)
    assert w['start_step']==step and w['target_step']==target
    assert w['start_cursor']==cursor and w['stop_cursor']==stop
    assert w['source_ids']==fresh[cursor:stop]
    assert w['historical']==original+fresh[:cursor]
    assert len(set(w['historical']+w['source_ids']))==3000+stop
    assert parent==before


@pytest.mark.parametrize('damage',['step','format','batch','history','missing','repeat','cursor','root'])
def test_segment_rejects_changed_lineage_or_reused_sources(damage):
    parent,original,fresh = ledger(6625)
    if damage=='step': parent['optimizer_step']+=1
    elif damage=='format': parent['format']=joint.previous.VERSION
    elif damage=='batch': parent['accumulation']=3
    elif damage=='history': parent['additional_sources_seen'][:2]=reversed(parent['additional_sources_seen'][:2])
    elif damage=='missing': fresh=fresh[:47999]
    elif damage=='repeat': fresh[36000]=original[0]
    elif damage=='cursor': parent['fresh_cursor']+=12
    elif damage=='root': parent['identity']['root_checkpoint_sha256']='other'
    with pytest.raises(ValueError): joint.segment_window(parent,original,fresh)


@pytest.mark.parametrize('updates',[True,0,250,1001,1500])
def test_invalid_segment_budget_rejected(updates):
    parent,original,fresh=ledger()
    with pytest.raises(ValueError): joint.segment_window(parent,original,fresh,updates)


def decision_fixture():
    parent,original,fresh=ledger(6625); w=joint.segment_window(parent,original,fresh)
    inputs=dict(parent_sha256='parent',window=w,plan_sha256='plan',plan_identity='identity',runner_sha256='runner',previous_action='continue')
    decision={'version':joint.DECISION_VERSION,'action':'continue','parent_checkpoint_sha256':'parent',
        'start_step':6625,'fresh_cursor':36000,'updates':1000,'target_step':7625,
        'source_plan_sha256':'plan','source_plan_identity_sha256':'identity','runner_sha256':'runner',
        'previous_gate_action':'continue','architecture_changed':False,'loss_changed':False,
        'optimizer_changed':False,'reviewed_by':'explicit scheduled review'}
    return decision,inputs


def test_explicit_review_binds_all_next_segment_identifiers():
    decision,inputs=decision_fixture()
    assert joint.validate_decision(decision,**inputs)==decision
    for key in decision:
        bad=copy.deepcopy(decision); bad[key]=None
        with pytest.raises(ValueError): joint.validate_decision(bad,**inputs)
    inputs['previous_action']='pause_for_diagnosis'
    with pytest.raises(ValueError): joint.validate_decision(decision,**inputs)


def test_shared_tensorboard_path_reuses_only_new_campaign_and_rejects_old_alias(tmp_path):
    old=tmp_path/'old'; old.mkdir(); campaign=tmp_path/'new-campaign'
    args=SimpleNamespace(tensorboard_logdir=None,out=campaign/'segment5625',joint_run=old/'joint',
        candidate=old/'candidate'/'final.pt',checkpoint=old/'original'/'checkpoint.pt',
        anchor_checkpoint=old/'anchor'/'checkpoint.pt',screen_out=old/'screen',base_out=old/'base')
    assert joint.tensorboard_path(args)==(args.out/'tensorboard'/'raw').resolve()
    shared=campaign/'tensorboard'/'raw'; shared.mkdir(parents=True)
    args.tensorboard_logdir=shared
    assert joint.tensorboard_path(args)==shared.resolve()
    args.joint_run.mkdir(); alias=tmp_path/'old-alias'; alias.symlink_to(args.joint_run,target_is_directory=True)
    args.tensorboard_logdir=alias/'tensorboard'/'raw'
    with pytest.raises(ValueError): joint.tensorboard_path(args)


def populated(monkeypatch):
    model,teacher,crops,objective,old,_,_=old_tests.populated_candidate(monkeypatch)
    parent,original,fresh=ledger()
    parent.update({k:copy.deepcopy(old[k]) for k in ['group','optimizer','rng','original_training_identity']})
    for state in parent['optimizer']['state'].values(): state['step'].fill_(5625)
    return model,teacher,crops,objective,parent,original,fresh


def test_first_update_matches_original_engine_exactly_including_saved_adam_moments(monkeypatch):
    model,teacher,crops,objective,parent,_,_=populated(monkeypatch)
    expected_model=gm.build_student(teacher.model.decoder,*selections())
    expected_opt,_=joint.restart_checkpoint(expected_model,parent)
    values=joint.base.training_update(expected_model,teacher,crops,objective,
        parent['original_training_identity']['coefficients'],expected_opt,record_diagnostics=True)
    expected_rng=joint.screen.rng_state(); parent_before=copy.deepcopy(parent)
    optimizer,checks=joint.restart_checkpoint(model,parent)
    assert all(c['equal'] for c in checks.values())
    frozen=joint.screen.frozen_versions(model); teacher_before=snapshot(teacher)
    actual=joint.screen.training_update(model,teacher,crops,'current',objective,
        parent['original_training_identity']['coefficients'],optimizer,record_diagnostics=True)
    assert actual==values
    assert_snapshot(model,snapshot(expected_model)); assert_snapshot(teacher,teacher_before)
    assert joint.replay.compare_tree(optimizer.state_dict(),expected_opt.state_dict())['equal']
    assert joint.replay.compare_tree(joint.screen.rng_state(),expected_rng)['equal']
    assert joint.replay.compare_tree(parent,parent_before)['equal']
    assert joint.screen.frozen_versions(model)==frozen
    assert all(float(s['step'])==5626 for s in optimizer.state.values())


def test_restore_rejects_missing_moments_without_mutating_any_model_or_rng(monkeypatch):
    model,_,_,_,parent,_,_=populated(monkeypatch)
    parent['optimizer']['state'].pop(next(iter(parent['optimizer']['state'])))
    before=snapshot(model); rng=joint.screen.rng_state()
    with pytest.raises(ValueError): joint.restart_checkpoint(model,parent)
    assert_snapshot(model,before); assert joint.replay.compare_tree(joint.screen.rng_state(),rng)['equal']


def saved_fixture(monkeypatch,tmp_path):
    model,teacher,crops,objective,parent,original,fresh=populated(monkeypatch)
    optimizer,_=joint.restart_checkpoint(model,parent)
    for s in optimizer.state.values(): s['step'].fill_(6125)
    history=[]
    for step in (5625,6125):
        path=tmp_path/f'development-step{step}.json'; path.write_text(json.dumps({'step':step}))
        history.append(joint.report_entry(path,step))
    identity={'version':joint.VERSION,'starting_optimizer_step':5625,'target_optimizer_step':6625,
        'source_interval':[24000,36000],'source_ids':fresh[24000:36000],
        'source_ids_sha256':joint.screen.digest(fresh[24000:36000]),'root_checkpoint_sha256':joint.ROOT_SHA256,
        'gradient_accumulation':12,'execution_batch_size':1,'learning_rate':3e-5,
        'coefficients':parent['original_training_identity']['coefficients'],
        'source_plan_identity_sha256':'extension-identity'}
    return model,optimizer,parent,fresh[24000:30000],identity,history,{'step':6125,'action':'continue'}


def test_save_preserves_exact_state_and_durable_review_source_cursor(monkeypatch,tmp_path):
    model,optimizer,parent,seen,identity,history,review=saved_fixture(monkeypatch,tmp_path)
    before=snapshot(model); moments=copy.deepcopy(optimizer.state_dict()); rng=joint.screen.rng_state()
    path=tmp_path/'checkpoint-step6125.pt'
    receipt=joint.save_state(path,model,optimizer,parent,seen,12345,identity,history,review)
    saved=torch.load(path,weights_only=True,map_location='cpu')
    assert saved['fresh_cursor']==30000 and saved['total_source_count']==33000
    assert saved['optimizer_step']==6125 and saved['accumulation']==12
    assert saved['historical_sources_seen']==parent['historical_sources_seen']+parent['additional_sources_seen']
    assert saved['additional_sources_seen']==seen
    assert saved['review_history']==history and saved['latest_review']==review
    assert joint.replay.compare_tree(saved['group'],dict(model.group_state_dict()))['equal']
    assert joint.replay.compare_tree(saved['optimizer'],moments)['equal']
    assert joint.replay.compare_tree(saved['rng'],rng)['equal']
    assert receipt['checkpoint_sha256']==joint.base.sha(path)
    assert_snapshot(model,before)
    assert joint.replay.compare_tree(optimizer.state_dict(),moments)['equal']
    assert joint.replay.compare_tree(joint.screen.rng_state(),rng)['equal']
    with pytest.raises(FileExistsError): joint.save_state(path,model,optimizer,parent,seen,12345,identity,history,review)


@pytest.mark.parametrize('damage',['reorder','partial','counter','history_hash','history_missing','review','coefficient'])
def test_invalid_checkpoint_never_publishes(monkeypatch,tmp_path,damage):
    model,opt,parent,seen,identity,history,review=saved_fixture(monkeypatch,tmp_path)
    if damage=='reorder': seen[:2]=reversed(seen[:2])
    elif damage=='partial': seen.pop()
    elif damage=='counter': next(iter(opt.state.values()))['step'].add_(1)
    elif damage=='history_hash': Path(history[0]['path']).write_text('{}')
    elif damage=='history_missing': history.pop(0)
    elif damage=='review': review['action']='unknown'
    elif damage=='coefficient': identity['coefficients']={**identity['coefficients'],'waveform':2.}
    path=tmp_path/'invalid.pt'
    with pytest.raises(ValueError): joint.save_state(path,model,opt,parent,seen,12345,identity,history,review)
    assert not path.exists() and not path.with_suffix('.tmp').exists()


def test_actual_checkpoint_payload_is_accepted_by_rolling_cache_provider(monkeypatch,tmp_path):
    from continuation_data_v2 import validate_committed_checkpoint, RETAIN_CONSUMED, SHARD_SIZE
    model,opt,parent,seen,identity,history,review=saved_fixture(monkeypatch,tmp_path)
    path=tmp_path/'checkpoint-step6125.pt'
    joint.save_state(path,model,opt,parent,seen,12345,identity,history,review)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    _,original,fresh=ledger()
    before=validate_committed_checkpoint(saved,30000,fresh,original,'extension-identity')
    assert before==max(24000,((30000-RETAIN_CONSUMED)//SHARD_SIZE)*SHARD_SIZE)
    for bad_cursor in (29988,30012):
        with pytest.raises(ValueError): validate_committed_checkpoint(saved,bad_cursor,fresh,original,'extension-identity')
    with pytest.raises(ValueError): validate_committed_checkpoint(saved,30000,fresh,original,'other-identity')


def test_later_parent_authenticates_full_history_completion_and_source_plan(monkeypatch,tmp_path):
    model,opt,parent,seen,identity,history,review=saved_fixture(monkeypatch,tmp_path)
    plan=tmp_path/'plan.json'; plan.write_text('{}')
    identity.update(runner_sha256=joint.base.sha(Path(joint.__file__)),source_plan_sha256=joint.base.sha(plan),protected={})
    _,original,fresh=ledger(); seen=fresh[24000:36000]
    for state in opt.state.values(): state['step'].fill_(6625)
    final_report={'aggregate':{'sentinel':1.},'rows':[{'source_id':'dev'}]}
    path=tmp_path/'development-step6625.json'; path.write_text(json.dumps(final_report))
    history.append(joint.report_entry(path,6625)); review={'step':6625,'action':'continue'}
    checkpoint=tmp_path/'checkpoint-step6625.pt'
    receipt=joint.save_state(checkpoint,model,opt,parent,seen,12345,identity,history,review)
    receipt.update(frozen_state_preserved=True,quality=final_report['aggregate'])
    checkpoint.with_suffix('.json').write_text(json.dumps(receipt))
    (tmp_path/'launch.json').write_text(json.dumps(identity))
    completed={'status':'awaiting_review','last_checkpoint_sha256':receipt['checkpoint_sha256'],
        'step':6625,'updates':1000,'sources':12000,'fresh_cursor':36000,'total_source_count':39000,
        'original_files_preserved':True,'final':final_report['aggregate'],'reviews':[review]}
    (tmp_path/'completed.json').write_text(json.dumps(completed))
    data=SimpleNamespace(identity='extension-identity',plan_path=plan)
    pools={'calibration':[],'development':[{'source_id':'dev'}]}
    saved,reference,checksum,entries,reports,steps=joint.authenticate_parent(checkpoint,parent,{},data,pools)
    assert reference==final_report and checksum==receipt['checkpoint_sha256']
    assert steps==[5625,6125,6625] and entries==history
    assert joint.segment_window(saved,original,fresh)['start_cursor']==36000
    completed['target_step']=123 # Unused informational fields cannot authorize a different actual counter.
    completed['step']=6125
    (tmp_path/'completed.json').write_text(json.dumps(completed))
    with pytest.raises(ValueError): joint.authenticate_parent(checkpoint,parent,{},data,pools)


def observer_fixture():
    t=torch.cat([torch.full((1,1,960),5e-6),torch.full((1,1,960),5e-4)],-1)
    p=t+torch.cat([torch.full((1,1,960),6e-6),torch.full((1,1,960),-2e-4)],-1)
    crop={'source_id':'source','context_frames':0,'start_frame':0,'valid_scored_samples':1920,
          'reference16k':torch.zeros(1,1,640)}
    return p,t,crop


def test_split_observer_adds_no_model_calls_and_preserves_original_window_values(monkeypatch):
    p,t,crop=observer_fixture(); original=joint.base.quiet_window_metrics; expected=original(p,t)
    captured=[]
    class Monitor:
        def evaluate(self,*args):
            observed=joint.base.quiet_window_metrics(p,t); captured.append(observed)
            return {'aggregate':{'quiet_windows':2,'quiet_failed_windows':2},'rows':[{'source_id':'source'}]}
    report=joint.evaluate_review(Monitor(),None,None,[crop],None,summarize=joint.quiet_audit.grouped_summary)
    assert captured==[expected] and joint.base.quiet_window_metrics is original
    regions=report['quiet_regions']
    assert regions['all']['failure_categories']=={'passed':0,'residual_only':1,'amplitude_only':1,'both':0}
    assert regions['source_reference_exact_zero']['True']['windows']==2
    assert regions['actual_startup']['True']['windows']==2
    assert regions['all']['centered_residual_rms']<1e-10
    assert report['recovery_window_metrics']['near_samples']==960


def test_split_observer_restores_original_helper_on_exception():
    p,t,crop=observer_fixture(); original=joint.base.quiet_window_metrics
    class Monitor:
        def evaluate(self,*args):
            joint.base.quiet_window_metrics(p,t)
            raise RuntimeError('injected')
    with pytest.raises(RuntimeError,match='injected'):
        joint.evaluate_review(Monitor(),None,None,[crop],None,summarize=joint.quiet_audit.grouped_summary)
    assert joint.base.quiet_window_metrics is original


def test_default_observer_uses_actual_gate_region_schema_without_altering_old_report():
    import joint_recovery_gates_v2 as gates
    p,t,crop=observer_fixture()
    class Monitor:
        def evaluate(self,*args):
            joint.base.quiet_window_metrics(p,t)
            return {'aggregate':{'quiet_windows':2,'quiet_failed_windows':2},'rows':[{'source_id':'source'}]}
    expected=joint.previous.evaluate_review(Monitor(),None,None,[crop],None)
    actual=joint.evaluate_review(Monitor(),None,None,[crop],None)
    assert {k:v for k,v in actual.items() if k!='quiet_regions'}==expected
    assert actual['quiet_regions']['version']==gates.VERSION
    assert set(actual['quiet_regions']['regions'])==set(gates.REGIONS)
    assert actual['quiet_regions']['regions']['all_quiet']['failure_categories']=={
        'passed':0,'residual_only':1,'amplitude_only':1,'both':0}
    gates._validate_regions([actual])
