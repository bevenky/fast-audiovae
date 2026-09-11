"""Paired perceptual accounting and review integration on synthetic CPU data."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from audiovae_student.perceptual_comparison import PerceptualComparisonConfig,run_perceptual_comparison
from audiovae_student.perceptual_readiness import collect_readiness_evidence,assess_perceptual_readiness
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import PilotWindow
from audiovae_student.teacher import CHECKPOINT_SHA256
from audiovae_student.training import _rng_state
from test_perceptual_readiness import fixture as readiness_fixture
from test_representative_pilot import Corpus
from test_restart_data import row,ledger_for,counts


def fixture():
    base,crops,current,previous,_,dev,panel_rows,panel_counts=readiness_fixture()
    base.config=replace(base.config,learning_rate=2e-6,final_learning_rate=2e-6)
    for optimizer in base.optimizer.optimizers.values():
        for group in optimizer.param_groups:
            group['lr']=2e-6
    evidence=collect_readiness_evidence(base,crops,corpus=dev,panel_rows=panel_rows,sample_counts=panel_counts)
    report=assess_perceptual_readiness(base,crops,current=current,previous=previous,evidence=evidence)
    assert report['ready_for_bounded_trial']
    training=[row('gan-a',samples=12*640),row('gan-b',samples=12*640,language='en')]
    corpus=Corpus(training)
    for r in training:
        record=corpus.records[r.source_id]
        with torch.no_grad():
            target=base.model(record.latents).detach()*1.5
        corpus.records[r.source_id]=replace(record,teacher_audio=target)
    windows=[PilotWindow(r.source_id,start,6,6*640,r.dataset,r.language) for start in (0,6) for r in training]
    parent={'engine':deepcopy(base.state_dict()),'rng':_rng_state(),'identity':{'data':{'teacher_checkpoint_sha256':CHECKPOINT_SHA256}}}
    ready={'report':report,'current':current,'previous':previous,'evidence':evidence,
        'policy':{'parent_checkpoint_sha256':'b'*64,'data_plan_sha256':'c'*64,'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'readiness_sha256':report['sha256']}}
    return parent,(corpus,windows,training,{**counts(training),**panel_counts},ledger_for(training),crops,panel_rows),dev,ready


def run(path,fixture,*,resume=None,bound=None,review=False):
    parent,data,dev,ready=fixture
    return run_perceptual_comparison(*data,path,parent=parent,parent_checkpoint_sha256='b'*64,
        data_identity={'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'source_corpus':data[0].identity,
            'teacher_batch_qualification':{'fixture':True},'plan':{'identity_sha256':'c'*64}},heldout_metadata={},heldout_corpus=dev,
        config=PerceptualComparisonConfig(total_steps=4,batch_size=1,expected_parent_step=3,evaluation_interval=2,
            checkpoint_interval=1,min_free_bytes=0),device='cpu',readiness_inputs=ready,
        resume_from=resume,max_updates=bound,review_before_continuation=review)


def test_real_pair_stops_at_review_with_identical_targets_and_exact_resume(tmp_path):
    data=fixture();fingerprint=state_fingerprint(data[0])
    result=run(tmp_path/'whole',data)
    assert result['state']=='paused_for_safety_review' and result['step']==2
    assert result['perceptual_training_started'] and result['consumed_windows_per_arm']==2
    run(tmp_path/'split',data,bound=1)
    run(tmp_path/'split',data,resume=tmp_path/'split/latest.pt')
    a=torch.load(tmp_path/'whole/latest.pt',weights_only=True)
    b=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    assert state_fingerprint(a['engines'])==state_fingerprint(b['engines'])
    assert a['paired_start']['control']==a['paired_start']['perceptual']
    assert state_fingerprint(a['engines']['control']['discriminators'])==state_fingerprint(data[0]['engine']['discriminators'])
    assert state_fingerprint(a['engines']['perceptual']['discriminators'])!=state_fingerprint(data[0]['engine']['discriminators'])
    assert state_fingerprint(data[0])==fingerprint
    assert len((tmp_path/'whole/exposure.jsonl').read_text().splitlines())==2
    assert not (tmp_path/'whole/inflight.json').exists()


def test_short_planned_window_fails_before_any_update(tmp_path):
    data=fixture();values=list(data[1]);values[1]=list(values[1]);values[1][0]=replace(values[1][0],valid_input_samples16k=2000)
    changed=(data[0],tuple(values),data[2],data[3])
    with pytest.raises(ValueError,match='adversarial window'):
        run(tmp_path/'short',changed)
    assert not (tmp_path/'short').exists()


def test_uncertain_pair_cannot_resume_or_lose_exposure(tmp_path):
    data=fixture();run(tmp_path/'run',data,bound=1)
    (tmp_path/'run/inflight.json').write_text('{}')
    with pytest.raises(ValueError,match='in-flight'):
        run(tmp_path/'run',data,resume=tmp_path/'run/latest.pt')


def test_wrong_plan_rejected_before_creating_output(tmp_path):
    data=fixture()
    data[3]['policy']['data_plan_sha256']='d'*64
    with pytest.raises(ValueError,match='another data plan'):
        run(tmp_path/'wrong-plan',data)
    assert not (tmp_path/'wrong-plan').exists()


def test_later_failed_safety_check_saves_and_stops_before_next_pair(tmp_path,monkeypatch):
    import audiovae_student.perceptual_comparison as module
    data=fixture()
    parent,args,dev,ready=data
    old=args[2]
    training=[replace(r,duration_seconds=18*640/16000) for r in old]
    corpus=Corpus(training)
    args=(corpus,[PilotWindow(r.source_id,start,6,6*640,r.dataset,r.language) for start in (0,6,12) for r in training],
          training,{**args[3],**counts(training)},ledger_for(training),args[5],args[6])
    # Keep the real synthetic targets already used by this fixture for its first
    #12frames; append a final new interval so a failed update4 can prevent5.
    for row_value in training:
        original=data[1][0].records[row_value.source_id]
        longer=corpus.records[row_value.source_id]
        corpus.records[row_value.source_id]=replace(longer,
            latents=torch.cat([original.latents,longer.latents[...,12:]],-1),
            teacher_audio=torch.cat([original.teacher_audio,longer.teacher_audio[...,12*1920:]],-1))
    def invoke(resume=None,review=False):
        return run_perceptual_comparison(*args,tmp_path/'rollback',parent=parent,parent_checkpoint_sha256='b'*64,
            data_identity={'teacher_checkpoint_sha256':CHECKPOINT_SHA256,'source_corpus':corpus.identity,
                'teacher_batch_qualification':{'fixture':True},'plan':{'identity_sha256':'c'*64}},heldout_metadata={},heldout_corpus=dev,
            config=PerceptualComparisonConfig(total_steps=6,batch_size=1,expected_parent_step=3,evaluation_interval=2,checkpoint_interval=1,min_free_bytes=0),
            device='cpu',readiness_inputs=ready,resume_from=resume,review_before_continuation=review)
    assert invoke()['step']==2
    original_compare=module.compare_evaluations
    def fail_four(reports,previous=None):
        report=original_compare(reports,previous)
        if reports['perceptual']['rows'][0]['evaluated_step']==4:
            report['checks']['injected_regression']=False
            report['safety_checks_passed']=False
        return report
    monkeypatch.setattr(module,'compare_evaluations',fail_four)
    result=invoke(tmp_path/'rollback/latest.pt',True)
    assert result['step']==4 and result['state']=='stopped_for_safety'
    saved=torch.load(tmp_path/'rollback/latest.pt',weights_only=True)
    assert saved['sampler']['cursor']==4 and saved['stop_reason']=='matched_control_rollback_limit'
    assert len((tmp_path/'rollback/exposure.jsonl').read_text().splitlines())==4
    with pytest.raises(ValueError,match='failed safety review'):
        invoke(tmp_path/'rollback/latest.pt')
