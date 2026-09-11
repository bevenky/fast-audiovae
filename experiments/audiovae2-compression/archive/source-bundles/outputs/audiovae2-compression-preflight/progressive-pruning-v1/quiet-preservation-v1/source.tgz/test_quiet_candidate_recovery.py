"""Fresh native B/C origin, matched exposure and projected-update observations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import quiet_candidate_recovery as run
import quiet_candidate_monitor as monitor_api
from joint_recovery_gates_v2 import summarize_regions
from test_group_model import TinyDecoder,gm,snapshot,assert_snapshot


@pytest.fixture(autouse=True)
def cpu_policy():
    rng=run.screen.rng_state();threads=torch.get_num_threads()
    torch.manual_seed(811);torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng);torch.set_num_threads(threads)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.decoder=TinyDecoder()
        self.eval().requires_grad_(False)


def setup():
    teacher=Teacher();selection={'stage2_indices':list(range(24)),'stage3_indices':list(range(16))}
    model=run.progressive.initialize_from_teacher(teacher.model.decoder,selection)
    bmodel=run.progressive.initialize_from_teacher(teacher.model.decoder,selection)
    with torch.no_grad():
        for i,path in enumerate(run.OPERATOR_PATHS):
            module=bmodel.decoder.get_submodule(path);module.weight_g.mul_(1+.01*(i+1));module.bias.add_(.001*(i+1))
    candidate=run.progressive.initialize_from_teacher(teacher.model.decoder,selection)
    candidate.load_group_state_dict(bmodel.group_state_dict())
    with torch.no_grad():candidate.decoder.get_submodule(run.OPERATOR_PATHS[-1]).bias.add_(.002)
    ids=[f'calibration-{i}' for i in range(72)]
    def operators(m):return {p:deepcopy(m.decoder.get_submodule(p).state_dict()) for p in run.OPERATOR_PATHS}
    ba={'format':run.b.INIT_VERSION,'variant':'B','base_step0_sha256':run.b.STEP0_SHA,
        'selection':selection,'fit_source_ids':ids,'ridge':1e-6,'operators':operators(bmodel)}
    br={'operators_sha256':run.b.B_SHA,'candidate_state_sha256':run.control.state_hash(bmodel.decoder),
        'changed_native_paths':list(run.OPERATOR_PATHS),'unchanged_frozen_and_unselected_tensors':True,
        'all9_residual_units_preserved':True,'neural_training_updates':0,'extra_inference_modules':0,
        'automatic_promotion':False,'original_step0_pristine':{'passed':True,
            'student_state_sha256':run.control.state_hash(model.decoder),
            'teacher_state_sha256':run.control.state_hash(teacher.model.decoder)}}
    ca={'format':run.combined.INIT_VERSION,'variant':'combined_B_startup','base_step0_sha256':run.b.STEP0_SHA,
        'b_operators_sha256':run.b.B_SHA,'selection':selection,'fit_source_ids':ids,'ridge':1e-6,
        'interface_atol':1e-5,'interface_rtol':1e-4,'automatic_promotion':False,'operators':operators(candidate)}
    cr={**deepcopy(br),'operators_sha256':'combined-sha','candidate_state_sha256':run.control.state_hash(candidate.decoder),
        'base_b_state_sha256':run.control.state_hash(bmodel.decoder),
        'native_fp32_constraints_passed':True,'stage2_b_operators_preserved':True}
    zero={'selection':selection,'cut_updates':0,'global_updates':0,'sources_seen':[],
        'group':{'must_never_load':torch.tensor([float('nan')])},
        'optimizer':run.progressive.fresh_optimizer(model).state_dict(),'rng':run.screen.rng_state()}
    return teacher,model,candidate,zero,ba,br,ca,cr,ids


def restore(items):
    teacher,model,_,zero,ba,br,ca,cr,ids=items
    return run.restore_quiet_start(model,teacher,zero,ba,br,ca,cr,ids,artifact_sha256='combined-sha')


def test_fresh_factory_b_c_never_reads_or_installs_old_group(monkeypatch):
    items=setup();teacher,model,candidate,zero,*_=items
    old=deepcopy(zero);teacher_before=snapshot(teacher)
    objects={n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    def forbidden(*a,**k):raise AssertionError('Historical group checkpoint was loaded')
    monkeypatch.setattr(model,'load_group_state_dict',forbidden)
    torch.randn(11);opt,report=restore(items)
    assert report['old_group_state_loaded'] is False and not opt.state
    assert report['original_step0_rng_restored_exactly'] and report['fresh_adam_matches_original']
    assert_snapshot(model.decoder,snapshot(candidate.decoder));assert_snapshot(teacher,teacher_before)
    assert objects=={n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    assert torch.isnan(zero['group']['must_never_load']).all()
    assert run.replay.compare_tree(opt.state_dict(),old['optimizer'])['equal']
    assert run.replay.compare_tree(run.screen.rng_state(),old['rng'])['equal']
    with torch.no_grad():
        z=torch.randn(1,8,2)*.01
        torch.testing.assert_close(model.forward_from_latents(z)['waveform'],candidate.forward_from_latents(z)['waveform'],atol=0,rtol=0)
    assert len(run.base.parameters(model))==90


@pytest.mark.parametrize('damage',['trained','moments','selection','teacher','adapted_factory'])
def test_rejects_nonfresh_or_foreign_start_before_native_install(damage):
    items=setup();_,model,_,zero,_,br,_,_,_=items
    if damage=='trained':zero['global_updates']=2000
    elif damage=='moments':zero['optimizer']['state']={0:{'step':torch.tensor(2000.)}}
    elif damage=='selection':zero['selection']={'stage2_indices':list(range(23)),'stage3_indices':list(range(16))}
    elif damage=='teacher':br['original_step0_pristine']['teacher_state_sha256']='foreign'
    else:
        with torch.no_grad():model.decoder.model[5].block[2].block[3].bias.add_(.02)
    before=snapshot(model.decoder)
    with pytest.raises(ValueError):restore(items)
    assert_snapshot(model.decoder,before)


def auth_fixture(tmp_path,monkeypatch):
    folder=tmp_path/'combined';folder.mkdir();ids=[f'source-{i}' for i in range(30000)]
    zero={'selection':{'stage2_indices':[1],'stage3_indices':[2]}}
    reference={'aggregate':{'mae':.02}};reports={s:deepcopy(reference) for s in run.REVIEW_STEPS}
    endpoint=folder/'checkpoint-step2500.pt';endpoint.write_bytes(b'sealed completed C2500')
    endpoint2000=folder/'checkpoint-step2000.pt';endpoint2000.write_bytes(b'sealed matched C2000')
    monkeypatch.setattr(run,'COMBINED2500_SHA',run.base.sha(endpoint))
    done={'version':run.combined.VERSION,'status':'awaiting_review','step':2500,'source_count':30000,
        'frozen_state_preserved':True,'original_files_preserved':True,'last_checkpoint_sha256':run.base.sha(endpoint)}
    launch={'initial_decoder_state_sha256':'state','combined_operators_sha256':'artifact',
        'selection':zero['selection'],'source_ids':ids,'protected':{},'coefficients':dict(run.prior.COEFFICIENTS),
        'learning_rate':3e-5,'gradient_accumulation':12,'execution_batch_size':1,'trainable_stages':[2,3,4]}
    journal=[{'step':i,'source_ids':ids[(i-1)*12:i*12]} for i in range(1,2501)]
    def write():
        for name,value in (('completed.json',done),('launch.json',launch),
            ('checkpoint-step2500.json',{'checkpoint_sha256':run.base.sha(endpoint)}),
            ('checkpoint-step2000.json',{'checkpoint_sha256':run.base.sha(endpoint2000),'quality':reports[2000]['aggregate']})):
            (folder/name).write_text(json.dumps(value))
        (folder/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journal))
        for step,report in reports.items():(folder/f'development-step{step}.json').write_text(json.dumps(report))
    write()
    inherited=(zero,{}, {},{}, {},tuple(ids),{},reference,{}, {},{})
    # The old authenticated inputs are a separate tested provider. This fixture
    # exercises the actual additional C journal/checkpoint/recipe binding here.
    monkeypatch.setattr(run.combined,'authenticate_inputs',lambda *args:deepcopy(inherited))
    args=SimpleNamespace(combined_run=folder,combined_state_sha256='state',combined_artifact_sha256='artifact')
    return args,ids,done,launch,journal,reports,write


@pytest.mark.parametrize('damage',[None,'incomplete','source_order','missing_update','initial_state','recipe','baseline'])
def test_actual_combined_reference_authentication_matches_first24000(tmp_path,monkeypatch,damage):
    args,ids,done,launch,journal,reports,write=auth_fixture(tmp_path,monkeypatch)
    if damage=='incomplete':done['step']=2000
    elif damage=='source_order':journal[0]['source_ids'].reverse()
    elif damage=='missing_update':journal.pop(123)
    elif damage=='initial_state':launch['initial_decoder_state_sha256']='adapted'
    elif damage=='recipe':launch['gradient_accumulation']=3
    elif damage=='baseline':reports[0]['aggregate']['mae']=.9
    write()
    if damage:
        with pytest.raises(ValueError):run.authenticate_inputs(args,None,None)
    else:
        result=run.authenticate_inputs(args,None,None)
        assert result[5]==tuple(ids[:24000]) and len(result[-1])==13


def test_checkpoints_count_adam_batches_even_when_weights_did_not_move(tmp_path):
    items=setup();model=items[1];opt,_=restore(items);initial=snapshot(model.decoder)
    ids=[f'source-{i}' for i in range(24000)]
    identity={'source_ids':ids,'source_ids_sha256':run.screen.digest(ids),'selection':model.selections,'schedule_sha256':'schedule'}
    for step in run.CHECKPOINT_STEPS:
        if step:
            for p in run.base.parameters(model):
                opt.state[p]={'step':torch.tensor(float(step)),'exp_avg':torch.full_like(p,.1),'exp_avg_sq':torch.full_like(p,.2)}
        prior_opt=deepcopy(opt.state_dict());rng=run.screen.rng_state()
        run.save_checkpoint(tmp_path/f'{step}.pt',model,opt,step,ids[:step*12],identity,model.selections,step*12)
        saved=torch.load(tmp_path/f'{step}.pt',weights_only=True)
        assert saved['step']==step and saved['sources_seen']==ids[:step*12] and not saved['optimizer_reset_after_start']
        assert run.replay.compare_tree(saved['optimizer'],prior_opt)['equal']
        assert run.replay.compare_tree(saved['rng'],rng)['equal']
        assert_snapshot(model.decoder,initial)
    with pytest.raises(ValueError):run.save_checkpoint(tmp_path/'2500.pt',model,opt,2500,ids,identity,model.selections,30000)
    opt.state[run.base.parameters(model)[0]]['step'].fill_(1000)
    with pytest.raises(ValueError):run.save_checkpoint(tmp_path/'reset.pt',model,opt,2000,ids,identity,model.selections,24000)


class Writer:
    def __init__(self,path):self.path=Path(path);self.scalars=[];self.text=[]
    def add_scalar(self,tag,value,step):self.scalars.append((tag,value,step))
    def add_text(self,tag,value,step):self.text.append((tag,value,step))
    def add_custom_scalars(self,layout):pass
    def flush(self):pass
    def close(self):pass


def quality(scale=1.):
    windows=[]
    for i,(start,teacher,zero) in enumerate(((0,9e-6,True),(960,6e-4,True),(1920,9e-6,True),(48000,9e-6,False),(48960,3e-4,False))):
        windows.append({'window_id':str(i),'source_id':'source','source_start_sample':start,'source_stop_sample':start+960,
            'valid_samples':960,'is_quiet':True,'teacher_rms':teacher,'student_rms':teacher,
            'residual_limit':max(.02**.5*teacher,1e-5),'output_rms_limit':max(10**.05*teacher,1e-5),
            'source_reference_exact_zero':zero,'failure_category':'passed',
            'residual_square_sum':4e-6**2*960,'centered_residual_square_sum':3e-6**2*960})
    return {'aggregate':{'sources':1,'samples':4800,'mae':.02*scale,'mel':.5*scale,'group_mse':.1*scale,
        'quiet_residual_rms_mean':4e-6,'nonquiet_cosine_mean':.875,'peak_abs_max':.75,
        'quiet_windows':5,'quiet_failed_windows':0,'overshoot_samples':0},'rows':[{'source_id':'source'}],
        'quiet_regions':summarize_regions(windows),
        'overview_window_metrics':{'by_source':{'source':{'active_teacher_energy':4.,'active_student_energy':1.}}}}


def test_monitor_counts_all2000_optimizer_updates_and_keeps_c_reference_separate(tmp_path):
    baseline=quality();original=quality(2);c=quality(.7);before=deepcopy((baseline,original,c));rng=run.screen.rng_state()
    monitor=monitor_api.RecoveryMonitor(tmp_path/'tb',writer_factory=Writer,
        source_metadata={'source':{'verified_source_labels':['human_whistling_source_description']}})
    monitor.log_validation(baseline,0,original,c)
    for step in range(1,2001):
        monitor.log_training({'step':step,'total':.1,'waveform':.2,'mel':.3,'feature':.4,'unique_sources':step*12,
            'q_projected':int(step%2==0),'q_zero_displacement':int(step%3==0),
            'q_accepted_fraction':0 if step%3==0 else 1,'q_projection_cosine':None},step)
        if step in run.REVIEW_STEPS:monitor.log_validation(quality(.5),step,None if step==250 else original,c)
    progress=next(w for n,w in monitor.writers.items() if ' 13 ' in n).scalars
    assert len(monitor.writers)==16 and [s for _,_,s in progress]==list(range(2001))
    assert progress[1000][1]==50. and progress[-1][1]==100.
    details=monitor.details.scalars
    assert [s for n,_,s in details if n=='matched_combined/quality/mae']==list(run.REVIEW_STEPS)
    assert [s for n,_,s in details if n=='matched_original/quality/mae']==[0,500,1000,1500,2000]
    assert len([s for n,_,s in details if n=='loss/teacher_waveform'])==2000
    assert len([s for n,_,s in details if n=='quiet_update/zero_displacement'])==2000
    for region in monitor_api.original.REGIONS:assert any(n=='quiet/'+region+'/residual_rms' for n,_,_ in details)
    assert (baseline,original,c)==before and run.replay.compare_tree(run.screen.rng_state(),rng)['equal']
    with pytest.raises(ValueError):monitor.log_training({'step':2001},2001)
    with pytest.raises(ValueError):monitor.log_validation(baseline,2000)
    monitor.close()


def test_monitor_rejects_foreign_quiet_identity_before_logging_baseline(tmp_path):
    monitor=monitor_api.RecoveryMonitor(tmp_path/'tb',writer_factory=Writer)
    report=quality();foreign=quality();foreign['quiet_regions']['window_identity_sha256']='foreign'
    with pytest.raises(ValueError):monitor.log_validation(report,0,c_reference=foreign)
    assert monitor.baseline is None
    monitor.close()


def test_counter_helper_distinguishes_projection_backtracking_zero_and_optimizer_updates():
    totals={}
    ordinary={'q_projected':0,'q_zero_displacement':0,'q_accepted_fraction':1.,
        'q_auxiliary_seconds':.25,'q_ordinary_update_seconds':.1}
    for extra in ({},{'q_projected':1},{'q_projected':1,'q_accepted_fraction':.5},
                  {'q_zero_displacement':1,'q_accepted_fraction':0.}, {'q_zero_displacement':1}):
        assert run.accumulate_update_totals(totals,{**ordinary,**extra}) is totals
    assert totals['updates']==5 and totals['projected_updates']==2
    assert totals['zero_displacements']==2 and totals['backtracking_updates']==2
    assert totals['updates']-totals['zero_displacements']==3
    assert totals['auxiliary_seconds']==1.25 and totals['ordinary_seconds']==.5


@pytest.mark.parametrize('key,value',[('q_projected',2),('q_zero_displacement',float('nan')),
    ('q_accepted_fraction',.3),('q_ordinary_update_seconds',float('inf')),
    ('q_auxiliary_seconds',-.1),('q_projected','1'),('q_accepted_fraction',0.)])
def test_invalid_counter_record_is_rejected_without_partial_accounting(key,value):
    record={'q_projected':0,'q_zero_displacement':0,'q_accepted_fraction':1.,
        'q_auxiliary_seconds':.25,'q_ordinary_update_seconds':.1}
    totals={};run.accumulate_update_totals(totals,record);before=deepcopy(totals)
    record[key]=value
    with pytest.raises(ValueError):run.accumulate_update_totals(totals,record)
    assert totals==before
