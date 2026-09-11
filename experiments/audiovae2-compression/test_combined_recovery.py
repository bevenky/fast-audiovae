"""Real native-state, optimizer, source-prefix and observation-only contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'convnext'))
import combined_recovery as run
import combined_recovery_monitor as monitor_api
from test_reconstruction_b_recovery import setup as b_setup, ledger as b_ledger, Writer, quality
from test_group_model import snapshot, assert_snapshot, gm


@pytest.fixture(autouse=True)
def cpu_policy():
    rng=run.screen.rng_state();threads=torch.get_num_threads()
    torch.manual_seed(771);torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng);torch.set_num_threads(threads)


def setup():
    teacher,model,b_candidate,b_artifact,b_receipt,ids,zero=b_setup()
    combined=run.progressive.initialize_from_teacher(teacher.model.decoder,model.selections)
    combined.load_group_state_dict(b_candidate.group_state_dict())
    with torch.no_grad():combined.decoder.get_submodule(run.OPERATOR_PATHS[-1]).bias.add_(.002)
    artifact={'format':run.INIT_VERSION,'variant':'combined_B_startup','base_step0_sha256':run.b.STEP0_SHA,
        'b_operators_sha256':'fixture-file-sha','selection':model.selections,'fit_source_ids':ids,'ridge':1e-6,
        'interface_atol':1e-5,'interface_rtol':1e-4,'automatic_promotion':False,
        'operators':{path:deepcopy(combined.decoder.get_submodule(path).state_dict()) for path in run.OPERATOR_PATHS}}
    receipt={**deepcopy(b_receipt),'operators_sha256':'combined-sha','candidate_state_sha256':run.control.state_hash(combined.decoder),
        'base_b_state_sha256':run.control.state_hash(b_candidate.decoder),
        'native_fp32_constraints_passed':True,'stage2_b_operators_preserved':True}
    return teacher,model,b_candidate,combined,b_artifact,b_receipt,artifact,receipt,ids,zero


def restore(items):
    t,m,_,_,ba,br,a,r,ids,z=items
    return run.restore_combined_start(m,t,z,ba,br,a,r,ids,artifact_sha256='combined-sha',b_operators_sha256='fixture-file-sha')


def test_actual_combined_state_restores_from_original_zero_with_fresh_adam_and_rng():
    items=setup();teacher,model,bc,combined,ba,br,a,r,ids,zero=items
    before=snapshot(teacher);objects={n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    torch.randn(29);optimizer,receipt=restore(items)
    assert receipt['original_rng_and_fresh_adam_exact'] and not optimizer.state
    assert_snapshot(model.decoder,snapshot(combined.decoder));assert_snapshot(teacher,before)
    assert objects=={n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    assert run.replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
    assert run.replay.compare_tree(run.screen.rng_state(),zero['rng'])['equal']
    with torch.no_grad():
        z=torch.randn(1,8,2)*.05
        torch.testing.assert_close(model.forward_from_latents(z)['waveform'],combined.forward_from_latents(z)['waveform'],rtol=0,atol=0)


@pytest.mark.parametrize('damage',['stage2','nan','shape','candidate','base_b','constraints','source_order','ridge'])
def test_combined_rejects_bad_or_adapted_coefficients_before_any_write(damage):
    _,_,model,_,ba,_,a,r,ids,_=setup()
    path=run.OPERATOR_PATHS[-1]
    if damage=='stage2':a['operators'][run.OPERATOR_PATHS[0]]['bias'][0]+=.001
    elif damage=='nan':a['operators'][path]['bias'][0]=float('nan')
    elif damage=='shape':a['operators'][path]['weight_v']=a['operators'][path]['weight_v'][...,:-1]
    elif damage=='candidate':r['candidate_state_sha256']='foreign'
    elif damage=='base_b':
        with torch.no_grad():model.decoder.model[5].block[4].block[3].bias.add_(.03)
    elif damage=='constraints':r['native_fp32_constraints_passed']=False
    elif damage=='source_order':a['fit_source_ids']=list(reversed(ids))
    else:a['ridge']=1e-3
    before=snapshot(model.decoder)
    with pytest.raises(ValueError):
        run.install_combined_operators(model,a,r,ba,artifact_sha256='combined-sha',b_operators_sha256='fixture-file-sha',calibration_ids=ids)
    assert_snapshot(model.decoder,before)


def ledger():
    zero,two,ids=b_ledger();zero['identity']={'original':'cut1'}
    allids=ids+[f'source-{i}' for i in range(24000,60000)]
    five={**deepcopy(two),'format':'audiovae2_progressive_same_cut_5000_v1','cut_updates':5000,'global_updates':5000,
          'sources_seen':allids,'original_cut_identity':zero['identity']}
    b2000={**deepcopy(two),'format':run.b.VERSION,'step':2000,'optimizer_reset_after_start':False}
    return zero,two,five,b2000,allids[:30000]


@pytest.mark.parametrize('damage',[None,'next_order','b_order','duplicate','teacher','recipe','short'])
def test_exact30000_prefix_matches_original5000_and_b24000(damage):
    zero,two,five,b2,ids=ledger()
    if damage=='next_order':five['sources_seen'][24000],five['sources_seen'][24001]=five['sources_seen'][24001],five['sources_seen'][24000]
    elif damage=='b_order':b2['sources_seen'][0],b2['sources_seen'][1]=b2['sources_seen'][1],b2['sources_seen'][0]
    elif damage=='duplicate':ids[-1]=ids[0]
    elif damage=='teacher':five['teacher_checkpoint_sha256']='foreign'
    elif damage=='recipe':b2['accumulation']=3
    elif damage=='short':ids.pop()
    if damage:
        with pytest.raises(ValueError):run.validate_source_schedule(zero,two,five,b2,ids)
    else:assert run.validate_source_schedule(zero,two,five,b2,ids)==tuple(ids)


def test_real_combined_update_matches_original_pooled_objective_and_adam(monkeypatch):
    items=setup();teacher,model=items[:2];optimizer,_=restore(items)
    oracle=run.progressive.initialize_from_teacher(teacher.model.decoder,model.selections)
    oracle.load_group_state_dict(model.group_state_dict());other=run.progressive.fresh_optimizer(oracle)
    batch=run.base.batch;monkeypatch.setattr(run.base,'batch',lambda rows:batch(rows,device='cpu'))
    @torch.no_grad()
    def forward(instance,z):return gm.teacher_trace(instance.model.decoder,z[:,:8])
    monkeypatch.setattr(run.base,'teacher_forward',forward)
    crops=[]
    for i in range(12):
        frames=2+i%2;context=i%2;z=torch.randn(1,64,frames)*(.001 if i%3 else .05)
        crops.append({'source_id':f'crop-{i}','latents':z,'teacher_audio':forward(teacher,z)['waveform'].clone(),
            'context_frames':context,'context_start_frame':0,'start_frame':context,'valid_scored_samples':(frames-context)*1920-i*9})
    objective=run.base.ReconstructionV2(run.base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    teacher_before=snapshot(teacher);outer=run.screen.frozen_versions(model);rng=run.screen.rng_state()
    expected=run.screen.training_update(oracle,teacher,crops,'current',objective,run.prior.COEFFICIENTS,other,record_diagnostics=True)
    run.screen.restore_rng(rng)
    actual,checks=run.perform_update(model,teacher,crops,objective,optimizer,diagnostics=True)
    assert actual==expected and len(checks)==12 and all(c['allclose_original_tolerance'] for c in checks)
    assert_snapshot(model.decoder,snapshot(oracle.decoder));assert_snapshot(teacher,teacher_before)
    assert run.replay.compare_tree(optimizer.state_dict(),other.state_dict())['equal']
    assert run.screen.frozen_versions(model)==outer


def test_all_four_checkpoint_milestones_keep_moments_and_complete_prefix(tmp_path):
    items=setup();model=items[1];optimizer,_=restore(items)
    ids=[f'source-{i}' for i in range(30000)]
    identity={'source_ids':ids,'source_ids_sha256':run.screen.digest(ids),'selection':model.selections,'schedule_sha256':'schedule'}
    for step in run.CHECKPOINT_STEPS:
        if step:
            for p in run.base.parameters(model):
                optimizer.state[p]={'step':torch.tensor(float(step)),'exp_avg':torch.full_like(p,.02),'exp_avg_sq':torch.full_like(p,.03)}
        before=deepcopy(optimizer.state_dict());rng=run.screen.rng_state()
        run.save_checkpoint(tmp_path/f'{step}.pt',model,optimizer,step,ids[:12*step],identity,model.selections,step*12)
        saved=torch.load(tmp_path/f'{step}.pt',weights_only=True)
        assert saved['sources_seen']==ids[:12*step] and saved['step']==step and not saved['optimizer_reset_after_start']
        assert run.replay.compare_tree(saved['optimizer'],before)['equal'] and run.replay.compare_tree(saved['rng'],rng)['equal']
    with pytest.raises(ValueError):run.save_checkpoint(tmp_path/'late.pt',model,optimizer,3000,ids,identity,model.selections,30000)
    with pytest.raises(ValueError):run.save_checkpoint(tmp_path/'reordered.pt',model,optimizer,2500,list(reversed(ids)),identity,model.selections,30000)
    optimizer.state[run.base.parameters(model)[0]]['step'].fill_(2000)
    with pytest.raises(ValueError):run.save_checkpoint(tmp_path/'reset.pt',model,optimizer,2500,ids,identity,model.selections,30000)


def test_monitor2500_has_seven_reviews_correct_progress_and_matched_b_without_replay(tmp_path):
    rng=run.screen.rng_state();baseline=quality();original=quality(2);b_ref=quality(.9)
    before=deepcopy((baseline,original,b_ref))
    m=monitor_api.RecoveryMonitor(tmp_path/'chart',writer_factory=Writer,
        source_metadata={'source':{'verified_source_labels':['human_whistling_source_description']}})
    m.log_validation(baseline,0,original,b_ref)
    for step in range(1,2501):
        m.log_training({'step':step,'total':.2,'waveform':.02,'mel':.4,'feature':.1,'unique_sources':step*12},step)
        if step in run.REVIEW_STEPS:m.log_validation(quality(.5),step,None if step==250 else original,b_ref if step<=2000 else None)
    progress=next(w for name,w in m.writers.items() if ' 13 ' in name).scalars
    quality_points=next(w for name,w in m.writers.items() if ' 01 ' in name).scalars
    assert [s for _,_,s in progress]==list(range(2501)) and progress[2000][1]==80. and progress[-1][1]==100.
    assert [s for _,_,s in quality_points]==list(run.REVIEW_STEPS)
    assert [s for t,_,s in m.details.scalars if t=='matched_b/quality/mae']==list(run.b.REVIEW_STEPS)
    assert len([s for t,_,s in m.details.scalars if t=='loss/teacher_waveform'])==2500
    for region in monitor_api.original.REGIONS:
        assert any(t=='quiet/'+region+'/residual_rms' for t,_,s in m.details.scalars)
    assert len(m.writers)==16 and (baseline,original,b_ref)==before
    assert run.replay.compare_tree(run.screen.rng_state(),rng)['equal']
    with pytest.raises(ValueError):m.log_training({'step':2501},2501)
    m.close()


def test_nonfinite_or_different_teacher_reference_does_not_enter_monitor(tmp_path):
    m=monitor_api.RecoveryMonitor(tmp_path/'chart',writer_factory=Writer);q=quality();bad=quality()
    bad['quiet_regions']['window_identity_sha256']='different'
    with pytest.raises(ValueError):m.log_validation(q,0,b_reference=bad)
    assert m.baseline is None
    bad=quality();bad['aggregate']['mae']=float('nan')
    with pytest.raises(ValueError):m.log_validation(bad,0)
    m.close()
