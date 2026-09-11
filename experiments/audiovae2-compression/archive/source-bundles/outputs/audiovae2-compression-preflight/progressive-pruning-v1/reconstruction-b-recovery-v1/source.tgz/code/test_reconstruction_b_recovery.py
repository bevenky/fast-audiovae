"""Native B installation, paired recipe and observational history contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'convnext'))
import reconstruction_b_recovery as run
import reconstruction_b_monitor as monitoring
from joint_recovery_gates_v2 import summarize_regions
from test_group_model import TinyDecoder, snapshot, assert_snapshot, gm


@pytest.fixture(autouse=True)
def cpu_policy():
    rng = run.screen.rng_state(); threads = torch.get_num_threads()
    torch.manual_seed(876); torch.set_num_threads(1)
    yield
    run.screen.restore_rng(rng); torch.set_num_threads(threads)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__(); self.model = nn.Module(); self.model.decoder = TinyDecoder()
        self.eval().requires_grad_(False)


def setup():
    teacher = Teacher()
    selection = {'stage2_indices':list(range(24)),'stage3_indices':list(range(16))}
    model = run.progressive.initialize_from_teacher(teacher.model.decoder,selection)
    pristine = run.control.state_hash(model.decoder)
    candidate = run.progressive.initialize_from_teacher(teacher.model.decoder,selection)
    with torch.no_grad():
        for i,path in enumerate(run.OPERATOR_PATHS):
            module = candidate.decoder.get_submodule(path)
            module.weight_g.mul_(1+.01*(i+1)); module.bias.add_(.001*(i+1))
    ids = [f'calibration-{i}' for i in range(72)]
    artifact = {'format':run.INIT_VERSION,'variant':'B','base_step0_sha256':run.STEP0_SHA,
        'selection':selection,'fit_source_ids':ids,'ridge':1e-6,
        'operators':{path:deepcopy(candidate.decoder.get_submodule(path).state_dict()) for path in run.OPERATOR_PATHS}}
    receipt = {'operators_sha256':'fixture-file-sha','candidate_state_sha256':run.control.state_hash(candidate.decoder),
        'changed_native_paths':list(run.OPERATOR_PATHS),'unchanged_frozen_and_unselected_tensors':True,
        'all9_residual_units_preserved':True,'neural_training_updates':0,'extra_inference_modules':0,
        'automatic_promotion':False,'original_step0_pristine':{'passed':True,'student_state_sha256':pristine,
            'teacher_state_sha256':run.control.state_hash(teacher.model.decoder)}}
    zero = {'group':deepcopy(model.group_state_dict()),'optimizer':run.progressive.fresh_optimizer(model).state_dict(),
            'rng':run.screen.rng_state()}
    return teacher,model,candidate,artifact,receipt,ids,zero


def install(model,artifact,receipt,ids):
    return run.install_b_operators(model,artifact,receipt,artifact_sha256='fixture-file-sha',
                                  step0_sha256=run.STEP0_SHA,calibration_ids=ids)


def test_complete_native_install_preserves_other_tensors_teacher_and_parameter_objects():
    teacher,model,candidate,artifact,receipt,ids,_ = setup()
    original = snapshot(model.decoder); teacher_before = snapshot(teacher)
    objects = {n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    result = install(model,artifact,receipt,ids)
    assert result['candidate_state_sha256'] == run.control.state_hash(candidate.decoder)
    assert_snapshot(model.decoder,snapshot(candidate.decoder))
    assert objects == {n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    for name,value in model.decoder.state_dict().items():
        if not name.startswith(tuple(p+'.' for p in run.OPERATOR_PATHS)):
            torch.testing.assert_close(value,original[name],rtol=0,atol=0)
    assert_snapshot(teacher,teacher_before)
    with torch.no_grad():
        z = torch.randn(1,8,2)*.05
        torch.testing.assert_close(model.forward_from_latents(z)['waveform'],
                                   candidate.forward_from_latents(z)['waveform'],rtol=0,atol=0)


@pytest.mark.parametrize('damage',['nan','shape','keys','candidate_hash','source_order','already_adapted','scope','variant'])
def test_invalid_artifact_is_rejected_before_any_partial_install(damage):
    _,model,_,artifact,receipt,ids,_ = setup()
    path = run.OPERATOR_PATHS[-1]
    if damage == 'nan': artifact['operators'][path]['bias'][0] = float('nan')
    elif damage == 'shape': artifact['operators'][path]['weight_v'] = artifact['operators'][path]['weight_v'][...,:-1]
    elif damage == 'keys': del artifact['operators'][path]['weight_g']
    elif damage == 'candidate_hash': receipt['candidate_state_sha256'] = 'different'
    elif damage == 'source_order': artifact['fit_source_ids'] = list(reversed(ids))
    elif damage == 'already_adapted':
        with torch.no_grad(): model.decoder.model[5].block[4].block[3].bias.add_(.01)
    elif damage == 'scope': artifact['operators']['model.5.block.1'] = {}
    else: artifact['variant'] = 'A'
    before = snapshot(model.decoder)
    with pytest.raises(ValueError): install(model,artifact,receipt,ids)
    assert_snapshot(model.decoder,before)


def test_start_restores_original_rng_and_fresh_adam_while_installing_b_only():
    teacher,model,candidate,artifact,receipt,ids,zero = setup()
    torch.randn(27)
    optimizer,result = run.restore_b_start(model,teacher,zero,artifact,receipt,ids,artifact_sha256='fixture-file-sha')
    assert not optimizer.state and result['fresh_adam_matches_original']
    assert run.replay.compare_tree(run.screen.rng_state(),zero['rng'])['equal']
    assert run.replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
    assert_snapshot(model.decoder,snapshot(candidate.decoder))


def ledger():
    common = {'cut_index':1,'selection':{'stage2_indices':[1],'stage3_indices':[2]},'schedule_sha256':'schedule',
        'coefficients':dict(run.prior.COEFFICIENTS),'accumulation':12,
        'teacher_source_sha256':run.base.SOURCE_SHA256,'teacher_checkpoint_sha256':run.base.CHECKPOINT_SHA256}
    ids = [f'source-{i}' for i in range(24000)]
    zero = {**deepcopy(common),'format':run.prior.VERSION,'cut_updates':0,'global_updates':0,
        'sources_seen':[],'optimizer':{'state':{}}}
    end = {**deepcopy(common),'format':'audiovae2_progressive_same_cut_continuation_v1',
        'cut_updates':2000,'global_updates':2000,'sources_seen':ids.copy()}
    return zero,end,ids


@pytest.mark.parametrize('damage',[None,'order','duplicate','recipe','selection','nonzero_start','short'])
def test_exact_original24000_prefix_and_recipe_are_required(damage):
    zero,end,ids = ledger()
    if damage == 'order': end['sources_seen'][0],end['sources_seen'][1] = end['sources_seen'][1],end['sources_seen'][0]
    elif damage == 'duplicate': ids[-1] = ids[0]; end['sources_seen'] = ids.copy()
    elif damage == 'recipe': end['coefficients']['mel'] *= 2
    elif damage == 'selection': end['selection']['stage2_indices'] = [3]
    elif damage == 'nonzero_start': zero['cut_updates'] = 5000
    elif damage == 'short': ids.pop()
    if damage:
        with pytest.raises(ValueError): run.validate_source_schedule(zero,end,ids)
    else: assert run.validate_source_schedule(zero,end,ids+['unused']) == tuple(ids)


def test_actual_b_update_equals_unchanged_original_singleton_accumulation(monkeypatch):
    teacher,model,_,artifact,receipt,ids,zero = setup()
    optimizer,_ = run.restore_b_start(model,teacher,zero,artifact,receipt,ids,artifact_sha256='fixture-file-sha')
    oracle = run.progressive.initialize_from_teacher(teacher.model.decoder,model.selections)
    oracle.load_group_state_dict(model.group_state_dict()); oracle_optimizer = run.progressive.fresh_optimizer(oracle)
    original_batch = run.base.batch
    monkeypatch.setattr(run.base,'batch',lambda rows:original_batch(rows,device='cpu'))
    @torch.no_grad()
    def forward(instance,z): return gm.teacher_trace(instance.model.decoder,z[:,:8])
    monkeypatch.setattr(run.base,'teacher_forward',forward)
    crops = []
    for i in range(12):
        frames = 2+i%2; context = i%2
        z = torch.randn(1,64,frames)*(.001 if i%3 else .05)
        crops.append({'source_id':f'crop-{i}','latents':z,'teacher_audio':forward(teacher,z)['waveform'].clone(),
            'context_frames':context,'start_frame':context,'context_start_frame':0,
            'valid_scored_samples':(frames-context)*1920-i*7})
    objective = run.base.ReconstructionV2(run.base.ReconstructionV2Config(fft_sizes=(32,64),mel_bands=(4,8)))
    teacher_before = snapshot(teacher); outer = run.screen.frozen_versions(model)
    rng = run.screen.rng_state()
    expected = run.screen.training_update(oracle,teacher,crops,'current',objective,run.prior.COEFFICIENTS,
                                         oracle_optimizer,record_diagnostics=True)
    run.screen.restore_rng(rng)
    actual,checks = run.perform_update(model,teacher,crops,objective,optimizer,diagnostics=True)
    assert actual == expected and len(checks) == 12
    assert all(c['allclose_original_tolerance'] for c in checks)
    assert run.replay.compare_tree(optimizer.state_dict(),oracle_optimizer.state_dict())['equal']
    assert_snapshot(model.decoder,snapshot(oracle.decoder)); assert_snapshot(teacher,teacher_before)
    assert run.screen.frozen_versions(model) == outer
    assert all(float(s['step']) == 1 for s in optimizer.state.values())


def test_checkpoints_preserve_continuous_moments_rng_and_exact_prefix(tmp_path):
    teacher,model,_,artifact,receipt,ids,zero = setup()
    optimizer,_ = run.restore_b_start(model,teacher,zero,artifact,receipt,ids,artifact_sha256='fixture-file-sha')
    sources = [f'source-{i}' for i in range(24000)]
    identity = {'source_ids':sources,'source_ids_sha256':run.screen.digest(sources),
                'selection':model.selections,'schedule_sha256':'schedule'}
    run.save_checkpoint(tmp_path/'zero.pt',model,optimizer,0,[],identity,model.selections,0)
    # A controlled saved-state fixture verifies counters/moments without1000 fake forwards.
    for parameter in run.base.parameters(model):
        optimizer.state[parameter] = {'step':torch.tensor(1000.),'exp_avg':torch.full_like(parameter,.02),
                                      'exp_avg_sq':torch.full_like(parameter,.03)}
    before = deepcopy(optimizer.state_dict()); rng = run.screen.rng_state()
    report = run.save_checkpoint(tmp_path/'at1000.pt',model,optimizer,1000,sources[:12000],identity,model.selections,12000)
    state = torch.load(tmp_path/'at1000.pt',weights_only=True)
    assert state['sources_seen'] == sources[:12000] and report['optimizer_and_rng_saved']
    assert set(state['group']) == set(model.group_state_dict())
    assert run.replay.compare_tree(state['optimizer'],before)['equal']
    assert run.replay.compare_tree(state['rng'],rng)['equal']
    assert not state['optimizer_reset_after_start']
    with pytest.raises(ValueError): run.save_checkpoint(tmp_path/'bad.pt',model,optimizer,2000,sources,identity,model.selections,24000)
    with pytest.raises(ValueError): run.save_checkpoint(tmp_path/'bad2.pt',model,optimizer,1000,sources[1:12001],identity,model.selections,12000)
    with pytest.raises(ValueError): run.save_checkpoint(tmp_path/'bad3.pt',model,optimizer,500,sources[:6000],identity,model.selections,6000)


class Writer:
    def __init__(self,path): self.path=Path(path); self.scalars=[]; self.text=[]
    def add_scalar(self,tag,value,step): self.scalars.append((tag,value,step))
    def add_text(self,tag,value,step): self.text.append((tag,value,step))
    def add_custom_scalars(self,layout): pass
    def flush(self): pass
    def close(self): pass


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


def test_fresh_monitor2000_progress_15curves_allquiet_original_reference_and_no_rng_effect(tmp_path):
    rng = run.screen.rng_state(); baseline = quality(); original = quality(2); before = deepcopy((baseline,original))
    monitor = monitoring.RecoveryMonitor(tmp_path/'new',writer_factory=Writer,
        source_metadata={'source':{'verified_source_labels':['human_whistling_source_description']}})
    monitor.log_validation(baseline,0,original)
    for step in range(1,2001):
        monitor.log_training({'step':step,'total':1.,'waveform':.1,'mel':.2,'feature':.3,'unique_sources':12*step},step)
        if step in run.REVIEW_STEPS:
            monitor.log_validation(quality(.5),step,None if step == 250 else original)
    curves = {name.split(' ',1)[1]:writer for name,writer in monitor.writers.items()}
    assert len(curves) == 16
    progress = next(w for name,w in curves.items() if name.startswith('13')).scalars
    assert [s for _,_,s in progress] == list(range(2001))
    assert progress[1000][1] == 50. and progress[-1][1] == 100.
    error = next(w for name,w in curves.items() if name.startswith('03')).scalars
    assert [s for _,_,s in error] == list(run.REVIEW_STEPS)
    assert error[-1][1] == 50.
    tags = {tag for tag,_,_ in monitor.details.scalars}
    for region in monitoring.original.REGIONS:
        assert 'quiet/'+region+'/residual_rms' in tags
    assert len([s for t,v,s in monitor.details.scalars if t=='loss/teacher_waveform']) == 2000
    assert [s for t,v,s in monitor.details.scalars if t=='matched_original/quality/mae'] == [0,500,1000,1500,2000]
    assert (baseline,original) == before and run.replay.compare_tree(run.screen.rng_state(),rng)['equal']
    with pytest.raises(ValueError): monitor.log_training({'step':2001},2001)
    with pytest.raises(ValueError): monitor.log_validation(quality(),2000)
    monitor.close()
