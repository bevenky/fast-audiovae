"""Fresh native G-plus-startup restoration and bounded recovery contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'convnext'))
from test_grail_candidate_recovery import setup as g_setup, cpu_policy
from test_group_model import snapshot, assert_snapshot
from test_quiet_candidate_recovery import Writer, quality
import grail_startup_recovery as run
import grail_startup_monitor as monitor_api


def setup():
    teacher, model, candidate, _, _, ids, zero = g_setup()
    # Distinguish the startup refit from G: only its upsampler bias changes.
    with torch.no_grad(): candidate.decoder.get_submodule(run.OPERATOR_PATHS[-1]).bias.add_(.001)
    artifact = run.init.export_artifact(candidate, teacher.model.decoder, ids,
        g_sha256=run.init.G_PINS['grail-native-operators.pt'],
        base_g_state_sha256=run.init.G_STATE_SHA,
        base_selected_state_sha256=run.control.state_hash(model.decoder))
    receipt = {key: deepcopy(artifact[key]) for key in (
        'format', 'variant', 'teacher_state_sha256', 'selection', 'calibration_sources',
        'calibration_source_ids_sha256', 'base_selected_state_sha256', 'candidate_state_sha256')}
    receipt.update(operators_sha256='native-fixture-sha', changed_native_paths=list(run.OPERATOR_PATHS),
        changed_relative_to_g=[run.OPERATOR_PATHS[-1]], stage2_g_operators_preserved=True,
        native_fp32_constraints_passed=True, calibration_waveform_startup_passed=6,
        all9_residual_units_preserved=True, fresh_original_teacher_factory=True,
        original_checkpoint_weights_installed=False, unchanged_frozen_and_unselected_tensors=True,
        neural_training_updates=0, extra_inference_modules=0, automatic_promotion=False)
    return teacher, model, candidate, artifact, receipt, ids, zero


def restore(parts):
    teacher, model, _, artifact, receipt, ids, zero = parts
    return run.restore_start(model, teacher, zero, artifact, receipt, ids,
                             artifact_sha256='native-fixture-sha')


def test_fresh_restore_ignores_poisoned_group_preserves_teacher_objects_and_restores_rng(monkeypatch):
    parts = setup()
    teacher, model, candidate, artifact, _, ids, zero = parts
    zero['group'] = {name: torch.full_like(value, 123.) for name, value in zero['group'].items()}
    before_zero, teacher_before = deepcopy(zero), snapshot(teacher)
    before = snapshot(model.decoder)
    objects = {name: (id(p), p.data_ptr()) for name, p in model.named_parameters()}
    def forbidden(*args, **kwargs): raise AssertionError('Historical checkpoint group was accessed')
    monkeypatch.setattr(type(model), 'load_group_state_dict', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    torch.randn(19)
    optimizer, receipt = restore(parts)
    assert_snapshot(model.decoder, snapshot(candidate.decoder)); assert_snapshot(teacher, teacher_before)
    assert objects == {name: (id(p), p.data_ptr()) for name, p in model.named_parameters()}
    assert run.replay.compare_tree(zero, before_zero)['equal']
    assert run.replay.compare_tree(run.screen.rng_state(), zero['rng'])['equal']
    assert run.replay.compare_tree(optimizer.state_dict(), zero['optimizer'])['equal']
    assert not optimizer.state and receipt['old_group_state_loaded'] is False
    params = run.base.parameters(model)
    assert len(params) == 90 and set(params) == {p for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in teacher.parameters())
    for name, value in model.decoder.state_dict().items():
        if not name.startswith(tuple(path + '.' for path in run.OPERATOR_PATHS)):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for path in run.OPERATOR_PATHS[:3]:
        assert run.b._state_hash(model.decoder.get_submodule(path).state_dict()) == artifact['stage2_g_operator_hashes'][path]
    assert len(ids) == 72 and run.ACCUMULATION == 12 and run.UPDATES == 2000


@pytest.mark.parametrize('damage', ['adapted_basis', 'late_nan', 'wrong_calibration', 'wrong_g', 'stage2_tamper', 'unqualified_native'])
def test_foreign_or_partial_native_candidate_is_rejected_before_any_write(damage):
    parts = setup()
    teacher, model, _, artifact, receipt, ids, _ = parts
    if damage == 'adapted_basis':
        with torch.no_grad(): model.decoder.model[5].block[4].block[3].bias.add_(.01)
    elif damage == 'late_nan': artifact['operators'][run.OPERATOR_PATHS[-1]]['bias'][0] = float('nan')
    elif damage == 'wrong_calibration': ids.reverse()
    elif damage == 'wrong_g': artifact['base_g_state_sha256'] = 'foreign-g'
    elif damage == 'stage2_tamper':
        artifact['operators'][run.OPERATOR_PATHS[0]]['bias'][0] += .01
        # Even a self-consistent whole-candidate hash cannot replace sealed G mixers.
        proposed = model.decoder.state_dict().copy()
        for path, state in artifact['operators'].items(): proposed.update({path+'.'+key:value for key,value in state.items()})
        artifact['candidate_state_sha256'] = receipt['candidate_state_sha256'] = run.b._state_hash(proposed)
    else: receipt['native_fp32_constraints_passed'] = False
    before, teacher_before = snapshot(model), snapshot(teacher)
    with pytest.raises(ValueError): restore(parts)
    assert_snapshot(model, before); assert_snapshot(teacher, teacher_before)


def test_trained_state_or_foreign_support_cannot_be_a_fresh_template():
    for damage in ('counter', 'moments', 'selection'):
        parts = setup(); model, zero = parts[1], parts[-1]
        if damage == 'counter': zero['cut_updates'] = zero['global_updates'] = 64
        elif damage == 'moments': zero['optimizer']['state'] = {0: {'step': torch.tensor(64.)}}
        else: zero['selection'] = {'stage2_indices': list(range(24)), 'stage3_indices': list(range(16))}
        before = snapshot(model)
        with pytest.raises(ValueError): restore(parts)
        assert_snapshot(model, before)


def pilot_pair():
    shared = {key: key+'-fixture' for key in ('initial_state_sha256', 'initializer_artifact_sha256',
        'config_sha256', 'ordinary_source_prefix_sha256', 'source_plan_identity_sha256',
        'initial_rng_sha256', 'policy_source_sha256', 'runner_source_sha256')}
    shared.update(version=run.PILOT_VERSION, complete=True, updates=64, ordinary_unique_sources=768,
        checkpoint_written=False, all_preservation_checks_passed=True,
        installation={'old_group_state_loaded':False, 'fresh_adam_matches_original':True},
        development_before={'aggregate':quality()['aggregate']}, fixed_fitting_batch_objective_before={'total':.1},
        initial_calibration_startup={'passed':6}, initial_development_startup={'passed':13},
        development_after={'aggregate':quality()['aggregate']})
    ordinary, retention = deepcopy(shared), deepcopy(shared)
    ordinary['method'], retention['method'] = 'ordinary', 'corrected'
    ordinary['update_records'] = [{'step':step,'total':.1} for step in range(1,65)]
    retention['update_records'] = [{'step':step, 'q_constraints':12, 'q_caps_passed':1,
        'q_normal_budget_passed':1, 'startup_anchor_after_passed':6, 'q_normal_solves':int(step%2==0),
        'q_full_primal_verified':1, 'q_kkt_passed':1, 'q_normal_full_primal_verified':1, 'q_normal_kkt_passed':1,
        'q_zero_displacement':0, 'q_accepted_displacement_norm':.01} for step in range(1,65)]
    retention['calibration_after'] = {'passed':6}
    return ordinary, retention


def test_paired_qualification_binds_start_exposure_and_learning_without_auto_quality_promotion():
    ordinary, retention = pilot_pair()
    # A quality tradeoff must remain visible for review, not acquire a fabricated threshold.
    retention['development_after']['aggregate']['mae'] *= 1.1
    report = run.paired_qualification(ordinary, retention)
    assert report['eligible_for_review'] and not report['automatic_promotion']
    assert report['human_quality_review_required']
    assert report['matched_quality']['mae']['relative_change'] == pytest.approx(.1)
    for damage in ('rng', 'sources', 'anchor', 'late_zero', 'checkpoint', 'certificate', 'nonfinite'):
        changed = deepcopy(retention)
        if damage == 'rng': changed['initial_rng_sha256'] = 'other'
        elif damage == 'sources': changed['ordinary_source_prefix_sha256'] = 'other'
        elif damage == 'anchor': changed['update_records'][2]['startup_anchor_after_passed'] = 5
        elif damage == 'late_zero': changed['update_records'][-1].update(q_zero_displacement=1,q_accepted_displacement_norm=0.)
        elif damage == 'certificate': changed['update_records'][2]['q_normal_full_primal_verified'] = 0
        elif damage == 'nonfinite': changed['update_records'][2]['q_accepted_displacement_norm'] = float('inf')
        else: changed['checkpoint_written'] = True
        assert not run.paired_qualification(ordinary, changed)['eligible_for_review']


def test_compact_checkpoints_keep_90_moments_and_exact_source_prefix_without_reset(tmp_path):
    parts = setup(); model = parts[1]; optimizer, _ = restore(parts)
    ids = [f'ordinary-{i}' for i in range(24000)]
    identity = {'source_ids':ids, 'source_ids_sha256':run.screen.digest(ids), 'selection':model.selections}
    before = snapshot(model)
    for step in (0,1000,2000):
        if step:
            # Small state fixture represents completed Adam calls even if parameters did not move.
            for parameter in run.base.parameters(model):
                optimizer.state[parameter] = {'step':torch.tensor(float(step)),
                    'exp_avg':torch.full_like(parameter,.1), 'exp_avg_sq':torch.full_like(parameter,.2)}
        expected, rng = deepcopy(optimizer.state_dict()), run.screen.rng_state()
        path = tmp_path/f'checkpoint-{step}.pt'
        receipt = run.save_checkpoint(path,model,optimizer,step,ids[:step*12],identity,model.selections,step*12)
        saved = torch.load(path, weights_only=True)
        assert saved['format'] == run.VERSION and saved['step'] == step
        assert len(saved['group']) == 90 and len(saved['optimizer']['state']) == (90 if step else 0)
        assert saved['sources_seen'] == ids[:step*12] and saved['identity'] == identity
        assert saved['coefficients'] == run.prior.COEFFICIENTS and saved['accumulation'] == 12
        assert not saved['optimizer_reset_after_start'] and not saved['automatic_next_cut']
        assert run.replay.compare_tree(saved['optimizer'],expected)['equal']
        assert run.replay.compare_tree(saved['rng'],rng)['equal']
        assert receipt['checkpoint_sha256'] == run.base.sha(path)
        assert_snapshot(model, before)
        with pytest.raises(FileExistsError): run.save_checkpoint(path,model,optimizer,step,ids[:step*12],identity,model.selections,step*12)
    assert sum(path.stat().st_size for path in tmp_path.glob('*.pt')) < 16*1024**2
    for damage in ('step', 'source_order', 'moment', 'learning_rate'):
        candidate_opt = run.progressive.fresh_optimizer(model)
        candidate_opt.load_state_dict(deepcopy(optimizer.state_dict()))
        step, seen = 2000, ids.copy()
        if damage == 'step': step = 2500
        elif damage == 'source_order': seen[0],seen[1] = seen[1],seen[0]
        elif damage == 'moment': candidate_opt.state[run.base.parameters(model)[0]]['exp_avg'][0] = float('nan')
        else: candidate_opt.param_groups[0]['lr'] = 1e-4
        with pytest.raises(ValueError): run.save_checkpoint(tmp_path/f'{damage}.pt',model,candidate_opt,step,seen,identity,model.selections,24000)
    assert run.storage_plan(1024)['pilot_checkpoint_bytes'] == 0
    assert run.storage_plan(1024)['checkpoint_steps'] == [0,1000,2000]


def test_monitor_keeps_normal_motion_separate_and_uses_fresh_2000_baseline(tmp_path):
    monitor = monitor_api.RecoveryMonitor(tmp_path/'tb', writer_factory=Writer)
    monitor.log_validation(quality(),0)
    monitor.log_training({'step':1,'total':.1,'waveform':.2,'mel':.3,'feature':.4,'unique_sources':12,
        'q_accepted_fraction':1.,'q_accepted_displacement_norm':.02,'q_normal_accepted_norm':.00001,
        'q_base_fraction_has_normal_correction':1,'q_constraints':12,'startup_anchor_after_passed':6},1)
    scalars={tag:(value,step) for tag,value,step in monitor.details.scalars}
    assert scalars['quiet_update/accepted_displacement_norm'] == (.02,1)
    assert scalars['quiet_update/normal_accepted_norm'] == (.00001,1)
    assert scalars['quiet_update/base_fraction_has_normal_correction'] == (1.,1)
    assert scalars['startup_anchor/after_passed'] == (6.,1)
    guide=next(text for tag,text,_ in monitor.details.text if tag=='Guide/Reading this recovery')
    assert 'not an unchanged Adam step' in guide and 'No pilot state is resumed' in guide
    progress=next(w for name,w in monitor.writers.items() if ' 13 ' in name).scalars
    assert progress[-1][1:] == (.05,1)
    with pytest.raises(ValueError): monitor.log_validation(quality(),2000)
    monitor.close()
