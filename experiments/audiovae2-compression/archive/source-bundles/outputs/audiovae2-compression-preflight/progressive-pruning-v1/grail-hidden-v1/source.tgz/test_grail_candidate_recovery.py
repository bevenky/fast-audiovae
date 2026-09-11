"""Independent fixed-support GRAIL restoration and unchanged recovery contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import grail_candidate_recovery as run
import reconstruction_b_recovery as old
from test_group_model import TinyDecoder, snapshot, assert_snapshot, gm


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = old.screen.rng_state(), torch.get_num_threads()
    torch.manual_seed(876)
    torch.set_num_threads(1)
    yield
    old.screen.restore_rng(rng)
    torch.set_num_threads(threads)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = TinyDecoder()
        self.eval().requires_grad_(False)


def setup():
    teacher = Teacher()
    original_selection = {'stage2_indices': [i for i in range(32) if i % 4], 'stage3_indices': list(range(16))}
    selection = {'stage2_indices': [i for i in range(32) if i % 4],
                 'stage3_indices': list(range(16))}
    original = old.progressive.initialize_from_teacher(teacher.model.decoder, original_selection)
    model = old.progressive.initialize_from_teacher(teacher.model.decoder, selection)
    candidate = old.progressive.initialize_from_teacher(teacher.model.decoder, selection)
    with torch.no_grad():
        for i, path in enumerate(old.OPERATOR_PATHS):
            module = candidate.decoder.get_submodule(path)
            module.weight_g.mul_(1 + .01 * (i + 1))
    ids = [f'calibration-{i}' for i in range(72)]
    base_hash = old.control.state_hash(model.decoder)
    candidate_hash = old.control.state_hash(candidate.decoder)
    artifact = {
        'format': 'audiovae2_grail_hidden_initialization_v1',
        'variant': 'grail_shared_hidden', 'original_step0_sha256': old.STEP0_SHA,
        'teacher_checkpoint_sha256': old.base.CHECKPOINT_SHA256,
        'teacher_source_sha256': old.base.SOURCE_SHA256,
        'teacher_state_sha256': old.control.state_hash(teacher.model.decoder),
        'selection': deepcopy(selection), 'fit_source_ids': ids.copy(), 'ridge': 1e-6,
        'operators': {path: deepcopy(candidate.decoder.get_submodule(path).state_dict())
                      for path in old.OPERATOR_PATHS},
        'base_selected_state_sha256': base_hash, 'candidate_state_sha256': candidate_hash,
        'hidden_maps_sha256': 'hidden-maps-sha', 'hidden_maps_exported': False,
        'automatic_promotion': False,
    }
    receipt = {
        'operators_sha256': 'fixture-artifact-sha',
        'teacher_state_sha256': artifact['teacher_state_sha256'],
        'hidden_maps_sha256': 'hidden-maps-sha', 'hidden_maps_exported': False,
        'original_bias_bitwise_preserved': True,
        'fresh_factory_matches_original_step0': True, 'original_checkpoint_weights_installed': False,
        'base_selected_state_sha256': base_hash, 'candidate_state_sha256': candidate_hash,
        'changed_native_paths': list(old.OPERATOR_PATHS),
        'all9_residual_units_preserved': True, 'neural_training_updates': 0,
        'extra_inference_modules': 0, 'unchanged_frozen_and_unselected_tensors': True,
        'automatic_promotion': False,
    }
    zero = {'group': deepcopy(original.group_state_dict()),
            'optimizer': old.progressive.fresh_optimizer(original).state_dict(),
            'rng': old.screen.rng_state(), 'selection': original_selection,
            'cut_updates': 0, 'global_updates': 0, 'sources_seen': []}
    return teacher, model, candidate, artifact, receipt, ids, zero


def install(model, artifact, receipt, ids):
    return run.install_candidate_operators(model, artifact, receipt,
        artifact_sha256='fixture-artifact-sha', calibration_ids=ids)


def restore(items):
    teacher, model, _, artifact, receipt, ids, zero = items
    return run.restore_candidate_start(model, teacher, zero, artifact, receipt, ids,
        artifact_sha256='fixture-artifact-sha')


def test_portable_grail_installs_on_original_support_without_bias_or_other_writes(tmp_path):
    teacher, model, candidate, artifact, receipt, ids, zero = setup()
    assert model.selections == zero['selection']
    before, teacher_before = snapshot(model.decoder), snapshot(teacher)
    objects = {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    path = tmp_path / 'native.pt'
    torch.save(artifact, path)
    supplied = torch.load(path, weights_only=True, map_location='cpu')
    install(model, supplied, receipt, ids)
    assert_snapshot(model.decoder, snapshot(candidate.decoder))
    assert objects == {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    for name, value in model.decoder.state_dict().items():
        if not name.startswith(tuple(p + '.' for p in old.OPERATOR_PATHS)):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    selected = model.selections['stage2_indices']
    torch.testing.assert_close(model.decoder.model[3].block[2].block[0].alpha,
        teacher.model.decoder.model[3].block[2].block[0].alpha[:, selected], rtol=0, atol=0)
    torch.testing.assert_close(model.decoder.sr_cond_model[4].scale_embed.weight,
        teacher.model.decoder.sr_cond_model[4].scale_embed.weight[:, selected], rtol=0, atol=0)
    with torch.no_grad():
        z = torch.randn(1, 8, 2) * .05
        torch.testing.assert_close(model.forward_from_latents(z)['waveform'],
            candidate.forward_from_latents(z)['waveform'], rtol=0, atol=0)
    assert_snapshot(teacher, teacher_before)
    assert 'maps' not in supplied and not any('maps' in name for name in model.state_dict())
    for native in old.OPERATOR_PATHS:
        torch.testing.assert_close(model.decoder.get_submodule(native).bias,
            before[native+'.bias'], rtol=0, atol=0)


@pytest.mark.parametrize('damage', ['foreign_selection', 'adapted_start', 'late_nan',
    'shape', 'candidate_hash', 'base_hash', 'source_order', 'scope', 'teacher_checkpoint', 'ridge', 'bias', 'map_hash', 'exported_map', 'foreign_variant'])
def test_rejects_foreign_or_partial_artifact_before_mutation(damage):
    _, model, _, artifact, receipt, ids, zero = setup()
    path = old.OPERATOR_PATHS[-1]
    if damage == 'foreign_selection': artifact['selection']['stage2_indices'] = list(range(24))
    elif damage == 'adapted_start':
        with torch.no_grad(): model.decoder.model[5].block[4].block[3].bias.add_(.01)
    elif damage == 'late_nan': artifact['operators'][path]['bias'][0] = float('nan')
    elif damage == 'shape': artifact['operators'][path]['weight_v'] = artifact['operators'][path]['weight_v'][..., :-1]
    elif damage == 'candidate_hash': receipt['candidate_state_sha256'] = 'foreign'
    elif damage == 'base_hash': artifact['base_selected_state_sha256'] = 'foreign'
    elif damage == 'source_order': artifact['fit_source_ids'] = list(reversed(ids))
    elif damage == 'scope': artifact['operators']['model.5.block.1'] = {}
    elif damage == 'teacher_checkpoint': artifact['teacher_checkpoint_sha256'] = 'foreign'
    elif damage == 'ridge': artifact['ridge'] = 1e-3
    elif damage == 'bias':
        artifact['operators'][path]['bias'][0] += .01
        # Even a self-consistent candidate hash cannot authorize a changed bias.
        proposed = model.decoder.state_dict().copy()
        for native, state in artifact['operators'].items():
            proposed.update({native+'.'+key: value for key, value in state.items()})
        artifact['candidate_state_sha256'] = receipt['candidate_state_sha256'] = old._state_hash(proposed)
    elif damage == 'map_hash': receipt['hidden_maps_sha256'] = 'other-map'
    elif damage == 'exported_map': artifact['hidden_maps_exported'] = True
    else: artifact['variant'] = 'downstream_selection_B'
    before = snapshot(model.decoder)
    with pytest.raises(ValueError): install(model, artifact, receipt, ids)
    assert_snapshot(model.decoder, before)


def test_restore_ignores_poisoned_old_group_and_restores_fresh_adam_rng():
    items = setup()
    teacher, model, candidate, _, _, _, zero = items
    zero['group'] = {name: torch.full_like(value, 123.45) for name, value in zero['group'].items()}
    old_payload = deepcopy(zero)
    teacher_before = snapshot(teacher)
    assert any(not torch.equal(model.group_state_dict()[name], value)
               for name, value in zero['group'].items())
    torch.randn(31)
    optimizer, _ = restore(items)
    assert_snapshot(model.decoder, snapshot(candidate.decoder))
    assert_snapshot(teacher, teacher_before)
    assert old.replay.compare_tree(zero, old_payload)['equal']
    assert not optimizer.state
    assert old.replay.compare_tree(optimizer.state_dict(), zero['optimizer'])['equal']
    assert old.replay.compare_tree(old.screen.rng_state(), zero['rng'])['equal']
    params = old.base.parameters(model)
    assert len(params) == 90 and all(p.requires_grad for p in params)
    assert set(params) == {p for p in model.parameters() if p.requires_grad}
    assert model.selections == zero['selection']


@pytest.mark.parametrize('target', ['artifact', 'receipt'])
def test_teacher_state_mismatch_rejected_without_candidate_installation(target):
    items = setup()
    teacher, model, _, artifact, receipt, _, _ = items
    (artifact if target == 'artifact' else receipt)['teacher_state_sha256'] = 'different-teacher'
    before, teacher_before = snapshot(model.decoder), snapshot(teacher)
    with pytest.raises(ValueError): restore(items)
    assert_snapshot(model.decoder, before)
    assert_snapshot(teacher, teacher_before)


def test_another_same_width_teacher_support_cannot_replace_original_pivoted_support():
    items = list(setup())
    changed = {'stage2_indices': list(range(24)), 'stage3_indices': list(range(16))}
    model = old.progressive.initialize_from_teacher(items[0].model.decoder, changed)
    items[1] = model
    items[3]['selection'] = changed
    before = snapshot(model.decoder)
    with pytest.raises(ValueError, match='exact original pivoted support'):
        restore(items)
    assert_snapshot(model.decoder, before)


@pytest.mark.parametrize('damage', ['trained_counter', 'existing_moments'])
def test_trained_template_cannot_start_an_independent_recovery(damage):
    items = setup()
    model, zero = items[1], items[-1]
    if damage == 'trained_counter':
        zero['cut_updates'] = zero['global_updates'] = 2000
    else:
        zero['optimizer']['state'] = {0: {'step': torch.tensor(2000.)}}
    before = snapshot(model.decoder)
    with pytest.raises(ValueError): restore(items)
    assert_snapshot(model.decoder, before)


def test_first_update_matches_existing_pooled12_objective_moments_and_frozen_state(monkeypatch):
    items = setup()
    teacher, model = items[:2]
    optimizer, _ = restore(items)
    oracle = old.progressive.initialize_from_teacher(teacher.model.decoder, model.selections)
    oracle.load_group_state_dict(model.group_state_dict())
    other = old.progressive.fresh_optimizer(oracle)
    batch = old.base.batch
    monkeypatch.setattr(old.base, 'batch', lambda rows: batch(rows, device='cpu'))
    @torch.no_grad()
    def forward(instance, z):
        return gm.teacher_trace(instance.model.decoder, z[:, :8])
    monkeypatch.setattr(old.base, 'teacher_forward', forward)
    crops = []
    for i in range(12):
        frames, context = 2 + i % 2, i % 2
        z = torch.randn(1, 64, frames) * (.001 if i % 3 else .05)
        crops.append({'source_id': f'crop-{i}', 'latents': z,
            'teacher_audio': forward(teacher, z)['waveform'].clone(),
            'context_frames': context, 'context_start_frame': 0, 'start_frame': context,
            'valid_scored_samples': (frames - context) * 1920 - i * 7})
    objective = old.base.ReconstructionV2(old.base.ReconstructionV2Config(
        fft_sizes=(32, 64), mel_bands=(4, 8)))
    before, frozen = snapshot(teacher), old.screen.frozen_versions(model)
    rng = old.screen.rng_state()
    expected = old.screen.training_update(oracle, teacher, crops, 'current', objective,
        old.prior.COEFFICIENTS, other, record_diagnostics=True)
    old.screen.restore_rng(rng)
    actual, checks = run.perform_update(model, teacher, crops, objective, optimizer, diagnostics=True)
    assert actual == expected and len(checks) == 12
    assert all(c['allclose_original_tolerance'] for c in checks)
    assert_snapshot(model.decoder, snapshot(oracle.decoder))
    assert_snapshot(teacher, before)
    assert old.replay.compare_tree(optimizer.state_dict(), other.state_dict())['equal']
    assert old.screen.frozen_versions(model) == frozen
    assert len(optimizer.state) == 90 and all(float(s['step']) == 1 for s in optimizer.state.values())


def test_checkpoint_keeps_original_support_exact_24k_prefix_and_2000_bound(tmp_path):
    items = setup()
    model = items[1]
    optimizer, _ = restore(items)
    sources = [f'source-{i}' for i in range(24000)]
    identity = {'source_ids': sources, 'source_ids_sha256': old.screen.digest(sources),
                'selection': model.selections, 'schedule_sha256': 'reference-schedule'}
    run.save_checkpoint(tmp_path / 'zero.pt', model, optimizer, 0, [], identity, model.selections, 0)
    for parameter in old.base.parameters(model):
        optimizer.state[parameter] = {'step': torch.tensor(2000.),
            'exp_avg': torch.full_like(parameter, .02), 'exp_avg_sq': torch.full_like(parameter, .03)}
    before, rng = deepcopy(optimizer.state_dict()), old.screen.rng_state()
    run.save_checkpoint(tmp_path / 'final.pt', model, optimizer, 2000, sources,
                        identity, model.selections, 24000)
    saved = torch.load(tmp_path / 'final.pt', weights_only=True)
    assert saved['format'] == run.VERSION and saved['step'] == 2000
    assert saved['selection'] == model.selections == items[-1]['selection']
    assert saved['sources_seen'] == sources and len(set(saved['sources_seen'])) == 24000
    assert saved['coefficients'] == old.prior.COEFFICIENTS and saved['accumulation'] == 12
    assert old.replay.compare_tree(saved['optimizer'], before)['equal']
    assert old.replay.compare_tree(saved['rng'], rng)['equal']
    assert not saved['optimizer_reset_after_start'] and not saved['automatic_promotion']
    with pytest.raises(FileExistsError):
        run.save_checkpoint(tmp_path / 'final.pt', model, optimizer, 2000, sources,
                            identity, model.selections, 24000)
    with pytest.raises(ValueError):
        run.save_checkpoint(tmp_path / 'too_far.pt', model, optimizer, 2500, sources,
                            identity, model.selections, 24000)
    with pytest.raises(ValueError):
        run.save_checkpoint(tmp_path / 'wrong_order.pt', model, optimizer, 2000,
                            sources[::-1], identity, model.selections, 24000)
    optimizer.param_groups[0]['lr'] = 1e-4
    with pytest.raises(ValueError, match='optimizer settings'):
        run.save_checkpoint(tmp_path / 'wrong_recipe.pt', model, optimizer, 2000,
                            sources, identity, model.selections, 24000)


def test_monitor_uses_fresh_grail_history_matched_references_and_2000_budget(tmp_path):
    import grail_candidate_monitor as monitoring
    from joint_recovery_gates_v2 import summarize_regions

    class Writer:
        def __init__(self, path): self.scalars = []; self.text = []; self.layout = None
        def add_scalar(self, tag, value, step): self.scalars.append((tag, value, step))
        def add_text(self, tag, value, step): self.text.append((tag, value, step))
        def add_custom_scalars(self, layout): self.layout = layout
        def flush(self): pass
        def close(self): pass

    windows = []
    for i, (start, teacher_rms, zero) in enumerate(((0, 9e-6, True), (960, 6e-4, True),
            (1920, 9e-6, True), (48000, 9e-6, False), (48960, 3e-4, False))):
        windows.append({'window_id': str(i), 'source_id': 'fixture', 'source_start_sample': start,
            'source_stop_sample': start+960, 'valid_samples': 960, 'is_quiet': True,
            'teacher_rms': teacher_rms, 'student_rms': teacher_rms,
            'residual_limit': max(.02**.5*teacher_rms, 1e-5),
            'output_rms_limit': max(10**.05*teacher_rms, 1e-5),
            'source_reference_exact_zero': zero, 'failure_category': 'passed',
            'residual_square_sum': 4e-6**2*960, 'centered_residual_square_sum': 3e-6**2*960})
    report = {'aggregate': {'sources': 1, 'samples': 4800, 'mae': .02, 'mel': .5,
        'group_mse': .1, 'quiet_residual_rms_mean': 4e-6, 'nonquiet_cosine_mean': .875,
        'peak_abs_max': .75, 'quiet_windows': 5, 'quiet_failed_windows': 0, 'overshoot_samples': 0},
        'rows': [{'source_id': 'fixture'}], 'quiet_regions': summarize_regions(windows)}
    before, rng = deepcopy(report), old.screen.rng_state()
    monitor = monitoring.RecoveryMonitor(tmp_path/'events', writer_factory=Writer)
    monitor.log_validation(report, 0, report, report)
    for step in range(1, 2001):
        monitor.log_training({'step': step, 'total': 1., 'waveform': .1, 'mel': .2,
                             'feature': .3, 'unique_sources': step*12}, step)
        if step in run.REVIEW_STEPS:
            monitor.log_validation(report, step, None if step == 250 else report, report)
    writers = list(monitor.writers.items())
    progress = next(writer.scalars for name, writer in writers if '13 Training progress' in name)
    assert progress[0][1:] == (0., 0) and progress[-1][1:] == (100., 2000)
    assert progress[1000][1:] == (50., 1000)
    assert [step for tag, _, step in monitor.details.scalars if tag == 'matched_b/quality/mae'] == list(run.REVIEW_STEPS)
    assert [step for tag, _, step in monitor.details.scalars if tag == 'matched_original/quality/mae'] == [0, 500, 1000, 1500, 2000]
    assert all('GRAIL_shared_hidden_384x256' in name for name, _ in writers)
    assert report == before and old.replay.compare_tree(old.screen.rng_state(), rng)['equal']
    with pytest.raises(ValueError): monitor.log_training({'step': 2001}, 2001)
    with pytest.raises(ValueError): monitor.log_validation(report, 2000)
    monitor.close()
