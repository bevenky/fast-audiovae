"""Fresh G native composition and constrained refit on its actual inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'convnext'))
from test_grail_candidate_recovery import setup, cpu_policy, old
from test_group_model import snapshot, assert_snapshot, gm
import grail_startup_init as init


def initialize(parts):
    teacher, _, _, artifact, receipt, ids, _ = parts
    return init.initialize_g(teacher.model.decoder, artifact, receipt, ids,
                             artifact_sha256='fixture-artifact-sha')[0]


def test_fresh_teacher_factory_installs_only_native_g_and_exports_independent_states(monkeypatch):
    parts = setup()
    teacher, raw, g, artifact, receipt, ids, zero = parts
    # A poisoned unrelated old checkpoint is never an input to the new helper.
    zero['group'] = {name: torch.full_like(value, 123.) for name, value in zero['group'].items()}
    original_teacher, original_g = snapshot(teacher), snapshot(g)
    rng = old.screen.rng_state()
    def forbidden(*args, **kwargs): raise AssertionError('Initializer attempted a checkpoint/group installation')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(type(raw), 'load_group_state_dict', forbidden)
    model = initialize(parts)
    assert_snapshot(model, original_g)
    assert_snapshot(teacher, original_teacher)
    assert_snapshot(g, original_g)
    assert old.replay.compare_tree(old.screen.rng_state(), rng)['equal']
    assert all(not p.requires_grad for p in teacher.parameters())
    payload = init.export_artifact(model, teacher.model.decoder, ids,
        g_sha256='fixture-artifact-sha', base_g_state_sha256=receipt['candidate_state_sha256'],
        base_selected_state_sha256=receipt['base_selected_state_sha256'])
    assert set(payload['operators']) == set(old.OPERATOR_PATHS)
    assert payload['variant'] != artifact['variant']
    assert payload['candidate_state_sha256'] == old.control.state_hash(model.decoder)
    metadata = {k: v for k, v in payload.items() if k != 'operators'}
    serialized = json.dumps(metadata, allow_nan=False)
    assert all(source not in serialized for source in ids)
    for path, states in payload['operators'].items():
        for name, value in states.items():
            native = model.decoder.get_submodule(path).state_dict()[name]
            assert torch.equal(value, native) and value.data_ptr() != native.data_ptr()


@pytest.mark.parametrize('damage', ['teacher', 'source_order', 'candidate_hash'])
def test_wrong_lineage_or_calibration_is_rejected_without_changing_teacher_or_g(damage):
    parts = setup()
    teacher, _, g, artifact, receipt, ids, _ = parts
    if damage == 'teacher': artifact['teacher_state_sha256'] = 'foreign-teacher'
    elif damage == 'source_order': artifact['fit_source_ids'] = list(reversed(ids))
    else: receipt['candidate_state_sha256'] = 'foreign-candidate'
    teacher_before, g_before = snapshot(teacher), snapshot(g)
    with pytest.raises(ValueError): initialize(parts)
    assert_snapshot(teacher, teacher_before); assert_snapshot(g, g_before)


def test_refit_uses_actual_g_inputs_preserves_three_mixers_and_native_up3_geometry():
    parts = setup()
    teacher, raw, g, artifact, receipt, ids, _ = parts
    model = initialize(parts)
    site = init.audit.four_sites(teacher.model.decoder, model.selections)[-1]
    module = model.decoder.get_submodule(site.path)
    assert module.stride == (5,) and module.kernel_size == (10,)
    weight = gm.effective_weight(module).detach().clone()
    assert tuple(weight.shape) == (24, 16, 10) and tuple(module.bias.shape) == (16,)
    desired, bias = weight * 1.01, module.bias.detach().clone() + .002
    crops = [{'source_id': str(i), 'z': torch.randn(1, 8, 3) * .1} for i in range(2)]
    before, teacher_before, g_before = snapshot(model), snapshot(teacher), snapshot(g)
    objects = {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    def capture(student, crop):
        with torch.no_grad(), init.audit.capture_sites(student.decoder, [site]) as trace:
            student.forward_from_latents(crop['z'])
        return trace[site.name]
    raw_inputs = [capture(raw, crop)['input'].clone() for crop in crops]
    seen = []
    def observe(student, original, crop, actual_site):
        assert original is teacher and actual_site.path == site.path
        trace = capture(student, crop)
        x, output = trace['input'], trace['output']
        assert output.shape[-1] == x.shape[-1] * 5
        target = init.common.linear_response(module, x, desired, False) + bias[None, :, None]
        weights = torch.full((1, 1, output.shape[-1]), 8)
        weights[..., :5] = 0
        weights[..., -1] = 3
        mask = torch.zeros_like(weights, dtype=torch.bool); mask[..., 5:10] = True
        target[..., :5] += 1000  # Context has no fit/equality weight and is not reset.
        seen.append(x.clone())
        return {'source_id': crop['source_id'], 'input': x, 'target': target,
                'current_output': output, 'weights': weights, 'constraint_mask': mask}
    report = init.refit_g_upsampler(model, teacher, crops, observe_fn=observe)
    assert report['feasible'] and report['native_fp32_constraints_passed'] and report['native_fold_allclose']
    assert report['stage2_g_operators_preserved']
    assert len(seen) == 4
    assert all(not torch.equal(seen[i], raw_inputs[i]) for i in range(2))
    assert all(torch.equal(seen[i], seen[i + 2]) for i in range(2))
    changed = [name for name, value in snapshot(model).items() if not torch.equal(before[name], value)]
    assert changed and all(name.startswith('decoder.' + site.path + '.') for name in changed)
    assert objects == {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    for path in old.OPERATOR_PATHS[:3]:
        for name, value in model.decoder.get_submodule(path).state_dict().items():
            assert torch.equal(value, artifact['operators'][path][name])
    assert_snapshot(teacher, teacher_before); assert_snapshot(g, g_before)


def test_fit_and_quality_exports_keep_pooled_numbers_and_exclude_sources_windows_and_tensors():
    from test_independent_trial_aggregate import quality
    equality = init.constrained.equality_metrics(torch.zeros(2), torch.zeros(2))
    full_fit = {'feasible': True, 'observations': 72,
        'source_ids': ['PRIVATE_FIT_SOURCE'], 'input': torch.zeros(3),
        'equality_before': equality, 'equality_fp64': equality,
        'equality_native_fp32': [deepcopy(equality) for _ in range(6)],
        'constraint_receipts': [{'source_id': 'PRIVATE_FIT_SOURCE',
            'windows': [{'window_id': 'PRIVATE_CALIB_WINDOW'}], 'observed_near_startup_samples': 960,
            'constraint_cells': 120, 'excluded_partial_cell_samples': 0} for _ in range(6)]}
    fit = init.summarize_fit(full_fit)
    assert fit['native_equalities']['windows'] == fit['native_equalities']['passed'] == 6
    assert fit['calibration_startup'] == {'windows': 6, 'samples': 5760,
        'native_constraint_cells': 720, 'excluded_partial_cell_samples': 0}
    full, windows = quality(1.)
    starts = [w for w in windows if w['source_start_sample'] == 0]
    for window in starts[13:]:
        window.update(source_start_sample=1920, source_stop_sample=2880)
    full['quiet_regions'] = init.aggregate.summarize_regions(windows)
    before = deepcopy(full)
    summary = init.summarize_quality(full)
    assert summary['regions']['near_startup_first20ms']['windows'] == 13
    assert summary['aggregate']['mae'] == full['aggregate']['mae']
    assert summary['active_rms_ratio'] == 1.
    serialized = json.dumps({'fit': fit, 'quality': summary}, allow_nan=False)
    assert 'PRIVATE' not in serialized and 'DO_NOT_EXPORT' not in serialized
    assert '987654321' not in serialized and 'source_ids' not in serialized
    assert full == before
