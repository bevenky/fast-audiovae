"""Sealed B composition and refitting its actual chain without A transplantation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'convnext'))
import combined_startup_init as combined
from test_group_model import TinyDecoder, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def cpu_policy():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(633); torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def fixture():
    decoder = TinyDecoder().eval().requires_grad_(False)
    teacher = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    selection = {'stage2_indices': list(range(24)), 'stage3_indices': list(range(16))}
    raw = combined.progressive.initialize_from_teacher(decoder, selection)
    b = combined.progressive.initialize_from_teacher(decoder, selection)
    with torch.no_grad():
        for i, path in enumerate(combined.PATHS):
            module = b.decoder.get_submodule(path)
            module.weight_g.mul_(1+.04*(i+1)); module.bias.add_(.01*(i+1))
    ids = ['fit-'+str(i) for i in range(72)]
    zero = {'selection': selection, 'group': deepcopy(raw.group_state_dict())}
    artifact = {'format': combined.b_recovery.INIT_VERSION, 'variant': 'B',
        'base_step0_sha256': combined.audit.STEP0_SHA, 'selection': selection,
        'fit_source_ids': ids, 'ridge': 1e-6,
        'operators': {path: deepcopy(b.decoder.get_submodule(path).state_dict()) for path in combined.PATHS}}
    receipt = {'operators_sha256': 'fixture-sha', 'candidate_state_sha256': combined.control.state_hash(b.decoder),
        'changed_native_paths': list(combined.PATHS), 'unchanged_frozen_and_unselected_tensors': True,
        'all9_residual_units_preserved': True, 'neural_training_updates': 0,
        'extra_inference_modules': 0, 'automatic_promotion': False,
        'original_step0_pristine': {'passed': True, 'student_state_sha256': combined.control.state_hash(raw.decoder),
                                   'teacher_state_sha256': combined.control.state_hash(decoder)}}
    return teacher, raw, b, zero, artifact, receipt, ids


def initialize(parts):
    teacher, _, _, zero, artifact, receipt, ids = parts
    return combined.initialize_b(teacher.model.decoder, zero, artifact, receipt, ids, artifact_sha256='fixture-sha')[0]


def test_combined_starts_from_exact_sealed_b_and_exports_four_operators_without_aliases():
    parts = fixture(); teacher, raw, b, zero, artifact, receipt, ids = parts
    raw_before = snapshot(raw); teacher_before = snapshot(teacher.model.decoder)
    rng = torch.get_rng_state().clone()
    model = initialize(parts)
    assert_snapshot(model, snapshot(b))
    assert_snapshot(raw, raw_before); assert_snapshot(teacher.model.decoder, teacher_before)
    assert torch.equal(rng, torch.get_rng_state())
    payload = combined.export_artifact(model, ids, 'fixture-sha')
    assert payload['variant'] == 'combined_B_startup' and set(payload['operators']) == set(combined.PATHS)
    assert payload['b_operators_sha256'] == 'fixture-sha'
    for path, state in payload['operators'].items():
        for name, value in state.items():
            actual = model.decoder.get_submodule(path).state_dict()[name]
            assert torch.equal(value, actual) and value.data_ptr() != actual.data_ptr()


@pytest.mark.parametrize('damage', ['teacher', 'adapted_start', 'wrong_b_hash'])
def test_foreign_teacher_adapted_start_or_wrong_b_state_is_rejected_without_touching_originals(damage):
    parts = list(fixture()); teacher, raw, b, zero, artifact, receipt, _ = parts
    if damage == 'teacher': receipt['original_step0_pristine']['teacher_state_sha256'] = 'foreign'
    elif damage == 'adapted_start':
        first = next(iter(zero['group'])); zero['group'][first].add_(.03)
    else: receipt['candidate_state_sha256'] = 'foreign'
    raw_before = snapshot(raw); teacher_before = snapshot(teacher.model.decoder); b_before = snapshot(b)
    with pytest.raises(ValueError): initialize(parts)
    assert_snapshot(raw, raw_before); assert_snapshot(b, b_before); assert_snapshot(teacher.model.decoder, teacher_before)


def test_refit_uses_changed_b_inputs_preserves_stage2_and_roundtrips_full_combined_native_state():
    parts = fixture(); teacher, raw, b, zero, artifact, receipt, ids = parts
    model = initialize(parts); site = combined.audit.four_sites(teacher.model.decoder, model.selections)[-1]
    module = model.decoder.get_submodule(site.path)
    desired = combined.group.effective_weight(module).detach().clone() * 1.01
    desired_bias = module.bias.detach().clone() + .002
    crops = [{'source_id': str(i), 'z': torch.randn(1, 8, 3)*.1} for i in range(2)]
    before = snapshot(model); b_before = snapshot(b); teacher_before = snapshot(teacher.model.decoder)
    objects = {n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    def captured(student, crop):
        with torch.no_grad(), combined.audit.capture_sites(student.decoder, [site]) as values:
            student.forward_from_latents(crop['z'])
        return values[site.name]
    raw_inputs = [captured(raw, crop)['input'].clone() for crop in crops]
    seen = []
    def observe(student, original, crop, actual_site):
        assert actual_site.path == site.path and original is teacher
        value = captured(student, crop); x = value['input']; length = value['output'].shape[-1]
        target = combined.common.linear_response(module, x, desired, False) + desired_bias[None,:,None]
        mask = torch.zeros(1, 1, length, dtype=torch.bool); mask[..., :10] = True
        seen.append((crop['source_id'], x.clone()))
        return {'source_id': crop['source_id'], 'input': x, 'target': target,
                'current_output': value['output'], 'weights': torch.full((1,1,length),8), 'constraint_mask': mask}
    report = combined.refit_b_upsampler(model, teacher, crops, observe_fn=observe)
    assert report['feasible'] and report['native_fp32_constraints_passed'] and report['native_fold_allclose']
    assert [source for source,_ in seen] == ['0','1','0','1']
    assert all(not torch.equal(seen[i][1], raw_inputs[i]) for i in range(2))
    assert all(torch.equal(seen[i][1], seen[i+2][1]) for i in range(2))
    after = snapshot(model)
    changed = [name for name in after if not torch.equal(before[name],after[name])]
    assert changed and all(name.startswith('decoder.'+site.path+'.') for name in changed)
    assert objects == {n:(id(p),p.data_ptr()) for n,p in model.named_parameters()}
    assert_snapshot(b,b_before); assert_snapshot(teacher.model.decoder,teacher_before)
    payload = combined.export_artifact(model, ids, 'fixture-sha')
    for path in combined.PATHS[:3]:
        assert all(torch.equal(value,artifact['operators'][path][name]) for name,value in payload['operators'][path].items())
    reconstructed = combined.progressive.initialize_from_teacher(teacher.model.decoder, zero['selection'])
    reconstructed.load_group_state_dict(zero['group'])
    for path,state in payload['operators'].items(): reconstructed.decoder.get_submodule(path).load_state_dict(state)
    assert combined.control.state_hash(reconstructed.decoder) == combined.control.state_hash(model.decoder)


def test_infeasible_combined_fit_keeps_complete_sealed_b_state():
    parts = fixture(); teacher = parts[0]; model = initialize(parts); before = snapshot(model)
    site = combined.audit.four_sites(teacher.model.decoder,model.selections)[-1]
    def observe(student, original, crop, actual_site):
        x = torch.zeros(1,24,3); target = torch.zeros(1,16,15)
        return {'source_id': crop['source_id'], 'input': x, 'target': target,
                'current_output': target, 'weights': torch.full((1,1,15),8),
                'constraint_mask': torch.zeros(1,1,15,dtype=torch.bool)}
    report = combined.refit_b_upsampler(model,teacher,[{'source_id':'no-startup'}],observe_fn=observe)
    assert not report['feasible'] and report['status'] == 'no_stable_feasible_solution'
    assert report['stage2_b_operators_preserved']
    assert_snapshot(model,before)
