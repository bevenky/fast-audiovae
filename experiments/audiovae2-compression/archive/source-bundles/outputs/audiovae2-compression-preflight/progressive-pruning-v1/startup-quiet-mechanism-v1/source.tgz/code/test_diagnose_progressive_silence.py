"""CPU accounting and restoration controls for the current single width cut."""
from __future__ import annotations

from pathlib import Path
import copy
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))

import diagnose_progressive_silence as audit
import group_model as gm
import progressive_model as progressive
from test_group_model import TinyDecoder, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(764)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def teacher():
    decoder = TinyDecoder().double().eval().requires_grad_(False)
    # Nonzero, unequal biases expose accidental double-addition of bias/skip.
    with torch.no_grad():
        for stage in (3, 4):
            for unit in decoder.model[stage].block[2:]:
                bias = unit.block[3].bias
                bias.copy_(torch.linspace(.013, .051, bias.numel(), dtype=bias.dtype))
    return decoder


def selection():
    return {'stage2_indices': [i for i in range(32) if i % 4 != 1],
            'stage3_indices': list(range(16))}


def test_current_cut_has_only_three_stage2_mixers_and_one_stage3_input_site(teacher):
    selected = selection()
    sites = audit.four_sites(teacher, selected)
    assert [s.name for s in sites] == ['stage2_ru1', 'stage2_ru2', 'stage2_ru3', 'stage3_up']
    assert [s.path for s in sites] == [f'model.3.block.{i}.block.3' for i in (2, 3, 4)] + ['model.4.block.1']
    assert all(s.kept_inputs == selected['stage2_indices'] for s in sites)
    assert [s.kept_outputs for s in sites] == [selected['stage2_indices']] * 3 + [list(range(16))]
    assert [(s.rate, s.stride) for s in sites] == [(1200, 1)] * 3 + [(6000, 5)]
    assert teacher.model[4].block[1].kernel_size == (10,)


@pytest.mark.parametrize('change', ['stage3_pruned', 'stage3_reordered', 'stage2_full', 'duplicate'])
def test_four_site_contract_rejects_other_cuts(teacher, change):
    selected = selection()
    if change == 'stage3_pruned': selected['stage3_indices'].pop()
    elif change == 'stage3_reordered': selected['stage3_indices'].reverse()
    elif change == 'stage2_full': selected['stage2_indices'] = list(range(32))
    else: selected['stage2_indices'][0] = selected['stage2_indices'][1]
    with pytest.raises((ValueError, RuntimeError)):
        audit.four_sites(teacher, selected)


def prepared(teacher):
    selected = selection()
    student = progressive.initialize_from_teacher(teacher, selected)
    sites = audit.four_sites(teacher, selected)
    z = torch.randn(1, 8, 3, dtype=torch.float64) * .1
    with torch.no_grad(), audit.common.capture_operations(teacher, sites) as captured:
        native = teacher(z)
    return student, sites, z, captured, native


def test_all_four_initial_restorations_recover_native_output_without_double_bias_or_skip(teacher):
    student, sites, z, captured, native = prepared(teacher)
    pristine = audit.validate_pristine(teacher, student, selection())
    student.train()
    for p in student.trainable_group_parameters(): p.grad = torch.full_like(p, .25)
    before, teacher_before = snapshot(student), snapshot(teacher)
    gradients = {n: p.grad.clone() for n, p in student.group_named_parameters()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        plain = student.forward_from_latents(z)['waveform']
        with audit.restoration(student, teacher, captured, sites, [sites[0].name], pristine_report=pristine):
            partial = student.forward_from_latents(z)['waveform']
        with audit.restoration(student, teacher, captured, sites, [s.name for s in sites], pristine_report=pristine):
            restored = student.forward_from_latents(z)['waveform']
        after = student.forward_from_latents(z)['waveform']
    assert (plain-native).abs().max() > 1e-7
    assert (partial-native).abs().max() > 1e-7
    torch.testing.assert_close(restored, native, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(after, plain, rtol=0, atol=0)
    assert_snapshot(student, before)
    assert_snapshot(teacher, teacher_before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert student.training and not teacher.training
    for name, p in student.group_named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in student.modules())
    assert all(not m._forward_hooks for m in teacher.modules())


def test_restoration_rejects_adapted_weights_and_stale_pristine_receipt(teacher):
    student, sites, z, captured, _ = prepared(teacher)
    pristine = audit.validate_pristine(teacher, student, selection())
    with torch.no_grad(): student.decoder.model[3].block[2].block[3].bias.add_(.2)
    before = snapshot(student)
    with pytest.raises((ValueError, RuntimeError)):
        audit.validate_pristine(teacher, student, selection())
    with pytest.raises((ValueError, RuntimeError)):
        with audit.restoration(student, teacher, captured, sites, [s.name for s in sites], pristine_report=pristine):
            student.forward_from_latents(z)
    assert_snapshot(student, before)
    assert all(not m._forward_hooks for m in student.modules())


def test_restoration_hooks_are_removed_when_forward_fails(teacher):
    student, sites, _, captured, _ = prepared(teacher)
    pristine = audit.validate_pristine(teacher, student, selection())
    before, teacher_before = snapshot(student), snapshot(teacher)
    with pytest.raises(RuntimeError, match='intentional failure'):
        with audit.restoration(student, teacher, captured, sites, [s.name for s in sites], pristine_report=pristine):
            raise RuntimeError('intentional failure')
    assert_snapshot(student, before)
    assert_snapshot(teacher, teacher_before)
    assert all(not m._forward_hooks for m in student.modules())


def test_four_site_accounting_separates_dropped_inputs_retained_drift_and_residual_skips(teacher):
    student = progressive.initialize_from_teacher(teacher, selection())
    sites = audit.four_sites(teacher, selection())
    z = torch.randn(1, 8, 3, dtype=torch.float64) * .1
    with torch.no_grad(), audit.capture_sites(teacher, sites) as tc:
        teacher(z)
    with torch.no_grad(), audit.capture_sites(student.decoder, sites) as sc:
        student.forward_from_latents(z)
    for site in sites:
        terms = audit.decompose_site(site, teacher, student, tc, sc)
        torch.testing.assert_close(terms['kept'] + terms['dropped'] + terms['bias'],
                                   tc[site.name]['output'][:, site.kept_outputs], rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(terms['dropped'] + terms['retained_drift'],
                                   terms['gap'], rtol=1e-9, atol=1e-11)
        assert terms['dropped'].abs().max() > 1e-7
        if site.name != 'stage2_ru1':
            assert terms['retained_drift'].abs().max() > 1e-7
        if site.stride == 1:
            full = audit.residual_terms(site, terms, tc, sc)
            torch.testing.assert_close(full['kept'] + full['dropped'] + full['bias'],
                                       full['teacher_selected'], rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(terms['dropped'] + terms['retained_drift'] + full['skip_drift'],
                                       full['gap'], rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(full['reconstructed_gap'], full['gap'], rtol=1e-9, atol=1e-11)
            if site.name != 'stage2_ru1': assert full['skip_drift'].abs().max() > 1e-7
        else:
            torch.testing.assert_close(terms['dropped_current'] + terms['dropped_previous'],
                                       terms['dropped'], rtol=1e-10, atol=1e-11)
            assert torch.count_nonzero(terms['dropped_previous'][..., :5]) == 0
            assert terms['dropped'].shape[-1] == tc[site.name]['input'].shape[-1] * 5
    assert all(not m._forward_hooks for m in teacher.modules())
    assert all(not m._forward_hooks for m in student.modules())


def test_absolute_window_mapping_keeps_source_startup_distinct_from_late_crop_start():
    startup = {'context_frames': 0, 'context_start_frame': 0, 'start_frame': 0,
               'valid_scored_samples': 973}
    assert audit.absolute_interval(startup, 0, 960) == {
        'tensor_start': 0, 'tensor_stop': 960, 'source_start': 0, 'source_stop': 960}
    assert audit.absolute_interval(startup, 960, 973)['source_stop'] == 973
    interior = {'context_frames': 2, 'context_start_frame': 20, 'start_frame': 22,
                'valid_scored_samples': 973}
    interval = audit.absolute_interval(interior, 960, 973)
    assert interval == {'tensor_start': 4800, 'tensor_stop': 4813,
                        'source_start': 43200, 'source_stop': 43213}
    assert interval['source_start'] >= 38400
    with pytest.raises(ValueError): audit.absolute_interval(interior, 960, 974)
    with pytest.raises(ValueError): audit.absolute_interval({**interior, 'start_frame': 0}, 0, 960)


def test_interval_masks_and_feature_weights_preserve_context_and_partial_cells():
    valid = torch.zeros(1, 1, 4000, dtype=torch.bool)
    valid[..., 1920:3883] = True
    all_mask = audit.interval_mask(valid, 0, 4000)
    assert torch.equal(all_mask, valid) and all_mask.data_ptr() != valid.data_ptr()
    part = audit.interval_mask(valid, 3840, 3960)
    assert int(part.sum()) == 43
    for samples_per_cell in (40, 8):
        weights = audit.common.cell_weights(part, 4000 // samples_per_cell)
        assert int(weights.sum()) == 43
        assert int(weights.max()) == samples_per_cell
        assert weights[..., 3840 // samples_per_cell].item() == samples_per_cell
        assert weights[..., 3880 // samples_per_cell].item() == 3
    assert not audit.interval_mask(valid, 0, 1920).any()
    with pytest.raises(ValueError): audit.interval_mask(valid, -1, 960)
    with pytest.raises(ValueError): audit.interval_mask(valid.float(), 0, 960)


def test_window_error_preserves_dc_and_ac_separately_and_avoids_zero_denominator_claims():
    target = torch.zeros(1, 1, 7, dtype=torch.float64)
    prediction = torch.tensor([[[99., 1., 3., 1., 3., -99., 99.]]], dtype=torch.float64)
    score = audit.window_summary(prediction, target, 1, 5)
    assert score['elements'] == 4
    assert score['dc_error'] == 2 and score['dc_mean_square'] == 4
    assert score['ac_mean_square'] == 1 and score['residual_mean_square'] == 5
    assert score['residual_mean_square'] == score['dc_mean_square'] + score['ac_mean_square']
    assert score['dc_energy_fraction'] == pytest.approx(.8)
    assert score['rms_ratio'] is None and score['least_squares_gain'] is None and score['cosine'] is None
    zero = audit.window_summary(target, target, 1, 5)
    assert zero['dc_energy_fraction'] is None
    assert zero['residual_mean_square'] == zero['dc_mean_square'] == zero['ac_mean_square'] == 0


def test_weighted_feature_dc_uses_each_channel_not_cancellation_of_global_means():
    tensor = torch.tensor([[[1., 3., 99.], [-1., -3., -99.]]], dtype=torch.float64)
    weights = torch.tensor([[[1, 3, 0]]])
    score = audit.common.weighted_tensor_summary(tensor, weights)
    assert score['weighted_samples'] == 4 and score['channels'] == 2
    assert score['mean'] == 0
    assert score['channel_mean_rms'] == pytest.approx(2.5)
    assert score['channel_centered_rms'] ** 2 == pytest.approx(.75)
    assert score['rms'] ** 2 == pytest.approx(7.)
    assert score['rms'] ** 2 == pytest.approx(score['channel_mean_rms'] ** 2 + score['channel_centered_rms'] ** 2)


def panel_fixture():
    crops = [dict(source_id='startup', context_frames=0, context_start_frame=0, start_frame=0,
                  valid_scored_samples=973),
             dict(source_id='interior', context_frames=2, context_start_frame=20, start_frame=22,
                  valid_scored_samples=973)]
    observed = [dict(source_id='startup', source_start_sample=0, source_stop_sample=960,
                     teacher_rms=5e-6, is_quiet=True),
                dict(source_id='interior', source_start_sample=43200, source_stop_sample=43213,
                     teacher_rms=.02, is_quiet=False)]
    rows = [dict(source_id=w['source_id'], source_start_sample=w['source_start_sample'],
                 source_stop_sample=w['source_stop_sample'], roles=['fixed'], teacher_rms=999.)
            for w in observed]
    return {'version': 'progressive_silence_panel_v1', 'rows': rows}, crops, observed


def test_panel_resolver_preserves_exact_source_windows_and_uses_target_classification():
    panel, crops, observed = panel_fixture()
    original = copy.deepcopy((panel, crops, observed))
    result = audit.resolve_panel_rows(panel, crops, observed)
    assert [(r['source_id'], r['source_start_sample'], r['source_stop_sample']) for r in result] == [
        ('startup', 0, 960), ('interior', 43200, 43213)]
    assert [(r['tensor_start'], r['tensor_stop']) for r in result] == [(0, 960), (4800, 4813)]
    assert [(r['scored_start'], r['scored_stop']) for r in result] == [(0, 960), (960, 973)]
    assert [(r['teacher_rms'], r['is_quiet']) for r in result] == [(5e-6, True), (.02, False)]
    assert (panel, crops, observed) == original


@pytest.mark.parametrize('damage', ['duplicate', 'shifted', 'absent', 'missing_role'])
def test_panel_resolver_rejects_duplicates_shifted_windows_and_unregistered_selection(damage):
    panel, crops, observed = panel_fixture()
    if damage == 'duplicate': panel['rows'].append(copy.deepcopy(panel['rows'][0]))
    elif damage == 'shifted': panel['rows'][1]['source_start_sample'] += 1
    elif damage == 'absent': panel['rows'][1]['source_id'] = 'unregistered-source'
    else: panel['rows'][1]['roles'] = []
    with pytest.raises(ValueError): audit.resolve_panel_rows(panel, crops, observed)
