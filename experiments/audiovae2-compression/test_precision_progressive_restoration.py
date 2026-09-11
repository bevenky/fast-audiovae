"""Bounded proof that FP64 forwards preserve the actual FP32 operators."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'convnext'))
import precision_progressive_restoration as proof
from test_group_model import TinyDecoder, snapshot, assert_snapshot


@pytest.fixture(autouse=True)
def deterministic_cpu():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(396)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def fixture():
    teacher = TinyDecoder().eval().requires_grad_(False)
    selection = {'stage2_indices': [i for i in range(32) if i % 4 != 1],
                 'stage3_indices': list(range(16))}
    student = proof.audit.progressive.initialize_from_teacher(teacher, selection)
    return teacher, student, selection


def test_actual_fp32_coefficients_are_materialized_before_cast_without_renormalization():
    teacher, student, selection = fixture()
    originals = (snapshot(teacher), snapshot(student))
    rng = torch.get_rng_state().clone()
    teacher64, student64, receipt = proof.fold_existing_pair(teacher, student, selection)
    assert receipt['all_fp32_coefficients_preserved_exactly']
    assert receipt['student_resliced_in_fp64'] is False
    differs_from_renormalizing = False
    for original, folded in ((teacher, teacher64), (student.decoder, student64.decoder)):
        for name, layer in original.named_modules():
            if not isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)): continue
            actual = folded.get_submodule(name)
            expected = proof.group.effective_weight(layer).detach().double()
            assert actual.weight.dtype == torch.float64
            assert torch.equal(actual.weight, expected)
            assert torch.equal(actual.bias, layer.bias.double())
            assert not hasattr(actual, 'weight_g') and not hasattr(actual, 'weight_v')
            assert proof.group._legacy_weight_hook(actual) is None
            if hasattr(layer, 'weight_v'):
                v = layer.weight_v.detach().double()
                renormalized = v * (layer.weight_g.detach().double() / v.norm(dim=(1, 2), keepdim=True))
                differs_from_renormalizing |= not torch.equal(expected, renormalized)
        assert not any(p.requires_grad for p in folded.parameters())
    assert differs_from_renormalizing, 'The fixture must distinguish casting from recomputing normalization in FP64'
    assert_snapshot(teacher, originals[0]); assert_snapshot(student, originals[1])
    assert torch.equal(torch.get_rng_state(), rng)


def test_original_step0_guard_and_materialized_pair_token_reject_adaptation():
    teacher, student, selection = fixture()
    teacher64, student64, _ = proof.fold_existing_pair(teacher, student, selection)
    with torch.no_grad(): student.decoder.model[3].block[2].block[3].bias.add_(.01)
    before = snapshot(student)
    with pytest.raises(ValueError, match='pristine'):
        proof.fold_existing_pair(teacher, student, selection)
    assert_snapshot(student, before)
    with torch.no_grad(): student64.decoder.model[3].block[2].block[3].bias.add_(.01)
    before64 = snapshot(student64)
    with pytest.raises(ValueError, match='unchanged'):
        proof.precision_forward(teacher64, student64, torch.zeros(1, 8, 2, dtype=torch.float64), selection)
    assert_snapshot(student64, before64)
    assert all(not m._forward_hooks for m in student64.modules())
    assert all(not m._forward_hooks for m in teacher64.modules())


def test_precision_forward_closes_at_original_tolerance_and_reports_inherited_coefficient_differences():
    teacher, student, selection = fixture()
    teacher64, student64, _ = proof.fold_existing_pair(teacher, student, selection)
    state = snapshot(teacher64), snapshot(student64)
    z = (torch.randn(1, 8, 3) * .1).double()
    report = proof.precision_forward(teacher64, student64, z, selection)
    assert report['all_boundaries_original_tolerance']
    assert set(report['restored_local_mixers']) == {'stage2_ru1', 'stage2_ru2', 'stage2_ru3', 'stage3_up'}
    measured_weight_difference = False
    for name, entry in report['actual_selected_weight_differences'].items():
        assert entry['bias']['exact']
        measured_weight_difference |= not entry['weight']['exact']
        assert entry['weight']['allclose_existing']
    assert measured_weight_difference, 'FP32 WN slicing roundoff must remain visible, not be repaired by a new slice'
    for result in report['boundaries'].values():
        assert result['existing_tolerance_failed_elements'] == 0
        assert result['tight_atol'] == result['tight_rtol'] == 1e-10
    # Tight FP64 equality is reported independently, never required by this test:
    # the inherited FP32 operator differences have deliberately been retained.
    assert isinstance(report['all_boundaries_tight_fp64'], bool)
    assert_snapshot(teacher64, state[0]); assert_snapshot(student64, state[1])
    assert all(not m._forward_hooks for m in teacher64.modules())
    assert all(not m._forward_hooks for m in student64.modules())
