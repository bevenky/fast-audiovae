"""Analytic reduction checks and tiny full-decoder diagnostic integration."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))

import boundary_diagnostics as diagnostics
import group_model as gm
from test_group_model import TinyDecoder


@pytest.fixture(autouse=True)
def cpu_determinism():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(713)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def test_weighted_feature_cells_use_partial_sample_counts_and_exclude_nonfinite_padding():
    teacher = torch.tensor([[[1.,2.,float("nan")],[3.,4.,float("inf")]]])
    student = torch.tensor([[[2.,0.,float("inf")],[2.,5.,float("nan")]]])
    valid = torch.tensor([[[1,1,1,1,1,1,0,0,0,0,0,0]]],dtype=torch.bool)
    stats = diagnostics.weighted_statistics(student,teacher,valid,12000)
    assert stats["error_square_sum"] == 18
    assert stats["teacher_square_sum"] == 80
    assert stats["student_square_sum"] == 82
    assert stats["dot_sum"] == 72
    assert stats["valid_audio_samples"] == 6
    assert stats["weighted_feature_elements"] == 12
    assert stats["valid_feature_cells"] == 4
    assert stats["partial_feature_cells"] == 2
    metrics = diagnostics.summarize_statistics(stats)
    assert metrics["mse"] == 1.5
    assert metrics["nrmse"] == pytest.approx(math.sqrt(18/80))
    assert metrics["cosine"] == pytest.approx(72/math.sqrt(80*82))
    assert metrics["max_abs_error"] == 2


def test_region_masks_keep_teacher_grid_holes_and_short_tail():
    target = torch.cat([torch.zeros(960),torch.full((960,),.05),torch.full((7,),.0005)]).reshape(1,1,-1)
    valid = torch.ones_like(target,dtype=torch.bool)
    valid[...,:10] = False
    masks, config = diagnostics.region_masks(target,valid)
    assert config["window_samples"] == 960
    assert int(masks["quiet"].sum()) == 957
    assert int(masks["active"].sum()) == 960
    assert torch.equal(masks["all"],masks["quiet"]|masks["active"])
    assert not bool((masks["quiet"]&masks["active"]).any())


def test_empty_and_zero_energy_regions_are_explicit():
    target = torch.zeros(1,2,3)
    student = torch.ones_like(target)
    empty = diagnostics.summarize_statistics(diagnostics.weighted_statistics(student,target,torch.zeros(1,1,3,dtype=torch.bool),48000))
    assert empty["weighted_feature_elements"] == 0
    assert empty["mse"] is None and empty["cosine"] is None
    full = diagnostics.summarize_statistics(diagnostics.weighted_statistics(student,target,torch.ones(1,1,3,dtype=torch.bool),48000))
    assert full["rmse"] == 1
    assert full["nrmse"] is None and full["cosine"] is None


def test_source_aggregate_pools_counts_instead_of_source_means():
    one = diagnostics.weighted_statistics(torch.ones(1,1,1)*3,torch.ones(1,1,1),torch.ones(1,1,1,dtype=torch.bool),48000)
    three = diagnostics.weighted_statistics(torch.ones(1,1,3)*2,torch.ones(1,1,3),torch.ones(1,1,3,dtype=torch.bool),48000)
    merged = diagnostics.summarize_statistics(diagnostics.merge_statistics([one,three]))
    assert merged["mse"] == 7/4
    assert merged["mae"] == 5/4
    assert merged["nrmse"] == pytest.approx(math.sqrt(7/4))


@pytest.mark.parametrize("rate,mask", [(123,torch.ones(1,1,12,dtype=torch.bool)), (12000,torch.ones(1,1,11,dtype=torch.bool)), (12000,torch.ones(1,1,12))])
def test_misaligned_or_nonboolean_masks_are_rejected(rate,mask):
    with pytest.raises(ValueError):
        diagnostics.weighted_statistics(torch.ones(1,2,3),torch.ones(1,2,3),mask,rate)


def test_extended_teacher_trace_preserves_exact_execution_and_boundary_aliases():
    decoder = TinyDecoder().eval().requires_grad_(False)
    z = torch.randn(1,8,2)
    with torch.no_grad():
        trace = gm.teacher_trace(decoder,z)
        direct = decoder(z)
    assert torch.equal(trace["waveform"],direct)
    assert trace["stage1_output"] is trace["group_input"]
    assert trace["stage4_output"] is trace["group_output"]
    assert torch.equal(torch.tanh(trace["pre_tanh"]),trace["waveform"])
    assert [trace[f"stage{i}_output"].shape[-1] for i in range(1,7)] == [16,96,480,960,1920,3840]


@pytest.mark.parametrize("narrow", [False,True])
def test_boundary_panel_is_read_only_and_marks_partial_internal_coordinates(narrow):
    decoder = TinyDecoder().eval().requires_grad_(False)
    teacher = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    selection = {"stage2_indices":list(range(16 if narrow else 32)),"stage3_indices":list(range(8 if narrow else 16))}
    student = gm.build_student(decoder,**selection)
    crops = [{"source_id":str(i),"z":torch.randn(1,8,frames)} for i,frames in enumerate((1,2))]
    with torch.no_grad():
        for crop in crops:
            crop["target"] = decoder(crop["z"])
            crop["valid"] = torch.ones_like(crop["target"],dtype=torch.bool)
            crop["valid"][...,-7:] = False
    seen=[]
    def batch(rows):
        assert len(rows)==1
        crop=rows[0]
        return crop["z"],crop["target"],crop["valid"],[(0,crop["target"].shape[-1]-7)]
    def teacher_forward(model,z):
        assert not torch.is_grad_enabled()
        seen.append(z.data_ptr())
        return gm.teacher_trace(model.model.decoder,z)
    state={name:value.clone() for name,value in student.state_dict().items()}
    grads={name:torch.randn_like(param) for name,param in student.named_parameters() if param.requires_grad}
    for name,param in student.named_parameters():
        if name in grads:param.grad=grads[name].clone()
    modes={name:module.training for name,module in student.named_modules()}
    rng=torch.get_rng_state().clone()
    report=diagnostics.evaluate_boundaries(teacher,student,crops,selection,batch,teacher_forward)
    assert seen == [crop["z"].data_ptr() for crop in crops]
    assert report["measurement_only"] and report["batch_size"]==1
    assert report["source_ids"]==["0","1"]
    for stage in diagnostics.STAGE_RATES:
        assert set(report["aggregate"][stage])=={"all","quiet","active"}
        assert report["aggregate"][stage]["all"]["valid_audio_samples"]==5746
        scope=report["comparisons"][stage]["scope"]
        assert scope == ("selected_teacher_coordinates_only" if stage in ("stage2_output","stage3_output") else "shared_full_boundary")
        if not narrow or stage=="stage1_output":
            assert report["aggregate"][stage]["all"]["mse"] == 0
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in student.state_dict().items():assert torch.equal(value,state[name])
    for name,param in student.named_parameters():
        if name in grads:assert torch.equal(param.grad,grads[name])
    assert modes=={name:module.training for name,module in student.named_modules()}


def test_diagnostic_selection_mismatch_rejected_before_forward():
    decoder=TinyDecoder().eval().requires_grad_(False)
    student=gm.build_student(decoder,list(range(16)),list(range(8)))
    bad={"stage2_indices":list(reversed(range(16))),"stage3_indices":list(range(8))}
    def fail(*args):raise AssertionError("No forward should run")
    with pytest.raises(ValueError,match="selections differ"):
        diagnostics.evaluate_boundaries(None,student,[{"source_id":"one"}],bad,fail,fail)
