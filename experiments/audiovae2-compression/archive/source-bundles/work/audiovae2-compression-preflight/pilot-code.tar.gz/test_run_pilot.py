"""Small CPU checks for pilot pooling, masks and disposable calibration."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))
SPEC = importlib.util.spec_from_file_location("compression_pilot_under_test", HERE / "run_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    state = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31091)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


@pytest.fixture
def spectral():
    return pilot.ReconstructionV2(pilot.ReconstructionV2Config(fft_sizes=(32, 64), mel_bands=(4, 8)))


def features(z):
    h = z[:, :1].repeat_interleave(480, dim=-1)
    time = torch.arange(h.shape[-1], device=h.device, dtype=h.dtype)
    return h * (.75 + .2 * torch.sin(time * .27))[None, None]


def crop(source, frames=4, context=0, valid=None, amplitude=.1):
    z = torch.full((1, 64, frames), amplitude)
    target = .9 * features(z).repeat_interleave(4, dim=-1)
    return {"source_id": source, "start_frame": context, "context_start_frame": 0,
            "context_frames": context, "valid_scored_samples": valid or (frames-context)*1920,
            "latents": z, "teacher_audio": target}


class TinyGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.6))
        self.register_buffer("frozen_suffix_scale", torch.tensor(1.))

    def group_from_input(self, value):
        return self.gain * value

    def suffix_from_group(self, value):
        return value.repeat_interleave(4, dim=-1) * self.frozen_suffix_scale

    def group_named_parameters(self):
        return [("gain", self.gain)]

    def group_state_dict(self):
        return {"gain": self.gain.detach()}

    @torch.no_grad()
    def load_group_state_dict(self, state):
        self.gain.copy_(state["gain"])


@pytest.fixture
def cpu_pipeline(monkeypatch):
    real_batch = pilot.batch
    observed=[]
    monkeypatch.setattr(pilot, "batch", lambda crops: real_batch(crops, device="cpu"))
    def teacher_forward(teacher,z):
        observed.append(z.shape[0])
        return {"group_input":features(z),"group_output":.9*features(z),
                "waveform":.9*features(z).repeat_interleave(4,dim=-1)}
    monkeypatch.setattr(pilot, "teacher_forward",teacher_forward)
    return observed


def tensors(crops, model):
    z, target, valid, spans = pilot.batch(crops, device="cpu")
    ht = .9 * features(z)
    h = model.group_from_input(features(z))
    return model.suffix_from_group(h), target, h, ht, valid, spans


def test_batch_keeps_context_and_short_tail_counts_exact():
    c1 = crop("long", frames=5, context=2, valid=4101)
    c2 = crop("short", frames=4, context=0, valid=7000)
    z, target, valid, spans = pilot.batch([c1, c2], device="cpu")
    assert spans == [(3840, 7941), (0, 7000)]
    assert valid.sum().item() == 11101
    assert z.shape == (2, 64, 5) and target.shape == (2, 1, 9600)
    assert not valid[0, :, :3840].any() and not valid[0, :, 7941:].any()
    assert not valid[1, :, 7000:].any()
    assert torch.count_nonzero(z[1, :, 4:]) == 0
    assert torch.count_nonzero(target[1, :, 7680:]) == 0


def test_batch_cannot_count_padding_of_a_short_crop_as_valid():
    invalid = crop("short", frames=3, valid=5761)
    with pytest.raises(ValueError):
        pilot.batch([invalid, crop("long", frames=5)], device="cpu")


def test_feature_partial_cell_weight_and_zero_gradient_outside_mask(spectral):
    count = 192
    p = torch.zeros(1, 1, count, requires_grad=True)
    target = torch.zeros_like(p)
    h = torch.ones(1, 2, count//4, requires_grad=True)
    ht = torch.zeros_like(h, requires_grad=True)
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 3:100] = True
    result = pilot.losses(p,target,h,ht,valid,[(3,100)],spectral)
    torch.testing.assert_close(result["feature"], torch.tensor(1.))
    result["feature"].backward()
    expected = valid.reshape(1,1,count//4,4).sum(-1).expand_as(h) / 97
    torch.testing.assert_close(h.grad, expected)
    assert ht.grad is None


def test_invalid_feature_padding_nan_is_excluded_before_arithmetic(spectral):
    p = torch.zeros(1,1,128,requires_grad=True)
    h = torch.ones(1,2,32,requires_grad=True)
    ht = torch.zeros_like(h)
    with torch.no_grad():
        h[...,24:] = float("nan")
        ht[...,24:] = float("inf")
    valid = torch.zeros_like(p,dtype=torch.bool)
    valid[...,:96] = True
    value = pilot.losses(p,torch.zeros_like(p),h,ht,valid,[(0,96)],spectral)["feature"]
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(h.grad).all() and not h.grad[...,24:].any()


def test_waveform_and_feature_masks_cannot_disagree(spectral):
    p = torch.zeros(1,1,128)
    h = torch.zeros(1,2,32)
    valid = torch.ones_like(p,dtype=torch.bool)
    valid[...,17] = False
    with pytest.raises(ValueError,match="disagree"):
        pilot.losses(p,p,h,h,valid,[(0,128)],spectral)


def test_perfect_teacher_is_zero_for_all_objectives(spectral):
    p = torch.randn(2,1,256)*.01
    h = torch.randn(2,3,64)
    valid = torch.ones_like(p,dtype=torch.bool)
    values = pilot.losses(p,p,h,h,valid,[(0,256),(0,256)],spectral)
    assert all(float(v) == 0 for v in values.values())


def test_denominators_and_partitioned_gradients_match_one_pooled_batch(spectral):
    crops = [crop("a",frames=3,valid=4101,amplitude=.08),
             crop("b",frames=5,context=1,valid=7103,amplitude=.2),
             crop("c",frames=4,valid=7019,amplitude=.13)]
    model = TinyGroup()
    denom = pilot.reconstruction_denominators(crops,spectral)
    assert denom["samples"] == 18223
    expected_counts = tuple(sum(((c["valid_scored_samples"]-size)//(size//4)+1)*bands for c in crops)
                            for size,bands in zip(spectral.config.fft_sizes,spectral.config.mel_bands))
    assert denom["mel_elements"] == expected_counts
    full = pilot.losses(*tensors(crops,model),spectral)
    split = [pilot.losses(*tensors(part,model),spectral,denom) for part in (crops[:1],crops[1:])]
    for name in full:
        combined = sum(r[name] for r in split)
        torch.testing.assert_close(combined,full[name],rtol=2e-6,atol=1e-7)
        actual_grad, = torch.autograd.grad(combined,model.gain,retain_graph=True)
        expected_grad, = torch.autograd.grad(full[name],model.gain,retain_graph=True)
        torch.testing.assert_close(actual_grad,expected_grad,rtol=2e-6,atol=1e-6)


def test_pivot_selection_prefers_independent_coordinates_not_duplicate_energy():
    gram = torch.tensor([[9.,9.,0.],[9.,9.,0.],[0.,0.,4.]])
    assert pilot.pivoted_subset(gram,2) == [0,2]
    assert pilot.pivoted_subset(torch.zeros(4,4),3) == [0,1,2]


@pytest.mark.parametrize("rank", [0, -1, 4])
def test_invalid_pivot_rank_is_rejected(rank):
    with pytest.raises(ValueError):
        pilot.pivoted_subset(torch.eye(3),rank)


def test_calibration_uses_global_parameter_gradients_and_restores_probe_state(cpu_pipeline,spectral,tmp_path):
    crops=[crop(str(i),frames=3+i,valid=4101+i*1093,amplitude=.03+i*.06) for i in range(4)]
    model=TinyGroup()
    # Unequal source lengths expose equal-source averaging errors relative
    # to this independent corpus-pooled gradient oracle.
    z,t,valid,spans=pilot.batch(crops)
    h=model.group_from_input(features(z));p=model.suffix_from_group(h)
    expected=pilot.losses(p,t,h,.9*features(z),valid,spans,spectral)
    gradients={k:abs(float(torch.autograd.grad(v,model.gain,retain_graph=True)[0])) for k,v in expected.items()}
    before=model.gain.detach().clone()
    model.gain.grad=torch.tensor(.123)
    rng=torch.get_rng_state().clone()
    report=pilot.calibration(model,SimpleNamespace(),crops,spectral,tmp_path)
    assert report["passed"] and report["state_restored"]
    for k in expected:
        assert report["before"][k] == pytest.approx(float(expected[k].detach()),rel=2e-6,abs=1e-7)
        assert report["pooled_parameter_gradient_norms"][k] == pytest.approx(gradients[k],rel=3e-6)
    torch.testing.assert_close(model.gain,before,rtol=0,atol=0)
    torch.testing.assert_close(model.gain.grad,torch.tensor(.123),rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    assert (tmp_path/"calibration.json").is_file()
    assert cpu_pipeline and set(cpu_pipeline)=={1}


def test_failed_disposable_probe_restores_parameters_gradients_and_rng(cpu_pipeline,spectral,tmp_path,monkeypatch):
    crops=[crop(str(i),frames=3,valid=4301,amplitude=.1+i*.03) for i in range(3)]
    model=TinyGroup()
    before=model.gain.detach().clone()
    model.gain.grad=torch.tensor(.321)
    rng=torch.get_rng_state().clone()
    original=model.suffix_from_group
    def fail_after_update(value):
        if not torch.equal(model.gain.detach(),before):
            torch.rand(7)
            raise RuntimeError("injected post-update error")
        return original(value)
    monkeypatch.setattr(model,"suffix_from_group",fail_after_update)
    with pytest.raises(RuntimeError,match="injected"):
        pilot.calibration(model,SimpleNamespace(),crops,spectral,tmp_path)
    torch.testing.assert_close(model.gain,before,rtol=0,atol=0)
    torch.testing.assert_close(model.gain.grad,torch.tensor(.321),rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    assert not (tmp_path/"calibration.json").exists()


def test_evaluation_mel_is_element_pooled_and_independent_of_batch_partition(cpu_pipeline,spectral):
    crops=[crop("a",frames=3,valid=4097,amplitude=.03),crop("b",frames=5,valid=9533,amplitude=.27)]
    model=TinyGroup()
    together=pilot.evaluate(model,SimpleNamespace(),crops,spectral,batch_size=2)
    separate=pilot.evaluate(model,SimpleNamespace(),crops,spectral,batch_size=1)
    for name in ("mae","mse","mel","mel_linear","mel_log"):
        assert together["aggregate"][name] == pytest.approx(separate["aggregate"][name],rel=2e-6)
    terms=[]
    for c in crops:
        z,t,_,spans=pilot.batch([c]);a,b=spans[0]
        p=model.suffix_from_group(model.group_from_input(features(z)))
        terms.append(spectral.group_terms(p[...,a:b],t[...,a:b]))
    expected=spectral.aggregate(terms)
    assert together["aggregate"]["mel"] == pytest.approx(float(expected.losses["teacher_mel"].detach()),rel=2e-6)
    assert together["mel_elements"] == list(expected.counts["mel_elements_by_resolution"])


def test_nonquiet_cosine_excludes_quiet_windows_in_mixed_recording(cpu_pipeline,spectral):
    c=crop("mixed",frames=3)
    c["latents"][...,0]=1e-5
    c["teacher_audio"]=.9*features(c["latents"]).repeat_interleave(4,dim=-1)
    class QuietNoise(TinyGroup):
        def suffix_from_group(self,h):
            result=super().suffix_from_group(h)
            return torch.where(result.abs()<1e-4,torch.full_like(result,.3),result)
    result=pilot.evaluate(QuietNoise(),SimpleNamespace(),[c],spectral)
    row=result["rows"][0]
    assert row["quiet_samples"]==1920
    assert row["nonquiet_samples"]==3840
    assert row["cosine"]==pytest.approx(1.,abs=1e-6)
    assert row["quiet_failed"]==2


def test_singleton_accumulation_matches_pooled_gradient_and_one_adamw_update(cpu_pipeline,spectral,monkeypatch):
    crops=[crop("a",frames=3,valid=4101,amplitude=.08),
           crop("b",frames=5,context=1,valid=7103,amplitude=.2),
           crop("c",frames=4,valid=7019,amplitude=.13)]
    candidate,reference=TinyGroup(),TinyGroup()
    coefficients={"waveform":1.,"mel":.014,"feature":.5}
    options={"lr":3e-5,"betas":(.9,.99),"eps":1e-8,"weight_decay":0.}
    actual_opt=torch.optim.AdamW(candidate.parameters(),**options)
    reference_opt=torch.optim.AdamW(reference.parameters(),**options)
    # Existing moments exercise resumed optimizer semantics, not just the
    # first AdamW step whose normalized displacement hides scaling errors.
    for opt,model in ((actual_opt,candidate),(reference_opt,reference)):
        opt.state[model.gain]={"step":torch.tensor(7.),"exp_avg":torch.tensor(.02),"exp_avg_sq":torch.tensor(.03)}
    z,target,valid,spans=pilot.batch(crops)
    h=reference.group_from_input(features(z))
    branches=pilot.losses(reference.suffix_from_group(h),target,h,.9*features(z),valid,spans,spectral)
    total=sum(coefficients[k]*v for k,v in branches.items())
    reference_opt.zero_grad(set_to_none=True);total.backward()
    expected_gradient=reference.gain.grad.clone()
    reference_opt.step()
    calls=[]
    original_step=actual_opt.step
    def counted_step(*args,**kwargs):
        calls.append(1)
        return original_step(*args,**kwargs)
    monkeypatch.setattr(actual_opt,"step",counted_step)
    result=pilot.training_update(candidate,SimpleNamespace(),crops,spectral,coefficients,actual_opt)
    assert cpu_pipeline==[1,1,1] and calls==[1]
    assert result["total"]==pytest.approx(float(total.detach()),rel=2e-6)
    for key,value in branches.items():
        assert result[key]==pytest.approx(float(value.detach()),rel=2e-6,abs=1e-7)
    torch.testing.assert_close(candidate.gain.grad,expected_gradient,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(candidate.gain,reference.gain,rtol=0,atol=1e-7)
    for name,value in reference_opt.state[reference.gain].items():
        torch.testing.assert_close(actual_opt.state[candidate.gain][name],value,rtol=2e-6,atol=1e-7)


def test_numerical_preflight_and_role_check_never_execute_a_batch(cpu_pipeline,monkeypatch):
    observed=[]
    class WithForward(TinyGroup):
        def forward_latents(self,z):
            observed.append(z.shape[0])
            h=self.group_from_input(features(z))
            return self.suffix_from_group(h),h
    copy_model=WithForward()
    with torch.no_grad():copy_model.gain.fill_(.9)
    monkeypatch.setattr(pilot,"build_student",lambda *a,**kw:copy_model)
    monkeypatch.setattr(pilot.torch.cuda,"empty_cache",lambda:None)
    monkeypatch.setattr(pilot,"event",lambda *a,**kw:None)
    crops=[crop("a",frames=3,valid=4101),crop("b",frames=4,valid=7333)]
    teacher=SimpleNamespace(model=SimpleNamespace(decoder=None))
    result=pilot.numerical_preflight(teacher,crops)
    assert len(result)==2 and all(r["execution_batch_size"]==1 for r in result)
    assert all(r["copy_wave_max_abs"]==0 and r["sealed_wave_max_abs"]==0 for r in result)
    candidate=WithForward()
    check=pilot.singleton_forward_check(candidate,teacher,crops)
    assert check["passed"] and len(check["records"])==2
    assert observed and set(observed)=={1}
    assert cpu_pipeline and set(cpu_pipeline)=={1}


def test_feature_diagnostics_use_partial_cell_weights_and_flag_zero_reference():
    prediction=torch.tensor([[[1.,2.,90.],[3.,4.,90.]]])
    target=torch.tensor([[[.5,1.,90.],[1.5,2.,90.]]])
    valid=torch.zeros(1,1,12,dtype=torch.bool);valid[...,:6]=True
    report=pilot.feature_statistics(prediction,target,valid)
    expected_error=((prediction[...,:2]-target[...,:2]).square()*torch.tensor([4.,2.])).sum()/12
    assert report["group_elements"]==12
    assert report["group_mse"]==pytest.approx(float(expected_error))
    assert report["group_nrmse"]==pytest.approx(1.)
    assert report["group_cosine"]==pytest.approx(1.)
    assert not report["group_nrmse_floor_applied"]
    silent=pilot.feature_statistics(torch.zeros_like(prediction),torch.zeros_like(target),valid)
    assert silent["group_nrmse_floor_applied"]
    assert silent["group_nrmse"]==0 and silent["group_cosine"] is None


def test_evaluation_zero_reference_does_not_fabricate_correlation(cpu_pipeline,spectral):
    c=crop("silence",frames=3,amplitude=0.)
    report=pilot.evaluate(TinyGroup(),SimpleNamespace(),[c],spectral)
    for values in (report["rows"][0],report["aggregate"]):
        assert values["waveform_nrmse_floor_applied"] and values["group_nrmse_floor_applied"]
        assert values["waveform_nrmse"]==0 and values["waveform_residual_rms"]==0
        assert values["group_cosine"] is None
    assert report["aggregate"]["nonquiet_cosine_mean"] is None
    assert report["rows"][0]["cosine"] is None


def test_boundary_panel_is_fixed_metadata_only_and_stratified():
    metadata={}
    for name,label in (("whistle","human_whistling_source_description"),("loud","Screaming"),
                       ("laugh","Laughter"),("cry","Crying_and_sobbing")):
        metadata[name]={"verified_source_labels":[label],"normalized_language":"und","selection_kind":"explicit_event"}
    for kind in ("quiet","transition"):
        for i in range(3):metadata[f"{kind}{i}"]={"selection_kind":kind,"normalized_language":"und"}
    for language in ("hi","ta","en","fr"):
        metadata[language]={"normalized_language":language,"selection_kind":"ordinary"}
    crops=[{"source_id":key,"fake_loss":i} for i,key in enumerate(metadata)]
    selected,receipt=pilot.select_boundary_panel(crops,metadata)
    reversed_selected,reversed_receipt=pilot.select_boundary_panel(list(reversed(crops)),metadata)
    assert len(selected)==12 and len({row["source_id"] for row in selected})==12
    assert [row["source_id"] for row in selected]==[row["source_id"] for row in reversed_selected]
    assert receipt==reversed_receipt
    assert [row["reason"] for row in receipt["sources"]].count("quiet")==2
    assert [row["reason"] for row in receipt["sources"]].count("transition")==2
    for category in ("indic","international"):
        sources=[row["source_id"] for row in receipt["sources"] if row["reason"]==category]
        assert len(sources)==2 and len({metadata[key]["normalized_language"] for key in sources})==2
    with pytest.raises(ValueError,match="metadata"):
        pilot.select_boundary_panel(crops,{})


class ScalarRecorder:
    def __init__(self):self.values={}
    def add_scalar(self,tag,value,step):self.values[tag]=(value,step)


def test_monitoring_total_includes_all_three_weighted_objectives():
    writer=ScalarRecorder()
    record={"waveform":2.,"mel":3.,"feature":5.,"total":33.,
            "unique_sources":6,"audio_hours":.001,"elapsed_seconds":10.}
    coeff={"waveform":2.,"mel":3.,"feature":4.}
    pilot.log_training(writer,record,coeff,1e-4,2)
    assert writer.values["loss/total"]==(33.,2)
    assert writer.values["loss/teacher_group_mse"]==(5.,2)
    assert writer.values["loss_weighted/teacher_group_mse"]==(20.,2)
    with pytest.raises(ValueError,match="excludes"):
        pilot.log_training(writer,{**record,"total":13.},coeff,1e-4,2)


def test_boundary_monitoring_logs_nested_regions_with_partial_coordinate_labels():
    writer=ScalarRecorder()
    pilot.log_boundaries(writer,{"aggregate":{
        "stage2_output":{"all":{"mse":.2},"quiet":{"cosine":None}},
        "stage3_output":{"active":{"cosine":.8}},
        "stage4_output":{"all":{"mse":.03}}}},256)
    assert writer.values=={
        "quality_boundaries/stage2_output_selected_coordinates/all/mse":(.2,256),
        "quality_boundaries/stage3_output_selected_coordinates/active/cosine":(.8,256),
        "quality_boundaries/stage4_output/all/mse":(.03,256)}


def streaming_receipt():
    expected={"initial_sha256":"i"*64,"preflight_sha256":"p"*64,"model_sha256":"m"*64,
              "streaming_checker_sha256":"s"*64,"manifest_sha256":"f"*64,"cache_sha256":"c"*64,
              "runner_sha256":"r"*64}
    model={"passed":True,"model_state_sha256_before":"a"*64,"model_state_sha256_after":"a"*64}
    report={**expected,"version":"audiovae2_group_streaming_v1","passed":True,"device":"cpu","threads":1,
            "chunk_sizes_latent_frames":[1,2,4],"teacher_state_sha256_before":"t"*64,"teacher_state_sha256_after":"t"*64,
            "models":{"full_width_control":copy.deepcopy(model),"narrowed_initial":copy.deepcopy(model)}}
    return report,expected


def test_authenticated_streaming_check_accepts_both_preserved_models():
    report,expected=streaming_receipt()
    pilot.authenticate_streaming_check(report,expected)


@pytest.mark.parametrize("mutation",[
    lambda report:report.update(initial_sha256="changed"),
    lambda report:report.update(streaming_checker_sha256="changed"),
    lambda report:report.update(passed=False),
    lambda report:report.update(chunk_sizes_latent_frames=[2,4]),
    lambda report:report.update(teacher_state_sha256_after="changed"),
    lambda report:report["models"].pop("full_width_control"),
    lambda report:report["models"]["narrowed_initial"].update(passed=False),
    lambda report:report["models"]["narrowed_initial"].update(model_state_sha256_after="changed"),
])
def test_streaming_gate_rejects_mismatched_or_failed_receipts(mutation):
    report,expected=streaming_receipt();mutation(report)
    with pytest.raises(ValueError):pilot.authenticate_streaming_check(report,expected)
