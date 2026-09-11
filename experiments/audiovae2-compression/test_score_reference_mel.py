from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent/"convnext")]
import score_reference_mel as score
from author_mel import AuthorMelLoss
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_quiet_classification_dc_centered_energy_startup_and_teacher_replay():
    target = torch.cat([torch.full((1,1,960),v) for v in (0.,2e-4,2e-4,2e-4,0.)],-1)
    p = torch.cat([torch.full((1,1,960),v) for v in (2e-5,-2e-4,2.26e-4,4e-4,0.)],-1)
    p[0,0,13] = .005
    crop = {"source_id":"example","start_frame":0,"context_frames":0}
    report = score.quiet_diagnostics(p,target,crop)
    assert [r["failure_category"] for r in report["windows"]] == ["both","residual_only","amplitude_only","both","passed"]
    assert report["aggregate"]["failed_windows"] == 4
    maximum = report["aggregate"]["maximum_absolute_error"]
    assert maximum["actual_startup"] is True
    assert maximum["maximum_error_source_sample"] == 13
    assert maximum["maximum_error_source_seconds"] == 13/48000
    assert report["windows"][1]["centered_residual_rms"] == 0
    for row in report["windows"]:
        assert row["residual_rms_double"]**2 == pytest.approx(row["residual_mean"]**2+row["centered_residual_rms"]**2,rel=1e-12,abs=1e-20)
    teacher = score.quiet_diagnostics(target,target,crop)
    assert teacher["aggregate"]["failed_windows"] == 0
    assert teacher["aggregate"]["residual_rms"] == 0
    offset = score.quiet_diagnostics(p,target,{**crop,"start_frame":35,"context_frames":20})
    assert offset["aggregate"]["maximum_absolute_error"]["maximum_error_source_sample"] == 35*1920+13
    assert offset["aggregate"]["maximum_absolute_error"]["actual_startup"] is False


def test_fit_occupancy_counts_actual_samples_with_partial_tails_and_no_model(monkeypatch):
    monkeypatch.setattr(score.base,"teacher_forward",lambda *_:pytest.fail("Teacher must not run"))
    crops = []
    for i,(quiet,active) in enumerate(((960,960),(123,0))):
        scored = torch.cat((torch.zeros(1,1,quiet),torch.full((1,1,active),.01)),-1)
        crops.append({"source_id":str(i),"context_frames":1,"valid_scored_samples":quiet+active,
                      "teacher_audio":torch.cat((torch.full((1,1,1920),.5),scored),-1)})
    got = score.fit_quiet_occupancy(crops)
    assert got["aggregate"]["quiet_samples"] == 1083
    assert got["aggregate"]["active_samples"] == 960
    assert got["aggregate"]["quiet_windows"] == 2
    assert got["aggregate"]["active_windows"] == 1
    assert got["source_ids_sha256"] == score.screen.digest(["0","1"])


def evaluate_fixture(monkeypatch):
    generator = torch.Generator().manual_seed(91)
    crops = []
    for i,length in enumerate((1501,2011)):
        target = torch.randn(1,1,length,generator=generator)*.0003
        prediction = target+.00002
        crops.append({"source_id":str(i),"target":target,"prediction":prediction,
                      "start_frame":0,"context_frames":0})
    active = {}
    def batch(chosen):
        assert not torch.is_grad_enabled()
        assert len(chosen)==1
        active["crop"] = chosen[0]
        p,t = chosen[0]["prediction"],chosen[0]["target"]
        return p,t,torch.ones_like(t,dtype=torch.bool),[(0,t.shape[-1])]
    def teacher_forward(teacher,z):
        assert not torch.is_grad_enabled()
        return {"group_input":z,"waveform":active["crop"]["target"]}
    monkeypatch.setattr(score.base,"batch",batch)
    monkeypatch.setattr(score.base,"teacher_forward",teacher_forward)
    monkeypatch.setattr(torch.optim,"AdamW",lambda *_args,**_kwargs:pytest.fail("No optimizer is allowed"))
    model = SimpleNamespace(group_from_input=lambda x:x,suffix_from_group=lambda x:x)
    author = AuthorMelLoss()
    current = ReconstructionV2(ReconstructionV2Config(fft_sizes=(64,128),mel_bands=(3,4)))
    return score.evaluate_reference(model,None,crops,author,current),crops,author,current


def test_evaluation_uses_singleton_pooled_counts_and_both_metric_definitions(monkeypatch):
    got,crops,author,current = evaluate_fixture(monkeypatch)
    terms = [author.group_terms(c["prediction"],c["target"]) for c in crops]
    expected = author.aggregate(terms)
    assert got["aggregate"]["author_mel"] == float(expected.losses["teacher_mel"])
    assert got["author_mel_elements"] == list(expected.counts["mel_elements_by_resolution"])
    assert got["aggregate"]["samples"] == 3512
    assert got["quiet_diagnostics"]["teacher_replay"]["aggregate"]["failed_windows"] == 0
    assert [r["source_id"] for r in got["rows"]] == ["0","1"]
    assert all(r["teacher_cache_max_absolute_error"]==0 for r in got["rows"])


def test_common_reproduction_rejects_metric_counts_and_source_changes(monkeypatch):
    got,*_ = evaluate_fixture(monkeypatch)
    aggregate = {key:got["aggregate"]["common_"+key] for key in ("mel","mel_linear","mel_log")}
    aggregate.update({k:got["aggregate"][k] for k in ("sources","samples")})
    q = got["quiet_diagnostics"]["student"]["aggregate"]
    aggregate.update(quiet_windows=q["quiet_windows"],quiet_failed_windows=q["failed_windows"],quiet_residual_rms_mean=q["residual_rms"])
    rows = [{"source_id":r["source_id"],"samples":r["samples"],"mel_elements":r["common_mel_elements"],
             **{key:r["common_"+key] for key in ("mel","mel_linear","mel_log")}} for r in got["rows"]]
    saved = {"aggregate":aggregate,"rows":rows,"mel_elements":got["common_mel_elements"]}
    assert score.verify_common(got,saved)["maximum_absolute_difference"] == 0
    wrong = copy.deepcopy(saved);wrong["rows"][0]["mel"] += .01
    with pytest.raises(RuntimeError,match="not reproduced"):score.verify_common(got,wrong)
    wrong = copy.deepcopy(saved);wrong["rows"].reverse()
    with pytest.raises(ValueError,match="order"):score.verify_common(got,wrong)
    wrong = copy.deepcopy(saved);wrong["mel_elements"][0] += 1
    with pytest.raises(ValueError,match="denominators"):score.verify_common(got,wrong)


def checkpoint_fixture(directory):
    arm,definition,rate = score.screen.ARMS[0]
    ids = [str(i) for i in range(768)]
    identity = {"original_identity":{"teacher":"frozen"},"initial_group_sha256":"initial",
                "channel_selection":{"x":[0]},"fit_source_ids":ids,"fit_source_ids_sha256":score.screen.digest(ids)}
    launch = {"arm":arm,"definition":definition,"learning_rate":rate,"screen_identity_sha256":score.screen.digest(identity),
              "original_identity":identity["original_identity"],"initial_group_sha256":"initial",
              "channel_selection":identity["channel_selection"],"fit_source_ids_sha256":identity["fit_source_ids_sha256"],
              "optimizer_initial_state_entries":0}
    payload = {"format":score.screen.VERSION,"step":256,"fit_cursor":768,"sources_seen":ids,
               "identity":launch,"group":{"weight":torch.ones(2)}}
    torch.save(payload,directory/"final.pt")
    completion = {"arm":arm,"step":256,"unique_sources":768,"frozen_decoder_state_preserved":True,
                  "automatic_promotion":False,"checkpoint_sha256":score.base.sha(directory/"final.pt")}
    (directory/"completed.json").write_text(json.dumps(completion))
    (directory/"launch.json").write_text(json.dumps(launch))
    return arm,identity,payload,completion


def test_checkpoint_authentication_accepts_only_recorded_final_identity(tmp_path):
    arm,identity,payload,completion = checkpoint_fixture(tmp_path)
    got = score.load_authenticated_arm(tmp_path,arm,identity)
    assert torch.equal(got["group"]["weight"],payload["group"]["weight"])
    payload["sources_seen"] = payload["sources_seen"][:-1]
    torch.save(payload,tmp_path/"final.pt")
    with pytest.raises(ValueError,match="bytes changed"):score.load_authenticated_arm(tmp_path,arm,identity)
    completion["checkpoint_sha256"] = score.base.sha(tmp_path/"final.pt")
    (tmp_path/"completed.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError,match="authenticated launch"):score.load_authenticated_arm(tmp_path,arm,identity)


def test_mismatched_launch_rejects_before_checkpoint_load(tmp_path,monkeypatch):
    arm,identity,_,_ = checkpoint_fixture(tmp_path)
    launch = json.loads((tmp_path/"launch.json").read_text())
    launch["learning_rate"] = 9
    (tmp_path/"launch.json").write_text(json.dumps(launch))
    monkeypatch.setattr(torch,"load",lambda *_args,**_kwargs:pytest.fail("Bad launch must be rejected before loading weights"))
    with pytest.raises(ValueError,match="launch identity"):score.load_authenticated_arm(tmp_path,arm,identity)
