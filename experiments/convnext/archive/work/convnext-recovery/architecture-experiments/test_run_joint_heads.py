"""CPU-only correctness checks of the actual isolated joint-head runner.

These use small real decoder blocks. They do not load production checkpoints,
connect to a remote host, launch training jobs, or time inference.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
import run_joint_heads as joint


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(713902)


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=3, *, source="fixture", target_offset=.0002):
    z = torch.randn(1, 64, frames)
    features, baseline = joint.capture_prehead(model, z)
    target = baseline + target_offset
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :19] = False
    valid[..., -17:] = False
    quiet = valid.clone()
    quiet[..., target.shape[-1]//3:] = False
    return {"features": features, "target": target, "baseline": baseline,
        "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
        "outside_samples": int((valid & ~quiet).sum()), "source_id": source,
        "start_frame": 0, "language": "fixture", "condition": "speech", "dataset": "unit"}


@pytest.mark.parametrize("arm", joint.ARMS)
def test_pooled_loss_and_gradient_use_samples_not_equal_clip_weights(arm):
    # Unequal lengths, invalid gaps and large invalid sentinels detect accidental
    # per-example weighting or padding/context leakage.
    predictions = [torch.tensor([[[1., 3., 5000., 5.]]], requires_grad=True),
                   torch.tensor([[[2., 7., 4., 9., 6., -5000., 8.]]], requires_grad=True)]
    targets = [torch.zeros_like(p) for p in predictions]
    baselines = [torch.ones_like(p)*.5 for p in predictions]
    valids = [torch.tensor([[[1, 1, 0, 1]]], dtype=torch.bool),
              torch.tensor([[[1, 1, 1, 1, 1, 0, 1]]], dtype=torch.bool)]
    quiets = [torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool),
              torch.tensor([[[1, 0, 1, 0, 0, 0, 0]]], dtype=torch.bool)]
    totals = {"quiet_samples": 3, "outside_samples": 6}
    scales = {"quiet_mse": 2., "outside_mse": 4.}
    weight = 7.
    loss = sum(joint.normalized_loss(joint.masked_sums(p,t,b,v,q), totals, scales, arm, weight)
               for p,t,b,v,q in zip(predictions, targets, baselines, valids, quiets))
    all_quiet_error = torch.cat([(p-t)[q] for p,t,q in zip(predictions, targets, quiets)])
    outside = [v & ~q for v,q in zip(valids, quiets)]
    reference = targets if arm == joint.ARMS[0] else baselines
    all_outside_error = torch.cat([(p-r)[m] for p,r,m in zip(predictions, reference, outside)])
    factor = 1. if arm == joint.ARMS[0] else weight
    oracle = all_quiet_error.square().mean()/2. + factor*all_outside_error.square().mean()/4.
    torch.testing.assert_close(loss, oracle, rtol=0, atol=1e-5)
    actual_gradients = torch.autograd.grad(loss, predictions, retain_graph=True)
    expected_gradients = torch.autograd.grad(oracle, predictions)
    for observed, expected, valid in zip(actual_gradients, expected_gradients, valids):
        torch.testing.assert_close(observed, expected, rtol=0, atol=1e-6)
        assert not bool(observed[~valid].any())


@pytest.mark.parametrize("all_quiet", [False, True])
def test_empty_partition_is_finite_without_discarding_other_samples(all_quiet):
    p = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    valid = torch.ones_like(p, dtype=torch.bool)
    quiet = valid.clone() if all_quiet else ~valid
    sums = joint.masked_sums(p, torch.zeros_like(p), torch.ones_like(p), valid, quiet)
    totals = {k:sums[k] for k in ("quiet_samples", "outside_samples")}
    loss = joint.normalized_loss(sums, totals, {"quiet_mse":1., "outside_mse":1.}, joint.ARMS[0], 100.)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, p.square().mean())
    loss.backward()
    torch.testing.assert_close(p.grad, 2*p.detach()/3)


def test_quiet_mask_cannot_leak_into_padding_or_change_geometry():
    p = torch.zeros(1,1,5)
    valid = torch.ones_like(p, dtype=torch.bool)
    valid[...,0] = False
    with pytest.raises(ValueError, match="subset"):
        joint.masked_sums(p,p,p,valid,torch.ones_like(valid))
    with pytest.raises(ValueError, match="subset"):
        joint.masked_sums(p,p,p,valid,valid.float())
    with pytest.raises(ValueError, match="geometry"):
        joint.masked_sums(p,p[...,:-1],p,valid,valid)
    with pytest.raises(ValueError, match="shape"):
        joint.masked_sums(p,p,p,valid[...,:-1],valid)


def test_capture_replay_is_exact_state_preserving_and_has_same_head_gradients():
    model = tiny_decoder()
    named = joint.head_parameters(model)
    parameters = [p for _,p in named]
    initial = deepcopy(model.state_dict())
    modes = tuple(m.training for m in model.modules())
    rng = torch.random.get_rng_state().clone()
    z = torch.linspace(-.2,.3,64*5).reshape(1,64,5)
    features, original = joint.capture_prehead(model,z)
    assert torch.equal(original,model(z))
    assert not features.requires_grad and not original.requires_grad
    assert not model.head._forward_pre_hooks
    assert torch.equal(rng,torch.random.get_rng_state())
    assert tuple(m.training for m in model.modules()) == modes
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    target = original + .0002
    full_grad = torch.autograd.grad((model(z)-target).square().mean(),parameters)
    replay_grad = torch.autograd.grad((joint.head_forward(model,features)-target).square().mean(),parameters)
    assert all(torch.equal(a,b) for a,b in zip(full_grad,replay_grad))
    # The nonlinear head is allowed to adapt; its stable input remains the same.
    with torch.no_grad():
        model.activation.weight.add_(.01)
        model.head.conv.weight.mul_(1.01)
    after_features, after_audio = joint.capture_prehead(model,z)
    assert torch.equal(after_features,features)
    assert not torch.equal(after_audio,original)
    assert torch.equal(after_audio,joint.head_forward(model,features))


@pytest.mark.parametrize("arm", joint.ARMS)
def test_real_update_changes_only_declared_head_and_preserves_norm_buffers(arm):
    model = tiny_decoder()
    named = joint.head_parameters(model)
    assert {n for n,p in model.named_parameters() if p.requires_grad} == joint.HEAD_NAMES
    bank = [bank_row(model,3,source="a"),bank_row(model,5,source="b",target_offset=-.0003)]
    before = deepcopy(model.state_dict())
    flags = tuple(m.training for m in model.modules())
    scales,_ = joint.baseline_scales(bank)
    optimizer = joint.new_optimizer([p for _,p in named],1e-7)
    result = joint.train_batch(model,bank,[0,1],optimizer,scales,arm,100.,device="cpu")
    assert result["parameter_displacement"] > 0
    changed = {name for name,value in model.state_dict().items() if not torch.equal(value,before[name])}
    assert changed and changed.issubset(joint.HEAD_NAMES)
    assert tuple(m.training for m in model.modules()) == flags
    assert all(p.grad is None for n,p in model.named_parameters() if n not in joint.HEAD_NAMES)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in named)
    assert result["quiet_samples"] == sum(r["quiet_samples"] for r in bank)
    assert result["outside_samples"] == sum(r["outside_samples"] for r in bank)


def test_baseline_scales_and_validation_keep_tail_samples_and_pool_denominators():
    model = tiny_decoder()
    bank = [bank_row(model,2,source="short",target_offset=.002),
            bank_row(model,7,source="long",target_offset=.0001)]
    scales,counts = joint.baseline_scales(bank)
    score = joint.bank_score(model,bank,"cpu")
    quiet = torch.cat([(r["baseline"].double()-r["target"].double())[r["quiet"]] for r in bank])
    outside = torch.cat([(r["baseline"].double()-r["target"].double())[r["valid"] & ~r["quiet"]] for r in bank])
    assert counts["quiet_samples"] == quiet.numel()
    assert counts["outside_samples"] == outside.numel()
    assert scales["quiet_mse"] == pytest.approx(float(quiet.square().mean()),rel=1e-14)
    assert scales["outside_mse"] == pytest.approx(float(outside.square().mean()),rel=1e-14)
    assert score["aggregate"]["quiet_mse"] == scales["quiet_mse"]
    assert score["aggregate"]["outside_mse"] == scales["outside_mse"]
    assert score["aggregate"]["outside_drift_mse"] == 0.
    assert score["aggregate"]["samples"] == sum(int(r["valid"].sum()) for r in bank)
    reversed_score = joint.bank_score(model,list(reversed(bank)),"cpu")
    assert reversed_score["aggregate"] == score["aggregate"]


def test_calibration_discards_trial_updates_and_starts_each_optimizer_fresh(monkeypatch):
    model = tiny_decoder()
    parameters = [p for _,p in joint.head_parameters(model)]
    bank = [bank_row(model,3,source="a"),bank_row(model,2,source="b")]
    initial = joint.head_state(model)
    frozen = deepcopy(model.state_dict())
    scales,_ = joint.baseline_scales(bank)
    rng = torch.random.get_rng_state().clone()
    original_new = joint.new_optimizer
    created = []
    def observed_new(*args,**kwargs):
        assert all(torch.equal(value,dict(model.named_parameters())[name]) for name,value in initial.items())
        opt = original_new(*args,**kwargs)
        assert not opt.state
        created.append(opt)
        return opt
    monkeypatch.setattr(joint,"new_optimizer",observed_new)
    result = joint.calibrate_rate(model,bank,initial,parameters,scales,
        SimpleNamespace(batch_size=2,learning_rate=1e-10,preservation_weight=100.),"cpu")
    assert result["chosen_learning_rate"] is not None
    assert len(created) == 2*len(result["trials"])
    assert len({id(opt) for opt in created}) == len(created)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in frozen.items())
    assert all(p.grad is None for p in parameters)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_calibration_failure_restores_weights_and_clears_transient_gradients(monkeypatch):
    model = tiny_decoder()
    parameters = [p for _,p in joint.head_parameters(model)]
    bank = [bank_row(model)]
    initial = joint.head_state(model)
    scales,_ = joint.baseline_scales(bank)
    def broken_step(*args,**kwargs):
        with torch.no_grad():
            parameters[0].add_(1.)
        parameters[0].grad = torch.ones_like(parameters[0])
        raise FloatingPointError("injected finite-update failure")
    monkeypatch.setattr(joint,"train_batch",broken_step)
    with pytest.raises(FloatingPointError,match="injected"):
        joint.calibrate_rate(model,bank,initial,parameters,scales,
            SimpleNamespace(batch_size=1,learning_rate=1e-7,preservation_weight=100.),"cpu")
    assert all(torch.equal(value,dict(model.named_parameters())[name]) for name,value in initial.items())
    assert all(p.grad is None for p in parameters)


def selection_report():
    aggregate = {"samples":1000,"quiet_samples":200,"outside_samples":800,"sources":4,
        "mae":.01,"mse":.002,"quiet_mse":1e-6,"outside_mse":.003,
        "outside_drift_mse":0.,"peak":1.2,"overshoot_samples":4}
    return {"aggregate":deepcopy(aggregate),"groups":{"all":deepcopy(aggregate),
        "condition/speech":deepcopy(aggregate)}}


@pytest.mark.parametrize("failure",["quiet_mse","mae","mse","outside_drift_mse","peak","overshoot_samples","group"])
def test_selection_cannot_qualify_a_failed_candidate(failure):
    baseline = selection_report()
    candidate = deepcopy(baseline)
    candidate["aggregate"]["quiet_mse"] = baseline["aggregate"]["quiet_mse"]*.8**2
    assert joint.selection_checks(candidate,baseline)["qualified"]
    if failure == "quiet_mse":
        candidate["aggregate"][failure] = baseline["aggregate"][failure]*.91**2
    elif failure == "group":
        candidate["groups"]["condition/speech"]["mae"] *= 1.051
    elif failure == "outside_drift_mse":
        candidate["aggregate"][failure] = baseline["aggregate"]["outside_mse"]*.0101
    elif failure in ("mae","mse"):
        candidate["aggregate"][failure] *= 1.011
    else:
        candidate["aggregate"][failure] += .01 if failure == "peak" else 1
    outcome = joint.selection_checks(candidate,baseline)
    assert not outcome["qualified"]
    assert any(not check["passed"] for check in outcome["checks"])


def test_selection_rejects_changed_mask_counts():
    baseline = selection_report()
    candidate = deepcopy(baseline)
    candidate["aggregate"]["quiet_samples"] -= 1
    with pytest.raises(ValueError,match="counts"):
        joint.selection_checks(candidate,baseline)


def peak_report(peaks, counts):
    return {"rows":[{"source_id":str(i),"start_frame":0,"student_peak_abs":peak,
        "student_overshoot_samples":count} for i,(peak,count) in enumerate(zip(peaks,counts))],
        "recovery_metrics":{"all":{"maximum_peak":max(peaks),"scored_overshoot_samples":sum(counts)}}}


def test_new_peak_crop_cannot_hide_behind_a_lower_global_maximum():
    before = peak_report([1.3,.9],[20,0])
    after = peak_report([1.1,1.01],[5,1])
    result = joint.canonical_peak_screen(before,after)
    assert not result["passed"]
    assert result["regressing_crops"][0]["source_id"] == "1"
    assert joint.canonical_peak_screen(before,peak_report([1.2,.99],[15,0]))["passed"]


def test_wider_overshoot_cannot_hide_behind_improvement_in_other_crops():
    before = peak_report([1.3,1.1],[20,2])
    after = peak_report([1.2,1.1],[5,3])
    assert not joint.canonical_peak_screen(before,after)["passed"]


def test_peak_screen_rejects_duplicate_candidate_crop_identity():
    before = peak_report([1.3,.9],[20,0])
    after = deepcopy(before)
    after["rows"].append(deepcopy(after["rows"][0]))
    with pytest.raises(ValueError,match="identit"):
        joint.canonical_peak_screen(before,after)
