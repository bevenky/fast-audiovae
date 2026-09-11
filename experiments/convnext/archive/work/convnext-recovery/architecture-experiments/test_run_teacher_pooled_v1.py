"""CPU checks of the isolated sample-pooled teacher reconstruction pilot."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config
import run_joint_heads as joint
import run_spectral_heads as spectral
import run_teacher_refinements_v2 as reference
import run_teacher_pooled_v1 as pooled


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(813925)


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=3, *, source="fixture", target_offset=.0002):
    features, baseline = joint.capture_prehead(model, torch.randn(1, 64, frames))
    target = baseline + target_offset
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :19] = False
    valid[..., -17:] = False
    quiet = valid.clone()
    quiet[..., target.shape[-1] // 3:] = False
    return {"features": features, "target": target, "baseline": baseline,
        "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
        "outside_samples": int((valid & ~quiet).sum()), "source_id": source,
        "start_frame": 0, "language": "fixture", "condition": "speech", "dataset": "unit"}


def synthetic_wave_rows():
    predictions = [torch.tensor([[[1., 3., 5000., 5.]]], requires_grad=True),
                   torch.tensor([[[2., 7., 4., 9., 6., -5000., 8.]]], requires_grad=True)]
    targets = [torch.tensor([[[.1, .5, 100., .2]]]),
               torch.tensor([[[.4, -.6, .9, .3, .1, -100., .8]]])]
    valid = [torch.tensor([[[1, 1, 0, 1]]], dtype=torch.bool),
             torch.tensor([[[1, 1, 1, 1, 1, 0, 1]]], dtype=torch.bool)]
    return predictions, targets, valid


@pytest.mark.parametrize("quiet_mode", ["none", "all", "alternating", "one_sample"])
def test_waveform_value_and_gradient_are_invariant_to_quiet_classification(quiet_mode):
    predictions, targets, masks = synthetic_wave_rows()
    quiets = []
    for mask in masks:
        quiet = mask.clone()
        if quiet_mode == "none":
            quiet.zero_()
        elif quiet_mode == "alternating":
            quiet[..., 1::2] = False
        elif quiet_mode == "one_sample":
            quiet.zero_(); quiet[..., 0] = True
        quiets.append(quiet)
    totals = {"quiet_samples": sum(int(q.sum()) for q in quiets),
              "outside_samples": sum(int((v & ~q).sum()) for v, q in zip(masks, quiets))}
    fixed_scale = {"teacher_mean_square": .7}
    observed = sum(pooled.pooled_teacher_wave_loss(
        joint.masked_sums(p, t, p.detach()+100, v, q), totals, fixed_scale)
        for p, t, v, q in zip(predictions, targets, masks, quiets))
    errors = torch.cat([(p-t)[v] for p, t, v in zip(predictions, targets, masks)])
    oracle = errors.square().mean()/.7
    torch.testing.assert_close(observed, oracle, rtol=0, atol=2e-5)
    actual_gradients = torch.autograd.grad(observed, predictions, retain_graph=True)
    expected_gradients = torch.autograd.grad(oracle, predictions)
    for actual, expected, valid in zip(actual_gradients, expected_gradients, masks):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
        assert not bool(actual[~valid].any())


def test_equal_errors_get_equal_per_sample_gradient_regardless_of_region_or_clip_length():
    predictions = [torch.ones(1, 1, 3, requires_grad=True), torch.ones(1, 1, 11, requires_grad=True)]
    quiets = [torch.ones_like(predictions[0], dtype=torch.bool), torch.zeros_like(predictions[1], dtype=torch.bool)]
    totals = {"quiet_samples": 3, "outside_samples": 11}
    loss = sum(pooled.pooled_teacher_wave_loss(joint.masked_sums(p, torch.zeros_like(p),
        torch.randn_like(p), torch.ones_like(p, dtype=torch.bool), q), totals, {"teacher_mean_square": .25})
        for p, q in zip(predictions, quiets))
    gradients = torch.autograd.grad(loss, predictions)
    expected = 2/14/.25
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.full_like(gradient, expected), rtol=0, atol=1e-7)


def test_teacher_perfect_is_zero_gradient_even_when_baseline_and_quiet_partition_differ():
    teacher = torch.randn(1, 1, 20)
    p = teacher.clone().requires_grad_(True)
    valid = torch.ones_like(p, dtype=torch.bool)
    quiet = valid.clone(); quiet[..., 3:] = False
    sums = joint.masked_sums(p, teacher, teacher+10., valid, quiet)
    loss = pooled.pooled_teacher_wave_loss(sums, {"quiet_samples": 3, "outside_samples": 17}, {"teacher_mean_square": .2})
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(p.grad, torch.zeros_like(p))


def test_fixed_fit_target_scale_is_sample_pooled_valid_only_and_quiet_independent():
    _, targets, valids = synthetic_wave_rows()
    bank = [{"source_id": f"train-{i}", "target": t, "valid": v,
             "quiet": v.clone(), "quiet_samples": int(v.sum()), "outside_samples": 0}
            for i, (t, v) in enumerate(zip(targets, valids))]
    before = deepcopy(bank)
    scale, accounting = pooled.pooled_target_scale(bank)
    oracle = torch.cat([t.double()[v] for t, v in zip(targets, valids)]).square().mean().item()
    assert scale["teacher_mean_square"] == pytest.approx(oracle, rel=1e-14)
    assert oracle != pytest.approx(sum(t.double()[v].square().mean().item() for t, v in zip(targets, valids))/2)
    other = deepcopy(bank)
    for row in other:
        row["target"][~row["valid"]] = 1e7
        row["quiet"].zero_(); row["quiet_samples"] = 0
        row["outside_samples"] = int(row["valid"].sum())
    assert pooled.pooled_target_scale(other)[0] == scale
    for actual, saved in zip(bank, before):
        for key, value in saved.items():
            if torch.is_tensor(value):
                assert torch.equal(actual[key], value)


def test_final_quality_and_peak_gates_remain_identical_to_reference():
    for name in ("teacher_selection_checks", "refinement_peak_screen", "finite_update_checks"):
        assert inspect.getsource(getattr(pooled, name)) == inspect.getsource(getattr(reference, name))


def test_only_existing_plain_head_scope_and_inference_function_is_unchanged():
    assert pooled.ARMS == ("pooled",)
    model = tiny_decoder()
    before = deepcopy(model.state_dict())
    z = torch.randn(1, 64, 4)
    with torch.no_grad():
        audio = model(z)
    named = pooled.configure_head(model, "pooled")
    assert {name for name, _ in named} == joint.HEAD_NAMES
    assert {name for name, p in model.named_parameters() if p.requires_grad} == joint.HEAD_NAMES
    assert not any("parametrizations" in name for name, _ in model.named_parameters())
    assert set(model.state_dict()) == set(before)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert torch.equal(model(z), audio)
    assert all(not module.training for module in model.modules())


@pytest.fixture
def objectives():
    return {"long": spectral.SpectralObjective(ReconstructionV2Config(fft_sizes=(64, 128), mel_bands=(3, 4)))}


def test_unchanged_teacher_mel_and_pooled_waveform_match_explicit_batch_gradient(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, frames=2, source="short"), bank_row(model, frames=4, source="long", target_offset=-.0003)]
    named = pooled.configure_head(model, "pooled")
    scales, _ = pooled.pooled_target_scale(bank)
    params = [p for _, p in named]
    predictions = [joint.head_forward(model, row["features"]) for row in bank]
    ns = sum(int(row["valid"].sum()) for row in bank)
    wave = sum((p-row["target"])[row["valid"]].square().sum() for p, row in zip(predictions, bank))/ns/scales["teacher_mean_square"]
    mel = ReconstructionV2(objectives["long"].config).forward_groups(
        (p[..., 19:-17], row["target"][..., 19:-17]) for p, row in zip(predictions, bank)).losses["teacher_mel"]
    oracle = wave+.3*mel
    gradients = torch.autograd.grad(oracle, params)
    result = pooled.backward_batch(model, bank, scales, objectives, "pooled", {"wave": 1., "long": .3},
        device="cpu", preservation_weight=100., active_weight=1.)
    assert result["loss"] == pytest.approx(float(oracle.detach()), rel=3e-6)
    assert set(result["branch_losses"]) == {"wave", "long"}
    for p, grad in zip(params, gradients):
        torch.testing.assert_close(p.grad, grad, rtol=5e-6, atol=1e-8)


def test_spectral_calibration_uses_first_four_fit_batches_and_independent_output_energy_oracle(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    scales, accounting = pooled.pooled_target_scale(bank)
    assert accounting["source_ids"] == [f"train-{i}" for i in range(8)]
    state = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("Coefficient calibration must not execute head or optimizer")
    monkeypatch.setattr(pooled, "head_forward", forbidden)
    monkeypatch.setattr(pooled, "new_optimizer", forbidden)
    result = pooled.calibrate_coefficients(bank + [{"source_id": "heldout-do-not-use"}], scales,
        objectives, SimpleNamespace(batch_size=2, preservation_weight=100., active_weight=1.), "cpu")
    assert [r["source_ids"] for r in result["batches"]] == [[f"train-{i}", f"train-{i+1}"] for i in range(0, 8, 2)]
    energy = {"wave": 0., "long": 0.}
    regions = {"quiet": 0., "active": 0.}
    for start in range(0, 8, 2):
        selected = bank[start:start+2]
        ps = [row["baseline"].clone().requires_grad_(True) for row in selected]
        ns = sum(int(row["valid"].sum()) for row in selected)
        wave = sum((p-row["target"])[row["valid"]].square().sum() for p, row in zip(ps, selected))/ns/scales["teacher_mean_square"]
        mel = ReconstructionV2(objectives["long"].config).forward_groups(
            (p[..., 19:-17], row["target"][..., 19:-17]) for p, row in zip(ps, selected)).losses["teacher_mel"]
        for name, loss in (("wave", wave), ("long", mel)):
            gradients = torch.autograd.grad(loss, ps, retain_graph=True)
            energy[name] += sum(float(g.double().square().sum()) for g in gradients)
            if name == "wave":
                for g, row in zip(gradients, selected):
                    regions["quiet"] += float(g.double()[row["quiet"]].square().sum())
                    regions["active"] += float(g.double()[row["valid"] & ~row["quiet"]].square().sum())
    for name in energy:
        assert result["summed_output_gradient_energies"][name] == pytest.approx(energy[name], rel=3e-6)
    for name in regions:
        assert result["summed_wave_output_gradient_energy_by_region"][name] == pytest.approx(regions[name], rel=3e-6)
    assert sum(regions.values()) == pytest.approx(energy["wave"], rel=3e-6)
    assert result["coefficients"]["wave"] == 1.
    assert result["coefficients"]["long"] == pytest.approx((energy["wave"]/energy["long"])**.5, rel=3e-6)
    assert set(result["coefficients"]) == {"wave", "long"}
    assert result["optimizer_updates"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_spectral_coefficient_does_not_change_when_only_quiet_labels_change(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    scales, _ = pooled.pooled_target_scale(bank)
    args = SimpleNamespace(batch_size=2, preservation_weight=100., active_weight=1.)
    before = pooled.calibrate_coefficients(bank, scales, objectives, args, "cpu")
    altered = deepcopy(bank)
    for row in altered:
        row["quiet"] = row["valid"].clone()
        row["quiet_samples"] = int(row["valid"].sum())
        row["outside_samples"] = 0
    after = pooled.calibrate_coefficients(altered, scales, objectives, args, "cpu")
    assert after["coefficients"]["long"] == pytest.approx(before["coefficients"]["long"], rel=3e-6)
    assert after["summed_wave_output_gradient_energy_by_region"]["active"] == 0


def test_real_update_only_changes_existing_head_with_frozen_norms_and_fresh_state(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source="a"), bank_row(model, frames=4, source="b", target_offset=-.0003)]
    scales, _ = pooled.pooled_target_scale(bank)
    named = pooled.configure_head(model, "pooled")
    before = deepcopy(model.state_dict())
    modes = tuple(module.training for module in model.modules())
    optimizer = pooled.new_optimizer([p for _, p in named], 1e-7)
    assert not optimizer.state
    result = pooled.train_batch(model, bank, [0, 1], optimizer, scales, "pooled", objectives,
        {"wave": 1., "long": .3}, device="cpu", preservation_weight=100., active_weight=1.)
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
    assert changed and changed.issubset(joint.HEAD_NAMES)
    assert result["parameter_displacement"] > 0
    assert set(result["branch_losses"]) == {"wave", "long"}
    assert tuple(module.training for module in model.modules()) == modes
    assert all(p.grad is None for name, p in model.named_parameters() if name not in joint.HEAD_NAMES)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)


def test_parent_head_hash_validation_and_effective_state_restore_remain_unchanged():
    for name in ("load_candidate_head", "restore_effective_head"):
        assert inspect.getsource(getattr(pooled, name)) == inspect.getsource(getattr(reference, name))


@pytest.mark.parametrize("problem", ["empty", "duplicate", "zero_teacher", "nonfinite_valid", "invalid_quiet_mask"])
def test_fit_scale_rejects_invalid_or_repeated_calibration_audio(problem):
    _, targets, valids = synthetic_wave_rows()
    bank = [{"source_id": f"train-{i}", "target": t, "valid": v, "quiet": v.clone()}
            for i, (t, v) in enumerate(zip(targets, valids))]
    if problem == "empty":
        bank = []
    elif problem == "duplicate":
        bank[1]["source_id"] = bank[0]["source_id"]
    elif problem == "zero_teacher":
        for row in bank:
            row["target"].zero_()
    elif problem == "nonfinite_valid":
        bank[0]["target"][..., 0] = float("nan")
    else:
        bank[0]["quiet"][..., 2] = True
    with pytest.raises(ValueError):
        pooled.pooled_target_scale(bank)


def test_rate_calibration_uses_fresh_single_arm_updates_and_training_only_panel(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    scales, _ = pooled.pooled_target_scale(bank)
    initial = pooled.effective_head_state(model)
    state = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    real_new, real_score, real_train = pooled.new_optimizer, pooled.bank_score, pooled.train_batch
    created, updates = [], []
    allowed = {id(row) for row in bank}
    def observed_new(*a, **kw):
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
        result = real_new(*a, **kw)
        assert not result.state
        created.append(result)
        return result
    def observed_score(model, selected, *a, **kw):
        assert len(selected) == 8 and {id(row) for row in selected} == allowed
        return real_score(model, selected, *a, **kw)
    def observed_train(model, bank, indices, optimizer, scales, arm, *a, **kw):
        assert arm == "pooled" and len(indices) == 2 and all(0 <= i < 8 for i in indices)
        updates.append(tuple(indices))
        return real_train(model, bank, indices, optimizer, scales, arm, *a, **kw)
    monkeypatch.setattr(pooled, "new_optimizer", observed_new)
    monkeypatch.setattr(pooled, "bank_score", observed_score)
    monkeypatch.setattr(pooled, "train_batch", observed_train)
    result = pooled.calibrate_rate(model, bank + [{"source_id": "heldout-do-not-use"}], initial, scales,
        objectives, {"wave": 1., "long": .3},
        SimpleNamespace(batch_size=2, learning_rate=1e-10, preservation_weight=100., active_weight=1.), "cpu")
    assert result["chosen_learning_rate"] is not None
    assert len(created) == 4*len(result["trials"])
    assert len({id(opt) for opt in created}) == len(created)
    assert updates == [(i, i+1) for i in range(0, 8, 2)]*len(result["trials"])
    assert result["retained_optimizer_updates"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_calibration_failure_restores_head_and_frozen_state(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}") for i in range(4)]
    scales, _ = pooled.pooled_target_scale(bank)
    initial = pooled.effective_head_state(model)
    state = deepcopy(model.state_dict())
    def broken(*a, **kw):
        p = model.output.weight
        with torch.no_grad():
            p.add_(1.)
        p.grad = torch.ones_like(p)
        raise FloatingPointError("injected failure")
    monkeypatch.setattr(pooled, "train_batch", broken)
    with pytest.raises(FloatingPointError, match="injected"):
        pooled.calibrate_rate(model, bank, initial, scales, objectives, {"wave": 1., "long": .3},
            SimpleNamespace(batch_size=1, learning_rate=1e-7, preservation_weight=100., active_weight=1.), "cpu")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())
