"""Independent CPU oracles for the bounded raw-L1 teacher head experiment."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import inspect
import random
import statistics
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config
import run_joint_heads as joint
import run_spectral_heads as spectral
import run_teacher_refinements_v2 as reference
import run_teacher_l1_v1 as trial


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(207331)
    random.seed(207331)
    np.random.seed(207331)


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=3, *, source="fit-0", target_offset=.003):
    features, baseline = joint.capture_prehead(model, torch.randn(1, 64, frames))
    positions = torch.arange(baseline.shape[-1], dtype=baseline.dtype)
    target = baseline + target_offset * (1.1 + torch.sin(positions*.081)).reshape(1, 1, -1)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :19] = False
    valid[..., -17:] = False
    quiet = valid.clone()
    quiet[..., target.shape[-1] // 3:] = False
    return {"features": features, "target": target, "baseline": baseline,
        "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
        "outside_samples": int((valid & ~quiet).sum()), "source_id": source,
        "start_frame": 0, "language": "fixture", "condition": "speech", "dataset": "unit"}


@pytest.fixture
def objectives():
    return {"long": spectral.SpectralObjective(ReconstructionV2Config(
        fft_sizes=(64, 128), mel_bands=(3, 4)))}


def args(batch_size=2):
    return SimpleNamespace(batch_size=batch_size, preservation_weight=100., active_weight=1.)


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def snapshot(model, optimizer=None):
    return {"model": deepcopy(model.state_dict()),
        "flags": [(name, parameter.requires_grad) for name, parameter in model.named_parameters()],
        "gradients": {name: None if parameter.grad is None else parameter.grad.clone()
                      for name, parameter in model.named_parameters()},
        "modes": [module.training for module in model.modules()],
        "optimizer": None if optimizer is None else deepcopy(optimizer.state_dict()),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().clone()}


def direct_branches(model, rows, objectives):
    predictions = [joint.head_forward(model, row["features"]) for row in rows]
    count = sum(int(row["valid"].sum()) for row in rows)
    wave = sum((prediction-row["target"])[row["valid"]].abs().sum()
               for prediction, row in zip(predictions, rows)) / count
    mel = ReconstructionV2(objectives["long"].config).forward_groups(
        (prediction[..., 19:-17], row["target"][..., 19:-17])
        for prediction, row in zip(predictions, rows)).losses["teacher_mel"]
    return {"wave": wave, "long": mel}, predictions


@pytest.mark.parametrize("quiet_mode", ["none", "all", "alternating", "single"])
def test_raw_l1_is_sample_pooled_with_equal_valid_derivative_and_no_quiet_weight(quiet_mode):
    model = tiny_decoder()
    rows = [bank_row(model, frames=2, source="short"), bank_row(model, frames=4, source="long")]
    predictions = []
    for row in rows:
        teacher = torch.zeros_like(row["target"])
        teacher[..., 30:100] = .9
        prediction = teacher.clone()
        # Errors span six orders of magnitude, including a genuine zero.
        prediction[..., ::2] += 1e-6
        prediction[..., 1::2] -= 1.
        prediction[..., 31] = teacher[..., 31]
        row["target"] = teacher
        row["baseline"] = torch.full_like(teacher, 123.)
        prediction[~row["valid"]] = 1e6
        predictions.append(prediction.requires_grad_(True))
        quiet = row["valid"].clone()
        if quiet_mode == "none": quiet.zero_()
        elif quiet_mode == "alternating": quiet[..., 1::2] = False
        elif quiet_mode == "single": quiet.zero_(); quiet[..., 20] = True
        row["quiet"] = quiet
        row["quiet_samples"] = int(quiet.sum())
        row["outside_samples"] = int((row["valid"] & ~quiet).sum())
    total = sum(int(row["valid"].sum()) for row in rows)
    loss = sum(trial.teacher_l1_loss(prediction, row["target"], row["valid"], total)
               for prediction, row in zip(predictions, rows))
    oracle = torch.cat([(prediction-row["target"])[row["valid"]]
                        for prediction, row in zip(predictions, rows)]).abs().mean()
    torch.testing.assert_close(loss, oracle)
    for gradient, prediction, row in zip(torch.autograd.grad(loss, predictions), predictions, rows):
        expected = torch.where(row["valid"], (prediction.detach()-row["target"]).sign()/total, 0.)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


def test_teacher_target_detached_and_teacher_perfect_has_zero_gradient():
    teacher = torch.randn(1, 1, 37, requires_grad=True)
    prediction = teacher.detach().clone().requires_grad_(True)
    valid = torch.ones_like(teacher, dtype=torch.bool)
    loss = trial.teacher_l1_loss(prediction, teacher, valid, int(valid.sum()))
    grads = torch.autograd.grad(loss, [prediction, teacher], allow_unused=True)
    assert loss.item() == 0
    assert torch.equal(grads[0], torch.zeros_like(prediction))
    assert grads[1] is None


def test_empty_microbatch_contributes_zero_without_changing_global_sample_denominator():
    prediction = torch.randn(1, 1, 13, requires_grad=True)
    teacher = torch.randn_like(prediction)
    loss = trial.teacher_l1_loss(prediction, teacher, torch.zeros_like(prediction, dtype=torch.bool), 100)
    assert loss.item() == 0
    assert torch.equal(torch.autograd.grad(loss, prediction)[0], torch.zeros_like(prediction))


def test_nonfinite_padding_is_excluded_before_arithmetic_but_valid_nonfinite_is_rejected():
    prediction = torch.tensor([[[float("nan"), 3., float("inf"), -4.]]], requires_grad=True)
    teacher = torch.tensor([[[float("inf"), 1., float("nan"), -1.]]])
    valid = torch.tensor([[[False, True, False, True]]])
    loss = trial.teacher_l1_loss(prediction, teacher, valid, 2)
    assert loss.item() == 2.5
    assert torch.equal(torch.autograd.grad(loss, prediction)[0], torch.tensor([[[0., .5, 0., -.5]]]))
    valid[..., 0] = True
    with pytest.raises(FloatingPointError):
        trial.teacher_l1_loss(prediction, teacher, valid, 3)


def test_entire_objective_ignores_quiet_partition_baseline_and_retired_normalizations(objectives):
    model = tiny_decoder()
    parameters = [p for _, p in trial.configure_head(model, "l1")]
    bank = [bank_row(model, frames=2, source="short"), bank_row(model, frames=4, source="long")]
    result = trial.backward_batch(model, bank, {}, objectives, "l1", {"wave": 1., "long": .3},
        device="cpu", preservation_weight=100., active_weight=1.)
    gradients = [p.grad.clone() for p in parameters]
    alternate = deepcopy(bank)
    for row in alternate:
        row["quiet"] = row["valid"] & ~row["quiet"]
        row["quiet_samples"] = int(row["quiet"].sum())
        row["outside_samples"] = int((row["valid"] & ~row["quiet"]).sum())
        row["baseline"].fill_(5000.)
    model.zero_grad(set_to_none=True)
    changed = trial.backward_batch(model, alternate,
        {"quiet_mse": 1e-30, "outside_mse": 1e30, "teacher_mean_square": 1e-30},
        objectives, "l1", {"wave": 1., "long": .3}, device="cpu",
        preservation_weight=1e15, active_weight=1e-15)
    assert result["branch_losses"] == changed["branch_losses"]
    for parameter, gradient in zip(parameters, gradients):
        assert torch.equal(parameter.grad, gradient)


def test_sole_existing_head_scope_preserves_exact_frozen_prefix_and_normalization():
    model = tiny_decoder()
    before = deepcopy(model.state_dict())
    z = torch.randn(1, 64, 3)
    expected = model(z).detach()
    assert trial.ARMS == ("l1",)
    named = trial.configure_head(model, "l1")
    assert {name for name, _ in named} == joint.HEAD_NAMES
    assert {name for name, p in model.named_parameters() if p.requires_grad} == joint.HEAD_NAMES
    assert_nested_equal(model.state_dict(), before)
    assert torch.equal(model(z), expected)
    assert all(not module.training for module in model.modules())


def test_long_mel_unchanged_and_raw_l1_backward_matches_independent_oracle(objectives):
    model = tiny_decoder()
    rows = [bank_row(model, frames=2, source="short"), bank_row(model, frames=4, source="long", target_offset=-.005)]
    named = trial.configure_head(model, "l1")
    branches, _ = direct_branches(model, rows, objectives)
    oracle = branches["wave"] + .3*branches["long"]
    expected = torch.autograd.grad(oracle, [p for _, p in named])
    result = trial.backward_batch(model, rows, {}, objectives, "l1", {"wave": 1., "long": .3},
        device="cpu", preservation_weight=100., active_weight=1.)
    assert result["loss"] == pytest.approx(float(oracle.detach()), rel=3e-6)
    assert set(result["branch_losses"]) == {"wave", "long"}
    for (_, parameter), gradient in zip(named, expected):
        torch.testing.assert_close(parameter.grad, gradient, rtol=5e-6, atol=1e-8)


def test_strict_selection_and_retained_candidate_provenance_unchanged():
    for name in ("teacher_selection_checks", "refinement_peak_screen", "load_candidate_head",
                 "restore_effective_head"):
        assert inspect.getsource(getattr(trial, name)) == inspect.getsource(getattr(reference, name))


def test_explicit_calibration_source_ids_resolve_fit_only_in_requested_order():
    bank = [{"source_id": f"fit-{i}"} for i in range(11)]
    wanted = [f"fit-{i}" for i in (8, 3, 0, 10, 2, 9, 1, 7)]
    actual = trial.resolve_calibration_indices(bank, wanted, 2)
    assert actual == [8, 3, 0, 10, 2, 9, 1, 7]


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "short", "incomplete", "ambiguous_fit_id"])
def test_calibration_source_resolution_rejects_invalid_or_leaky_panel(problem):
    bank = [{"source_id": f"fit-{i}"} for i in range(10)]
    wanted = [f"fit-{i}" for i in range(8)]
    if problem == "unknown": wanted[-1] = "heldout-only"
    elif problem == "duplicate": wanted[-1] = wanted[0]
    elif problem == "short": wanted = wanted[:6]
    elif problem == "incomplete": wanted = wanted[:7]
    elif problem == "ambiguous_fit_id": bank[-1]["source_id"] = bank[0]["source_id"]
    with pytest.raises((ValueError, RuntimeError)):
        trial.resolve_calibration_indices(bank, wanted, 2)


def test_calibration_median_uses_actual_head_gradients_not_output_gradients(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, frames=2+i%3, source=f"fit-{i}", target_offset=.0005*(i+1))
            for i in range(9)]
    named = trial.configure_head(model, "l1")
    parameters = [p for _, p in named]
    for index, parameter in enumerate(model.parameters()):
        if index % 3 == 0:
            parameter.grad = torch.full_like(parameter, .123)
    before = snapshot(model)
    indices = [6, 1, 7, 3, 0, 5, 2, 4]
    ratios = []
    output_ratios = []
    energies = []
    for start in range(0, len(indices), 2):
        rows = [bank[i] for i in indices[start:start+2]]
        branches, predictions = direct_branches(model, rows, objectives)
        parameter_energy = {}
        output_energy = {}
        for name, loss in branches.items():
            gradients = torch.autograd.grad(loss, [*parameters, *predictions], retain_graph=True)
            parameter_energy[name] = sum(float(g.double().square().sum()) for g in gradients[:len(parameters)])
            output_energy[name] = sum(float(g.double().square().sum()) for g in gradients[len(parameters):])
        energies.append(parameter_energy)
        ratios.append((parameter_energy["wave"]/parameter_energy["long"])**.5)
        output_ratios.append((output_energy["wave"]/output_energy["long"])**.5)
    result = trial.calibrate_coefficients(model, bank, {}, objectives, args(), "cpu", indices)
    assert result["coefficients"] == pytest.approx({"wave": 1., "long": statistics.median(ratios)}, rel=5e-6)
    assert statistics.median(ratios) != pytest.approx(statistics.median(output_ratios), rel=.01)
    assert result["source_ids"] == [bank[i]["source_id"] for i in indices]
    assert "fit-8" not in result["source_ids"]
    assert result["eligible_batches"] == 4 and result["skipped_batches"] == 0
    assert result["optimizer_updates"] == 0
    for actual, expected in zip(result["batches"], energies):
        assert actual["parameter_gradient_energies"] == pytest.approx(expected, rel=5e-6)
        for branch in ("wave", "long"):
            assert sum(value[branch] for value in actual["parameter_layer_gradient_energies"].values()) == pytest.approx(expected[branch], rel=5e-6)
        assert set(actual["parameter_layer_gradient_energies"]) == joint.HEAD_NAMES
    assert_nested_equal(snapshot(model), before)


def test_genuinely_perfect_batch_is_disclosed_and_excluded_from_calibration(objectives):
    model = tiny_decoder()
    trial.configure_head(model, "l1")
    bank = [bank_row(model, source=f"fit-{i}", target_offset=0. if i == 0 else .002*i) for i in range(4)]
    result = trial.calibrate_coefficients(model, bank, {}, objectives, args(1), "cpu", list(range(4)))
    assert result["eligible_batches"] == 3
    assert result["skipped_batches"] == 1
    first = result["batches"][0]
    assert first["ratio"] is None
    assert first["skipped_reason"] == "zero_wave_and_mel"
    assert result["coefficients"]["long"] == statistics.median(row["ratio"] for row in result["batches"][1:])


def test_median_robust_to_finite_outlier_and_skips_zero_without_epsilon(monkeypatch, objectives):
    model = tiny_decoder()
    trial.configure_head(model, "l1")
    bank = [{"source_id": f"fit-{i}"} for i in range(6)]
    # Norm ratios 1, 2, 3, 1e200, then two genuine zero cases.
    energies = [(1., 1.), (4., 1.), (9., 1.), (1e200, 1e-200), (0., 1.), (1., 0.)]
    def probe(model, bank, indices, *unused, **kwargs):
        wave, mel = energies[indices[0]]
        return {"source_ids": [bank[indices[0]]["source_id"]],
                "parameter_gradient_energies": {"wave": wave, "long": mel}}
    monkeypatch.setattr(trial, "gradient_probe", probe)
    result = trial.calibrate_coefficients(model, bank, {}, objectives, args(1), "cpu", list(range(6)))
    assert result["coefficients"]["long"] == 2.5
    assert result["eligible_batches"] == 4 and result["skipped_batches"] == 2
    assert [row["skipped_reason"] for row in result["batches"][-2:]] == ["zero_wave", "zero_mel"]


@pytest.mark.parametrize("bad_energy", [0., float("nan"), float("inf"), -1.])
def test_unusable_calibration_fails_and_restores_all_disposable_state(bad_energy, monkeypatch, objectives):
    model = tiny_decoder()
    trial.configure_head(model, "l1")
    bank = [{"source_id": f"fit-{i}"} for i in range(4)]
    before = snapshot(model)
    def broken_probe(*unused, **kwargs):
        random.random(); np.random.rand(); torch.rand(1)
        with torch.no_grad():
            model.output.weight.add_(.25)
        model.activation.training = True
        return {"parameter_gradient_energies": {"wave": bad_energy, "long": 1.}}
    monkeypatch.setattr(trial, "gradient_probe", broken_probe)
    with pytest.raises((ValueError, FloatingPointError)):
        trial.calibrate_coefficients(model, bank, {}, objectives, args(1), "cpu", list(range(4)))
    assert_nested_equal(snapshot(model), before)


def primed_optimizer(model):
    named = trial.configure_head(model, "l1")
    optimizer = torch.optim.AdamW([p for _, p in named], lr=3e-5, betas=(.7, .8), eps=1e-5,
        weight_decay=.17, foreach=False, fused=False)
    for index, (_, parameter) in enumerate(named):
        parameter.grad = torch.full_like(parameter, .04*(index+1))
    optimizer.step()
    # Preserve mixed pre-existing gradients too, including a frozen parameter.
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = None if index % 2 else torch.full_like(parameter, .0123)
    model.activation.training = True  # Harmless mixed mode must still be restored.
    return optimizer


def oracle_adamw_step(model, optimizer, bank, update_indices, panel_indices, objectives, coefficients):
    copy_model = deepcopy(model)
    named = [(name, p) for name, p in copy_model.named_parameters() if p.requires_grad]
    parameter_lookup = dict(named)
    group_names = [[name for parameter in group["params"] for name, original in model.named_parameters()
                    if original is parameter] for group in optimizer.param_groups]
    groups = [{**{key: deepcopy(value) for key, value in group.items() if key != "params"},
               "params": [parameter_lookup[name] for name in names]}
              for group, names in zip(optimizer.param_groups, group_names)]
    copy_optimizer = torch.optim.AdamW(groups)
    copy_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    before = {name: parameter.detach().clone() for name, parameter in named}
    panel_branches, _ = direct_branches(copy_model, [bank[i] for i in panel_indices], objectives)
    gradients = {name: torch.autograd.grad(loss, [p for _, p in named], retain_graph=True)
                 for name, loss in panel_branches.items()}
    before_losses = {name: float(loss.detach()) for name, loss in panel_branches.items()}
    copy_optimizer.zero_grad(set_to_none=True)
    update_branches, _ = direct_branches(copy_model, [bank[i] for i in update_indices], objectives)
    batch_before = {name: float(loss.detach()) for name, loss in update_branches.items()}
    sum(coefficients[name]*loss for name, loss in update_branches.items()).backward()
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.)
    copy_optimizer.step()
    delta = [parameter.detach().double()-before[name].double() for name, parameter in named]
    after_losses = {name: float(loss.detach()) for name, loss in
                    direct_branches(copy_model, [bank[i] for i in panel_indices], objectives)[0].items()}
    batch_after = {name: float(loss.detach()) for name, loss in
                   direct_branches(copy_model, [bank[i] for i in update_indices], objectives)[0].items()}
    return {"before": before_losses, "after": after_losses,
        "batch_before": batch_before, "batch_after": batch_after,
        "predicted": {name: sum(float((gradient.double()*change).sum()) for gradient, change in zip(grads, delta))
                      for name, grads in gradients.items()},
        "absolute_products": {name: sum(float((gradient.double()*change).abs().sum()) for gradient, change in zip(grads, delta))
                              for name, grads in gradients.items()},
        "norm": float(norm), "displacement": sum(float(change.square().sum()) for change in delta)**.5,
        "max_change": max(float(change.abs().max()) for change in delta)}


@pytest.mark.parametrize("primed", [False, True])
def test_disposable_probe_measures_actual_adamw_moments_clipping_decay_and_restores_state(primed, objectives):
    model = tiny_decoder()
    if primed:
        optimizer = primed_optimizer(model)
    else:
        optimizer = trial.new_optimizer([p for _, p in trial.configure_head(model, "l1")], 3e-5)
    bank = [bank_row(model, source=f"fit-{i}", target_offset=.002*(i+1)) for i in range(4)]
    coefficients = {"wave": 10., "long": 100.}
    before = snapshot(model, optimizer)
    expected = oracle_adamw_step(model, optimizer, bank, [0, 1], [2, 3], objectives, coefficients)
    actual = trial.disposable_update_probe(model, bank, [0, 1], [2, 3], optimizer, {}, "l1", objectives,
        coefficients, device="cpu")
    assert actual["update_source_ids"] == ["fit-0", "fit-1"]
    assert actual["panel_source_ids"] == ["fit-2", "fit-3"]
    assert actual["before"] == pytest.approx(expected["before"], rel=3e-6)
    assert actual["after"] == pytest.approx(expected["after"], rel=3e-6)
    assert actual["predicted_loss_changes"] == pytest.approx(expected["predicted"], rel=4e-4, abs=1e-9)
    assert actual["predicted_absolute_product_sums"] == pytest.approx(expected["absolute_products"], rel=4e-4, abs=1e-9)
    assert actual["update_batch_before"] == pytest.approx(expected["batch_before"], rel=3e-6)
    assert actual["update_batch_after"] == pytest.approx(expected["batch_after"], rel=3e-6)
    assert actual["actual_loss_changes"] == pytest.approx(
        {name: expected["after"][name]-expected["before"][name] for name in expected["before"]}, rel=4e-4, abs=1e-7)
    assert actual["parameter_displacement"] == pytest.approx(expected["displacement"], rel=3e-5)
    assert actual["maximum_parameter_change"] == pytest.approx(expected["max_change"], rel=3e-5)
    assert expected["norm"] > 1.  # This witness actually exercises clipping.
    assert actual["update"]["gradient_norm_before_clip"] == pytest.approx(expected["norm"], rel=3e-6)
    for branch in ("wave", "long"):
        assert actual["weighted_predicted_loss_changes"][branch] == pytest.approx(coefficients[branch]*expected["predicted"][branch], rel=4e-4, abs=1e-8)
        assert sum(layer[branch] for layer in actual["parameter_layer_predicted_loss_changes"].values()) == pytest.approx(actual["predicted_loss_changes"][branch], abs=1e-12)
    assert set(actual["parameter_layer_predicted_loss_changes"]) == joint.HEAD_NAMES
    assert actual["optimizer_steps_before"] == ([1.]*4 if primed else [])
    assert actual["retained_optimizer_updates"] == 0
    assert actual["discarded_optimizer_updates"] == 1
    assert_nested_equal(snapshot(model, optimizer), before)


def test_disposable_probe_restores_weights_moments_rng_buffers_and_flags_after_step_exception(objectives, monkeypatch):
    model = tiny_decoder()
    optimizer = primed_optimizer(model)
    bank = [bank_row(model, source=f"fit-{i}") for i in range(2)]
    before = snapshot(model, optimizer)
    real_step = optimizer.step
    def explode(*step_args, **step_kwargs):
        real_step(*step_args, **step_kwargs)
        random.random(); np.random.rand(); torch.rand(1)
        with torch.no_grad():
            next(iter(model.buffers())).add_(1)
        next(iter(model.parameters())).requires_grad_(True)
        model.activation.training = False
        optimizer.param_groups[0]["lr"] = 1.
        raise RuntimeError("intentional failure after optimizer mutation")
    monkeypatch.setattr(optimizer, "step", explode)
    with pytest.raises(RuntimeError, match="intentional failure"):
        trial.disposable_update_probe(model, bank, [0], [0, 1], optimizer, {}, "l1", objectives,
            {"wave": 1., "long": .3}, device="cpu")
    assert_nested_equal(snapshot(model, optimizer), before)


def test_retained_real_update_only_changes_existing_head_and_frozen_buffers_stay_exact(objectives):
    model = tiny_decoder()
    parameters = [p for _, p in trial.configure_head(model, "l1")]
    bank = [bank_row(model, source="a"), bank_row(model, source="b")]
    before = deepcopy(model.state_dict())
    optimizer = trial.new_optimizer(parameters, 1e-6)
    result = trial.train_batch(model, bank, [0, 1], optimizer, {}, "l1", objectives,
        {"wave": 1., "long": .1}, device="cpu", preservation_weight=100., active_weight=1.)
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
    assert changed and changed.issubset(joint.HEAD_NAMES)
    assert result["parameter_displacement"] > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in joint.HEAD_NAMES)


def direction_probe(*, predicted_wave=-.1, predicted_mel=-.2, actual_wave=-.05, actual_mel=-.1,
                    panel=("fit-0", "fit-1")):
    before = {"wave": 1., "long": 2.}
    changes = {"wave": actual_wave, "long": actual_mel}
    return {"panel_source_ids": list(panel), "before": before,
        "after": {name: before[name]+change for name, change in changes.items()},
        "predicted_loss_changes": {"wave": predicted_wave, "long": predicted_mel},
        "predicted_absolute_product_sums": {"wave": 1., "long": 1.},
        "actual_loss_changes": changes, "quality_checks": {"passed": True}}


def test_direction_uses_mean_over_common_panel_without_requiring_every_update_improve():
    good = direction_probe(predicted_wave=-.3, actual_wave=-.3)
    bad = direction_probe(predicted_wave=.1, actual_wave=.1)
    result = trial.assess_update_directions([good, bad])
    assert result["passed"]
    assert result["branches"]["wave"]["predicted_mean_change"] == pytest.approx(-.1)
    assert result["branches"]["wave"]["actual_mean_change"] == pytest.approx(-.1)


def test_direction_separates_adverse_local_prediction_from_finite_step_curvature():
    local = trial.assess_update_directions([direction_probe(predicted_mel=.01)])
    assert local["predicted_adverse"] and not local["actual_adverse"]
    curvature = trial.assess_update_directions([direction_probe(actual_mel=.01)])
    assert not curvature["predicted_adverse"] and curvature["actual_adverse"]
    assert not local["passed"] and not curvature["passed"]


def test_direction_numeric_allowance_does_not_turn_small_real_adverse_prediction_into_success():
    roundoff = direction_probe(predicted_wave=1e-15, actual_wave=1e-7)
    assert trial.assess_update_directions([roundoff])["passed"]
    real = direction_probe(predicted_wave=1e-9)
    assert trial.assess_update_directions([real])["predicted_adverse"]


def test_direction_rejects_nonfinite_or_different_comparison_panels():
    with pytest.raises(ValueError):
        trial.assess_update_directions([direction_probe(), direction_probe(panel=("heldout",))])
    with pytest.raises(FloatingPointError):
        trial.assess_update_directions([direction_probe(predicted_wave=float("nan"))])


@pytest.mark.parametrize("first_failure", ["prediction", "finite", "quality", "none"])
def test_rate_calibration_halts_bad_direction_and_only_halves_finite_or_quality_failure(first_failure, objectives, monkeypatch):
    model = tiny_decoder()
    trial.configure_head(model, "l1")
    initial = trial.effective_head_state(model)
    before = snapshot(model)
    bank = [{"source_id": f"fit-{i}"} for i in range(6)]
    selected = [4, 1, 3, 0]
    calls = []
    monkeypatch.setattr(trial, "migration_parity", lambda *a: {"passed": True})
    def fake_probe(model, bank, update_indices, panel_indices, optimizer, *unused, **kwargs):
        rate = optimizer.param_groups[0]["lr"]
        calls.append((list(update_indices), list(panel_indices), rate))
        assert not optimizer.state, "Every calibration batch must get fresh Adam moments"
        row = direction_probe(panel=[bank[i]["source_id"] for i in panel_indices])
        if rate == 1e-6:
            if first_failure == "prediction": row["predicted_loss_changes"]["long"] = .1
            elif first_failure == "finite": row["actual_loss_changes"]["long"] = .1; row["after"]["long"] = 2.1
            elif first_failure == "quality": row["quality_checks"]["passed"] = False
        return row
    monkeypatch.setattr(trial, "disposable_update_probe", fake_probe)
    configuration = args(1); configuration.learning_rate = 1e-6
    result = trial.calibrate_rate(model, bank, initial, {}, objectives, {"wave": 1., "long": .3},
        configuration, "cpu", selected)
    expected = {"prediction": None, "finite": 5e-7, "quality": 5e-7, "none": 1e-6}[first_failure]
    assert result["chosen_learning_rate"] == expected
    assert len(calls) == (8 if first_failure in ("finite", "quality") else 4)
    assert [call[0] for call in calls[:4]] == [[i] for i in selected]
    assert all(call[1] == selected for call in calls)
    assert_nested_equal(snapshot(model), before)
