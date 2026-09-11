"""CPU correctness checks for staged teacher-target head refinements.

Only small synthetic fixtures are used. No production checkpoint is modified,
no remote job is launched, and these tests do not measure inference speed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config
import run_joint_heads as joint
import run_spectral_heads as spectral
import run_teacher_refinements as refine


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(813923)


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=2, *, source="fixture", target_offset=.0002):
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


@pytest.mark.parametrize("active_weight", [.25, 1., 4.])
def test_teacher_objective_matches_both_regions_with_sample_pooled_gradients(active_weight):
    predictions = [torch.tensor([[[1., 3., 5000., 5.]]], requires_grad=True),
                   torch.tensor([[[2., 7., 4., 9., 6., -5000., 8.]]], requires_grad=True)]
    targets = [torch.zeros_like(p) for p in predictions]
    # Equal baseline/prediction makes preservation derivative zero, so this
    # fixture catches accidentally retaining that old target on active samples.
    baselines = [p.detach().clone() for p in predictions]
    valids = [torch.tensor([[[1, 1, 0, 1]]], dtype=torch.bool),
              torch.tensor([[[1, 1, 1, 1, 1, 0, 1]]], dtype=torch.bool)]
    quiets = [torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool),
              torch.tensor([[[1, 0, 1, 0, 0, 0, 0]]], dtype=torch.bool)]
    totals, scales = {"quiet_samples": 3, "outside_samples": 6}, {"quiet_mse": 2., "outside_mse": 4.}
    loss = sum(refine.teacher_wave_loss(joint.masked_sums(p, t, b, v, q), totals, scales,
                                       active_weight=active_weight)
               for p, t, b, v, q in zip(predictions, targets, baselines, valids, quiets))
    qe = torch.cat([(p-t)[q] for p, t, q in zip(predictions, targets, quiets)])
    oe = torch.cat([(p-t)[v & ~q] for p, t, v, q in zip(predictions, targets, valids, quiets)])
    oracle = qe.square().mean()/2. + active_weight*oe.square().mean()/4.
    torch.testing.assert_close(loss, oracle, rtol=0, atol=1e-5)
    actual = torch.autograd.grad(loss, predictions, retain_graph=True)
    expected = torch.autograd.grad(oracle, predictions)
    for a, b, v in zip(actual, expected, valids):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        assert not bool(a[~v].any())
        assert bool(a[v].abs().sum())


@pytest.mark.parametrize("quiet_pattern", ["mixed", "all", "none"])
def test_exact_teacher_is_zero_loss_zero_gradient_even_far_from_old_student(quiet_pattern):
    target = torch.linspace(-.7, .7, 9).reshape(1, 1, -1)
    prediction = target.clone().requires_grad_(True)
    baseline = target + .8
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., 0] = False
    quiet = valid.clone()
    if quiet_pattern == "mixed":
        quiet[..., 4:] = False
    elif quiet_pattern == "none":
        quiet.zero_()
    sums = joint.masked_sums(prediction, target, baseline, valid, quiet)
    totals = {k: sums[k] for k in ("quiet_samples", "outside_samples")}
    loss = refine.teacher_wave_loss(sums, totals, {"quiet_mse": .01, "outside_mse": .02})
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("arm", refine.ARMS)
def test_configure_head_keeps_body_fixed_and_has_only_head_trainable_parameters(arm):
    model = tiny_decoder()
    initial = deepcopy(model.state_dict())
    x = torch.randn(1, 64, 4)
    with torch.no_grad():
        before = model(x)
    named = refine.configure_head(model, arm)
    assert named and {n for n, _ in named} == {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(("head.conv.", "activation.", "output.")) for n, _ in named)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in initial.items()
               if name not in joint.HEAD_NAMES)
    effective = refine.effective_head_state(model)
    assert set(effective) == joint.HEAD_NAMES
    for name, value in effective.items():
        torch.testing.assert_close(value, initial[name], rtol=2e-6, atol=1e-7)
    with torch.no_grad():
        after = model(x)
    torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-6)
    assert all(not m.training for m in model.modules())


@pytest.mark.parametrize("zero_rows", [False, True])
def test_weight_norm_initialization_and_removal_preserve_waveform_and_plain_keys(zero_rows):
    model = tiny_decoder()
    if zero_rows:
        with torch.no_grad():
            model.head.conv.weight[0].zero_()
            model.output.weight[3].zero_()
    x = torch.randn(1, 64, 5)
    initial = deepcopy(model.state_dict())
    with torch.no_grad():
        before = model(x)
    named = refine.configure_head(model, "wn")
    assert any("parametrizations" in n for n, _ in named)
    with torch.no_grad():
        parametrized = model(x)
    assert torch.isfinite(parametrized).all()
    torch.testing.assert_close(parametrized, before, rtol=2e-6, atol=2e-6)
    effective = refine.effective_head_state(model)
    refine.fold_weight_norm(model)
    assert set(model.state_dict()) == set(initial)
    assert not any("parametrizations" in n for n, _ in model.named_parameters())
    assert isinstance(model.head.conv.weight, torch.nn.Parameter)
    assert isinstance(model.output.weight, torch.nn.Parameter)
    assert {"head.conv.weight", "output.weight"}.issubset(dict(model.named_parameters()))
    with torch.no_grad():
        folded = model(x)
    torch.testing.assert_close(folded, parametrized, rtol=0, atol=0)
    for name, value in effective.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in initial.items():
        if name not in joint.HEAD_NAMES:
            assert torch.equal(model.state_dict()[name], value)


def quality_report():
    aggregate = {"samples": 1000, "quiet_samples": 200, "outside_samples": 800, "sources": 4,
        "mae": .01, "mse": .002, "quiet_mse": 1e-6, "outside_mse": .003,
        "outside_drift_mse": 0., "peak": 1.2, "overshoot_samples": 4}
    source = {"source_id": "a", "regions": {r: {"element_counts": [100, 100], "mel": .5}
                                              for r in spectral.REGIONS}}
    return {"aggregate": deepcopy(aggregate), "groups": {"all": deepcopy(aggregate),
        "condition/speech": deepcopy(aggregate)}, "spectral_sources": [source]}


def test_teacher_relative_gate_accepts_exact_teacher_despite_large_baseline_displacement():
    baseline = quality_report()
    candidate = deepcopy(baseline)
    for row in [candidate["aggregate"], *candidate["groups"].values()]:
        for key in ("mae", "mse", "quiet_mse", "outside_mse", "overshoot_samples"):
            row[key] = 0
        row["outside_drift_mse"] = 123.
        row["peak"] = .995
    for region in candidate["spectral_sources"][0]["regions"].values():
        region["mel"] = 0.
    checks = refine.teacher_selection_checks(candidate, baseline)
    assert checks["qualified"]
    assert checks["quiet_rms_ratio"] == 0
    assert all("preservation" not in c["name"] for c in checks["checks"])
    candidate["aggregate"]["outside_mse"] = baseline["aggregate"]["outside_mse"]*1.02
    checks = refine.teacher_selection_checks(candidate, baseline)
    assert not checks["qualified"]
    assert any(c["name"] == "active/teacher_mse" and not c["passed"] for c in checks["checks"])


@pytest.fixture
def objectives():
    return {"long": spectral.SpectralObjective(ReconstructionV2Config(fft_sizes=(64, 128), mel_bands=(3, 4))),
            "short": spectral.SpectralObjective(ReconstructionV2Config(fft_sizes=(16, 32), mel_bands=(1, 1)))}


def test_repair_control_update_matches_existing_spectral_runner_exactly(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source="a"), bank_row(model, frames=3, source="b", target_offset=-.0003)]
    previous = deepcopy(model)
    scales, _ = joint.baseline_scales(bank)
    params = refine.configure_head(model, "repair_control")
    old_params = spectral.select_parameters(previous, "joint_spectral")
    optimizer = refine.new_optimizer([p for _, p in params], 1e-7)
    old_optimizer = spectral.new_optimizer([p for _, p in old_params], 1e-7)
    new = refine.train_batch(model, bank, [0, 1], optimizer, scales, "repair_control", objectives,
        {"wave": 1., "long": .37, "short": 2.}, device="cpu", preservation_weight=100., active_weight=1.)
    old = spectral.train_batch(previous, bank, [0, 1], old_optimizer, scales, "joint_spectral",
        100., objectives["long"], .37, device="cpu")
    assert new["loss"] == old["loss"]
    assert new["branch_losses"]["wave"] == old["wave_loss"]
    assert new["branch_losses"]["long"] == old["spectral_loss"]
    assert "short" not in new["branch_losses"]
    assert all(torch.equal(value, previous.state_dict()[name]) for name, value in model.state_dict().items())


def test_short_mel_is_separate_and_long_branch_is_unchanged_in_value_and_gradient(objectives):
    p = torch.randn(1, 1, 300, requires_grad=True)
    target = torch.randn_like(p)*.2
    valid = torch.zeros_like(p, dtype=torch.bool)
    valid[..., 5:145] = True
    valid[..., 200:250] = True  # Eligible short span, but no long-scale spectrum.
    quiet = valid.clone()
    quiet[..., 75:] = False
    row = {"target": target, "baseline": p.detach().clone()*.9, "valid": valid, "quiet": quiet,
           "quiet_samples": int(quiet.sum()), "outside_samples": int((valid & ~quiet).sum())}
    totals, spectral_totals = refine.batch_totals([row], objectives)
    assert totals == {"quiet_samples": 70, "outside_samples": 120}
    assert spectral_totals["long"] == (15, 4)
    assert spectral_totals["short"] == (41, 17)
    args = dict(device="cpu", preservation_weight=100., active_weight=1.)
    base = refine.loss_branches(p, row, totals, spectral_totals,
        {"quiet_mse": 2., "outside_mse": 4.}, objectives, "teacher", **args)
    extended = refine.loss_branches(p, row, totals, spectral_totals,
        {"quiet_mse": 2., "outside_mse": 4.}, objectives, "short", **args)
    assert set(base) == {"wave", "long"}
    assert set(extended) == {"wave", "long", "short"}
    for name in base:
        torch.testing.assert_close(base[name], extended[name], rtol=0, atol=0)
    long_grad = torch.autograd.grad(base["long"], p, retain_graph=True)[0]
    new_long_grad = torch.autograd.grad(extended["long"], p, retain_graph=True)[0]
    assert torch.equal(long_grad, new_long_grad)
    assert not bool(long_grad[..., 200:250].any())
    short_grad = torch.autograd.grad(extended["short"], p)[0]
    assert bool(short_grad[..., 200:250].any())
    assert not bool(short_grad[~valid].any())


@pytest.mark.parametrize("arm", refine.ARMS)
def test_real_update_preserves_frozen_body_and_normalization_with_fresh_optimizer(arm, objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source="a"), bank_row(model, source="b", target_offset=-.0003)]
    scales, _ = joint.baseline_scales(bank)
    named = refine.configure_head(model, arm)
    state = deepcopy(model.state_dict())
    flags = tuple(m.training for m in model.modules())
    optimizer = refine.new_optimizer([p for _, p in named], 1e-8)
    assert not optimizer.state
    result = refine.train_batch(model, bank, [0, 1], optimizer, scales, arm, objectives,
        {"wave": 1., "long": .37, "short": .11}, device="cpu", preservation_weight=100., active_weight=1.)
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, state[name])}
    assert changed and changed.issubset({name for name, _ in named})
    assert result["parameter_displacement"] > 0
    assert tuple(m.training for m in model.modules()) == flags
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
    assert all(p.grad is None for name, p in model.named_parameters() if name not in dict(named))
    assert optimizer.state
    if arm in ("short", "wn_short"):
        assert result["branch_losses"]["short"] > 0
    else:
        assert "short" not in result["branch_losses"]
    before_fold = joint.head_forward(model, bank[0]["features"]).detach()
    refine.fold_weight_norm(model)
    after_fold = joint.head_forward(model, bank[0]["features"]).detach()
    assert torch.equal(before_fold, after_fold)


def saved_head(model):
    return {"format_version": "spectral_head_factorial_v1", "base_checkpoint_sha256": "a"*64,
        "arm": "joint_spectral", "mode": "raw", "updates": 256,
        "head_parameters": joint.head_state(model)}


def test_candidate_load_verifies_parent_and_head_bytes_without_mutating_artifact(tmp_path):
    model = tiny_decoder()
    parent = deepcopy(model.state_dict())
    payload = saved_head(model)
    payload["head_parameters"]["output.weight"] *= 1.01
    path = tmp_path/"head.pt"
    torch.save(payload, path)
    original_bytes = path.read_bytes()
    sha = hashlib.sha256(original_bytes).hexdigest()
    receipt = refine.load_candidate_head(model, path, sha, "a"*64)
    assert receipt["sha256"] == sha
    assert path.read_bytes() == original_bytes
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload["head_parameters"].get(name, parent[name]))


@pytest.mark.parametrize("problem", ["file_hash", "parent_hash", "arm", "shape", "dtype", "nonfinite", "missing", "updates"])
def test_invalid_candidate_load_fails_before_any_model_mutation(tmp_path, problem):
    model = tiny_decoder()
    initial = deepcopy(model.state_dict())
    payload = saved_head(model)
    # All valid head values differ from the model. If a late check copies any
    # early tensor first, the unchanged-state assertion detects partial restore.
    for value in payload["head_parameters"].values():
        value.add_(.1)
    if problem == "parent_hash":
        payload["base_checkpoint_sha256"] = "b"*64
    elif problem == "arm":
        payload["arm"] = "projection_spectral"
    elif problem == "shape":
        payload["head_parameters"]["output.weight"] = torch.ones(2)
    elif problem == "dtype":
        payload["head_parameters"]["output.weight"] = payload["head_parameters"]["output.weight"].double()
    elif problem == "nonfinite":
        payload["head_parameters"]["output.weight"].view(-1)[0] = float("nan")
    elif problem == "missing":
        del payload["head_parameters"]["output.weight"]
    elif problem == "updates":
        payload["updates"] = 255
    path = tmp_path/"head.pt"
    torch.save(payload, path)
    before_bytes = path.read_bytes()
    sha = hashlib.sha256(before_bytes).hexdigest() if problem != "file_hash" else "0"*64
    with pytest.raises(ValueError):
        refine.load_candidate_head(model, path, sha, "a"*64)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in initial.items())
    assert path.read_bytes() == before_bytes


@pytest.mark.parametrize("arm", [arm for arm in refine.ARMS if arm != "repair_control"])
def test_all_teacher_branches_share_teacher_waveform_as_optimum(arm, objectives):
    target = torch.randn(1, 1, 300)*.2
    prediction = target.clone().requires_grad_(True)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :10] = False
    quiet = valid.clone()
    quiet[..., 150:] = False
    row = {"target": target, "baseline": target+.7, "valid": valid, "quiet": quiet,
           "quiet_samples": int(quiet.sum()), "outside_samples": int((valid & ~quiet).sum())}
    totals, mel_counts = refine.batch_totals([row], objectives)
    branches = refine.loss_branches(prediction, row, totals, mel_counts,
        {"quiet_mse": .01, "outside_mse": .02}, objectives, arm, device="cpu",
        preservation_weight=100., active_weight=1.)
    assert all(value.item() == 0 for value in branches.values())
    sum(branches.values()).backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_gradient_probe_uses_only_declared_training_examples_without_state_or_grad_changes(monkeypatch, objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(3)]
    scales, _ = joint.baseline_scales(bank)
    named = refine.configure_head(model, "wn_short")
    for _, parameter in named:
        parameter.grad = torch.full_like(parameter, .125)
    before = deepcopy(model.state_dict())
    grads = {name: parameter.grad.clone() for name, parameter in named}
    modes = tuple(m.training for m in model.modules())
    rng = torch.random.get_rng_state().clone()
    real_forward = refine.head_forward
    allowed = {id(bank[i]["features"]) for i in (0, 2)}
    used = []
    def observed_forward(model, features):
        assert id(features) in allowed
        used.append(id(features))
        return real_forward(model, features)
    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic gradient probe must not create an optimizer")
    monkeypatch.setattr(refine, "head_forward", observed_forward)
    monkeypatch.setattr(refine, "new_optimizer", forbidden)
    result = refine.gradient_probe(model, bank + [{"source_id": "heldout-do-not-use"}], [0, 2],
        scales, "wn_short", objectives, {"wave": 1., "long": .3, "short": .1}, device="cpu",
        preservation_weight=100., active_weight=1.)
    assert result["source_ids"] == ["train-0", "train-2"]
    assert result["optimizer_updates"] == 0 and len(used) == 2
    assert set(result["output_gradient_energies"]) == {"wave", "long", "short"}
    assert all(value > 0 for value in result["output_gradient_energies"].values())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(torch.equal(parameter.grad, grads[name]) for name, parameter in named)
    assert tuple(m.training for m in model.modules()) == modes
    assert torch.equal(rng, torch.random.get_rng_state())


def test_coefficient_calibration_uses_first_four_training_batches_and_keeps_long_weight(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    scales, _ = joint.baseline_scales(bank)
    tensor_copies = [{key: value.clone() for key, value in row.items() if torch.is_tensor(value)} for row in bank]
    rng = torch.random.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("Output-gradient coefficient calibration must not execute a head or optimizer")
    monkeypatch.setattr(refine, "head_forward", forbidden)
    monkeypatch.setattr(refine, "new_optimizer", forbidden)
    result = refine.calibrate_coefficients(bank + [{"source_id": "do-not-use"}], scales, objectives,
        SimpleNamespace(batch_size=2, preservation_weight=100., active_weight=1.), "cpu")
    assert result["optimizer_updates"] == 0
    assert [r["source_ids"] for r in result["batches"]] == [[f"train-{i}", f"train-{i+1}"] for i in range(0, 8, 2)]
    oracle_energy = {"wave": 0., "long": 0., "short": 0.}
    for start in range(0, 8, 2):
        rows = bank[start:start+2]
        ps = [row["baseline"].clone().requires_grad_(True) for row in rows]
        qcount = sum(row["quiet_samples"] for row in rows)
        ocount = sum(row["outside_samples"] for row in rows)
        quiet = sum((p-row["target"])[row["quiet"]].square().sum() for p, row in zip(ps, rows))/qcount/scales["quiet_mse"]
        active = sum((p-row["target"])[row["valid"] & ~row["quiet"]].square().sum() for p, row in zip(ps, rows))/ocount/scales["outside_mse"]
        terms = {"wave": quiet+active}
        for name, objective in objectives.items():
            terms[name] = ReconstructionV2(objective.config).forward_groups(
                (p[..., 19:-17], row["target"][..., 19:-17]) for p, row in zip(ps, rows)).losses["teacher_mel"]
        for name, value in terms.items():
            grads = torch.autograd.grad(value, ps, retain_graph=True)
            oracle_energy[name] += sum(float(g.double().square().sum()) for g in grads)
    for name, energy in oracle_energy.items():
        assert result["summed_output_gradient_energies"][name] == pytest.approx(energy, rel=3e-6)
    weights = result["coefficients"]
    assert weights["wave"] == 1.
    assert weights["long"] == pytest.approx((oracle_energy["wave"]/oracle_energy["long"])**.5, rel=3e-6)
    assert weights["short"] == pytest.approx(.25*weights["long"]*(oracle_energy["long"]/oracle_energy["short"])**.5, rel=3e-6)
    assert weights["long"]**2*oracle_energy["long"] == pytest.approx(oracle_energy["wave"], rel=3e-6)
    assert weights["short"]**2*oracle_energy["short"] == pytest.approx(.25**2*oracle_energy["wave"], rel=3e-6)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(tensor, bank[i][name]) for i, row in enumerate(tensor_copies) for name, tensor in row.items())


def test_rate_calibration_isolates_every_arm_update_and_rejects_nontraining_data(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    initial = refine.effective_head_state(model)
    state = deepcopy(model.state_dict())
    scales, _ = joint.baseline_scales(bank)
    rng = torch.random.get_rng_state().clone()
    real_new, real_score, real_train = refine.new_optimizer, refine.bank_score, refine.train_batch
    created, updates = [], []
    allowed = {id(row) for row in bank}
    def observe_new(*args, **kwargs):
        for name, value in refine.effective_head_state(model).items():
            torch.testing.assert_close(value, initial[name], rtol=2e-6, atol=1e-7)
        result = real_new(*args, **kwargs)
        assert not result.state
        created.append(result)
        return result
    def observe_score(model, selected, *args, **kwargs):
        assert len(selected) == 8 and {id(row) for row in selected} == allowed
        return real_score(model, selected, *args, **kwargs)
    def observe_train(model, bank, indices, optimizer, scales, arm, *args, **kwargs):
        assert len(indices) == 2 and all(0 <= i < 8 for i in indices)
        updates.append((arm, tuple(indices)))
        return real_train(model, bank, indices, optimizer, scales, arm, *args, **kwargs)
    monkeypatch.setattr(refine, "new_optimizer", observe_new)
    monkeypatch.setattr(refine, "bank_score", observe_score)
    monkeypatch.setattr(refine, "train_batch", observe_train)
    result = refine.calibrate_rate(model, bank + [{"source_id": "heldout-do-not-use"}], initial, scales,
        objectives, {"wave": 1., "long": .3, "short": .1},
        SimpleNamespace(batch_size=2, learning_rate=1e-10, preservation_weight=100., active_weight=1.), "cpu")
    assert result["chosen_learning_rate"] is not None
    assert len(created) == len(refine.ARMS)*4*len(result["trials"])
    assert len({id(opt) for opt in created}) == len(created)
    expected_updates = [(arm, (i, i+1)) for arm in refine.ARMS for i in range(0, 8, 2)]
    assert updates == expected_updates*len(result["trials"])
    assert result["retained_optimizer_updates"] == 0
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_rate_calibration_failure_restores_candidate_exactly(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}") for i in range(4)]
    initial = refine.effective_head_state(model)
    state = deepcopy(model.state_dict())
    scales, _ = joint.baseline_scales(bank)
    def broken(*args, **kwargs):
        p = model.output.weight
        with torch.no_grad():
            p.add_(1.)
        p.grad = torch.ones_like(p)
        raise FloatingPointError("injected training failure")
    monkeypatch.setattr(refine, "train_batch", broken)
    with pytest.raises(FloatingPointError, match="injected"):
        refine.calibrate_rate(model, bank, initial, scales, objectives, {"wave": 1., "long": .3, "short": .1},
            SimpleNamespace(batch_size=1, learning_rate=1e-7, preservation_weight=100., active_weight=1.), "cpu")
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("baseline_quiet_error", [1e-6, 0.])
def test_teacher_perfect_gate_allows_restoring_under_fullscale_amplitude_and_preserving_exact_quiet(baseline_quiet_error):
    baseline = quality_report()
    candidate = deepcopy(baseline)
    for row in [baseline["aggregate"], *baseline["groups"].values()]:
        row["peak"] = .8
        row["overshoot_samples"] = 0
        row["quiet_mse"] = baseline_quiet_error
    for row in [candidate["aggregate"], *candidate["groups"].values()]:
        for key in ("mae", "mse", "quiet_mse", "outside_mse", "overshoot_samples"):
            row[key] = 0
        row["outside_drift_mse"] = 123.
        row["peak"] = .9
    for region in candidate["spectral_sources"][0]["regions"].values():
        region["mel"] = 0.
    assert refine.teacher_selection_checks(candidate, baseline)["qualified"]
    candidate["aggregate"]["peak"] = 1.00001
    candidate["aggregate"]["overshoot_samples"] = 1
    assert not refine.teacher_selection_checks(candidate, baseline)["qualified"]
    if baseline_quiet_error == 0:
        candidate["aggregate"]["peak"] = .9
        candidate["aggregate"]["overshoot_samples"] = 0
        candidate["aggregate"]["quiet_mse"] = 1e-10
        assert not refine.teacher_selection_checks(candidate, baseline)["qualified"]


def test_folding_frozen_weight_norm_keeps_plain_frozen_parameters():
    model = tiny_decoder()
    refine.configure_head(model, "wn")
    model.requires_grad_(False)
    before = refine.effective_head_state(model)
    refine.fold_weight_norm(model)
    for module in (model.head.conv, model.output):
        assert isinstance(module.weight, torch.nn.Parameter)
        assert not module.weight.requires_grad
        assert "weight" not in module._buffers
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)


def peak_report(peaks, counts):
    return {"rows": [{"source_id": str(i), "start_frame": 0,
        "student_peak_abs": peak, "student_overshoot_samples": count}
        for i, (peak, count) in enumerate(zip(peaks, counts))],
        "recovery_metrics": {"all": {"maximum_peak": max(peaks), "scored_overshoot_samples": sum(counts)}}}


def test_canonical_peak_screen_allows_under_fullscale_restoration_but_no_new_overshoot():
    baseline = peak_report([.8, .6], [0, 0])
    assert refine.refinement_peak_screen(baseline, peak_report([.9, .99], [0, 0]))["passed"]
    assert not refine.refinement_peak_screen(baseline, peak_report([.9, 1.001], [0, 1]))["passed"]
    baseline = peak_report([1.3, .6], [20, 0])
    # A lower overall maximum cannot hide a new overshooting recording.
    assert not refine.refinement_peak_screen(baseline, peak_report([1.2, 1.001], [10, 1]))["passed"]
    # Nor may a wider old overshoot be hidden by fewer observations elsewhere.
    baseline = peak_report([1.3, 1.1], [20, 2])
    assert not refine.refinement_peak_screen(baseline, peak_report([1.2, 1.1], [10, 3]))["passed"]
