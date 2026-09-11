"""CPU integration checks for matched last-block and auxiliary-feature pilots.

The existing feature-method tests cover source authentication and affine-fit
math. These tests check their integration into actual runner scopes, masks,
shared reconstruction, calibration and state restoration.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fast-audiovae/experiments/convnext"))

from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.reconstruction_v2 import ReconstructionV2Config
import refinement_features as feature
import run_joint_heads as joint
import run_spectral_heads as spectral
import run_teacher_refinements_v2 as refine
import run_late_refinements as late


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(813924)


def tiny_decoder():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16,
        head_channels=12, layer_scale_init=1., normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias"))
    with torch.no_grad():
        model(torch.randn(2, 64, 11))
    return model.freeze_normalization_statistics().eval()


def bank_row(model, frames=3, *, source="fixture", target_offset=.0002):
    z = torch.randn(1, 64, frames)
    prefix, baseline = late.capture_last_block_input(model, z)
    _, hidden = feature.last_block_forward(model, prefix, return_features=True)
    hidden = hidden.detach()
    target = baseline + target_offset
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :19] = False
    valid[..., -17:] = False
    quiet = valid.clone()
    quiet[..., target.shape[-1] // 3:] = False
    teacher_features = torch.stack((hidden[:, :3], hidden[:, :3]*1.3+.02), -1).flatten(-2)
    teacher_features += .0005*torch.randn_like(teacher_features)
    return {"features": prefix, "target": target, "baseline": baseline,
        "valid": valid, "quiet": quiet, "quiet_samples": int(quiet.sum()),
        "outside_samples": int((valid & ~quiet).sum()), "source_id": source,
        "start_frame": 0, "language": "fixture", "condition": "speech", "dataset": "unit",
        "pool_index": 0, "teacher_features": teacher_features,
        "feature_valid": feature.complete_teacher_feature_mask(valid),
        "_latents": z, "student_features": hidden}


@pytest.fixture
def objectives():
    return {"long": spectral.SpectralObjective(ReconstructionV2Config(fft_sizes=(64, 128), mel_bands=(3, 4))),
            "short": spectral.SpectralObjective(ReconstructionV2Config(fft_sizes=(16, 32), mel_bands=(1, 1)))}


def make_auxiliary(bank):
    normalization = feature.fit_feature_normalization(bank)
    adapter = feature.AuxiliaryPhaseReadout(bank[0]["student_features"].shape[1],
                                           bank[0]["teacher_features"].shape[1])
    feature.fit_auxiliary_ridge(adapter, bank, normalization)
    return {"adapter": adapter, "normalization": normalization}


def args(**overrides):
    return SimpleNamespace(**{"batch_size": 2, "preservation_weight": 100., "active_weight": 1.,
                              "learning_rate": 1e-10, **overrides})


def test_reconstruction_gates_and_acoustic_calibration_are_shared_helpers():
    assert late.teacher_selection_checks is refine.teacher_selection_checks
    assert late.finite_update_checks is refine.finite_update_checks
    assert late.refinement_peak_screen is refine.refinement_peak_screen
    assert late.calibrate_coefficients is refine.calibrate_coefficients
    assert late.loss_branches is refine.loss_branches


def test_prefix_replay_and_score_view_match_full_decoder_and_existing_scoring(objectives):
    model = tiny_decoder()
    rows = [bank_row(model, source="a"), bank_row(model, frames=4, source="b")]
    original = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    previous = []
    for row in rows:
        head_input, full_audio = joint.capture_prehead(model, row["_latents"])
        assert torch.equal(full_audio, row["baseline"])
        assert torch.equal(late.last_block_forward(model, row["features"]), full_audio)
        previous.append({**row, "features": head_input})
    old = spectral.bank_score(model, previous, "cpu", objectives["long"])
    observed = late.bank_score(model, rows, "cpu", objectives["long"])
    assert observed == old
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not model.blocks[9]._forward_pre_hooks


def test_late_state_is_exactly_block9_with_layernorm_affine_and_head():
    model = tiny_decoder()
    named = late.select_last_block_parameters(model)
    expected = joint.HEAD_NAMES | {name for name, _ in model.named_parameters() if name.startswith("blocks.9.")}
    assert {name for name, _ in named} == expected
    assert {"blocks.9.norm.weight", "blocks.9.norm.bias"}.issubset(expected)
    assert not any(name.startswith(("affine.", "stem.", "stem_norm.", "adapter.")) for name in expected)
    assert set(late.late_state(model)) == expected
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    assert all(not module.training for module in model.modules())


def test_full_and_prefix_replay_have_identical_trainable_gradients(objectives):
    model = tiny_decoder()
    row = bank_row(model)
    named = late.select_last_block_parameters(model)
    parameters = [p for _, p in named]
    scales, _ = joint.baseline_scales([row])
    totals, mel_counts = late.batch_totals([row], objectives)
    kwargs = dict(device="cpu", preservation_weight=100., active_weight=1.)
    full = late.loss_branches(model(row["_latents"]), row, totals, mel_counts, scales, objectives, "teacher", **kwargs)
    replay_audio, replay = late.sample_branches(model, row, totals, mel_counts, 0, scales,
        objectives, "late", None, device="cpu", args=args())
    assert torch.equal(replay_audio, model(row["_latents"]))
    expected = torch.autograd.grad(full["wave"]+.3*full["long"], parameters)
    actual = torch.autograd.grad(replay["wave"]+.3*replay["long"], parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_teacher_feature_attachment_accepts_only_exact_waveform_and_240_sample_masks(monkeypatch):
    model = tiny_decoder()
    source = bank_row(model)
    source.pop("teacher_features"); source.pop("feature_valid")
    source["valid"][..., 2*240+137] = False
    fixture = {"teacher_features": torch.randn(1, 1024, source["valid"].shape[-1]//240),
               "valid": source["valid"].clone(),
               "feature_valid": source["valid"].reshape(1, 1, -1, 240).all(-1),
               "receipt": {"source_id": "fixture", "target_changed": False}}
    calls = []
    crop = object()
    def capture(teacher, actual_crop, data, **kwargs):
        assert actual_crop is crop and kwargs["expected_teacher_state_sha256"] == "a"*64
        calls.append(1)
        return deepcopy(fixture)
    monkeypatch.setattr(late, "capture_full_source_teacher_features", capture)
    target = source["target"].clone()
    valid = source["valid"].clone()
    receipt = late.attach_teacher_features([source], [0], None, [crop], {}, "a"*64, status=lambda *a, **kw: None)
    assert len(calls) == 1 and receipt == [fixture["receipt"]]
    assert torch.equal(source["valid"], valid) and torch.equal(source["target"], target)
    assert torch.equal(source["feature_valid"], valid.reshape(1, 1, -1, 240).all(-1))
    assert not source["feature_valid"][..., 2].item()
    assert not source["teacher_features"].requires_grad


@pytest.mark.parametrize("problem", ["wave_mask", "feature_mask"])
def test_teacher_feature_attachment_rejects_mask_mismatch_before_adding_targets(monkeypatch, problem):
    model = tiny_decoder()
    row = bank_row(model)
    row.pop("teacher_features"); row.pop("feature_valid")
    result = {"teacher_features": torch.randn(1, 1024, row["valid"].shape[-1]//240),
              "valid": row["valid"].clone(),
              "feature_valid": feature.complete_teacher_feature_mask(row["valid"]), "receipt": {}}
    if problem == "wave_mask":
        result["valid"][..., 300] = ~result["valid"][..., 300]
    else:
        result["feature_valid"][..., 2] = ~result["feature_valid"][..., 2]
    monkeypatch.setattr(late, "capture_full_source_teacher_features", lambda *a, **kw: result)
    with pytest.raises(RuntimeError, match="sealed score mask"):
        late.attach_teacher_features([row], [0], None, [object()], {}, "a"*64, status=lambda *a, **kw: None)
    assert "teacher_features" not in row and "feature_valid" not in row


@pytest.mark.parametrize("arm", late.ARMS)
def test_actual_update_preserves_frozen_prefix_norms_and_training_only_adapter(arm, objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source="a"), bank_row(model, frames=4, source="b", target_offset=-.0003)]
    auxiliary = make_auxiliary(bank)
    adapter_state = deepcopy(auxiliary["adapter"].state_dict())
    norm_mean, norm_scale = auxiliary["normalization"].mean.clone(), auxiliary["normalization"].scale.clone()
    state = deepcopy(model.state_dict())
    named = late.select_last_block_parameters(model)
    optimizer = late.new_optimizer([p for _, p in named], 1e-7)
    scales, _ = joint.baseline_scales(bank)
    result = late.train_batch(model, bank, [0, 1], optimizer, scales, arm, objectives,
        {"wave": 1., "long": .3, "feature": .1}, auxiliary, device="cpu", args=args())
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, state[name])}
    allowed = {name for name, _ in named}
    assert changed and changed.issubset(allowed)
    assert result["parameter_displacement"] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
    assert all(p.grad is None for name, p in model.named_parameters() if name not in allowed)
    assert all(not m.training for m in model.modules())
    assert all(torch.equal(value, auxiliary["adapter"].state_dict()[name]) for name, value in adapter_state.items())
    assert not any(p.requires_grad or p.grad is not None for p in auxiliary["adapter"].parameters())
    assert torch.equal(norm_mean, auxiliary["normalization"].mean)
    assert torch.equal(norm_scale, auxiliary["normalization"].scale)
    assert not any("aux" in name or "projection" in name for name in late.late_state(model))
    assert not any(module is auxiliary["adapter"] for module in model.modules())
    assert set(late.late_state(model)) == allowed
    if arm == "aux":
        assert result["branch_losses"]["feature"] > 0
        assert result["feature_cells"] == sum(int(row["feature_valid"].sum()) for row in bank)
    else:
        assert "feature" not in result["branch_losses"]


def test_auxiliary_adds_feature_loss_without_changing_teacher_reconstruction(objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source="a"), bank_row(model, source="b")]
    aux = make_auxiliary(bank)
    late.select_last_block_parameters(model)
    scales, _ = joint.baseline_scales(bank)
    totals, counts = late.batch_totals(bank, objectives)
    cells = sum(int(row["feature_valid"].sum()) for row in bank)
    plain_audio, plain = late.sample_branches(model, bank[0], totals, counts, 0, scales, objectives,
        "late", None, device="cpu", args=args())
    aux_audio, extra = late.sample_branches(model, bank[0], totals, counts, cells, scales, objectives,
        "aux", aux, device="cpu", args=args())
    assert torch.equal(plain_audio, aux_audio)
    assert set(plain) == {"wave", "long"} and set(extra) == {"wave", "long", "feature"}
    assert all(torch.equal(plain[name], extra[name]) for name in plain)
    # The auxiliary loss enters through block features, never through waveform
    # head parameters or a new deployed output layer.
    names = dict(model.named_parameters())
    grads = torch.autograd.grad(extra["feature"], [names["output.weight"], names["blocks.9.norm.weight"]], allow_unused=True)
    assert grads[0] is None
    assert grads[1] is not None and grads[1].abs().sum() > 0


def test_auxiliary_coefficient_uses_only_first_four_training_batches_and_shared_block_gradients(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    auxiliary = make_auxiliary(bank)
    late.select_last_block_parameters(model)
    state = deepcopy(model.state_dict())
    aux_state = deepcopy(auxiliary["adapter"].state_dict())
    scales, _ = joint.baseline_scales(bank)
    coeffs = {"wave": 1., "long": .3}
    rng = torch.random.get_rng_state().clone()
    real_branches = late.sample_branches
    used = []
    allowed = {id(row) for row in bank}
    def observed_branches(model, row, *a, **kw):
        assert id(row) in allowed
        used.append(row["source_id"])
        return real_branches(model, row, *a, **kw)
    def forbidden(*a, **kw):
        raise AssertionError("Auxiliary coefficient calibration may not create an optimizer")
    monkeypatch.setattr(late, "sample_branches", observed_branches)
    monkeypatch.setattr(late, "new_optimizer", forbidden)
    result = late.calibrate_auxiliary_weight(model, bank + [{"source_id": "heldout-do-not-use"}],
        scales, objectives, coeffs, auxiliary, args(), "cpu")
    assert used == [f"train-{i}" for i in range(8)]
    assert [row["source_ids"] for row in result["batches"]] == [[f"train-{i}", f"train-{i+1}"] for i in range(0, 8, 2)]
    params = [p for name, p in model.named_parameters() if name.startswith("blocks.9.")]
    energy = {"reconstruction": 0., "feature": 0.}
    for start in range(0, 8, 2):
        selected = bank[start:start+2]
        totals, counts = late.batch_totals(selected, objectives)
        cells = sum(int(row["feature_valid"].sum()) for row in selected)
        combined = {"reconstruction": 0., "feature": 0.}
        for row in selected:
            hidden = model.blocks[9](row["features"])
            prediction = model._waveform(model.output(model.activation(model.head(model.affine(hidden)))))
            parts = refine.loss_branches(prediction, row, totals, counts, scales, objectives,
                "teacher", device="cpu", preservation_weight=100., active_weight=1.)
            combined["reconstruction"] += parts["wave"]+.3*parts["long"]
            combined["feature"] += feature.auxiliary_feature_loss(auxiliary["adapter"], hidden,
                row["teacher_features"], row["feature_valid"], auxiliary["normalization"], total_valid_cells=cells)
        for name, value in combined.items():
            grads = torch.autograd.grad(value, params, retain_graph=True)
            energy[name] += sum(float(g.double().square().sum()) for g in grads)
    for name in energy:
        assert result["summed_shared_block_energies"][name] == pytest.approx(energy[name], rel=4e-6)
    assert result["feature_weight"] == pytest.approx(.1*(energy["reconstruction"]/energy["feature"])**.5, rel=4e-6)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(torch.equal(value, auxiliary["adapter"].state_dict()[name]) for name, value in aux_state.items())
    assert all(p.grad is None for p in model.parameters())
    assert not any(p.requires_grad or p.grad is not None for p in auxiliary["adapter"].parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("which", ["coefficient", "rate"])
def test_calibration_requires_all_four_complete_training_batches(which, objectives):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}") for i in range(3)]
    auxiliary = make_auxiliary(bank)
    late.select_last_block_parameters(model)
    scales, _ = joint.baseline_scales(bank)
    coeffs = {"wave": 1., "long": .3, "feature": .1}
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        if which == "coefficient":
            late.calibrate_auxiliary_weight(model, bank, scales, objectives, coeffs, auxiliary, args(batch_size=1), "cpu")
        else:
            late.calibrate_rate(model, bank, late.late_state(model), scales, objectives, coeffs, auxiliary, args(batch_size=1), "cpu")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(p.grad is None for p in model.parameters())


def test_rate_calibration_has_matched_fresh_states_training_only_panel_and_exact_restore(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}", target_offset=.0001*(i+1)) for i in range(8)]
    auxiliary = make_auxiliary(bank)
    late.select_last_block_parameters(model)
    state = deepcopy(model.state_dict())
    initial = late.late_state(model)
    scales, _ = joint.baseline_scales(bank)
    rng = torch.random.get_rng_state().clone()
    real_new, real_score, real_train = late.new_optimizer, late.bank_score, late.train_batch
    created, updates = [], []
    allowed = {id(row) for row in bank}
    def observe_new(*a, **kw):
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
        result = real_new(*a, **kw)
        assert not result.state
        created.append(result)
        return result
    def observe_score(model, selected, *a, **kw):
        assert len(selected) == 8 and {id(row) for row in selected} == allowed
        return real_score(model, selected, *a, **kw)
    def observe_train(model, bank, indices, optimizer, scales, arm, *a, **kw):
        assert len(indices) == 2 and all(0 <= i < 8 for i in indices)
        updates.append((arm, tuple(indices)))
        return real_train(model, bank, indices, optimizer, scales, arm, *a, **kw)
    monkeypatch.setattr(late, "new_optimizer", observe_new)
    monkeypatch.setattr(late, "bank_score", observe_score)
    monkeypatch.setattr(late, "train_batch", observe_train)
    result = late.calibrate_rate(model, bank + [{"source_id": "heldout-do-not-use"}], initial, scales,
        objectives, {"wave": 1., "long": .3, "feature": .1}, auxiliary, args(), "cpu")
    assert result["chosen_learning_rate"] is not None
    assert len(created) == len(late.ARMS)*4*len(result["trials"])
    assert len({id(opt) for opt in created}) == len(created)
    assert updates == [(arm, (i, i+1)) for arm in late.ARMS for i in range(0, 8, 2)]*len(result["trials"])
    assert result["retained_optimizer_updates"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_exception_during_auxiliary_trial_restores_full_late_state(objectives, monkeypatch):
    model = tiny_decoder()
    bank = [bank_row(model, source=f"train-{i}") for i in range(4)]
    auxiliary = make_auxiliary(bank)
    aux_state = deepcopy(auxiliary["adapter"].state_dict())
    late.select_last_block_parameters(model)
    state = deepcopy(model.state_dict())
    initial = late.late_state(model)
    scales, _ = joint.baseline_scales(bank)
    real_train = late.train_batch
    def broken(model, bank, indices, optimizer, scales, arm, *a, **kw):
        if arm == "aux":
            p = model.blocks[9].norm.weight
            with torch.no_grad():
                p.add_(1.)
            p.grad = torch.ones_like(p)
            raise FloatingPointError("injected auxiliary failure")
        return real_train(model, bank, indices, optimizer, scales, arm, *a, **kw)
    monkeypatch.setattr(late, "train_batch", broken)
    with pytest.raises(FloatingPointError, match="injected"):
        late.calibrate_rate(model, bank, initial, scales, objectives,
            {"wave": 1., "long": .3, "feature": .1}, auxiliary, args(batch_size=1), "cpu")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(torch.equal(value, auxiliary["adapter"].state_dict()[name]) for name, value in aux_state.items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype", "nonfinite"])
def test_restore_validates_complete_late_state_before_mutating_any_tensor(failure):
    model = tiny_decoder()
    before = deepcopy(model.state_dict())
    changed = late.late_state(model)
    for value in changed.values():
        value.add_(.1)
    if failure == "missing":
        del changed["output.weight"]
    elif failure == "shape":
        changed["output.weight"] = torch.ones(1)
    elif failure == "dtype":
        changed["output.weight"] = changed["output.weight"].double()
    else:
        changed["output.weight"].view(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        late.restore_late_state(model, changed)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
