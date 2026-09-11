from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.training import _rng_state
from render_canonical_targets import digest as inventory_digest
from run_comparison import (METHODS, apply_radius, baseline_mse_vjps, calibrate_radius,
    calibration_guards, generate_directions, heldout_hashes, select_batches)
from test_diagnose_objectives_updates import prepared, deterministic


def test_inventory_digest_matches_renderer_without_newline(tmp_path):
    sha = "a" * 64
    inventory = {"ready": True, "sources": [{"source_id": "natural", "manifest_row": {"audio_sha256": sha}}]}
    inventory["identity_sha256"] = inventory_digest(inventory)
    path = tmp_path / "inventory.json"; path.write_text(json.dumps(inventory))
    panel = {"crops": [SimpleNamespace(source_id="natural"), SimpleNamespace(source_id="encoded_zero")],
        "receipt": {"contract": {"source_inventory_identity_sha256": inventory["identity_sha256"]}}}
    assert heldout_hashes(panel, path) == ({sha}, inventory["identity_sha256"])
    inventory["sources"][0]["manifest_row"]["audio_sha256"] = "b" * 64
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="sealed"):
        heldout_hashes(panel, path)


def test_split_is_deterministic_source_hash_disjoint_and_teacher_stratified(prepared):
    _, original = prepared
    crops = [replace(original[0], source_id=str(i), teacher_audio=torch.zeros_like(original[0].teacher_audio)) for i in range(14)]
    rows = {str(i): {"audio_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(14)}
    rows["13"] = dict(rows["0"])  # Different ID, identical recording.
    excluded_hash = rows["2"]["audio_sha256"]
    options = dict(calibration_batches=1, comparison_batches=1, batch_size=4, min_quiet=1)
    a = select_batches(crops, rows, {"1"}, {excluded_hash}, **options)
    assert a == select_batches(crops, rows, {"1"}, {excluded_hash}, **options)
    selected = [r for b in a["batches"] for r in b["sources"]]
    assert len({r["source_id"] for r in selected}) == len({r["audio_sha256"] for r in selected}) == 8
    assert all(r["source_id"] != "1" and r["audio_sha256"] != excluded_hash for r in selected)
    assert all(b["quiet_sources"] >= 1 for b in a["batches"])


def guard(full, half, linear):
    p0 = torch.ones(1, 1, 4)
    mask = torch.ones_like(p0, dtype=torch.bool)
    return calibration_guards(p0, torch.full_like(p0, full), torch.full_like(p0, half),
        torch.zeros_like(p0), mask, mask, linear_terms={"valid": linear, "teacher_quiet": linear})


def test_quadratic_guard_rejects_overshoot_even_for_perfectly_linear_waveform():
    result = guard(-2., -.5, -6.)
    assert result["regions"]["valid"]["nonlinear_waveform_ratio"] == 0
    assert result["regions"]["valid"]["exact_squared_displacement_term"] == 9
    assert not result["passed"]
    assert guard(.8, .9, -.4)["passed"]


def test_descent_classification_uses_exact_parameter_tangent_not_half_output():
    result = guard(2., 1.1, -1.)
    assert result["regions"]["valid"]["half_step_estimated_linear_term"] > 0
    assert result["regions"]["valid"]["direction_classification"] == "descending"
    assert not result["passed"]
    positive = guard(1.2, 1.1, .4)
    assert positive["passed"]
    assert positive["regions"]["valid"]["direction_classification"] == "non_descending"
    assert positive["regions"]["valid"]["non_descending_caution"]


def test_small_waveform_nonlinearity_cannot_hide_measured_mse_reversal():
    p0 = torch.tensor([[[1., 0.]]]); full = torch.tensor([[[1.0001, .002]]]); half = torch.tensor([[[1., .001]]])
    mask = torch.ones_like(p0, dtype=torch.bool)
    result = calibration_guards(p0, full, half, torch.zeros_like(p0), mask, mask,
        linear_terms={"valid": -.0001, "teacher_quiet": -.0001})
    row = result["regions"]["valid"]
    assert row["nonlinear_waveform_ratio"] < .1
    assert row["quadratic_to_absolute_linear_ratio"] < .25
    assert not row["observed_descent_or_roundoff"]
    assert not result["passed"]


class TinyScale(torch.nn.Module):
    def __init__(self, value=1., squared=False):
        super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(value)); self.squared = squared
    def forward(self, z, scored_latent_mask=None):
        return z * (self.scale.square() if self.squared else self.scale)


def tiny_pack():
    z = torch.ones(1, 1, 4)
    return {"z": z, "latent_mask": torch.ones(1, 4, dtype=torch.bool), "target": torch.zeros_like(z),
            "valid": torch.ones_like(z, dtype=torch.bool), "quiet": torch.ones_like(z, dtype=torch.bool)}


def test_exact_training_vjp_and_rounded_parameter_radius_do_not_mutate_gradients():
    model = TinyScale(2.)
    model.scale.grad = torch.tensor(3.)
    engine = SimpleNamespace(model=model)
    prediction, gradients = baseline_mse_vjps(engine, tiny_pack())
    assert torch.all(prediction == 2) and float(gradients["valid"][0]) == 4
    assert float(model.scale.grad) == 3
    baseline = {"scale": model.scale.detach().clone()}
    result = apply_radius(model, baseline, (torch.tensor(-1.),), .2, metric_vectors=gradients)
    assert result["realized_parameter_l2"] == pytest.approx(.2, abs=1e-6)
    assert result["exact_parameter_first_order_mse_change"]["valid"] == pytest.approx(-.8, abs=1e-6)


def test_training_only_radius_grid_selects_guarded_value_and_restores_parameters():
    model = TinyScale(); engine = SimpleNamespace(model=model)
    pack = tiny_pack(); baseline, vectors = baseline_mse_vjps(engine, pack)
    directions = {name: (torch.tensor(-4.),) for name in METHODS}
    item = {"id": "calibration_fixture", "pack": pack, "baseline": baseline,
        "metric_vectors": vectors, "generated": {"directions": directions,
        "receipt": {"direction_norms": {"retained": 4.}}}}
    result = calibrate_radius(engine, {"scale": torch.tensor(1.)}, [item])
    assert result["resolved"] and result["selected_radius"] == .5
    assert result["selected_fraction"] == .125
    assert float(model.scale.detach()) == 1


def test_unresolved_grid_stops_without_expanding_candidates():
    model = TinyScale(0., squared=True); engine = SimpleNamespace(model=model)
    pack = tiny_pack(); baseline, vectors = baseline_mse_vjps(engine, pack)
    item = {"id": "nonlinear_fixture", "pack": pack, "baseline": baseline,
        "metric_vectors": vectors, "generated": {"directions": {name: (torch.tensor(1.),) for name in METHODS},
        "receipt": {"direction_norms": {"retained": 1.}}}}
    result = calibrate_radius(engine, {"scale": torch.tensor(0.)}, [item])
    assert not result["resolved"] and result["selected_radius"] is None
    assert len(result["attempts"]) == 8
    assert float(model.scale.detach()) == 0


def test_directions_share_actual_gradient_and_restore_full_engine(prepared):
    engine, crops = prepared
    before, rng = state_fingerprint(engine.state_dict()), state_fingerprint(_rng_state())
    old_gradient = next(engine.model.parameters()).grad
    saved_gradient = old_gradient.clone()
    result = generate_directions(engine, crops, global_rng=_rng_state())
    assert set(result["directions"]) == set(METHODS)
    assert result["receipt"]["generator_step_calls"] == 2
    assert result["receipt"]["discriminator_step_calls"] == 1
    assert result["receipt"]["shared_gradient_identical"] and result["receipt"]["D_and_balancer_shared"]
    assert result["receipt"]["state_restored"]
    assert any(not torch.equal(a, b) for a, b in zip(result["directions"]["retained"], result["directions"]["fresh_generator_state"]))
    assert state_fingerprint(engine.state_dict()) == before and state_fingerprint(_rng_state()) == rng
    assert next(engine.model.parameters()).grad is old_gradient and torch.equal(old_gradient, saved_gradient)
