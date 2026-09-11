from copy import deepcopy

import pytest
import torch

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.training import _rng_state
from diagnose_objectives_updates import _capture_replay_state
from diagnose_step_path import set_parameter_fraction, one_step_path, FRACTIONS
# Reuse the small, calibrated real-engine fixture. It runs no retained model.
from test_diagnose_objectives_updates import prepared, deterministic


def test_interpolation_preserves_endpoints_and_buffers():
    model = torch.nn.Linear(3, 2)
    model.register_buffer("unchanged", torch.tensor([4.]))
    before = deepcopy(model.state_dict())
    after = {name: value + .1 for name, value in before.items()}
    set_parameter_fraction(model, before, after, 1)
    assert all(torch.equal(p, after[name]) for name, p in model.named_parameters())
    assert model.unchanged == 4
    set_parameter_fraction(model, before, after, .25)
    for name, p in model.named_parameters():
        assert torch.equal(p, torch.lerp(before[name].double(), after[name].double(), .25).float())
    set_parameter_fraction(model, before, after, 0)
    assert state_fingerprint(model.state_dict()) == state_fingerprint(before)
    set_parameter_fraction(model, before, after, 1., {"bias"})
    assert torch.equal(model.bias, after["bias"])
    assert torch.equal(model.weight, before["weight"])
    with pytest.raises(ValueError):
        set_parameter_fraction(model, before, after, 2)


def test_native_path_has_identical_zero_endpoint_and_restores_full_state(prepared):
    engine, crops = prepared
    original_state = state_fingerprint(_capture_replay_state(engine))
    rng = state_fingerprint(_rng_state())
    gradient = next(engine.model.parameters()).grad
    saved_gradient = gradient.clone()
    report = one_step_path(engine, crops, crops)
    assert tuple(row["fraction"] for row in report["fractions"]) == FRACTIONS
    assert all(value == 0 for value in report["fractions"][0]["absolute_change"].values())
    assert report["native_training_step_calls"] == 1 and report["retained_updates"] == 0
    assert report["state_restored"]
    assert report["model_buffers_unchanged_after_native_step"]
    assert state_fingerprint(_capture_replay_state(engine)) == original_state
    assert state_fingerprint(_rng_state()) == rng
    assert next(engine.model.parameters()).grad is gradient and torch.equal(gradient, saved_gradient)
    assert any(value != 0 for value in report["fractions"][-1]["absolute_change"].values())
    groups = report["delta_localization"]
    parts = [set(groups[name]["parameter_names"]) for name in ("output_projection", "late_blocks_8_9", "remaining_parameters")]
    assert len(set.union(*parts)) == sum(len(part) for part in parts)
    assert set.union(*parts) == set(dict(engine.model.named_parameters()))
    # This test fixture uses AdamW exclusively, so its subset is the full delta.
    assert groups["adamw_parameters"]["absolute_change"] == report["fractions"][-1]["absolute_change"]
    assert all(value == 0 for value in groups["muon_parameters"]["absolute_change"].values())
