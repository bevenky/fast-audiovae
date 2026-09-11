from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from benchmark_runtime import timing_summary, unique_batch_identity, finite_metrics, state_delta
from compare_runtime import compare_reports


def test_timing_reports_variance_and_rejects_invalid_values():
    row = timing_summary([1., 2., 3.])
    assert row["mean_seconds"] == 2
    assert row["stdev_seconds"] == 1
    assert row["coefficient_of_variation"] == .5
    for times in ([0., 1.], [float("nan"), 1.], [1.]):
        with pytest.raises(ValueError):
            timing_summary(times)


def test_scored_crops_cannot_repeat_but_context_can_overlap():
    a = SimpleNamespace(source_id="a", start_frame=1, context_start_frame=0,
                        valid_scored_samples=1920, latents=torch.zeros(1, 64, 2))
    b = SimpleNamespace(source_id="a", start_frame=2, context_start_frame=0,
                        valid_scored_samples=1920, latents=torch.zeros(1, 64, 3))
    assert len(unique_batch_identity([a, b])) == 2
    with pytest.raises(ValueError):
        unique_batch_identity([a, a])


def test_parameter_delta_is_read_only_and_requires_finite_gradient():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    initial = {"weight": torch.zeros_like(model.weight)}
    # Nonzero previous state permits relative norm; all coordinates change0.5.
    initial["weight"].fill_(.5)
    model.weight.grad = torch.ones_like(model.weight) * .2
    before = model.weight.clone()
    gradient = model.weight.grad.clone()
    row = state_delta(model, initial)
    assert row["update_rms"] == .5
    assert row["relative_update_l2"] == 1.
    assert torch.equal(model.weight, before) and torch.equal(model.weight.grad, gradient)
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        state_delta(model, initial)
    with pytest.raises(FloatingPointError):
        finite_metrics({"loss": float("inf")})


def test_comparison_rejects_changed_training_identity_before_speed_claim():
    old = {"checkpoint_sha256": "old"}
    new = {"checkpoint_sha256": "different"}
    with pytest.raises(ValueError, match="checkpoint"):
        compare_reports(old, new)
