from copy import deepcopy

import pytest
import torch

from audit_training_backend import compare_tensor, compare_maps, _grad_map
from audiovae_student.corrected_calibration import _fixed_engine, _scheduled_weights
from audiovae_student.discriminators import discriminator_loss
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.recipe_v2 import scored_batch_v2
from audiovae_student.training import _rng_state
from diagnose_objectives_updates import _losses_for_prediction
from test_diagnose_objectives_updates import prepared, deterministic


def test_last_channel_corruption_is_visible_in_full_tensor_comparison():
    a = torch.randn(8, 64, 100)
    b = a.clone()
    b[3, 63, 17] += 1.19
    row = compare_tensor(a, b)
    assert row["max_abs_index"] == [3, 63, 17]
    assert row["max_abs"] == pytest.approx(1.19, abs=1e-6)
    assert row["channel_max_abs"][63] == row["max_abs"]
    assert not row["screen_within_1e-5_abs_plus_relative"]


def test_zero_and_nonfinite_and_incompatible_are_not_hidden():
    zero = torch.zeros(2)
    assert compare_tensor(zero, zero)["relative_l2_symmetric"] == 0
    assert not compare_tensor(zero, torch.ones(2))["exact_equal"]
    assert not compare_tensor(zero, torch.tensor([float("nan"), 0.]))["finite"]
    assert not compare_tensor(zero, torch.zeros(3))["compatible"]
    with pytest.raises(ValueError):
        compare_maps({"a": zero}, {"b": zero})


def test_gradient_path_preserves_every_trained_state_and_existing_grad(prepared):
    engine, crops = prepared
    before = state_fingerprint(engine.state_dict())
    rng = state_fingerprint(_rng_state())
    existing = [(p.grad, p.grad.clone() if p.grad is not None else None)
                for p in engine.model.parameters()]
    with _fixed_engine(engine, 1861):
        batch, _ = scored_batch_v2(engine.model, crops, engine.reconstruction, engine.device)
        named = tuple((n, p) for n, p in engine.model.named_parameters() if p.requires_grad)
        params = tuple(p for _, p in named)
        fixed = torch.randn_like(batch.prediction).masked_fill(~batch.score_mask, 0)
        grads = torch.autograd.grad(batch.prediction, params, grad_outputs=fixed,
                                    allow_unused=True, retain_graph=True)
        saved = _grad_map(named, grads)
        assert len(saved) == len(named)
        batch, losses = _losses_for_prediction(engine, batch, batch.prediction,
                                                crop_rng=engine.crop_generator.get_state().clone())
        clone = deepcopy(engine.balancer)
        clone.weights = _scheduled_weights(engine)
        balanced = clone.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        grads = torch.autograd.grad(batch.prediction, params, grad_outputs=balanced.gradient, allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
        target = batch.targets[0][..., :engine.recipe.adversarial_samples].detach()
        d_loss = discriminator_loss(engine.discriminators, target + .01, target)
        d_named = tuple(engine.discriminators.named_parameters())
        d_grads = torch.autograd.grad(d_loss, tuple(p for _, p in d_named), allow_unused=True)
        assert _grad_map(d_named, d_grads)
    assert state_fingerprint(engine.state_dict()) == before
    assert state_fingerprint(_rng_state()) == rng
    for p, (old, value) in zip(engine.model.parameters(), existing):
        assert p.grad is old
        if old is not None:
            assert torch.equal(p.grad, value)
