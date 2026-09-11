"""Read-only diagnostics must distinguish output and actual parameter geometry."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig, generator_losses
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.optimizers import build_optimizer_bundle
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine, calibration_batch
from audiovae_student.training import _rng_state
from diagnose_objectives_updates import (objective_geometry, routing_diagnostic, native_replays,
    _parameter_gradient, vector_dot, vector_summary, _capture_replay_state, diagnostic_values)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(1943)


@pytest.fixture
def prepared():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=12,
        layer_scale_init=.1, normalization_mode="masked_batch_norm", adapter_mode="raw_repeat_phase_bias"))
    recipe = RecipeV2Config(total_steps=12, reconstruction_warmup_steps=2, perceptual_ramp_steps=2,
        learning_rate_warmup_steps=1, optimizer="adamw")
    discriminators = AudioDiscriminators(DiscriminatorConfig(periods=(2,),
        mpd_channels=(2, 2, 2, 2, 2), fft_sizes=(128,), mrd_channels=2))
    engine = RecipeV2Engine(model, recipe=recipe, discriminators=discriminators)
    target = torch.randn(1, 1, 7 * DECODER_HOP) * .03
    target[..., :3 * DECODER_HOP] = 0
    crops = [TrainingCrop(torch.randn(1, 64, 7), target, None, "a" * 64, "fixture", 1, 0, 1, 6,
                           6 * DECODER_HOP - 138)]
    engine.train_step(crops)
    engine.train_step(crops)
    engine.calibrate([calibration_batch(crops, "cpu")], provenance={"split": "train", "purpose": "synthetic"})
    engine.train_step(crops)
    engine.train_step(crops)
    return engine, crops


def test_shared_parameter_can_reverse_local_quiet_audio_gradient():
    parameter = torch.nn.Parameter(torch.tensor(0.))
    audio = torch.stack([parameter, -parameter]).reshape(1, 1, 2)
    # Audio-space descent improves the first (quiet) sample by moving it down,
    # but the stronger active component routes through the same parameter.
    audio_gradient = torch.tensor([[[1., 3.]]])
    grad, = _parameter_gradient(audio, (parameter,), audio_gradient)
    assert grad == -2
    native_sgd_delta = -.1 * grad
    assert native_sgd_delta > 0  # The quiet sample rises instead.
    quiet_grad, = _parameter_gradient(audio, (parameter,), torch.tensor([[[1., 0.]]]))
    active_grad, = _parameter_gradient(audio, (parameter,), torch.tensor([[[0., 3.]]]))
    assert torch.equal(quiet_grad + active_grad, grad)


def test_nonzero_gan_loss_does_not_imply_nonzero_gradient():
    class ConstantD(torch.nn.Module):
        def forward(self, audio):
            logits = audio.mean(dim=-1) * 0
            return (SimpleNamespace(logits=logits, features=(audio,)),)
    prediction = torch.ones(1, 1, 20, requires_grad=True)
    losses = generator_losses(ConstantD(), prediction, prediction.detach())
    gradient, = torch.autograd.grad(losses["adversarial"], prediction)
    assert losses["adversarial"] == 1
    assert gradient.count_nonzero() == 0
    assert losses["feature_matching"] == 0


def test_teacher_interpolation_and_parameter_audit_are_immutable(prepared):
    engine, crops = prepared
    before, rng = state_fingerprint(engine.state_dict()), state_fingerprint(_rng_state())
    original_grad = next(engine.model.parameters()).grad
    saved_grad = original_grad.clone()
    report = objective_geometry(engine, crops)
    assert report["alphas"][-1]["teacher_identity_max_abs"] == 0
    assert all(report["alphas"][-1]["nonadversarial_exact_zero_losses"].values())
    assert all(value == 0 for value in report["alphas"][-1]["nonadversarial_max_abs_gradients"].values())
    routing = routing_diagnostic(engine, crops)
    assert routing["gradient_linearity_max_abs_error"] < 1e-5
    assert state_fingerprint(engine.state_dict()) == before
    assert state_fingerprint(_rng_state()) == rng
    assert next(engine.model.parameters()).grad is original_grad
    assert torch.equal(original_grad, saved_grad)


def test_disposable_native_replays_restore_models_moments_gradients_and_rng(prepared):
    engine, crops = prepared
    before = state_fingerprint(_capture_replay_state(engine))
    rng = state_fingerprint(_rng_state())
    original_grad = next(engine.model.parameters()).grad
    saved_grad = original_grad.clone()
    report = native_replays(engine, crops, crops)
    assert report["retained_optimizer_updates"] == 0 and report["state_restored"]
    assert len(report["variants"]) == 7
    assert report["variants"]["exact_training_step"]["training_metrics"]["step"] == 5
    assert report["variants"]["zero_current_gradient"]["delta_norm"] > 0  # Existing Adam moments still act.
    assert state_fingerprint(_capture_replay_state(engine)) == before
    assert state_fingerprint(_rng_state()) == rng
    assert next(engine.model.parameters()).grad is original_grad and torch.equal(original_grad, saved_grad)
    assert any(abs(value) > 0 for value in report["variants"]["all"]["observed_probe_metric_change"].values())


def test_vector_summary_uses_actual_parameter_groups():
    p = torch.nn.Parameter(torch.zeros(2))
    named = (("blocks.0.expand.weight", p), ("output.bias", p))
    vectors = {"a": (torch.ones(2), -torch.ones(2)), "b": (-torch.ones(2), -torch.ones(2))}
    result = vector_summary(named, vectors)
    assert result["blocks.0"]["cosines"]["a"]["b"] == pytest.approx(-1)
    assert result["output"]["cosines"]["a"]["b"] == pytest.approx(1)
    assert vector_dot(vectors["a"], vectors["b"]) == 0


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Native Muon unavailable in local validation runtime")
def test_native_muon_moments_and_adamw_moments_restore(prepared):
    engine, crops = prepared
    engine.optimizer = build_optimizer_bundle(engine.model, optimizer="muon_adamw")
    engine.train_step(crops)  # Populate both native optimizers with actual moments.
    before = state_fingerprint(_capture_replay_state(engine))
    report = native_replays(engine, crops, crops)
    assert report["state_restored"]
    assert state_fingerprint(_capture_replay_state(engine)) == before
    assert set(engine.optimizer.optimizers) == {"muon", "adamw"}
