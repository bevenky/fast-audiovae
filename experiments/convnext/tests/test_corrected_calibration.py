"""Fixed-weight calibration must not silently restart any learned state."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.corrected_calibration import audit_output_gradients, calibrate_changed_losses
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig, generator_losses
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine, calibration_batch, scored_batch_v2
from audiovae_student.training import _rng_state


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(736)


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


def frozen_state(engine):
    state = deepcopy(engine.state_dict())
    state.pop("balancer")
    return state_fingerprint(state)


def test_selected_calibration_preserves_weights_moments_flags_clocks_rng_and_unselected_ema(prepared):
    parent, crops = prepared
    engine, _ = build_fusion_engine(deepcopy(parent.state_dict()), "complex")
    engine.model.eval()
    engine.discriminators.eval()
    next(engine.discriminators.parameters()).requires_grad_(False)
    parameter = next(engine.model.parameters())
    parameter.grad = torch.full_like(parameter, .17)
    original_grad = parameter.grad
    flags = [p.requires_grad for p in engine.discriminators.parameters()]
    before, rng = frozen_state(engine), state_fingerprint(_rng_state())
    original = deepcopy(engine.balancer.state_dict())
    report = calibrate_changed_losses(engine, [crops, crops], provenance={"split": "train", "purpose": "test"})
    assert frozen_state(engine) == before
    assert state_fingerprint(_rng_state()) == rng
    assert parameter.grad is original_grad and torch.equal(parameter.grad, torch.full_like(parameter, .17))
    assert [p.requires_grad for p in engine.discriminators.parameters()] == flags
    assert not engine.model.training and not engine.discriminators.training
    after = engine.balancer.state_dict()
    assert after["updates"] == original["updates"]
    assert after["weights"] == original["weights"]
    for name in ("teacher_waveform", "teacher_mel"):
        assert after["ema"][name] == original["ema"][name]
    for name in ("feature_matching", "adversarial"):
        norms = [row["regions"]["all"]["raw_norms"][name] for row in report["after"]]
        assert after["ema"][name]["total"] == pytest.approx(.999 * norms[0] + norms[1])
        assert after["ema"][name]["weight"] == pytest.approx(1.999)
    assert report["parameter_updates"] == 0 and report["optimizer_updates"] == 0
    # The ordinary optimizer and checkpoint path still work afterwards.
    assert engine.train_step(crops)["step"] == 5


def test_no_stale_history_and_all_loss_calibration_attains_target_on_fixed_batch(prepared):
    engine, crops = prepared
    old = engine.balancer.state_dict()
    for name in old["ema"]:
        old["ema"][name] = {"total": 1e8 if name == "adversarial" else 1e5, "weight": 1000.}
    engine.balancer.load_state_dict(old)
    report = calibrate_changed_losses(engine, [crops], loss_names=tuple(old["ema"]),
                                     provenance={"split": "train"})
    before, after = report["before"][0], report["after"][0]
    assert before["achieved_presum_shares"]["adversarial"] < .01
    for name, target in after["target_shares"].items():
        assert after["achieved_presum_shares"][name] == pytest.approx(target)
        assert report["balancer_after"]["ema"][name]["weight"] == 1


def test_audit_matches_training_loss_definitions_valid_lengths_and_crop_rng(prepared):
    engine, crops = prepared
    # Different valid sample counts must retain sample-weighted losses.
    other = replace(crops[0], source_id="other", cache_key="b" * 64,
                    teacher_audio=crops[0].teacher_audio * .7, valid_scored_samples=6 * DECODER_HOP)
    crops = [*crops, other]
    state = state_fingerprint(engine.state_dict())
    rng = state_fingerprint(_rng_state())
    report = audit_output_gradients(engine, crops, seed=725)
    assert state_fingerprint(engine.state_dict()) == state and state_fingerprint(_rng_state()) == rng
    # Compute the real train-step generator objectives without a D/G update.
    engine.crop_generator.manual_seed(725)
    batch, _ = scored_batch_v2(engine.model, crops, engine.reconstruction, engine.device)
    prediction, target, count = engine._perceptual_audio(batch)
    losses = {name: batch.losses[name] for name in ("teacher_waveform", "teacher_mel")}
    losses.update(generator_losses(engine.discriminators, prediction, target,
        example_weights=prediction.new_tensor([crop.valid_scored_samples for crop in crops])))
    measured = report["measurement"]
    assert count == len(crops)
    assert measured["scored_samples"] == sum(c.valid_scored_samples for c in crops)
    for name, loss in losses.items():
        grad, = torch.autograd.grad(loss, batch.prediction, retain_graph=True)
        grad = grad.float().masked_fill(~batch.score_mask, 0)
        assert measured["losses"][name] == float(loss.detach())
        assert measured["regions"]["all"]["raw_norms"][name] == float(grad.flatten(1).norm(dim=1).mean())
    regions = measured["regions"]
    # Two scored quiet frames per example; excluded context is also zero.
    assert regions["quiet"]["samples"] == 2 * 2 * DECODER_HOP
    assert regions["quiet"]["samples"] + regions["active"]["samples"] == measured["scored_samples"]
    for name in losses:
        assert regions["quiet"]["gradient_energy"][name] + regions["active"]["gradient_energy"][name] == pytest.approx(
            regions["all"]["gradient_energy"][name])
    assert regions["quiet"]["cosine_to_residual"]["teacher_waveform"] > 0


def test_failure_rolls_back_balancer_rng_modes_and_fixed_states(prepared):
    engine, crops = prepared
    before = state_fingerprint(engine.state_dict())
    rng = state_fingerprint(_rng_state())
    engine.model.eval()
    with pytest.raises(ValueError, match="adversarial sample count"):
        calibrate_changed_losses(engine, [crops, []], provenance={"split": "train"})
    assert state_fingerprint(engine.state_dict()) == before
    assert state_fingerprint(_rng_state()) == rng
    assert not engine.model.training


def test_empty_or_invalid_calibration_cannot_be_committed(prepared):
    engine, crops = prepared
    before = state_fingerprint(engine.state_dict())
    with pytest.raises(ValueError, match="nonzero gradient"):
        calibrate_changed_losses(engine, [], provenance={"split": "train"})
    with pytest.raises(ValueError, match="training-only"):
        calibrate_changed_losses(engine, [crops], provenance={"split": "validation"})
    with pytest.raises(ValueError, match="unique existing"):
        calibrate_changed_losses(engine, [crops], loss_names=("teacher_mel", "teacher_mel"), provenance={"split": "train"})
    assert state_fingerprint(engine.state_dict()) == before


def test_requires_frozen_calibrated_statistics(prepared):
    engine, crops = prepared
    engine.model.stem_norm.statistics_frozen.fill_(False)
    with pytest.raises(ValueError, match="frozen normalization"):
        audit_output_gradients(engine, crops)


def test_injected_event_view_is_used_without_consuming_crop_rng(prepared):
    engine, crops = prepared
    calls = []
    def event_view(batch):
        calls.append(tuple(c.source_id for c in batch.crops))
        size = engine.recipe.adversarial_samples
        return (torch.cat([p[..., :size] for p in batch.predictions]),
                torch.cat([t[..., :size] for t in batch.targets]), len(batch.crops))
    engine._perceptual_audio = event_view
    rng = engine.crop_generator.get_state().clone()
    result = audit_output_gradients(engine, crops)
    assert calls == [("fixture",)]
    assert torch.equal(engine.crop_generator.get_state(), rng)
    assert all(math.isfinite(v) for v in result["measurement"]["losses"].values())


def test_explicit_event_audit_counters_restore_identity(prepared):
    engine, crops = prepared
    engine.view_counts = {"calls": 8, "labels": []}
    original = engine.view_counts
    select = engine._perceptual_audio
    def event_view(batch):
        engine.view_counts["calls"] += 1
        engine.view_counts["labels"].append("quiet")
        return select(batch)
    engine._perceptual_audio = event_view
    report = calibrate_changed_losses(engine, [crops], provenance={"split": "train"},
                                     preserve_attributes=("view_counts",))
    assert engine.view_counts is original and original == {"calls": 8, "labels": []}
    assert report["loss_statistics"]["feature_matching"]["positive_observations"] == 1
    assert report["loss_statistics"]["teacher_waveform"]["recalibrated"] is False
    assert "adversarial" in report["after"][0]["scale_saturated"]
