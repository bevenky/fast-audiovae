"""Strict restoration and isolated optimizer migration with real CPU updates."""
from copy import deepcopy

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.fusion_architecture import FusionStudentDecoder
from audiovae_student.fusion_migration import FUSION_VARIANTS, build_fusion_engine
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine, calibration_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(927)


def fixture_parent(optimizer="adamw"):
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=12,
        layer_scale_init=.1, normalization_mode="masked_batch_norm", adapter_mode="raw_repeat_phase_bias"))
    recipe = RecipeV2Config(total_steps=10, reconstruction_warmup_steps=2, perceptual_ramp_steps=2,
        learning_rate_warmup_steps=1, optimizer=optimizer)
    discriminators = AudioDiscriminators(DiscriminatorConfig(periods=(2,),
        mpd_channels=(2, 2, 2, 2, 2), fft_sizes=(128,), mrd_channels=2))
    engine = RecipeV2Engine(model, recipe=recipe, discriminators=discriminators)
    crops = [TrainingCrop(torch.randn(1, 64, 6), torch.randn(1, 1, 6 * DECODER_HOP) * .03,
        None, "a" * 64, "fixture", 0, 0, 0, 6, 6 * DECODER_HOP)]
    engine.train_step(crops)
    engine.train_step(crops)
    engine.calibrate([calibration_batch(crops, "cpu")], provenance={"split": "train", "purpose": "synthetic"})
    engine.train_step(crops)
    return engine, crops


@pytest.mark.parametrize("variant", FUSION_VARIANTS)
def test_each_migration_preserves_parent_clocks_weights_moments_and_can_train_and_save(variant):
    parent, crops = fixture_parent()
    saved = deepcopy(parent.state_dict())
    fingerprint = state_fingerprint(saved)
    engine, receipt = build_fusion_engine(saved, variant)
    assert state_fingerprint(saved) == fingerprint
    assert receipt["original_student_state_preserved"]
    assert receipt["original_student_optimizers_preserved"]
    assert receipt["calibration_preserved"] and receipt["crop_rng_preserved"] and receipt["balancer_preserved"]
    assert engine.step == 3 and engine.discriminator_updates == 1
    for name, parameter in engine.model.named_parameters():
        if name in dict(parent.model.named_parameters()):
            assert parameter.data_ptr() != parent.model.get_parameter(name).data_ptr()
    if variant == "filter":
        assert receipt["required_context_latent_frames"] == 30
        assert receipt["new_student_optimizer_groups"] == ["filter_adamw"]
        assert engine.optimizer.optimizers["filter_adamw"].state_dict()["state"] == {}
    else:
        assert receipt["required_context_latent_frames"] == 29
        assert receipt["new_student_optimizer_groups"] == []
    metrics = engine.train_step(crops)
    assert metrics["step"] == 4 and metrics["discriminator_updates"] == 2
    result = engine.state_dict()
    assert result["format_version"] == "fusion_recipe_v1" and result["fusion_variant"] == variant
    assert result["fusion_migration"] == receipt
    assert state_fingerprint(saved) == fingerprint
    if variant == "filter":
        assert engine.optimizer.optimizers["filter_adamw"].state_dict()["state"]
    with pytest.raises(ValueError, match="exact resume"):
        parent.load_state_dict(result)
    with pytest.raises(ValueError, match="experimental resume"):
        engine.load_state_dict(result)


@pytest.mark.parametrize("variant", ("control", "tanh", "filter", "zero_padding", "short_mel"))
def test_nonrandom_candidates_preserve_global_rng_and_optimizer_parameter_references(variant):
    parent, _ = fixture_parent()
    before = torch.get_rng_state().clone()
    engine, _ = build_fusion_engine(deepcopy(parent.state_dict()), variant)
    assert torch.equal(torch.get_rng_state(), before)
    optimized = [parameter for optimizer in engine.optimizer.optimizers.values()
                 for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len(list(engine.model.parameters()))
    assert {id(p) for p in optimized} == {id(p) for p in engine.model.parameters()}
    assert not {id(p) for p in optimized} & {id(p) for p in parent.model.parameters()}


@pytest.mark.parametrize("variant", ("fresh_magnitude", "complex"))
def test_fresh_spectral_heads_preserve_period_weights_and_named_moments_only(variant):
    parent, _ = fixture_parent()
    saved = deepcopy(parent.state_dict())
    torch.manual_seed(503)
    first, receipt = build_fusion_engine(saved, variant)
    torch.manual_seed(503)
    second, _ = build_fusion_engine(saved, variant)
    assert state_fingerprint(first.discriminators.state_dict()) == state_fingerprint(second.discriminators.state_dict())
    assert state_fingerprint(first.discriminators.periods.state_dict()) == state_fingerprint(parent.discriminators.periods.state_dict())
    for name, parameter in first.discriminators.named_parameters():
        if name.startswith("periods."):
            old = parent.discriminators.get_parameter(name)
            assert state_fingerprint(first.discriminator_optimizer.state[parameter]) == state_fingerprint(parent.discriminator_optimizer.state[old])
        else:
            assert parameter not in first.discriminator_optimizer.state
    assert receipt["discriminator_migration"]["fresh_spectral_optimizer_state"] == "empty"
    assert state_fingerprint(first.discriminators.resolutions.state_dict()) != state_fingerprint(parent.discriminators.resolutions.state_dict())


def test_control_update_is_identical_to_strict_original_resume():
    parent, crops = fixture_parent()
    candidate, _ = build_fusion_engine(deepcopy(parent.state_dict()), "control")
    expected = parent.train_step(crops)
    actual = candidate.train_step(crops)
    assert actual == expected
    for key in ("model", "optimizer", "discriminators", "discriminator_optimizer", "crop_rng", "balancer"):
        assert state_fingerprint(candidate.state_dict()[key]) == state_fingerprint(parent.state_dict()[key])


def test_bad_or_relabelled_parent_is_rejected_before_migration():
    parent, _ = fixture_parent()
    saved = deepcopy(parent.state_dict())
    with pytest.raises(ValueError, match="Unknown"):
        build_fusion_engine(saved, "combined")
    invalid = deepcopy(saved)
    invalid["calibration"]["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Calibration report"):
        build_fusion_engine(invalid, "tanh")
    invalid = deepcopy(saved)
    invalid["model"]["output.weight"].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        build_fusion_engine(invalid, "tanh")
    invalid = deepcopy(saved)
    invalid["fusion_variant"] = "tanh"
    with pytest.raises(ValueError, match="relabeled"):
        build_fusion_engine(invalid, "tanh")


@pytest.mark.skipif(not callable(getattr(torch.optim, "Muon", None)), reason="native Muon unavailable locally")
def test_native_muon_and_complementary_adamw_are_preserved_with_fresh_filter_group():
    parent, crops = fixture_parent("muon_adamw")
    saved = deepcopy(parent.state_dict())
    engine, receipt = build_fusion_engine(saved, "filter")
    assert isinstance(engine.model, FusionStudentDecoder)
    assert set(engine.optimizer.optimizers) == {"muon", "adamw", "filter_adamw"}
    for name in ("muon", "adamw"):
        assert state_fingerprint(engine.optimizer.optimizers[name].state_dict()) == state_fingerprint(saved["optimizer"]["optimizers"][name])
    assert receipt["new_student_optimizer_groups"] == ["filter_adamw"]
    engine.train_step(crops)
