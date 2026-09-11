"""Integration checks for update isolation, fixed-stat evaluation and resumes."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import (
    DistillationEngine, DistillationTrainingConfig, evaluate_crops, scored_batch, waveform_gate, model_state_fingerprint,
    reconstruction_gradient_report,
)
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student.teacher import CHECKPOINT_SHA256


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def crop(context=0, valid=None, source="a"):
    frames, scored = context + 3, 3
    return TrainingCrop(torch.randn(1, 64, frames), torch.randn(1, 1, frames * DECODER_HOP) * 0.03,
                        None, "a" * 64, source, context, 0, context, scored,
                        valid if valid is not None else scored * DECODER_HOP)


def engine():
    model = StudentDecoder(StudentConfig(hidden_channels=8, expansion_channels=16, head_channels=12,
        dilations=(1, 2), layer_scale_init=0.1, normalization_mode="masked_batch_norm"))
    return DistillationEngine(model, config=DistillationTrainingConfig(optimizer="adamw",
        total_steps=8, warmup_steps=1, freeze_normalization_step=1, adversarial_samples=768),
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=AudioDiscriminators(DiscriminatorConfig(periods=(2,),
            mpd_channels=(2, 4, 4, 4, 4), fft_sizes=(64,), mrd_channels=2)))


def passed_gate(trainer):
    # A fixture tests the state-transition contract, not learned quality.
    return {"passed": True, "teacher_checkpoint_sha256": CHECKPOINT_SHA256,
            "model_config": trainer.model.config.to_dict(), "evaluated_step": trainer.step,
            "model_state_sha256": model_state_fingerprint(trainer.model), "fixed_statistics": True}


def test_reconstruction_updates_student_but_not_discriminator_or_targets():
    trainer = engine()
    item = crop()
    item.teacher_audio.requires_grad_(True)
    item.latents.requires_grad_(True)
    before = {k: v.clone() for k, v in trainer.discriminators.state_dict().items()}
    student = trainer.model.output.weight.detach().clone()
    metrics = trainer.train_step([item])
    assert metrics["step"] == 1
    assert not torch.equal(student, trainer.model.output.weight)
    assert item.teacher_audio.grad is None
    assert item.latents.grad is None
    for key, value in trainer.discriminators.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_perceptual_stage_is_gated_and_keeps_gradient_ownership():
    trainer = engine()
    with pytest.raises(ValueError, match="passed"):
        trainer.enable_perceptual({"passed": False})
    trainer.train_step([crop()])
    gate = passed_gate(trainer)
    with pytest.raises(ValueError, match="exact model step"):
        trainer.enable_perceptual({**gate, "evaluated_step": 0})
    trainer.enable_perceptual(gate)
    before = {k: v.clone() for k, v in trainer.discriminators.state_dict().items()}
    metrics = trainer.train_step([crop(source="b")])
    assert metrics["adversarial_examples"] == 1
    assert metrics["feature_matching"] >= 0
    assert any(not torch.equal(value, before[key]) for key, value in trainer.discriminators.state_dict().items())
    assert all(p.grad is None for p in trainer.discriminators.parameters())
    assert all(p.requires_grad for p in trainer.discriminators.parameters())


def test_masking_ignores_context_and_tail_teacher_values():
    trainer = engine()
    trainer.model.freeze_normalization_statistics()
    trainer.model.eval()
    item = crop(4, valid=2 * DECODER_HOP + 123)
    original = scored_batch(trainer.model, [item], trainer.criterion, "cpu")
    poison = item.teacher_audio.clone()
    poison[..., :item.scored_slice.start] = float("nan")
    poison[..., item.scored_slice.stop:] = float("nan")
    changed = scored_batch(trainer.model, [replace(item, teacher_audio=poison)], trainer.criterion, "cpu")
    for key in original.losses:
        torch.testing.assert_close(original.losses[key], changed.losses[key], atol=0, rtol=0)
    assert original.score_mask.sum().item() == item.valid_scored_samples


def test_decoder_receives_exact_cached_raw_latents_before_its_adapter():
    trainer = engine()
    trainer.model.eval()
    items = [crop(0), crop(4, source="b")]
    seen = []
    hook = trainer.model.adapter.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    try:
        scored_batch(trainer.model, items, trainer.criterion, "cpu")
    finally:
        hook.remove()
    assert len(seen) == 1
    for index, item in enumerate(items):
        torch.testing.assert_close(seen[0][index:index + 1, :, :item.latents.shape[-1]], item.latents, atol=0, rtol=0)
        assert not seen[0][index:index + 1, :, item.latents.shape[-1]:].count_nonzero()


def test_resume_reproduces_next_update_and_rejects_changed_objective():
    trainer = engine()
    items = [crop(), crop(4, source="b")]
    trainer.train_step(items)
    trainer.enable_perceptual(passed_gate(trainer))
    saved = deepcopy(trainer.state_dict())
    trainer.train_step(items)
    restored = engine()
    restored.load_state_dict(saved)
    restored.train_step(items)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    for key, value in trainer.discriminators.state_dict().items():
        torch.testing.assert_close(value, restored.discriminators.state_dict()[key], rtol=0, atol=0)
    broken = deepcopy(saved)
    broken["loss_config"]["waveform_rms_floor"] = 0.1
    with pytest.raises(ValueError, match="exact resume"):
        restored.load_state_dict(broken)


def test_evaluation_does_not_change_normalization_or_mode():
    trainer = engine()
    trainer.train_step([crop()])
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    rows = evaluate_crops(trainer, [crop()])
    assert trainer.model.training
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    gate = waveform_gate(trainer, rows)
    assert gate["passed"] is False
    assert len(gate["failed_clips"]) == 1
    trainer.train_step([crop()])
    with pytest.raises(ValueError, match="stale"):
        waveform_gate(trainer, rows)


def test_budget_and_invalid_short_crops_fail_before_update():
    trainer = engine()
    with pytest.raises(ValueError, match="reconstruction FFTs"):
        trainer.train_step([crop(valid=192)])
    assert trainer.step == 0
    trainer.step = trainer.config.total_steps
    with pytest.raises(ValueError, match="budget"):
        trainer.train_step([crop()])


@pytest.mark.parametrize("cosine, passed, milestone", [(0.90, False, False), (0.949999, False, False),
                (0.95, False, True), (0.97, False, True), (0.989999, False, True),
                (0.99, True, True), (0.999, True, True)])
def test_reconstruction_requires_099_per_clip_with_095_milestone(cosine, passed, milestone):
    trainer = engine()
    # Isolate the acceptance boundary; this fixture is not learned-quality evidence.
    row = {"source_id": "boundary", "start_frame": 0,
           "evaluated_step": trainer.step,
           "model_state_sha256": model_state_fingerprint(trainer.model),
           "fixed_statistics": True, "teacher_rms": 0.1, "student_rms": 0.1,
           "waveform_to_silence_error_ratio": 0.2, "rms_db_error": 0.0,
           "waveform_cosine": cosine}
    gate = waveform_gate(trainer, [row])
    assert gate["passed"] is passed
    assert gate["criteria"]["waveform_cosine_min"] == 0.99
    assert gate["milestone_095_passed"] is milestone


def assert_same_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_state(a, b)
    else:
        assert left == right


def fork_config(trainer, **kwargs):
    return replace(trainer.config, total_steps=8, warmup_steps=2,
        freeze_normalization_step=0, learning_rate_schedule="constant_after_warmup",
        warmup_start_learning_rate=2e-5, **kwargs)


def test_legacy_checkpoint_without_new_fields_reproduces_next_update():
    trainer = engine()
    items = [crop(), crop(4, source="b")]
    trainer.train_step(items)
    saved = deepcopy(trainer.state_dict())
    for key in ("reconstruction_waveform_share", "learning_rate_schedule",
                "warmup_start_learning_rate", "parameter_update_metrics_interval"):
        del saved["config"][key]
    restored = engine()
    restored.load_state_dict(saved)
    assert_same_state(trainer.train_step(items), restored.train_step(items))
    assert_same_state(trainer.state_dict(), restored.state_dict())


def test_fork_preserves_learned_state_and_internal_clocks_but_changes_objective():
    trainer = engine()
    trainer.train_step([crop()])
    trainer.train_step([crop(4)])
    before = deepcopy(trainer.state_dict())
    config = fork_config(trainer, reconstruction_waveform_share=1.0)
    provenance = trainer.fork_reconstruction(config)
    after = trainer.state_dict()
    assert trainer.step == 0
    assert provenance["parent_step"] == before["step"] == 2
    assert provenance["parent_model_state_sha256"] == model_state_fingerprint(trainer.model)
    for key in ("model", "optimizer", "discriminators", "discriminator_optimizer", "crop_rng"):
        assert_same_state(before[key], after[key])
    assert_same_state(before["balancer"]["ema"], after["balancer"]["ema"])
    assert before["balancer"]["updates"] == after["balancer"]["updates"] == 2
    assert after["balancer"]["weights"]["teacher_waveform"] == 1.0
    assert after["balancer"]["weights"]["teacher_mel"] == 0.0
    assert trainer.learning_rate() == 2e-5
    for step in (1, 2, 6, 7):
        trainer.step = step
        assert trainer.learning_rate() == trainer.config.learning_rate


def test_controlled_fork_rejects_unfrozen_normalization_and_changed_optimizer():
    trainer = engine()
    with pytest.raises(ValueError, match="frozen normalization"):
        trainer.fork_reconstruction(fork_config(trainer))
    trainer.model.freeze_normalization_statistics()
    with pytest.raises(ValueError, match="cannot change optimizer"):
        trainer.fork_reconstruction(fork_config(trainer, weight_decay=0.2))
    trainer.enable_perceptual(passed_gate(trainer))
    with pytest.raises(ValueError, match="reconstruction-stage"):
        trainer.fork_reconstruction(fork_config(trainer))


def test_two_identical_forks_reproduce_updates_without_changing_running_statistics():
    parent = engine()
    items = [crop(), crop(4, source="b")]
    parent.train_step(items)
    parent.train_step(items)
    saved = deepcopy(parent.state_dict())
    arms = [engine(), engine()]
    for trainer in arms:
        trainer.load_state_dict(deepcopy(saved))
        trainer.fork_reconstruction(fork_config(trainer))
    before_buffers = {key: value.clone() for key, value in arms[0].model.named_buffers()}
    assert_same_state(arms[0].train_step(items), arms[1].train_step(items))
    assert_same_state(arms[0].state_dict(), arms[1].state_dict())
    for key, value in arms[0].model.named_buffers():
        torch.testing.assert_close(value, before_buffers[key], rtol=0, atol=0)


def test_waveform_only_fork_checkpoint_resumes_exactly_and_rejects_wrong_shares():
    trainer = engine()
    trainer.model.freeze_normalization_statistics()
    trainer.fork_reconstruction(fork_config(trainer, reconstruction_waveform_share=1.0))
    items = [crop(), crop(4, source="b")]
    trainer.train_step(items)
    saved = deepcopy(trainer.state_dict())
    restored = engine()
    restored.model.freeze_normalization_statistics()
    restored.fork_reconstruction(trainer.config)
    restored.load_state_dict(saved)
    assert_same_state(trainer.train_step(items), restored.train_step(items))
    assert_same_state(trainer.state_dict(), restored.state_dict())
    saved["balancer"]["weights"].update(teacher_waveform=0.5, teacher_mel=0.5)
    with pytest.raises(ValueError, match="weights disagree"):
        restored.load_state_dict(saved)


def test_sampled_metrics_measure_actual_update_and_clipped_gradient():
    trainer = engine()
    trainer.config = replace(trainer.config, parameter_update_metrics_interval=2)
    items = [crop(), crop(4, source="b")]
    first = trainer.train_step(items)
    assert not any(key.startswith("parameter_update/") for key in first)
    before = {key: trainer.model.get_parameter(key).detach().clone() for key in trainer.gradient_probe_names()}
    second = trainer.train_step(items)
    for name, value in before.items():
        prefix = f"parameter_update/{name}"
        delta = (trainer.model.get_parameter(name).detach() - value).norm().item()
        assert second[f"{prefix}/delta_norm"] == delta
        assert second[f"{prefix}/relative_delta_norm"] == pytest.approx(delta / value.norm().item())
        assert second[f"{prefix}/gradient_norm_after_clip"] <= second[f"{prefix}/gradient_norm_before_clip"]
        assert second[f"{prefix}/gradient_norm_after_clip"] <= trainer.config.gradient_clip


def test_waveform_only_gradient_report_preserves_state_and_reports_inactive_mel():
    trainer = engine()
    trainer.model.freeze_normalization_statistics()
    trainer.fork_reconstruction(fork_config(trainer, reconstruction_waveform_share=1.0))
    before = deepcopy(trainer.state_dict())
    report = reconstruction_gradient_report(trainer, [crop(), crop(4, source="b")])
    assert report["probe_examples"] == 2
    assert report["output_gradients"]["teacher_waveform/target_share"] == 1.0
    assert report["output_gradients"]["teacher_mel/scale"] == 0
    assert report["output_gradients"]["teacher_mel/scaled_norm"] == 0
    assert all(row["mel_scaled_norm"] == 0 and row["mel_raw_norm"] > 0 for row in report["parameters"])
    assert_same_state(before, trainer.state_dict())


@pytest.mark.parametrize("kwargs", [{"reconstruction_waveform_share": 0},
    {"reconstruction_waveform_share": 1.001}, {"reconstruction_waveform_share": float("nan")},
    {"learning_rate_schedule": "unknown"}, {"warmup_start_learning_rate": -1e-5},
    {"warmup_start_learning_rate": 0.01}, {"warmup_start_learning_rate": 1e-5, "warmup_steps": 1},
    {"parameter_update_metrics_interval": -1}, {"parameter_update_metrics_interval": 1.5}])
def test_invalid_experiment_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DistillationTrainingConfig(**kwargs)
