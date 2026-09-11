"""Integration checks for update isolation, fixed-stat evaluation and resumes."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import (
    DistillationEngine, DistillationTrainingConfig, evaluate_crops, scored_batch, waveform_gate, model_state_fingerprint,
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
    before = {k: v.clone() for k, v in trainer.discriminators.state_dict().items()}
    student = trainer.model.output.weight.detach().clone()
    metrics = trainer.train_step([item])
    assert metrics["step"] == 1
    assert not torch.equal(student, trainer.model.output.weight)
    assert item.teacher_audio.grad is None
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
