"""Small CPU contract checks; these fixtures establish no audio-quality claim."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from audiovae_student.cache import DECODER_HOP, TrainingCrop
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import DistillationEngine, DistillationTrainingConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig, StudentDecoder
from audiovae_student import objective_comparison as comparison
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.teacher import CHECKPOINT_SHA256
from audiovae_student.training import _rng_state

torch.set_num_threads(1)


def crops(source, seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for start in (0, 3):
        frames = start + 3
        result.append(TrainingCrop(torch.randn(1, 64, frames, generator=generator),
            torch.randn(1, 1, frames * DECODER_HOP, generator=generator) * .03,
            None, "a" * 64, source, start, 0, start, 3, 3 * DECODER_HOP))
    return tuple(result)


def fixture():
    torch.manual_seed(77)
    diagnostic, sentinel = crops("diagnostic", 9), crops("sentinel", 11)
    model = StudentDecoder(StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
        dilations=(1,), layer_scale_init=.1, normalization_mode="masked_batch_norm"))
    engine = DistillationEngine(model, config=DistillationTrainingConfig(optimizer="adamw",
        total_steps=4, warmup_steps=1, freeze_normalization_step=1),
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=AudioDiscriminators(DiscriminatorConfig(periods=(2,),
            mpd_channels=(2, 2, 4, 4, 4), fft_sizes=(64,), mrd_channels=2)))
    for _ in range(4):
        engine.train_step(diagnostic)
    identity = {"teacher_checkpoint_sha256": CHECKPOINT_SHA256, "fixture": "cpu-contract-check"}
    parent = {"format_version": 1, "engine": deepcopy(engine.state_dict()), "rng": _rng_state(),
        "identity": {"data": identity, "diagnostic": _crop_identity(diagnostic), "sentinel": _crop_identity(sentinel)}}
    return parent, diagnostic, sentinel, identity


def config(**overrides):
    values = dict(total_steps=4, evaluation_interval=2, checkpoint_interval=2,
        warmup_steps=2, parameter_update_metrics_interval=2, expected_parent_step=4,
        expected_crops_per_role=2, save_audio=False, check_fold_streaming=False, min_free_bytes=0)
    return comparison.ObjectiveComparisonConfig(**{**values, **overrides})


def run(parent, diagnostic, sentinel, identity, output, **kwargs):
    return comparison.run_comparison(parent, diagnostic, sentinel, output,
        parent_checkpoint_sha256="b" * 64, data_identity=identity, device="cpu", **kwargs)


def test_paired_forks_preserve_parent_states_crops_and_sentinels(tmp_path):
    parent, diagnostic, sentinel, identity = fixture()
    before = comparison.state_fingerprint(parent)
    for crop in sentinel:
        crop.latents.requires_grad_(True)
        crop.teacher_audio.requires_grad_(True)
    result = run(parent, diagnostic, sentinel, identity, tmp_path / "run", config=config())
    assert result["paired_start_verified"] and result["parent_payload_unchanged"]
    assert comparison.state_fingerprint(parent) == before
    assert not result["automatically_launched_training"] and result["main_exposure_hours"] == 0
    starts = []
    for arm, share in (("waveform-mel", .5), ("waveform-only", 1.0)):
        directory = tmp_path / "run" / arm
        outcome = result["arms"][arm]
        assert outcome["step"] == 4 and outcome["state"] == "budget_exhausted_below_gate"
        assert not outcome["perceptual_training_started"]
        assert not outcome["automatic_resume_supported"]
        assert not outcome["sentinel_used_for_acceptance"]
        initial = torch.load(directory / "checkpoint-step000000.pt", weights_only=True)
        final = torch.load(directory / "checkpoint-step000004.pt", weights_only=True)
        starts.append(initial["identity"]["starting_state_sha256"])
        assert initial["engine"]["step"] == 0
        assert initial["engine"]["config"]["reconstruction_waveform_share"] == share
        assert final["latest_metrics"]["learning_rate"] == 2e-4
        assert comparison.state_fingerprint(final["engine"]["discriminators"]) == comparison.state_fingerprint(parent["engine"]["discriminators"])
        for norm in ("stem_norm", "affine"):
            for statistic in ("running_mean", "running_var", "num_batches_tracked", "statistics_frozen"):
                key = f"{norm}.{statistic}"
                torch.testing.assert_close(final["engine"]["model"][key], parent["engine"]["model"][key], atol=0, rtol=0)
        report = json.loads((directory / "gradient-final.json").read_text())
        assert report["probe_examples"] == 2
        assert report["reconstruction_weights"]["teacher_waveform"] == share
        assert len(list(directory.glob("checkpoint-step*.pt"))) == 3
    assert starts[0] == starts[1]
    assert all(c.latents.grad is None and c.teacher_audio.grad is None for c in sentinel)
    checkpoint = tmp_path / "run" / "waveform-only" / "checkpoint-step000004.pt"
    preserved = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        run(parent, diagnostic, sentinel, identity, tmp_path / "run", config=config())
    assert checkpoint.read_bytes() == preserved


def test_rejects_changed_crops_parent_step_and_teacher_before_writing(tmp_path):
    parent, diagnostic, sentinel, identity = fixture()
    diagnostic[0].latents[0, 0, 0] += 1
    with pytest.raises(ValueError, match="crop values or order"):
        run(parent, diagnostic, sentinel, identity, tmp_path / "changed", config=config())
    assert not (tmp_path / "changed").exists()
    with pytest.raises(ValueError, match="declared step"):
        run(parent, diagnostic, sentinel, identity, tmp_path / "step", config=config(expected_parent_step=3))
    with pytest.raises(ValueError, match="original frozen teacher"):
        run(parent, diagnostic, sentinel, {"teacher_checkpoint_sha256": "c" * 64}, tmp_path / "teacher", config=config())


def test_acceptance_requires_two_consecutive_post_update_evaluations(tmp_path, monkeypatch):
    # The actual evaluation still runs. Only its acceptance booleans are
    # scheduled here to exercise the stopping state machine independently of
    # this intentionally unlearnable random fixture's audio quality.
    parent, diagnostic, sentinel, identity = fixture()
    original = comparison._evaluate
    def scheduled(engine, training, heldout):
        report = original(engine, training, heldout)
        passed = engine.step in (0, 2, 6, 8)
        report["diagnostic_passed"] = passed
        report["sentinel_passed"] = False
        report["gates"]["diagnostic"]["all"]["passed"] = passed
        report["gates"]["diagnostic"]["all"]["milestone_095_passed"] = engine.step >= 2
        return report
    monkeypatch.setattr(comparison, "_evaluate", scheduled)
    result = run(parent, diagnostic, sentinel, identity, tmp_path / "run", config=config(total_steps=10))
    for outcome in result["arms"].values():
        assert outcome["step"] == 8
        assert outcome["state"] == "gate_passed_awaiting_review"
        assert outcome["consecutive_acceptances"] == 2
        assert outcome["milestones"]["first_095_all_clips_step"] == 2
        assert not outcome["sentinel_passed"]


def test_instability_guard_distinguishes_inherited_quiet_failure():
    def report(rms):
        return {"rows": {"sentinel": [{"source_id": "quiet", "start_frame": 0,
            "teacher_rms": 1e-5, "student_rms": rms, "waveform_cosine": .01}]}}
    assert comparison._evaluation_safety(report(.1), report(.1)) == []
    assert comparison._evaluation_safety(report(1.5), report(.2)) == []
    failures = comparison._evaluation_safety(report(2.1), report(.2))
    assert failures[0]["reason"] == "catastrophic_amplitude"
    assert comparison._evaluation_safety(report(float("nan")), report(.1))[0]["reason"] == "nonfinite_evaluation"


def test_disk_capacity_checked_before_starting_either_arm(tmp_path, monkeypatch):
    parent, diagnostic, sentinel, identity = fixture()
    monkeypatch.setattr(comparison.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="both bounded arms"):
        run(parent, diagnostic, sentinel, identity, tmp_path / "run", config=config())
    assert not list((tmp_path / "run").glob("**/checkpoint*.pt"))


def test_tensorboard_records_minimum_target_and_fraction(tmp_path):
    events = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    parent, diagnostic, sentinel, identity = fixture()
    run(parent, diagnostic, sentinel, identity, tmp_path / "run", config=config(),
        log_dir=tmp_path / "logs", run_name="pair")
    reader = events.EventAccumulator(str(tmp_path / "logs" / "pair-waveform-only")).Reload()
    for tag in ("nonquiet_cosine_min", "nonquiet_cosine_mean", "nonquiet_fraction_passing_095",
                "nonquiet_fraction_passing_099", "waveform_cosine_target"):
        values = reader.Scalars(f"evaluation/diagnostic/{tag}")
        assert [v.step for v in values] == [0, 2, 4]
    assert reader.Scalars("evaluation/diagnostic/waveform_cosine_target")[-1].value == pytest.approx(.99)
    assert reader.Scalars("train/learning_rate")[-1].value == pytest.approx(2e-4)
