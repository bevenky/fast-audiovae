"""Bounded synthetic CPU runner checks, including real GAN optimizer updates."""
from copy import deepcopy
from dataclasses import replace
import fcntl
import json

import pytest
import torch

from audiovae_student.comparison_data import load_comparison_plan, write_comparison_plan
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.model import StudentConfig
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity, fixed_crops, PreflightConfig
from audiovae_student.recipe_v2 import RecipeV2Config, RecipeV2Engine
from audiovae_student.recipe_v2_pilot import run_recipe_v2
from audiovae_student.restart_data import PilotWindow
from audiovae_student.teacher import CHECKPOINT_SHA256
from test_representative_pilot import Corpus
from test_restart_data import counts, ledger_for, row


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(path):
    train = [row("train-a", samples=10*640), row("train-b", samples=10*640, language="en")]
    calibration = [row("cal-a", samples=10*640)]
    heldout = [replace(row("dev-a", samples=10*640), split="dev", source_split="dev")]
    corpus = Corpus(train + calibration)
    dev_corpus = Corpus(heldout)
    windows = [PilotWindow(source.source_id, start, 5, 5*640, source.dataset, source.language)
               for start in (0, 5) for source in train]
    calibration_windows = [PilotWindow(calibration[0].source_id, start, 5, 5*640,
                           calibration[0].dataset, calibration[0].language) for start in (0, 5)]
    plan = dict(rows=train, windows=windows, counts=counts(train), ledger=ledger_for(train + calibration),
        reserved=heldout, excluded=(), seed=47,
        metadata={"minimum_output_samples": 9120, "minimum_input_samples": 3040})
    write_comparison_plan(plan, path / "plan")
    plan = load_comparison_plan(path / "plan")
    crops = fixed_crops(dev_corpus.get("dev-a"), role="sentinel",
        config=PreflightConfig(scored_frames=5), minimum_samples=4096)
    data_identity = {"teacher_checkpoint_sha256": CHECKPOINT_SHA256,
        "teacher_batch_qualification": {"fixture": True}, "source_corpus": corpus.identity,
        "heldout": {"teacher_checkpoint_sha256": CHECKPOINT_SHA256, "source_corpus": dev_corpus.identity,
            "rows": [r.to_dict() for r in heldout], "crops": _crop_identity(crops),
            "input_sample_counts": counts(heldout)}}
    return dict(corpus=corpus, plan=plan, heldout_crops=crops, heldout_rows=heldout,
        calibration_windows=calibration_windows, calibration_rows=calibration,
        calibration_counts=counts(calibration), calibration_provenance={"split": "train", "fixture": True},
        data_identity=data_identity)


def run(path, data, *, resume=False, max_updates=None, **kwargs):
    torch.manual_seed(991)
    discriminator = AudioDiscriminators(DiscriminatorConfig(periods=(2,),
        mpd_channels=(2, 2, 4, 4, 4), fft_sizes=(64,), mrd_channels=2))
    return run_recipe_v2(output_dir=path, **data,
        recipe=RecipeV2Config(total_steps=4, reconstruction_warmup_steps=2,
            learning_rate_warmup_steps=1, perceptual_ramp_steps=2, optimizer="adamw"),
        model_config=StudentConfig(hidden_channels=4, expansion_channels=8, head_channels=8,
            dilations=(1,), layer_scale_init=.1, normalization_mode="masked_batch_norm",
            adapter_mode="raw_repeat_phase_bias"),
        device="cpu", batch_size=1, checkpoint_interval=2, min_free_bytes=0,
        discriminators=discriminator, max_updates=max_updates,
        resume_from=path / "latest.pt" if resume else None, **kwargs)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_single_recipe_consumes_unique_windows_and_starts_real_gan_after_calibration(tmp_path):
    data = fixture(tmp_path)
    result = run(tmp_path / "run", data)
    assert result["state"] == "budget_completed_awaiting_review"
    assert result["consumed_windows"] == 4 and result["remaining_windows"] == 0
    assert result["discriminator_updates"] == 2 and result["perceptual_training_started"]
    assert result["heldout_gate_passed"] is False  # No correlation threshold blocks the stage.
    assert result["unique_scored_hours"] == 4 * 9600 / 172_800_000
    assert result["teacher_coverage"]["valid_samples"] == 4 * 9600
    assert sorted(p.name for p in (tmp_path / "run").glob("evaluation-step*.json")) == [
        "evaluation-step000002.json", "evaluation-step000004.json"]
    metrics = read_lines(tmp_path / "run/metrics.jsonl")
    assert [m["discriminator_updates"] for m in metrics] == [0, 0, 1, 2]
    assert metrics[2]["feature_matching"] > 0 and metrics[2]["adversarial/raw_norm"] > 0
    checkpoint = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert checkpoint["engine"]["gate"] is None
    assert checkpoint["engine"]["discriminator_optimizer"]["state"]
    cal = json.loads((tmp_path / "run/calibration.json").read_text())
    assert cal["unique_windows"] == 2 and cal["passes"] == 2 and cal["optimizer_updates"] == 0
    assert cal["calibration"]["completed_step"] == 2
    assert cal["calibration"]["report"]["parameters_unchanged"]
    assert data["corpus"].access.count("cal-a") == 4  # Two windows, two passes.
    assert "dev-a" not in data["corpus"].access
    assert len(read_lines(tmp_path / "run/exposure.jsonl")) == 4
    assert not (tmp_path / "run/inflight.json").exists()


@pytest.mark.parametrize("pause", [1, 2, 3])
def test_clean_resume_matches_entire_model_optimizer_rng_and_calibrated_statistics(tmp_path, pause):
    data = fixture(tmp_path)
    run(tmp_path / "whole", data)
    partial = run(tmp_path / "split", data, max_updates=pause)
    assert partial["step"] == pause
    boundary = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    assert (boundary["engine"]["calibration"] is not None) == (pause >= 2)
    if pause == 2:
        assert boundary["latest_evaluation"]["step"] == 2
        assert boundary["engine"]["discriminator_updates"] == 0
    resumed = run(tmp_path / "split", data, resume=True)
    assert resumed["new_updates"] == 4 - pause
    left = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    right = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    for key in ("engine", "rng", "sampler", "latest_evaluation", "teacher_coverage", "journal_sha256"):
        assert state_fingerprint(left[key]) == state_fingerprint(right[key]), key


@pytest.mark.parametrize("damage", ["inflight", "journal", "metrics", "coverage", "heldout", "plan", "evaluation"])
def test_resume_rejects_changed_or_uncheckpointed_evidence(tmp_path, damage):
    data = fixture(tmp_path)
    run(tmp_path / "run", data, max_updates=2)
    if damage == "inflight":
        (tmp_path / "run/inflight.json").write_text('{}')
    elif damage in {"journal", "metrics"}:
        name = "exposure.jsonl" if damage == "journal" else "metrics.jsonl"
        with (tmp_path / "run" / name).open("a") as handle:
            handle.write('{"step":3}\n')
    elif damage == "coverage":
        checkpoint = torch.load(tmp_path / "run/latest.pt", weights_only=True)
        checkpoint["teacher_coverage"]["valid_samples"] += 1
        torch.save(checkpoint, tmp_path / "run/latest.pt")
    elif damage == "heldout":
        data["heldout_crops"][0].teacher_audio[..., 0] += .1
    elif damage == "plan":
        data["plan"]["metadata"]["annotation"] = "changed"
    else:
        (tmp_path / "run/evaluation-step000002.json").write_text('{}')
    with pytest.raises(ValueError):
        run(tmp_path / "run", data, resume=True)


@pytest.mark.parametrize("damage", ["overlap", "dev", "teacher", "short", "budget", "calibration_source"])
def test_invalid_inputs_fail_before_creating_run_directory(tmp_path, damage):
    data = fixture(tmp_path)
    if damage == "overlap":
        data["calibration_windows"] = [data["plan"]["windows"][0]]
    elif damage == "dev":
        source = data["calibration_rows"][0]
        data["heldout_rows"][0] = replace(data["heldout_rows"][0], speaker_id=source.speaker_id)
    elif damage == "teacher":
        data["data_identity"]["teacher_checkpoint_sha256"] = "f" * 64
    elif damage == "short":
        data["plan"]["windows"][0] = replace(data["plan"]["windows"][0], valid_input_samples16k=2000)
    elif damage == "budget":
        data["plan"]["windows"].pop()
    else:
        data["calibration_rows"][0] = replace(data["calibration_rows"][0], split="dev", source_split="dev")
    with pytest.raises(ValueError):
        run(tmp_path / "run", data)
    assert not (tmp_path / "run").exists()


def test_calibration_failure_retains_inflight_marker_and_never_consumes_next_window(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    def fail(self, batches, *, provenance):
        list(batches())
        raise RuntimeError("injected calibration failure")
    monkeypatch.setattr(RecipeV2Engine, "calibrate", fail)
    with pytest.raises(RuntimeError, match="calibration failure"):
        run(tmp_path / "run", data)
    assert len(read_lines(tmp_path / "run/exposure.jsonl")) == 2
    marker = json.loads((tmp_path / "run/inflight.json").read_text())
    assert marker["phase"] == "normalization_calibration" and marker["step"] == 2
    with pytest.raises(ValueError, match="in-flight"):
        run(tmp_path / "run", data, resume=True)


def test_optimizer_failure_after_update_is_not_silently_replayed(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    original = RecipeV2Engine.train_step
    def fail(self, crops):
        metrics = original(self, crops)
        if self.step == 3:
            raise RuntimeError("injected after optimizer step")
        return metrics
    monkeypatch.setattr(RecipeV2Engine, "train_step", fail)
    with pytest.raises(RuntimeError, match="after optimizer"):
        run(tmp_path / "run", data)
    assert len(read_lines(tmp_path / "run/exposure.jsonl")) == 2
    assert json.loads((tmp_path / "run/inflight.json").read_text())["step"] == 3
    with pytest.raises(ValueError, match="in-flight"):
        run(tmp_path / "run", data, resume=True)


def test_same_unknown_fleurs_placeholder_is_not_treated_as_identical_person(tmp_path):
    data = fixture(tmp_path)
    shared = "fleurs:unknown-session-group:hi_in"
    for key in ("heldout_rows", "calibration_rows"):
        data[key] = [replace(r, session_id=shared) for r in data[key]]
    data["plan"]["rows"] = [replace(r, session_id=shared) for r in data["plan"]["rows"]]
    data["corpus"] = Corpus(data["plan"]["rows"] + data["calibration_rows"])
    dev = Corpus(data["heldout_rows"])
    data["heldout_crops"] = fixed_crops(dev.get("dev-a"), role="sentinel",
        config=PreflightConfig(scored_frames=5), minimum_samples=4096)
    data["data_identity"]["heldout"].update(rows=[r.to_dict() for r in data["heldout_rows"]],
        crops=_crop_identity(data["heldout_crops"]))
    # The old generic validator rejected this documented unknown placeholder.
    result = run(tmp_path / "run", data, max_updates=1)
    assert result["step"] == 1


def test_concurrent_owner_is_rejected(tmp_path):
    data = fixture(tmp_path)
    with (tmp_path / ".run.runner.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            run(tmp_path / "run", data)


def test_tensorboard_logs_new_recipe_and_fixed_evaluation_calendar(tmp_path):
    events = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    data = fixture(tmp_path)
    run(tmp_path / "run", data, log_dir=tmp_path / "logs")
    reader = events.EventAccumulator(str(tmp_path / "logs/decoder-recipe-v2")).Reload()
    assert [v.step for v in reader.Scalars("validation/all/nonquiet_cosine_mean")] == [2, 4]
    assert reader.Scalars("train/discriminator_updates")[-1].value == 2
    assert reader.Scalars("coverage/teacher/valid_samples")[-1].value == 4*9600
    text = reader.Tensors("run/qualification/text_summary")[0].tensor_proto.string_val[0].decode()
    assert "declared calendar" in text and "No automatic GAN" not in text
