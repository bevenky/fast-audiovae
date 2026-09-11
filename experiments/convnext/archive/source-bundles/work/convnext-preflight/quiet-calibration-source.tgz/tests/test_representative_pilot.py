"""Small CPU control tests; synthetic audio establishes no quality claim."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from audiovae_student.cache import UtteranceCache
from audiovae_student.discriminators import AudioDiscriminators, DiscriminatorConfig
from audiovae_student.distillation_training import DistillationTrainingConfig
from audiovae_student.losses_distillation import DistillationLossConfig
from audiovae_student.model import StudentConfig
from audiovae_student.preflight_distillation import fixed_crops, PreflightConfig
from audiovae_student.representative_pilot import (RepresentativePilotConfig, run_representative_pilot,
    summarize_evaluation, load_restart_plan)
from audiovae_student.restart_data import PilotWindow, digest, write_plan
from audiovae_student.teacher import CHECKPOINT_SHA256, SOURCE_SHA256
from test_restart_data import row, ledger_for, counts

torch.set_num_threads(1)


class Corpus:
    identity = {"fixture": "small-verified-cpu-corpus"}

    def __init__(self, rows):
        self.records = {}
        self.access = []
        for index, source in enumerate(rows):
            n = round(source.duration_seconds * 16000)
            frames = (n + 639) // 640
            rng = torch.Generator().manual_seed(index + 200)
            meta = {"identity": {"source": source.to_dict(), "input_samples": n, "posterior": "raw_mu",
                "teacher": {"checkpoint_sha256": CHECKPOINT_SHA256, "source_sha256": SOURCE_SHA256,
                            "sample_rate_out": 48000, "latent_channels": 64}}}
            self.records[source.source_id] = UtteranceCache(torch.randn(1, 64, frames, generator=rng),
                torch.randn(1, 1, frames * 1920, generator=rng) * .01, None, meta, digest(source.to_dict()))

    def get(self, identifier):
        self.access.append(identifier)
        return self.records[identifier]

    def prefetch(self, identifiers, **kwargs):
        assert set(identifiers) <= set(self.records)

    def flush(self):
        pass


def fixture():
    training = [row("new-a", samples=6*640), row("new-b", samples=6*640, language="en")]
    heldout = [replace(row("dev-a", samples=6*640), split="dev", source_split="dev")]
    corpus = Corpus(training + heldout)
    windows = [PilotWindow(r.source_id, start, 3, 3*640, r.dataset, r.language)
               for start in (0, 3) for r in training]
    panel = fixed_crops(corpus.get("dev-a"), role="sentinel",
                       config=PreflightConfig(scored_frames=3), minimum_samples=256)
    return corpus, windows, training, counts(training + heldout), ledger_for(training), panel, heldout


def run(output, *, resume=None, max_updates=None, data=None, **overrides):
    args = fixture() if data is None else data
    corpus = args[0]
    config = RepresentativePilotConfig(batch_size=1, total_steps=4, evaluation_interval=2,
        checkpoint_interval=2, waveform_only_steps=1, mel_ramp_steps=2, min_free_bytes=0)
    train = DistillationTrainingConfig(total_steps=4, warmup_steps=1, freeze_normalization_step=1,
        optimizer="adamw", reconstruction_waveform_share=1.0, quiet_gradient_share=.05,
        quiet_window_gate=True, learning_rate_schedule="constant_after_warmup")
    model = __import__("audiovae_student.model", fromlist=["StudentConfig"]).StudentConfig(
        hidden_channels=4, expansion_channels=8, head_channels=8, dilations=(1,),
        layer_scale_init=.1, normalization_mode="masked_batch_norm")
    discriminator = AudioDiscriminators(DiscriminatorConfig(periods=(2,),
        mpd_channels=(2,2,4,4,4), fft_sizes=(64,), mrd_channels=2))
    return run_representative_pilot(*args, output,
        data_identity={"teacher_checkpoint_sha256": CHECKPOINT_SHA256,
                       "teacher_batch_qualification": {"fixture": True}, "source_corpus": corpus.identity},
        config=config, training_config=train, model_config=model,
        loss_config=DistillationLossConfig(fft_sizes=(256,), mel_bands=(8,)),
        discriminators=discriminator, device="cpu", resume_from=resume, max_updates=max_updates, **overrides)


def test_fresh_unique_window_pilot_stops_and_logs_grouped_quiet_metrics(tmp_path):
    result = run(tmp_path / "run")
    assert result["state"] == "budget_completed_awaiting_review"
    assert result["consumed_windows"] == 4 and result["remaining_windows"] == 0
    assert not result["perceptual_training_started"]
    assert "language/hi" in result["groups"] and "condition/unverified" in result["groups"]
    assert result["unique_scored_hours"] == 4*3*640 / 57_600_000
    checkpoint = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert checkpoint["engine"]["step"] == 4
    assert checkpoint["engine"]["config"]["reconstruction_waveform_share"] == .75
    assert checkpoint["engine"]["perceptual_start"] is None
    assert len(list((tmp_path / "run").glob("*.pt"))) == 1
    assert (tmp_path / "run/evaluation-step000000.json").exists()
    rows = [json.loads(line) for line in (tmp_path / "run/exposure.jsonl").read_text().splitlines()]
    assert [r["cursor"] for r in rows] == [1,2,3,4]
    assert not (tmp_path / "run/inflight.json").exists()
    with pytest.raises(FileExistsError):
        run(tmp_path / "run")


def test_exact_clean_resume_matches_uninterrupted_model_and_cursor(tmp_path):
    run(tmp_path / "whole")
    first = run(tmp_path / "split", max_updates=2)
    assert first["state"] == "paused_at_explicit_launch_bound"
    final = run(tmp_path / "split", resume=tmp_path / "split/latest.pt")
    assert final["new_updates"] == 2
    left = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    right = torch.load(tmp_path / "split/latest.pt", weights_only=True)
    assert left["sampler"] == right["sampler"]
    for key, value in left["engine"]["model"].items():
        torch.testing.assert_close(value, right["engine"]["model"][key], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["journal", "inflight", "curriculum"])
def test_interrupted_or_changed_resume_fails_closed(tmp_path, damage):
    run(tmp_path / "run", max_updates=2)
    if damage == "journal":
        with (tmp_path / "run/exposure.jsonl").open("a") as handle:
            handle.write(json.dumps({"step":3, "cursor":3, "window_identity":"changed"}) + "\n")
    elif damage == "inflight":
        (tmp_path / "run/inflight.json").write_text("{}")
    else:
        state = torch.load(tmp_path / "run/latest.pt", weights_only=True)
        state["engine"]["config"]["reconstruction_waveform_share"] = .123
        torch.save(state, tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="exposure|curriculum"):
        run(tmp_path / "run", resume=tmp_path / "run/latest.pt")


def test_cache_source_mutation_fails_before_scoring_and_reserves_no_audio(tmp_path):
    data = fixture()
    data[0].records["new-a"].metadata["identity"]["teacher"]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="cache changed"):
        run(tmp_path / "run", data=data)
    assert not (tmp_path / "run/exposure.jsonl").exists()
    assert not (tmp_path / "run/inflight.json").exists()


def test_repeated_window_and_heldout_leak_are_rejected_before_output(tmp_path):
    data = list(fixture())
    data[1][1] = data[1][0]
    with pytest.raises(ValueError, match="Repeated"):
        run(tmp_path / "duplicate", data=data)
    assert not (tmp_path / "duplicate").exists()
    data = list(fixture())
    data[6] = [replace(data[6][0], speaker_id=data[2][0].speaker_id)]
    with pytest.raises(ValueError, match="Reserved"):
        run(tmp_path / "leak", data=data)
    assert not (tmp_path / "leak").exists()


def test_curriculum_is_bounded_and_does_not_start_mel_early():
    config = RepresentativePilotConfig(waveform_only_steps=250, mel_ramp_steps=250)
    assert [config.waveform_share(i) for i in (0,249,250,375,500,999)] == [1,1,1,.875,.75,.75]
    with pytest.raises(ValueError):
        RepresentativePilotConfig(quiet_gradient_share=.3)
