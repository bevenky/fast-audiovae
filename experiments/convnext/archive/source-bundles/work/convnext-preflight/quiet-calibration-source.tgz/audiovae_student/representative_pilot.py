"""Fresh, finite reconstruction training on an immutable no-repeat window plan.

The caller owns a qualified frozen-teacher SourceCorpus (including its bounded
rolling cache). This runner never imports diagnostic weights or enables GANs.
An fsynced exposure journal and in-flight reservation prevent silent replay
after an interruption between model checkpoints.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from .cache import TrainingCrop, UtteranceCache, sample_training_crop
from .data import ManifestRow, load_manifest, validate_manifest
from .distillation_training import (DistillationEngine, DistillationTrainingConfig,
                                    evaluate_crops, model_state_fingerprint, waveform_gate)
from .losses_distillation import DistillationLossConfig
from .model import StudentConfig, StudentDecoder
from .preflight_distillation import _crop_identity, _writer
from .restart_data import (ConsumedLedger, FixedWindowSampler, PilotWindow, canonical,
                           digest, file_sha, identities, identity, verify_windows)
from .teacher import CHECKPOINT_SHA256, SOURCE_SHA256
from .training import _atomic_save, _restore_rng, _rng_state


@dataclass(frozen=True)
class RepresentativePilotConfig:
    batch_size: int = 32
    total_steps: int = 1009
    context_frames: int = 29
    seed: int = 23
    evaluation_interval: int = 100
    checkpoint_interval: int = 100
    waveform_only_steps: int = 250
    mel_ramp_steps: int = 250
    final_mel_share: float = 0.25
    quiet_gradient_share: float = 0.05
    min_free_bytes: int = 4 * 1024**3
    teacher_batch_size: int = 8
    teacher_batch_samples: int = 1_920_000

    def __post_init__(self):
        for name in ("batch_size", "total_steps", "evaluation_interval", "checkpoint_interval",
                     "mel_ramp_steps", "teacher_batch_size", "teacher_batch_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "waveform_only_steps", "min_free_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.context_frames) is not int or self.context_frames < 29:
            raise ValueError("At least 29 original latent context frames are required")
        if self.teacher_batch_size > 8:
            raise ValueError("Teacher batch size exceeds the qualified maximum")
        if not math.isfinite(self.final_mel_share) or not 0 <= self.final_mel_share < 1:
            raise ValueError("final_mel_share must be finite in [0, 1)")
        if not math.isfinite(self.quiet_gradient_share) or not 0 <= self.quiet_gradient_share <= .25:
            raise ValueError("Quiet gradient share must be finite and bounded by 0.25")

    def waveform_share(self, completed_steps: int) -> float:
        fraction = min(1.0, max(0.0, (completed_steps - self.waveform_only_steps) / self.mel_ramp_steps))
        return 1.0 - self.final_mel_share * fraction


def load_restart_plan(directory: str | Path) -> dict:
    """Read only a published checksum-complete planner output; recheck exclusions."""
    directory = Path(directory).resolve(strict=True)
    ready = json.loads((directory / "ready.json").read_text())
    body = {k: v for k, v in ready.items() if k != "identity_sha256"}
    if ready.get("state") != "ready" or ready.get("identity_sha256") != digest(body):
        raise ValueError("Restart plan is not a checksum-complete ready plan")
    required = {"train-manifest.jsonl", "windows.jsonl", "ledger.json", "input-sample-counts.json",
                "diagnostic.jsonl", "sentinel.jsonl", "dev-reserve.jsonl", "test-reserve.jsonl"}
    if not required <= set(ready["files_sha256"]):
        raise ValueError("Restart plan is missing required reservations")
    for name, expected in ready["files_sha256"].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or file_sha(path) != expected:
            raise ValueError("Restart plan file checksum or path changed")
    rows = load_manifest(directory / "train-manifest.jsonl")
    reserved = load_manifest(directory / "dev-reserve.jsonl") + load_manifest(directory / "test-reserve.jsonl")
    diagnostic = load_manifest(directory / "diagnostic.jsonl")
    sentinel = load_manifest(directory / "sentinel.jsonl")
    counts = json.loads((directory / "input-sample-counts.json").read_text())
    windows = []
    for line in (directory / "windows.jsonl").read_text().splitlines():
        value = json.loads(line)
        fields = {key: value[key] for key in PilotWindow.__dataclass_fields__}
        window = PilotWindow(**fields)
        if value != window.to_dict():
            raise ValueError("Pilot window has inconsistent sample accounting")
        windows.append(window)
    ledger = ConsumedLedger(json.loads((directory / "ledger.json").read_text()))
    verify_windows(windows, {r.source_id: r for r in rows}, counts, ledger,
                   reserved_rows=reserved + sentinel, excluded_sources=diagnostic)
    if FixedWindowSampler(windows).identity_sha256 != ready["fixed_sampler_identity"]:
        raise ValueError("Pilot window order differs from the published identity")
    return dict(rows=rows, windows=windows, counts=counts, ledger=ledger, reserved=reserved,
                diagnostic=diagnostic, sentinel=sentinel, identity=ready)


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append(path: Path, value):
    with path.open("ab") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def implementation_identity():
    root = Path(__file__).parent
    names = ("representative_pilot", "cache", "data", "source_corpus", "batched_teacher", "model",
             "distillation_training", "losses_distillation", "gradient_balancer", "optimizers",
             "quiet_audio", "restart_data", "training", "teacher")
    return {name: file_sha(root / (name + ".py")) for name in names}


def _record(corpus, row: ManifestRow, count: int) -> UtteranceCache:
    record = corpus.get(row.source_id)
    # SourceCorpus verifies full tensor/file checksums on preparation or load.
    # Check immutable provenance even for its in-memory cache-hit path.
    meta = record.metadata["identity"]
    source = ManifestRow.from_dict(meta["source"])
    teacher = meta["teacher"]
    if (identity(source) != identity(row) or record.input_samples != count or
            teacher.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
            teacher.get("source_sha256") != SOURCE_SHA256 or meta.get("posterior") != "raw_mu" or
            teacher.get("sample_rate_out") != 48000 or teacher.get("latent_channels") != 64 or
            record.latents.requires_grad or record.teacher_audio.requires_grad):
        raise ValueError("Teacher cache changed source, length, raw latent or teacher provenance")
    return record


def summarize_evaluation(rows: Sequence[dict], metadata: Mapping[str, dict]) -> dict:
    """Report each source/language/event; unknown event labels stay unknown."""
    groups = defaultdict(list)
    for row in rows:
        meta = metadata[row["source_id"]]
        for name in ("dataset", "language", "condition"):
            groups[f"{name}/{meta.get(name) or 'unverified'}"].append(row)
        groups["all"].append(row)
    result = {}
    for key, members in groups.items():
        nonquiet = [row for row in members if row["teacher_rms"] >= 1e-3]
        quiet = [row.get("quiet_windows", {}) for row in members]
        windows = sum(row.get("quiet_window_count", 0) for row in quiet)
        failures = sum(row.get("quiet_failed_count", 0) for row in quiet)
        result[key] = {"crops": len(members), "sources": len({r["source_id"] for r in members}),
            "scored_samples": sum(r["samples"] for r in members),
            "teacher_waveform_mean": sum(r["teacher_waveform"] for r in members) / len(members),
            "teacher_mel_mean": sum(r["teacher_mel"] for r in members) / len(members),
            "nonquiet_crops": len(nonquiet),
            "nonquiet_cosine_mean": sum(r["waveform_cosine"] for r in nonquiet) / len(nonquiet) if nonquiet else None,
            "nonquiet_cosine_min": min((r["waveform_cosine"] for r in nonquiet), default=None),
            "nonquiet_fraction_passing_099": sum(r["waveform_cosine"] >= .99 for r in nonquiet) / len(nonquiet) if nonquiet else None,
            "quiet_window_count": windows, "quiet_failed_count": failures,
            "quiet_fraction_passing": (windows - failures) / windows if windows else None}
    return result


def _verify_panel(crops, panel_rows, training_rows, counts):
    if not crops or len({r.source_id for r in panel_rows}) != len(panel_rows):
        raise ValueError("A nonempty uniquely identified development panel is required")
    by_id = {r.source_id: r for r in panel_rows}
    if {c.source_id for c in crops} != set(by_id) or any(r.split != "dev" for r in panel_rows):
        raise ValueError("Held-out crops must exactly match declared development sources")
    validate_manifest(training_rows, reserved_rows=panel_rows, training_only=True)
    train_keys = set().union(*(identities(r) for r in training_rows))
    if any(identities(r) & train_keys for r in panel_rows):
        raise ValueError("Held-out source, parent, hash, speaker or session overlaps training")
    for crop in crops:
        if crop.valid_scored_samples > 3 * counts[crop.source_id] - crop.start_frame * 1920:
            raise ValueError("Held-out crop exceeds the original valid output length")


def run_representative_pilot(corpus, windows: Sequence[PilotWindow], training_rows: Sequence[ManifestRow],
        sample_counts: Mapping[str, int], ledger: ConsumedLedger, heldout_crops: Sequence[TrainingCrop],
        heldout_rows: Sequence[ManifestRow], output_dir: str | Path, *, data_identity: dict,
        config=RepresentativePilotConfig(), training_config=None, model_config=None,
        loss_config=DistillationLossConfig(), device="cuda", resume_from=None, max_updates=None,
        heldout_metadata=None, reserved_rows=(), excluded_sources=(), discriminators=None,
        log_dir=None, run_name="representative-reconstruction-pilot-v1") -> dict:
    """Train at most the declared unique-window budget, then stop for review.

    ``max_updates`` allows an explicit clean checkpoint boundary for a bounded
    launch slice. Exact resume requires both that checkpoint and the unchanged
    on-disk journal. No initialization checkpoint is accepted on a fresh run.
    """
    windows, training_rows, heldout_rows = tuple(windows), tuple(training_rows), tuple(heldout_rows)
    by_id = {r.source_id: r for r in training_rows}
    verify_windows(windows, by_id, sample_counts, ledger,
                   reserved_rows=tuple(reserved_rows) + heldout_rows, excluded_sources=excluded_sources)
    _verify_panel(heldout_crops, heldout_rows, training_rows, sample_counts)
    if config.total_steps != math.ceil(len(windows) / config.batch_size):
        raise ValueError("Update budget must exactly consume the finite window plan, including its short final batch")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive explicit launch bound")
    if (data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256 or
            not data_identity.get("teacher_batch_qualification") or
            data_identity.get("source_corpus") != corpus.identity):
        raise ValueError("Pilot requires pinned teacher, immutable corpus and teacher-batching qualification evidence")
    model_config = model_config or StudentConfig(normalization_mode="masked_batch_norm")
    if config.context_frames * 4 < model_config.history_frames:
        raise ValueError("The planned real context is shorter than the student's receptive field")
    training_config = training_config or DistillationTrainingConfig(total_steps=config.total_steps,
        warmup_steps=50, freeze_normalization_step=200, reconstruction_waveform_share=1.0,
        quiet_gradient_share=config.quiet_gradient_share, quiet_window_gate=True,
        learning_rate_schedule="constant_after_warmup", parameter_update_metrics_interval=100)
    if training_config.total_steps != config.total_steps or not training_config.quiet_window_gate:
        raise ValueError("Engine budget must match the pilot and enable local quiet-window acceptance")
    metadata = {r.source_id: {"dataset": r.dataset, "language": r.language, "condition": None} for r in heldout_rows}
    for key, value in (heldout_metadata or {}).items():
        if key not in metadata or value.get("dataset", metadata[key]["dataset"]) != metadata[key]["dataset"]:
            raise ValueError("Evaluation metadata changes source identity")
        metadata[key].update(value)
    sampler = FixedWindowSampler(windows)
    run_identity = {"kind": "fresh_representative_reconstruction_pilot", "format_version": 1,
        "config": asdict(config), "initial_training_config": asdict(training_config),
        "model_config": model_config.to_dict(), "loss_config": asdict(loss_config),
        "data": data_identity, "window_identity": sampler.identity_sha256,
        "ledger_identity": ledger.identity_sha256, "heldout": _crop_identity(heldout_crops),
        "heldout_metadata": metadata, "implementation": implementation_identity(),
        "initialization": "fresh_seeded_student_and_optimizer_no_diagnostic_weights"}
    run_identity = json.loads(canonical(run_identity))
    directory = Path(output_dir)
    if resume_from is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise ValueError("Exact resume needs the original run directory and exposure journal")
    if shutil.disk_usage(directory).free < config.min_free_bytes:
        raise OSError("Insufficient disk reserve for the bounded pilot")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = StudentDecoder(model_config).to(device)
    engine = DistillationEngine(model, config=training_config, loss_config=loss_config, discriminators=discriminators)
    initial_fingerprint = model_state_fingerprint(model)
    journal = directory / "exposure.jsonl"
    inflight = directory / "inflight.json"
    latest = directory / "latest.pt"
    latest_evaluation = None
    seconds = 0.0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=True)
        if payload.get("identity") != run_identity or inflight.exists():
            raise ValueError("Resume identity changed or an interrupted in-flight batch has uncertain exposure")
        state = payload["engine"]
        expected_share = config.waveform_share(max(0, state["step"] - 1))
        if state["config"]["reconstruction_waveform_share"] != expected_share:
            raise ValueError("Resume objective differs from the immutable curriculum")
        engine.set_reconstruction_objective(waveform_share=expected_share, quiet_share=config.quiet_gradient_share)
        engine.load_state_dict(state)
        sampler.load_state_dict(payload["sampler"])
        records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        expected_cursor = min(engine.step * config.batch_size, len(windows))
        if (sampler.cursor != expected_cursor or len(records) != engine.step or
                any(row != {"step": i + 1, "cursor": min((i + 1) * config.batch_size, len(windows)),
                            "window_identity": sampler.identity_sha256} for i, row in enumerate(records))):
            raise ValueError("Uncheckpointed exposure cannot be replayed; resume only a matching committed cursor")
        if payload["initial_model_state_sha256"] != initial_fingerprint:
            raise ValueError("Fresh initialization changed")
        _restore_rng(payload["rng"])
        latest_evaluation, seconds = payload["latest_evaluation"], payload["seconds"]
    elif journal.exists() or latest.exists():
        raise ValueError("Fresh run directory already contains training state")
    writer = _writer(log_dir, run_name, run_identity, engine.step, resume_from is not None)

    def checkpoint():
        corpus.flush()
        if shutil.disk_usage(directory).free < config.min_free_bytes:
            raise OSError("Checkpoint disk reserve would be violated")
        _atomic_save({"format_version": 1, "identity": run_identity, "engine": engine.state_dict(),
            "sampler": sampler.state_dict(), "rng": _rng_state(), "latest_evaluation": latest_evaluation,
            "initial_model_state_sha256": initial_fingerprint, "seconds": seconds}, latest)

    def evaluate():
        nonlocal latest_evaluation
        rng = _rng_state()
        try:
            rows = evaluate_crops(engine, heldout_crops)
            latest_evaluation = {"step": engine.step, "rows": rows,
                "groups": summarize_evaluation(rows, metadata), "gate": waveform_gate(engine, rows)}
        finally:
            _restore_rng(rng)
        _atomic_json(directory / f"evaluation-step{engine.step:06d}.json", latest_evaluation)
        if writer:
            for group, metrics in latest_evaluation["groups"].items():
                for key, value in metrics.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"validation/{group}/{key}", value, engine.step)
            writer.add_scalar("validation/all/target_cosine", .99, engine.step)
            writer.flush()

    started_step = engine.step
    stop_step = min(config.total_steps, engine.step + max_updates) if max_updates else config.total_steps
    status = "training"
    try:
        if resume_from is None:
            evaluate()
            checkpoint()
        while engine.step < stop_step:
            started = time.monotonic()
            chosen = windows[sampler.cursor:sampler.cursor + config.batch_size]
            ids = list(dict.fromkeys(w.source_id for w in chosen))
            corpus.prefetch(ids, max_batch_size=config.teacher_batch_size,
                            max_total_input_samples=config.teacher_batch_samples)
            crops = []
            for window in chosen:
                record = _record(corpus, by_id[window.source_id], sample_counts[window.source_id])
                crop = sample_training_crop(record, window.start_frame, window.scored_frames, config.context_frames)
                if crop.valid_scored_samples != window.valid_output_samples48k:
                    # A quota-selected last window can be shorter than the source.
                    if crop.valid_scored_samples < window.valid_output_samples48k:
                        raise ValueError("Cached target is shorter than the immutable scored interval")
                    crop = replace(crop, valid_scored_samples=window.valid_output_samples48k)
                crops.append(crop)
            _atomic_json(inflight, {"step": engine.step + 1, "start_cursor": sampler.cursor,
                                   "windows": [w.to_dict() for w in chosen]})
            engine.set_reconstruction_objective(waveform_share=config.waveform_share(engine.step),
                                                quiet_share=config.quiet_gradient_share)
            metrics = engine.train_step(crops)
            sampler.take_batch(len(chosen))
            _append(journal, {"step": engine.step, "cursor": sampler.cursor,
                              "window_identity": sampler.identity_sha256})
            inflight.unlink()
            seconds += time.monotonic() - started
            metrics.update({"seconds_in_training_loop": seconds, "consumed_windows": sampler.cursor,
                            "unique_scored_hours": sum(w.valid_input_samples16k for w in windows[:sampler.cursor]) / 57_600_000})
            _append(directory / "metrics.jsonl", metrics)
            if writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar("train/" + key, value, engine.step)
            if engine.step % config.evaluation_interval == 0 or engine.step == stop_step:
                evaluate()
            if engine.step % config.checkpoint_interval == 0 or engine.step == stop_step:
                checkpoint()
            _atomic_json(directory / "status.json", {"state": status, "step": engine.step,
                "total_steps": config.total_steps, "consumed_windows": sampler.cursor,
                "latest_metrics": metrics, "perceptual_training_started": False})
        status = "budget_completed_awaiting_review" if engine.step == config.total_steps else "paused_at_explicit_launch_bound"
        result = {"state": status, "step": engine.step, "new_updates": engine.step - started_step,
            "total_steps": config.total_steps, "consumed_windows": sampler.cursor,
            "remaining_windows": sampler.remaining_segments,
            "unique_scored_hours": sum(w.valid_input_samples16k for w in windows[:sampler.cursor]) / 57_600_000,
            "seconds": seconds, "heldout_gate_passed": latest_evaluation["gate"]["passed"],
            "groups": latest_evaluation["groups"], "perceptual_training_started": False,
            "initialization": run_identity["initialization"], "checkpoint": str(latest)}
        _atomic_json(directory / "status.json", result)
        return result
    except BaseException as error:
        _atomic_json(directory / "status.json", {"state": "stopped_on_error", "step": engine.step,
            "consumed_windows": sampler.cursor, "error": f"{type(error).__name__}: {error}",
            "inflight_reserved": inflight.exists(), "automatic_resume": False})
        raise
    finally:
        if writer:
            writer.close()
