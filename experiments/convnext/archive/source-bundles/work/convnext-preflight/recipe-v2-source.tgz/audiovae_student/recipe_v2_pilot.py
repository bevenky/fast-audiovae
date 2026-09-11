"""One fresh, finite decoder run with fixed-calendar perceptual training.

Teacher targets retain the original latent, timing and gain contracts. The
optimization journal never replays a scored interval. Separate training-only
calibration intervals are read twice at fixed weights, outside optimizer work.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
from functools import wraps
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch

from .batching import _validate_crop
from .cache import sample_training_crop
from .comparison_data import assert_comparison_disjoint, verify_comparison_windows
from .distillation_training import evaluate_crops, model_state_fingerprint, waveform_gate
from .model import StudentConfig, StudentDecoder
from .preflight_distillation import _crop_identity
from .recipe_v2 import RecipeV2Config, RecipeV2Engine, calibration_batch
from .representative_pilot import (_append, _atomic_json, _checkpoint_reserve, _merge_coverage,
    _record, runtime_identity, summarize_evaluation, teacher_coverage)
from .restart_data import FixedWindowSampler, canonical, digest, file_sha
from .teacher import CHECKPOINT_SHA256
from .training import _atomic_save, _restore_rng, _rng_state


def _exclusive_run(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        directory = Path(kwargs.get("output_dir", args[4] if len(args) > 4 else ""))
        if not directory.name or directory.is_symlink():
            raise ValueError("An explicit nonsymlink output directory is required")
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock_path = directory.parent / ("." + directory.name + ".runner.lock")
        if lock_path.is_symlink():
            raise ValueError("Run lock must not be a symbolic link")
        with lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process owns this recipe run") from error
            return function(*args, **kwargs)
    return guarded


@contextmanager
def _writer(log_dir, name, identity, step, resuming):
    if log_dir is None:
        yield None
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("TensorBoard run name must be a safe directory name")
    path = Path(log_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("TensorBoard identity changed")
    if any(path.glob("events.out.tfevents.*")) and (not resuming or not manifest.exists()):
        raise ValueError("Existing TensorBoard events require exact resume")
    rng = _rng_state()
    try:
        from torch.utils.tensorboard import SummaryWriter
        _atomic_json(manifest, identity)
        writer = SummaryWriter(str(path), max_queue=1, flush_secs=1, purge_step=step + 1 if resuming else None)
        writer.add_text("run/qualification", "Fresh identity-adapter decoder; frozen AudioVAE2 raw latents "
            "and waveform targets. Unique optimizer windows; separate fixed-weight normalization calibration. "
            "Adversarial and feature-matching losses start on the declared calendar. Development quality "
            "is monitored independently and never used as a training entry gate.", step)
    finally:
        _restore_rng(rng)
    try:
        yield writer
    finally:
        writer.close()


def implementation_identity():
    # Bind the actual local implementations, including the reused durable I/O.
    names = ("recipe_v2_pilot", "recipe_v2", "reconstruction_v2", "normalization_calibration",
        "representative_pilot", "comparison_data", "cache", "batching", "data", "source_corpus",
        "batched_teacher", "model", "distillation_training", "losses_distillation", "discriminators",
        "gradient_balancer", "optimizers", "quiet_audio", "restart_data", "training", "teacher",
        "preflight_distillation")
    return {name: file_sha(Path(__file__).with_name(name + ".py")) for name in names}


def _read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _panel_identity(crops, rows, evidence, context_frames):
    if (not crops or len({r.source_id for r in rows}) != len(rows)
            or {c.source_id for c in crops} != {r.source_id for r in rows}
            or any(r.split != "dev" for r in rows)):
        raise ValueError("Heldout crops must exactly cover a nonempty unique development panel")
    if (evidence.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or not evidence.get("source_corpus") or evidence.get("rows") != [r.to_dict() for r in rows]
            or evidence.get("crops") != _crop_identity(crops)):
        raise ValueError("Pinned heldout teacher, sources or raw targets changed")
    counts = evidence.get("input_sample_counts", {})
    for crop in crops:
        _validate_crop(crop)
        count = counts.get(crop.source_id)
        if (type(count) is not int or count < 1
                or crop.valid_scored_samples > count * 3 - crop.start_frame * 1920
                or crop.context_start_frame != max(0, crop.start_frame - context_frames)):
            raise ValueError("Heldout source length or real causal context changed")
    return evidence


@_exclusive_run
def run_recipe_v2(corpus, plan, heldout_crops, heldout_rows, output_dir, *,
        calibration_windows, calibration_rows, calibration_counts, calibration_provenance,
        data_identity, recipe=RecipeV2Config(), model_config=None, device="cuda", resume_from=None,
        max_updates=None, heldout_metadata=None, log_dir=None, run_name="decoder-recipe-v2",
        batch_size=32, context_frames=29, evaluation_steps=None, checkpoint_interval=100,
        min_free_bytes=4 * 1024**3, discriminators=None):
    """Run one declared recipe, with exact resume only at committed boundaries.

    ``plan`` is a loaded comparison plan. The source corpus covers the union of
    optimization and calibration rows; the heldout corpus may be separate and
    must provide pinned rows, sample counts and ``_crop_identity`` evidence in
    ``data_identity['heldout']``. ``max_updates`` bounds only this invocation.
    The fixed-weight calibration still completes when that bound is warmup.
    """
    for name, value in (("batch_size", batch_size), ("checkpoint_interval", checkpoint_interval)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if type(context_frames) is not int or context_frames < 29:
        raise ValueError("At least 29 original latent context frames are required")
    if type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError("min_free_bytes must be a nonnegative integer")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive explicit launch bound")
    model_config = model_config or StudentConfig(normalization_mode="masked_batch_norm",
        adapter_mode="raw_repeat_phase_bias")
    if context_frames * 4 < model_config.history_frames:
        raise ValueError("Real context is shorter than the student receptive field")
    windows, rows = tuple(plan["windows"]), tuple(plan["rows"])
    calibration_windows, calibration_rows = tuple(calibration_windows), tuple(calibration_rows)
    heldout_crops, heldout_rows = tuple(heldout_crops), tuple(heldout_rows)
    if recipe.total_steps != math.ceil(len(windows) / batch_size):
        raise ValueError("Update budget must consume exactly the unique finite window plan")
    if not calibration_windows or any(w.valid_output_samples48k < recipe.adversarial_samples for w in windows):
        raise ValueError("Calibration must be nonempty and every training crop must support adversarial scoring")
    counts, by_id = dict(plan["counts"]), {r.source_id: r for r in rows}
    if len(by_id) != len(rows) or len({r.source_id for r in calibration_rows}) != len(calibration_rows):
        raise ValueError("Source rows must have unique identities")
    for row in calibration_rows:
        if row.source_id in by_id and row.to_dict() != by_id[row.source_id].to_dict():
            raise ValueError("Calibration changed a shared source identity")
        count = calibration_counts.get(row.source_id)
        if type(count) is not int or count < 1 or (row.source_id in counts and count != counts[row.source_id]):
            raise ValueError("Calibration sample counts changed")
        by_id[row.source_id], counts[row.source_id] = row, count
    reserved = (*plan.get("reserved", ()), *heldout_rows)
    verify_comparison_windows((*windows, *calibration_windows), by_id, counts, plan["ledger"],
        reserved_rows=reserved, excluded_sources=plan.get("excluded", ()))
    assert_comparison_disjoint(by_id.values(), heldout_rows)
    if calibration_provenance.get("split") != "train":
        raise ValueError("Normalization calibration must use training-only examples")
    if (data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or not data_identity.get("teacher_batch_qualification")
            or data_identity.get("source_corpus") != corpus.identity):
        raise ValueError("Pinned teacher, qualified batching and immutable SourceCorpus are required")
    _panel_identity(heldout_crops, heldout_rows, data_identity.get("heldout", {}), context_frames)
    sampler = FixedWindowSampler(windows)
    if (plan["identity"].get("fixed_sampler_identity") != sampler.identity_sha256
            or plan["identity"].get("ledger_identity") != plan["ledger"].identity_sha256):
        raise ValueError("Published plan window or ledger identity changed")
    if evaluation_steps is None:
        evaluation_steps = sorted({recipe.reconstruction_warmup_steps, recipe.total_steps,
                                   *(s for s in (1000, 5000) if s <= recipe.total_steps)})
    evaluation_steps = tuple(evaluation_steps)
    if (any(type(s) is not int or not 1 <= s <= recipe.total_steps for s in evaluation_steps)
            or tuple(sorted(set(evaluation_steps))) != evaluation_steps
            or not {recipe.reconstruction_warmup_steps, recipe.total_steps} <= set(evaluation_steps)):
        raise ValueError("Evaluation calendar must include postcalibration warmup and final budget")
    metadata = {r.source_id: {"dataset": r.dataset, "language": r.language, "condition": None}
                for r in heldout_rows}
    for key, value in (heldout_metadata or {}).items():
        if key not in metadata or any(value.get(k, metadata[key][k]) != metadata[key][k] for k in ("dataset", "language")):
            raise ValueError("Heldout metadata changes source identity")
        metadata[key].update(value)
    calibration_identity = {"windows": [w.to_dict() for w in calibration_windows],
        "rows": [r.to_dict() for r in calibration_rows], "counts": dict(calibration_counts),
        "provenance": calibration_provenance, "passes": 2, "batch_size": batch_size}
    provenance = {**calibration_provenance, "calibration_identity_sha256": digest(calibration_identity),
        "optimizer_window_identity": sampler.identity_sha256, "source_corpus": corpus.identity,
        "teacher_checkpoint_sha256": CHECKPOINT_SHA256}
    run_identity = json.loads(canonical({"kind": "fresh_decoder_recipe_v2", "format_version": 2,
        "recipe": asdict(recipe), "model_config": model_config.to_dict(), "data": data_identity,
        "plan": plan["identity"], "actual_plan": {"rows": [r.to_dict() for r in rows],
            "counts": plan["counts"], "metadata": plan["metadata"],
            "reserved": [r.to_dict() for r in plan.get("reserved", ())],
            "excluded": [r.to_dict() for r in plan.get("excluded", ())]},
        "window_identity": sampler.identity_sha256, "ledger_identity": plan["ledger"].identity_sha256,
        "calibration": calibration_identity, "heldout_metadata": metadata,
        "batch_size": batch_size, "context_frames": context_frames, "evaluation_steps": evaluation_steps,
        "checkpoint_interval": checkpoint_interval, "min_free_bytes": min_free_bytes,
        "implementation": implementation_identity(), "runtime": runtime_identity(device),
        "initialization": "fresh_seeded_student_discriminators_and_optimizers"}))
    directory = Path(output_dir)
    if resume_from is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise ValueError("Resume requires the original run directory and exposure journal")
    if shutil.disk_usage(directory).free < min_free_bytes:
        raise OSError("Insufficient disk reserve")
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    model = StudentDecoder(model_config).to(device)
    engine = RecipeV2Engine(model, recipe=recipe, discriminators=discriminators)
    initial_fingerprint = model_state_fingerprint(model)
    journal, inflight, latest = (directory / name for name in ("exposure.jsonl", "inflight.json", "latest.pt"))
    metrics_file = directory / "metrics.jsonl"
    latest_evaluation, latest_metrics = None, None
    seconds, calibration_seconds, journal_hash = 0., 0., digest([])
    coverage = teacher_coverage(())
    if resume_from is not None:
        if Path(resume_from).resolve() != latest.resolve():
            raise ValueError("Exact resume requires this run's latest committed checkpoint")
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload.get("format_version") != "recipe_v2_pilot" or payload.get("identity") != run_identity or inflight.exists():
            raise ValueError("Resume identity changed or interrupted in-flight exposure is uncertain")
        engine.load_state_dict(payload["engine"])
        sampler.load_state_dict(payload["sampler"])
        records, metrics_rows = _read_lines(journal), _read_lines(metrics_file)
        chain = digest([])
        for i, record in enumerate(records):
            selected = windows[i * batch_size:(i + 1) * batch_size]
            body = {key: value for key, value in record.items() if key != "sha256"}
            if (record.get("step") != i + 1 or record.get("cursor") != min((i + 1) * batch_size, len(windows))
                    or record.get("window_identity") != sampler.identity_sha256
                    or record.get("batch_windows_sha256") != digest([w.to_dict() for w in selected])
                    or record.get("previous_sha256") != chain or record.get("sha256") != digest(body)):
                raise ValueError("Exposure journal no longer matches the immutable window sequence")
            chain = record["sha256"]
        if (len(records) != engine.step or len(metrics_rows) != engine.step
                or any(row.get("step") != i + 1 for i, row in enumerate(metrics_rows))
                or sampler.cursor != min(engine.step * batch_size, len(windows))
                or chain != payload["journal_sha256"] or digest(metrics_rows) != payload["metrics_sha256"]):
            raise ValueError("Uncheckpointed exposure or metrics cannot be replayed")
        if payload["initial_model_state_sha256"] != initial_fingerprint:
            raise ValueError("Fresh initialization changed")
        if engine.calibration and engine.calibration["report"]["provenance"] != provenance:
            raise ValueError("Calibration provenance changed on resume")
        coverage = payload["teacher_coverage"]
        if coverage["valid_samples"] != sum(w.valid_output_samples48k for w in windows[:sampler.cursor]):
            raise ValueError("Saved teacher coverage does not match committed exposure")
        latest_evaluation, latest_metrics = payload["latest_evaluation"], payload["latest_metrics"]
        if latest_evaluation is not None:
            step = latest_evaluation["step"]
            saved = directory / f"evaluation-step{step:06d}.json"
            if step not in evaluation_steps or step > engine.step or not saved.exists() or json.loads(saved.read_text()) != latest_evaluation:
                raise ValueError("Saved evaluation no longer matches its committed calendar")
        seconds, calibration_seconds, journal_hash = payload["seconds"], payload["calibration_seconds"], chain
        _restore_rng(payload["rng"])

    def crops_for(chosen):
        corpus.prefetch(list(dict.fromkeys(w.source_id for w in chosen)), max_batch_size=8,
                        max_total_input_samples=1_920_000)
        crops = []
        for window in chosen:
            record = _record(corpus, by_id[window.source_id], counts[window.source_id])
            crop = sample_training_crop(record, window.start_frame, window.scored_frames, context_frames)
            if crop.valid_scored_samples < window.valid_output_samples48k:
                raise ValueError("Cached target is shorter than the immutable scored interval")
            crops.append(replace(crop, valid_scored_samples=window.valid_output_samples48k))
        return crops

    def checkpoint():
        corpus.flush()
        payload = {"format_version": "recipe_v2_pilot", "identity": run_identity, "engine": engine.state_dict(),
            "sampler": sampler.state_dict(), "rng": _rng_state(), "latest_evaluation": latest_evaluation,
            "latest_metrics": latest_metrics, "initial_model_state_sha256": initial_fingerprint,
            "seconds": seconds, "calibration_seconds": calibration_seconds, "teacher_coverage": coverage,
            "journal_sha256": journal_hash, "metrics_sha256": digest(_read_lines(metrics_file))}
        if shutil.disk_usage(directory).free < min_free_bytes + _checkpoint_reserve(payload):
            raise OSError("Checkpoint disk reserve would be violated")
        _atomic_save(payload, latest)

    def calibrate():
        nonlocal calibration_seconds
        started, rng = time.monotonic(), _rng_state()
        _atomic_json(inflight, {"phase": "normalization_calibration", "step": engine.step,
            "cursor": sampler.cursor, "identity_sha256": digest(calibration_identity)})
        def batches():
            for start in range(0, len(calibration_windows), batch_size):
                yield calibration_batch(crops_for(calibration_windows[start:start + batch_size]), device)
        try:
            calibration = engine.calibrate(batches, provenance=provenance)
        finally:
            _restore_rng(rng)
        calibration_seconds += time.monotonic() - started
        _atomic_json(directory / "calibration.json", {"calibration": calibration,
            "seconds": calibration_seconds, "unique_windows": len(calibration_windows), "passes": 2,
            "unique_scored_samples": sum(w.valid_output_samples48k for w in calibration_windows),
            "optimizer_updates": 0})
        checkpoint()
        inflight.unlink()

    def evaluate(writer):
        nonlocal latest_evaluation
        if _crop_identity(heldout_crops) != data_identity["heldout"]["crops"]:
            raise ValueError("Heldout raw targets changed during training")
        rng = _rng_state()
        try:
            values = evaluate_crops(engine, heldout_crops, include_signal_checks=True)
            latest_evaluation = {"step": engine.step, "rows": values,
                "groups": summarize_evaluation(values, metadata), "gate": waveform_gate(engine, values),
                "policy": "Original comparable reconstruction metrics; final quality goal monitored, never an entry gate"}
        finally:
            _restore_rng(rng)
        _atomic_json(directory / f"evaluation-step{engine.step:06d}.json", latest_evaluation)
        if writer:
            for group, metrics in latest_evaluation["groups"].items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"validation/{group}/{key}", value, engine.step)
            writer.add_scalar("validation/all/target_cosine", .99, engine.step)
            writer.flush()

    started_step = engine.step
    stop_step = min(recipe.total_steps, engine.step + max_updates) if max_updates else recipe.total_steps
    try:
        with _writer(log_dir, run_name, run_identity, engine.step, resume_from is not None) as writer:
            if resume_from is None:
                _atomic_json(directory / "run.json", run_identity)
                checkpoint()
            if engine.calibration_due:
                calibrate()
            if engine.step in evaluation_steps and (latest_evaluation is None or latest_evaluation["step"] != engine.step):
                evaluate(writer)
                checkpoint()
            while engine.step < stop_step:
                started = time.monotonic()
                chosen = windows[sampler.cursor:sampler.cursor + batch_size]
                crops = crops_for(chosen)
                targets_sha256 = digest(_crop_identity(crops))
                _atomic_json(inflight, {"phase": "optimizer", "step": engine.step + 1,
                    "start_cursor": sampler.cursor, "windows": [w.to_dict() for w in chosen],
                    "targets_sha256": targets_sha256})
                latest_metrics = engine.train_step(crops)
                if targets_sha256 != digest(_crop_identity(crops)):
                    raise RuntimeError("Optimizer mutated frozen teacher targets or raw latents")
                sampler.take_batch(len(chosen))
                coverage = _merge_coverage(coverage, teacher_coverage(crops))
                seconds += time.monotonic() - started
                latest_metrics.update(seconds_in_training_loop=seconds, consumed_windows=sampler.cursor,
                    unique_scored_hours=coverage["valid_samples"] / 172_800_000)
                body = {"step": engine.step, "cursor": sampler.cursor,
                    "window_identity": sampler.identity_sha256,
                    "batch_windows_sha256": digest([w.to_dict() for w in chosen]),
                    "targets_sha256": targets_sha256, "previous_sha256": journal_hash}
                journal_hash = digest(body)
                _append(journal, {**body, "sha256": journal_hash})
                _append(metrics_file, latest_metrics)
                inflight.unlink()
                _atomic_json(directory / "teacher-coverage.json", {"step": engine.step, **coverage,
                    "policy": "Optimizer scored intervals only; calibration/context/padding excluded"})
                if writer:
                    for key, value in latest_metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar("train/" + key, value, engine.step)
                    for key, value in coverage.items():
                        writer.add_scalar("coverage/teacher/" + key, value, engine.step)
                if engine.calibration_due:
                    calibrate()
                if engine.step in evaluation_steps:
                    evaluate(writer)
                if engine.step % checkpoint_interval == 0 or engine.step in evaluation_steps or engine.step == stop_step:
                    checkpoint()
                _atomic_json(directory / "status.json", {"state": "training", "step": engine.step,
                    "total_steps": recipe.total_steps, "consumed_windows": sampler.cursor,
                    "latest_metrics": latest_metrics, "perceptual_training_started": engine.discriminator_updates > 0})
            result = {"state": "budget_completed_awaiting_review" if engine.step == recipe.total_steps else "paused_at_explicit_launch_bound",
                "step": engine.step, "new_updates": engine.step - started_step, "total_steps": recipe.total_steps,
                "consumed_windows": sampler.cursor, "remaining_windows": sampler.remaining_segments,
                "unique_scored_hours": coverage["valid_samples"] / 172_800_000, "seconds": seconds,
                "calibration_seconds": calibration_seconds, "teacher_coverage": coverage,
                "heldout_gate_passed": latest_evaluation["gate"]["passed"] if latest_evaluation else None,
                "groups": latest_evaluation["groups"] if latest_evaluation else {},
                "latest_evaluation_step": latest_evaluation["step"] if latest_evaluation else None,
                "perceptual_training_started": engine.discriminator_updates > 0,
                "discriminator_updates": engine.discriminator_updates,
                "checkpoint": str(latest), "initialization": run_identity["initialization"]}
            _atomic_json(directory / "status.json", result)
            return result
    except BaseException as error:
        _atomic_json(directory / "status.json", {"state": "stopped_on_error", "step": engine.step,
            "consumed_windows": sampler.cursor, "error": f"{type(error).__name__}: {error}",
            "inflight_reserved": inflight.exists(), "automatic_resume": False})
        raise
