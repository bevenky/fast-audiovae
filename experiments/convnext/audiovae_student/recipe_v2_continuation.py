"""Continue a committed recipe-v2 student on a new finite, nonrepeating plan.

Only the total update budget may change. Model, objectives, calibration,
optimizers, loss balancer and random generators retain the parent's state.
The continuation has its own exposure journal and zero-based segment cursor;
engine and metric steps remain global. It never writes to the parent run.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from functools import wraps
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from .cache import sample_training_crop
from .comparison_data import assert_comparison_disjoint, verify_comparison_windows
from .data import ManifestRow
from .distillation_training import evaluate_crops, model_state_fingerprint, waveform_gate
from .model import StudentConfig, StudentDecoder
from .preflight_distillation import _crop_identity
from .recipe_v2 import RecipeV2Config, RecipeV2Engine
from .recipe_v2_pilot import (_exclusive_run, _panel_identity, _read_lines,
    implementation_identity as original_implementation_identity)
from .representative_pilot import (_append, _atomic_json, _checkpoint_reserve, _merge_coverage,
    _record, runtime_identity, summarize_evaluation, teacher_coverage)
from .restart_data import FixedWindowSampler, PilotWindow, canonical, digest, file_sha
from .teacher import CHECKPOINT_SHA256
from .training import _atomic_save, _restore_rng, _rng_state


FORMAT = "recipe_v2_continuation_v1"


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_json(path, value):
    _atomic_json(path, value)
    _sync_directory(path.parent)


def _durable_save(path, value, minimum):
    if shutil.disk_usage(path.parent).free < minimum + _checkpoint_reserve(value):
        raise OSError("Checkpoint disk reserve would be violated")
    _atomic_save(value, path)
    _sync_directory(path.parent)


def implementation_identity():
    return {**original_implementation_identity(), **{
        name: file_sha(Path(__file__).with_name(name + ".py"))
        for name in ("recipe_v2_continuation", "continuation_data")}}


def verify_loaded_plan(plan):
    """Check in-memory plan values against the sealed comparison-plan files."""
    identity = plan["identity"]
    if identity.get("identity_sha256") != digest({k: v for k, v in identity.items()
                                                 if k != "identity_sha256"}):
        raise ValueError("Loaded plan identity changed")
    values = {"train-manifest.jsonl": [r.to_dict() for r in plan["rows"]],
        "windows.jsonl": [w.to_dict() for w in plan["windows"]],
        "reserved.jsonl": [r.to_dict() for r in plan.get("reserved", ())],
        "excluded-sources.jsonl": [r.to_dict() for r in plan.get("excluded", ())],
        "input-sample-counts.json": plan["counts"], "ledger.json": plan["ledger"].document,
        "metadata.json": plan["metadata"]}
    if set(identity.get("files_sha256", {})) != set(values):
        raise ValueError("Loaded plan file inventory changed")
    for name, value in values.items():
        raw = b"".join(canonical(row) for row in value) if name.endswith(".jsonl") else canonical(value)
        if hashlib.sha256(raw).hexdigest() != identity["files_sha256"][name]:
            raise ValueError("Loaded plan data changed: " + name)


@contextmanager
def _parent_guard(directory):
    """Exclude simultaneous original-run resume without changing its files."""
    directory = Path(directory).resolve(strict=True)
    lock = directory.parent / ("." + directory.name + ".runner.lock")
    if lock.is_symlink() or not lock.is_file():
        raise ValueError("The original parent runner lock is missing or unsafe")
    with lock.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The original parent run is still active") from error
        yield directory


def load_committed_parent(checkpoint, expected_sha256, directory, original_plan, *, device):
    """Verify the complete original-run journal, checkpoint, targets and lineage."""
    directory, checkpoint = Path(directory), Path(checkpoint)
    if (checkpoint.is_symlink() or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64 or (directory / "inflight.json").exists()):
        raise ValueError("Parent checkpoint is invalid or parent in-flight exposure is uncertain")
    # Read one descriptor so even an unrelated replacement cannot mix snapshots.
    with checkpoint.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected_sha256:
            raise ValueError("Parent checkpoint SHA-256 changed")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if file_sha(directory / "latest.pt") != actual:
        raise ValueError("Parent snapshot is not its latest committed checkpoint")
    if payload.get("format_version") != "recipe_v2_pilot":
        raise ValueError("Continuation requires an original recipe-v2 parent")
    identity, state = payload["identity"], payload["engine"]
    if json.loads((directory / "run.json").read_text()) != identity:
        raise ValueError("Parent run manifest changed")
    if identity.get("implementation") != original_implementation_identity():
        raise ValueError("Parent training implementation changed")
    if identity.get("runtime") != runtime_identity(device):
        raise ValueError("Parent runtime differs; exact continuation is not established")
    verify_loaded_plan(original_plan)
    if identity.get("plan") != original_plan["identity"]:
        raise ValueError("Parent original plan changed")
    expected_actual = {"rows": [r.to_dict() for r in original_plan["rows"]],
        "counts": original_plan["counts"], "metadata": original_plan["metadata"],
        "reserved": [r.to_dict() for r in original_plan.get("reserved", ())],
        "excluded": [r.to_dict() for r in original_plan.get("excluded", ())]}
    if identity.get("actual_plan") != expected_actual:
        raise ValueError("Parent actual source plan changed")
    if (identity.get("recipe") != state.get("recipe")
            or identity.get("model_config") != state.get("model_config")):
        raise ValueError("Parent engine recipe or model differs from its run identity")
    step, batch_size = state["step"], identity["batch_size"]
    if type(step) is not int or type(batch_size) is not int or batch_size < 1 or step < 1:
        raise ValueError("Invalid parent global step or batch size")
    sampler = FixedWindowSampler(original_plan["windows"])
    sampler.load_state_dict(payload["sampler"])
    if (sampler.cursor != min(step * batch_size, len(original_plan["windows"]))
            or identity.get("window_identity") != sampler.identity_sha256
            or identity.get("ledger_identity") != original_plan["ledger"].identity_sha256):
        raise ValueError("Parent sampler, global step or historical ledger changed")
    records, metrics = _read_lines(directory / "exposure.jsonl"), _read_lines(directory / "metrics.jsonl")
    chain = digest([])
    for index, record in enumerate(records):
        selected = original_plan["windows"][index * batch_size:(index + 1) * batch_size]
        body = {k: v for k, v in record.items() if k != "sha256"}
        if (record.get("step") != index + 1
                or record.get("cursor") != min((index + 1) * batch_size, len(original_plan["windows"]))
                or record.get("window_identity") != sampler.identity_sha256
                or record.get("batch_windows_sha256") != digest([w.to_dict() for w in selected])
                or record.get("previous_sha256") != chain or record.get("sha256") != digest(body)):
            raise ValueError("Parent exposure journal differs from its immutable windows")
        chain = record["sha256"]
    if (len(records) != step or len(metrics) != step
            or any(row.get("step") != i + 1 for i, row in enumerate(metrics))
            or chain != payload["journal_sha256"] or digest(metrics) != payload["metrics_sha256"]
            or (metrics and metrics[-1] != payload["latest_metrics"])):
        raise ValueError("Parent has uncheckpointed exposure or changed metrics")
    if payload["teacher_coverage"]["valid_samples"] != sum(
            w.valid_output_samples48k for w in original_plan["windows"][:sampler.cursor]):
        raise ValueError("Parent teacher coverage does not match committed exposure")
    calibration = identity["calibration"]
    provenance = {**calibration["provenance"], "calibration_identity_sha256": digest(calibration),
        "optimizer_window_identity": sampler.identity_sha256,
        "source_corpus": identity["data"]["source_corpus"], "teacher_checkpoint_sha256": CHECKPOINT_SHA256}
    if (state.get("calibration") is None or state["calibration"]["report"]["provenance"] != provenance
            or step < state["recipe"]["reconstruction_warmup_steps"]):
        raise ValueError("Parent requires completed fixed-weight calibration")
    if json.loads((directory / "calibration.json").read_text())["calibration"] != state["calibration"]:
        raise ValueError("Parent saved calibration report changed")
    evaluation = payload.get("latest_evaluation")
    if evaluation is not None:
        path = directory / f"evaluation-step{evaluation['step']:06d}.json"
        if (evaluation["step"] > step or evaluation["step"] not in identity["evaluation_steps"]
                or not path.is_file() or json.loads(path.read_text()) != evaluation):
            raise ValueError("Parent committed evaluation changed")
    parent = {"checkpoint_sha256": actual, "run_identity_sha256": digest(identity),
        "journal_sha256": chain, "metrics_sha256": digest(metrics), "step": step, "batch_size": batch_size}
    return payload, parent


def extend_total_budget(state, total_steps):
    """Copy state and change only two budget fields under a constant LR schedule."""
    old = state["recipe"]["total_steps"]
    if (type(total_steps) is not int or total_steps < old or total_steps <= state["step"]
            or state["config"]["total_steps"] != old
            or state["config"].get("learning_rate_schedule") != "constant_after_warmup"
            or state["step"] < state["config"]["warmup_steps"]
            or state.get("calibration") is None):
        raise ValueError("Only a postcalibration constant-LR total-budget extension is supported")
    result = deepcopy(state)
    result["recipe"]["total_steps"] = total_steps
    result["config"]["total_steps"] = total_steps
    return result


@contextmanager
def observe_training_signal(model, crops):
    """Observe the existing single forward, excluding real history and padding."""
    metrics, calls = {}, 0

    @torch.no_grad()
    def capture(_module, _args, output):
        nonlocal calls
        calls += 1
        if calls != 1 or not isinstance(output, torch.Tensor) or output.shape[:2] != (len(crops), 1):
            raise RuntimeError("Training signal observer expected one waveform forward")
        student_peaks, student_counts, teacher_peaks, teacher_counts = [], [], [], []
        count = 0
        for i, crop in enumerate(crops):
            student = output[i:i + 1, :, crop.scored_slice].detach()
            teacher = crop.teacher_audio[..., crop.scored_slice].detach()
            if student.numel() != crop.valid_scored_samples or teacher.numel() != student.numel():
                raise RuntimeError("Training signal observer found an invalid scored slice")
            student_peaks.append(student.abs().amax())
            student_counts.append((student.abs() >= 1).sum())
            teacher_peaks.append(teacher.abs().amax())
            teacher_counts.append((teacher.abs() >= 1).sum())
            count += student.numel()
        metrics.update({"training_signal/student_peak_abs": float(torch.stack(student_peaks).amax()),
            "training_signal/student_full_scale_samples": int(torch.stack(student_counts).sum()),
            "training_signal/teacher_peak_abs": float(torch.stack(teacher_peaks).amax()),
            "training_signal/teacher_full_scale_samples": int(torch.stack(teacher_counts).sum()),
            "training_signal/scored_samples": count})
        # Returning None leaves the exact model output and autograd graph intact.

    handle = model.register_forward_hook(capture)
    try:
        yield metrics
        if calls != 1:
            raise RuntimeError("Training signal observer did not see the student forward")
    finally:
        handle.remove()


def _protect_parent_directory(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        # Reject overlap before the shared runner helper can create its lock.
        directory = Path(kwargs.get("output_dir", args[4] if len(args) > 4 else ""))
        parent = Path(kwargs["parent_run_dir"]).resolve(strict=True)
        resolved = directory.resolve()
        if resolved == parent or parent in resolved.parents or resolved in parent.parents:
            raise ValueError("Continuation output must be outside the original parent run")
        return function(*args, **kwargs)
    return guarded


@_protect_parent_directory
@_exclusive_run
def run_recipe_v2_continuation(corpus, plan, heldout_crops, heldout_rows, output_dir, *,
        parent_checkpoint, parent_checkpoint_sha256, parent_run_dir, parent_plan,
        expected_plan_identity_sha256,
        data_identity, supplement=None, device="cuda", resume_from=None, max_updates=None,
        checkpoint_interval=100, min_free_bytes=4 * 1024**3, discriminators=None):
    """Train only the declared continuation windows; pause between complete updates.

    ``expected_plan_identity_sha256`` must come from the independently reviewed
    handoff receipt, so rebuilding a plan cannot silently remove reservations.
    ``pause.request.json`` in output_dir must contain ``{"action":"pause"}``.
    A request is acknowledged in status without deleting it. Remove it explicitly
    before resuming. Exact resume rejects every uncertain in-flight interval and
    any journal work newer than the saved checkpoint. Only final-budget evaluation
    runs here; external observers can evaluate separate retained snapshots.
    """
    directory = Path(output_dir)
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError("min_free_bytes must be nonnegative")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive invocation bound")
    parent_directory = Path(parent_run_dir).resolve(strict=True)
    resolved = directory.resolve()
    if (resolved == parent_directory or parent_directory in resolved.parents
            or resolved in parent_directory.parents):
        raise ValueError("Continuation output must be outside the original parent run")
    with _parent_guard(parent_directory):
        payload, parent = load_committed_parent(parent_checkpoint, parent_checkpoint_sha256,
            parent_directory, parent_plan, device=device)
        return _run(corpus, plan, tuple(heldout_crops), tuple(heldout_rows), directory,
            payload=payload, parent=parent, parent_plan=parent_plan, data_identity=data_identity,
            expected_plan_identity_sha256=expected_plan_identity_sha256,
            supplement=supplement, device=device, resume_from=resume_from, max_updates=max_updates,
            checkpoint_interval=checkpoint_interval, min_free_bytes=min_free_bytes,
            discriminators=discriminators)


def _run(corpus, plan, heldout_crops, heldout_rows, directory, *, payload, parent, parent_plan,
         data_identity, expected_plan_identity_sha256, supplement, device, resume_from, max_updates, checkpoint_interval,
         min_free_bytes, discriminators):
    from .continuation_data import verify_continuation_plan

    if supplement is None:
        raise ValueError("The pinned supplemental catalog is required for continuation verification")
    verify_loaded_plan(plan)
    if plan["identity"]["identity_sha256"] != expected_plan_identity_sha256:
        raise ValueError("Continuation plan differs from the reviewed handoff identity")
    old_identity = payload["identity"]
    calibration = old_identity["calibration"]
    verify_continuation_plan(plan, parent_plan, payload["sampler"], parent=parent,
        calibration_windows=[PilotWindow(**{k: v[k] for k in PilotWindow.__dataclass_fields__})
                             for v in calibration["windows"]],
        calibration_rows=[ManifestRow.from_dict(v) for v in calibration["rows"]],
        calibration_counts=calibration["counts"], reserved_rows=heldout_rows, supplement=supplement)
    batch_size, context_frames = old_identity["batch_size"], old_identity["context_frames"]
    windows, rows = tuple(plan["windows"]), tuple(plan["rows"])
    counts, by_id = dict(plan["counts"]), {r.source_id: r for r in rows}
    sampler = FixedWindowSampler(windows)
    if sampler.identity_sha256 != plan["identity"].get("fixed_sampler_identity"):
        raise ValueError("Continuation fixed window identity changed")
    total_steps = plan["metadata"]["expected_total_steps"]
    if total_steps != parent["step"] + math.ceil(len(windows) / batch_size):
        raise ValueError("Continuation global budget does not consume its finite plan")
    migrated = extend_total_budget(payload["engine"], total_steps)
    recipe = RecipeV2Config(**migrated["recipe"])
    if not windows or any(w.valid_output_samples48k < recipe.adversarial_samples for w in windows):
        raise ValueError("Every continuation window must support the existing adversarial objective")
    verify_comparison_windows(windows, by_id, counts, plan["ledger"],
        reserved_rows=(*plan.get("reserved", ()), *heldout_rows), excluded_sources=plan.get("excluded", ()))
    assert_comparison_disjoint(rows, heldout_rows)
    old_data = old_identity["data"]
    if (data_identity.get("source_corpus") != corpus.identity
            or not data_identity.get("teacher_batch_qualification")
            or data_identity.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or any(data_identity.get(k) != old_data.get(k) for k in
                   ("teacher_checkpoint_sha256", "teacher_state_sha256", "heldout"))
            or data_identity["source_corpus"].get("teacher") != old_data["source_corpus"].get("teacher")):
        raise ValueError("Continuation changed the frozen teacher or original development targets")
    _panel_identity(heldout_crops, heldout_rows, data_identity["heldout"], context_frames)
    metadata = old_identity["heldout_metadata"]
    identity = json.loads(canonical({"kind": FORMAT, "format_version": 1, "parent": parent,
        "recipe": asdict(recipe), "model_config": migrated["model_config"], "data": data_identity,
        "plan": plan["identity"], "window_identity": sampler.identity_sha256,
        "ledger_identity": plan["ledger"].identity_sha256, "batch_size": batch_size,
        "context_frames": context_frames, "global_start_step": parent["step"],
        "global_total_steps": total_steps, "segment_window_count": len(windows),
        "evaluation_steps": [total_steps],
        "heldout_metadata": metadata, "checkpoint_interval": checkpoint_interval,
        "min_free_bytes": min_free_bytes, "implementation": implementation_identity(),
        "runtime": runtime_identity(device), "calibration_sha256": digest(payload["engine"]["calibration"]),
        "migration": {"changed_fields": ["engine.recipe.total_steps", "engine.config.total_steps"],
            "old_total_steps": payload["engine"]["recipe"]["total_steps"], "new_total_steps": total_steps,
            "learning_rate_schedule": "constant_after_warmup"}}))
    model = StudentDecoder(StudentConfig(**migrated["model_config"])).to(device)
    engine = RecipeV2Engine(model, recipe=recipe, discriminators=discriminators)
    engine.load_state_dict(migrated)
    del migrated
    if engine.calibration_due:
        raise ValueError("A continuation cannot initiate normalization calibration")
    initial_model = model_state_fingerprint(model)
    if resume_from is None:
        directory.mkdir(parents=True, exist_ok=False)
        _sync_directory(directory.parent)
    elif not directory.is_dir():
        raise ValueError("Continuation resume directory is missing")
    if shutil.disk_usage(directory).free < min_free_bytes:
        raise OSError("Insufficient continuation disk reserve")
    latest, journal, inflight, metrics_file = (directory / name for name in
        ("latest.pt", "exposure.jsonl", "inflight.json", "metrics.jsonl"))
    coverage, seconds, chain = teacher_coverage(()), 0., digest([])
    latest_metrics, latest_evaluation = None, None
    resume_rng = payload["rng"]
    if resume_from is not None:
        if Path(resume_from).resolve() != latest.resolve():
            raise ValueError("Exact continuation resume requires its own latest checkpoint")
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if (saved.get("format_version") != FORMAT or saved.get("identity") != identity
                or inflight.exists() or json.loads((directory / "run.json").read_text()) != identity):
            raise ValueError("Continuation identity changed or in-flight exposure is uncertain")
        engine.load_state_dict(saved["engine"])
        sampler.load_state_dict(saved["sampler"])
        records, metrics_rows = _read_lines(journal), _read_lines(metrics_file)
        for index, record in enumerate(records):
            chosen = windows[index * batch_size:(index + 1) * batch_size]
            body = {k: v for k, v in record.items() if k != "sha256"}
            if (record.get("step") != parent["step"] + index + 1
                    or record.get("cursor") != min((index + 1) * batch_size, len(windows))
                    or record.get("window_identity") != sampler.identity_sha256
                    or record.get("batch_windows_sha256") != digest([w.to_dict() for w in chosen])
                    or record.get("previous_sha256") != chain or record.get("sha256") != digest(body)):
                raise ValueError("Continuation journal changed or repeats an interval")
            chain = record["sha256"]
        updates = engine.step - parent["step"]
        if (len(records) != updates or len(metrics_rows) != updates
                or any(r.get("step") != parent["step"] + i + 1 for i, r in enumerate(metrics_rows))
                or sampler.cursor != min(updates * batch_size, len(windows))
                or chain != saved["journal_sha256"] or digest(metrics_rows) != saved["metrics_sha256"]
                or saved["initial_model_state_sha256"] != initial_model
                or (metrics_rows and metrics_rows[-1] != saved["latest_metrics"])):
            raise ValueError("Uncheckpointed continuation work cannot be replayed")
        coverage = saved["teacher_coverage"]
        if (saved.get("parent_teacher_coverage") != payload["teacher_coverage"]
                or digest(coverage) != (records[-1].get("teacher_coverage_sha256") if records
                                        else digest(teacher_coverage(())))):
            raise ValueError("Continuation teacher coverage journal or parent binding changed")
        if coverage["valid_samples"] != sum(w.valid_output_samples48k for w in windows[:sampler.cursor]):
            raise ValueError("Continuation coverage differs from committed intervals")
        latest_evaluation, latest_metrics = saved["latest_evaluation"], saved["latest_metrics"]
        if (latest_evaluation is not None) != (engine.step == total_steps):
            raise ValueError("Continuation final evaluation is missing or out of schedule")
        if latest_evaluation is not None:
            path = directory / f"evaluation-step{latest_evaluation['step']:06d}.json"
            if (latest_evaluation["step"] != total_steps or engine.step != total_steps
                    or not path.is_file() or json.loads(path.read_text()) != latest_evaluation):
                raise ValueError("Continuation final evaluation changed")
        seconds, resume_rng = saved["seconds"], saved["rng"]
    _restore_rng(resume_rng)

    def checkpoint():
        corpus.flush()
        saved = {"format_version": FORMAT, "identity": identity, "engine": engine.state_dict(),
            "sampler": sampler.state_dict(), "rng": _rng_state(), "initial_model_state_sha256": initial_model,
            "latest_metrics": latest_metrics, "latest_evaluation": latest_evaluation,
            "seconds": seconds, "teacher_coverage": coverage, "parent_teacher_coverage": payload["teacher_coverage"],
            "journal_sha256": chain, "metrics_sha256": digest(_read_lines(metrics_file))}
        _durable_save(latest, saved, min_free_bytes)

    def result(state, started_step):
        return {"state": state, "step": engine.step, "total_steps": total_steps,
            "parent_step": parent["step"], "new_updates": engine.step - started_step,
            "segment_updates": engine.step - parent["step"], "consumed_windows": sampler.cursor,
            "remaining_windows": sampler.remaining_segments, "teacher_coverage": coverage,
            "parent_teacher_coverage": payload["teacher_coverage"],
            "unique_scored_hours": (payload["teacher_coverage"]["valid_samples"] + coverage["valid_samples"]) / 172_800_000,
            "segment_unique_scored_hours": coverage["valid_samples"] / 172_800_000,
            "seconds": seconds, "latest_metrics": latest_metrics,
            "latest_evaluation_step": latest_evaluation["step"] if latest_evaluation else None,
            "heldout_gate_passed": latest_evaluation["gate"]["passed"] if latest_evaluation else None,
            "groups": latest_evaluation["groups"] if latest_evaluation else {},
            "perceptual_training_started": engine.discriminator_updates > 0,
            "discriminator_updates": engine.discriminator_updates, "checkpoint": str(latest),
            "calibration_preserved": True, "parent_run_modified": False}

    started_step = engine.step
    stop_step = min(total_steps, engine.step + max_updates) if max_updates else total_steps
    try:
        if resume_from is None:
            _durable_json(directory / "run.json", identity)
            checkpoint()
        _durable_json(directory / "status.json", result("training", started_step))
        paused = False
        while engine.step < stop_step:
            pause = directory / "pause.request.json"
            if pause.exists():
                if pause.is_symlink() or json.loads(pause.read_text()) != {"action": "pause"}:
                    raise ValueError("Invalid pause request; expected exactly action=pause")
                checkpoint()
                paused = True
                break
            started = time.monotonic()
            chosen = windows[sampler.cursor:sampler.cursor + batch_size]
            corpus.prefetch(list(dict.fromkeys(w.source_id for w in chosen)), max_batch_size=8,
                            max_total_input_samples=1_920_000)
            crops = []
            for window in chosen:
                record = _record(corpus, by_id[window.source_id], counts[window.source_id])
                crop = sample_training_crop(record, window.start_frame, window.scored_frames, context_frames)
                if crop.valid_scored_samples < window.valid_output_samples48k:
                    raise ValueError("Cached target is shorter than its declared interval")
                crops.append(replace(crop, valid_scored_samples=window.valid_output_samples48k))
            targets = digest(_crop_identity(crops))
            _durable_json(inflight, {"phase": "optimizer", "step": engine.step + 1,
                "start_cursor": sampler.cursor, "windows": [w.to_dict() for w in chosen], "targets_sha256": targets})
            if engine.step == parent["step"] or (engine.step + 1) % 100 == 0:
                with observe_training_signal(engine.model, crops) as signal_metrics:
                    latest_metrics = engine.train_step(crops)
                latest_metrics.update(signal_metrics)
            else:
                latest_metrics = engine.train_step(crops)
            if targets != digest(_crop_identity(crops)):
                raise RuntimeError("Optimizer mutated frozen teacher targets or latents")
            if digest(engine.calibration) != identity["calibration_sha256"] or engine.calibration_due:
                raise RuntimeError("Continuation changed calibration provenance")
            sampler.take_batch(len(chosen))
            coverage = _merge_coverage(coverage, teacher_coverage(crops))
            seconds += time.monotonic() - started
            latest_metrics.update(seconds_in_training_loop=seconds, consumed_windows=sampler.cursor,
                segment_step=engine.step - parent["step"],
                unique_scored_hours=(payload["teacher_coverage"]["valid_samples"] + coverage["valid_samples"]) / 172_800_000)
            body = {"step": engine.step, "cursor": sampler.cursor, "window_identity": sampler.identity_sha256,
                "batch_windows_sha256": digest([w.to_dict() for w in chosen]),
                "targets_sha256": targets, "teacher_coverage_sha256": digest(coverage), "previous_sha256": chain}
            chain = digest(body)
            _append(journal, {**body, "sha256": chain})
            _append(metrics_file, latest_metrics)
            inflight.unlink()
            _sync_directory(directory)
            _durable_json(directory / "teacher-coverage.json", {"step": engine.step, **coverage,
                "parent_valid_samples": payload["teacher_coverage"]["valid_samples"],
                "policy": "Continuation scored intervals only; parent/calibration/context/padding excluded"})
            if engine.step == total_steps:
                if _crop_identity(heldout_crops) != data_identity["heldout"]["crops"]:
                    raise ValueError("Original heldout targets changed during continuation")
                rng = _rng_state()
                try:
                    values = evaluate_crops(engine, heldout_crops, include_signal_checks=True)
                    latest_evaluation = {"step": engine.step, "rows": values,
                        "groups": summarize_evaluation(values, metadata), "gate": waveform_gate(engine, values),
                        "policy": "Unchanged original validation targets; final quality goal, never an entry gate"}
                finally:
                    _restore_rng(rng)
                _durable_json(directory / f"evaluation-step{engine.step:06d}.json", latest_evaluation)
            if engine.step % checkpoint_interval == 0 or engine.step == stop_step:
                checkpoint()
            _durable_json(directory / "status.json", result("training", started_step))
        state = ("paused_by_request" if paused else "budget_completed_awaiting_review"
                 if engine.step == total_steps else "paused_at_explicit_launch_bound")
        final = result(state, started_step)
        _durable_json(directory / "status.json", final)
        return final
    except BaseException as error:
        _durable_json(directory / "status.json", {"state": "stopped_on_error", "step": engine.step,
            "consumed_windows": sampler.cursor, "error": f"{type(error).__name__}: {error}",
            "inflight_reserved": inflight.exists(), "automatic_resume": False})
        raise
