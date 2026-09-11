"""Two bounded generator-rate continuations, released only by cache resolution.

The only experimental policy change is the joint generator learning rate.
Starting optimizer moments and EMA are preserved; they evolve normally in each
arm. Existing training windows are deliberately reused for debugging, without
repeating or overlapping any scored source interval within an arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import fcntl
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.fusion_evaluation import evaluate_fusion, diagnose_streaming
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.restart_data import digest, file_sha
from audiovae_student.training import _rng_state, _restore_rng
from run_corrected_screen import bind_views
from run_fusion_screen import cpu_tree
from evaluation_audit import build_evaluation, compare_reports, GATES

ARMS = (("current_rate", 1.), ("quarter_rate", .25))
UPDATES, BATCH = 400, 32
PARENT_STEP = 8490
HEALTH_UPDATES = (0, 25, 100, 400)
CHECKPOINT_EVERY = 100


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def status(out, stage, **fields):
    row = {"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    atomic_json(out / "status.json", row)
    print(json.dumps(row), flush=True)


def validate_gate(gate, *, parent_sha256, target_cache_sha256, data_plan_sha256):
    if gate.get("format_version") != 1 or gate.get("training_ready") is not True:
        raise ValueError("Cache reproduction has not been released for this experiment")
    expected = {"parent_checkpoint_sha256": parent_sha256,
                "target_cache_sha256": target_cache_sha256,
                "data_plan_sha256": data_plan_sha256}
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError("Cache resolution gate identifies different inputs: " + key)
    if not isinstance(gate.get("resolution"), str) or not gate["resolution"].strip():
        raise ValueError("The cache resolution must be documented")
    if not isinstance(gate.get("evidence"), list) or not gate["evidence"]:
        raise ValueError("The cache resolution needs evidence references")


def canonical_required(gate):
    declared = gate.get("requires_canonical_evaluation", False)
    if type(declared) is not bool:
        raise ValueError("requires_canonical_evaluation must be an explicit boolean")
    bindings = ("canonical_receipt_sha256", "canonical_target_contract_sha256", "canonical_target_cache_sha256")
    for name in bindings:
        if name in gate and (not isinstance(gate[name], str) or not gate[name]):
            raise ValueError("Empty or invalid canonical binding: " + name)
    return declared or any(name in gate for name in bindings)


def load_fresh_training(ctx, gate, receipt_path):
    required = gate.get("requires_fresh_training_pairs", False)
    if type(required) is not bool:
        raise ValueError("requires_fresh_training_pairs must be an explicit boolean")
    bindings = ("training_receipt_sha256", "training_target_contract_sha256", "training_target_cache_sha256")
    for name in bindings:
        if name in gate and (not isinstance(gate[name], str) or not gate[name]):
            raise ValueError("Invalid fresh-training binding: " + name)
    required = required or any(name in gate for name in bindings)
    if receipt_path is None:
        if required:
            raise ValueError("The resolution requires --training-receipt with fresh corrected pairs")
        return None
    from training_overlay import load_training_overlay
    overlay = load_training_overlay(ctx, receipt_path, required_counts={"targeted_generator": UPDATES * BATCH})
    identity = overlay["identity"]
    for name, value in (("training_receipt_sha256", identity["receipt_sha256"]),
                        ("training_target_contract_sha256", identity["contract_sha256"]),
                        ("training_target_cache_sha256", identity["cache_sha256"])):
        if name in gate and gate[name] != value:
            raise ValueError("Resolution binds a different fresh-training overlay: " + name)
    return overlay


def _geometry(crops):
    return sorted((c.source_id, c.start_frame, c.context_start_frame, c.context_frames,
                   c.scored_frames, c.valid_scored_samples, tuple(c.latents.shape), tuple(c.teacher_audio.shape))
                  for c in crops)


def load_canonical_panel(ctx, gate, receipt_path):
    required = canonical_required(gate)
    if receipt_path is None:
        if required:
            raise ValueError("The corrected-domain resolution requires --canonical-receipt before any updates")
        return None
    from canonical_evaluation import load_canonical_cache
    receipt_path = Path(receipt_path).resolve(strict=True)
    crops, metadata, receipt = load_canonical_cache(receipt_path)
    if _geometry(crops) != _geometry(ctx.heldout):
        raise ValueError("Canonical and historical panels must retain identical source/crop geometry")
    if digest(metadata) != digest(ctx.metadata):
        raise ValueError("Canonical source metadata differs from the sealed historical panel")
    identity = {"receipt_path": str(receipt_path), "receipt_sha256": file_sha(receipt_path),
        "cache_path": str(Path(receipt["path"]).resolve()), "cache_sha256": receipt["sha256"],
        "contract_sha256": receipt["contract"]["identity_sha256"],
        "heldout_identity_sha256": digest(_crop_identity(crops)), "heldout_crops": len(crops),
        "metadata_sha256": digest(metadata), "required_by_resolution": required,
        "evaluation_policy": "Separate corrected encoder domain; original historical tensors, counts and masks remain unchanged"}
    for name, value in (("canonical_receipt_sha256", identity["receipt_sha256"]),
                        ("canonical_target_contract_sha256", identity["contract_sha256"]),
                        ("canonical_target_cache_sha256", identity["cache_sha256"])):
        if name in gate and gate[name] != value:
            raise ValueError("Resolution binds a different canonical panel: " + name)
    return {"crops": crops, "metadata": metadata, "receipt": receipt, "identity": identity}


def evaluate_domains(engine, ctx, canonical):
    reports = {"historical": build_evaluation(engine, ctx.heldout, ctx.metadata)}
    if canonical is not None:
        from canonical_evaluation import build_canonical_evaluation
        reports["canonical"] = build_canonical_evaluation(engine, canonical["crops"],
                                                        canonical["metadata"], canonical["receipt"])
    return reports


def combined_domain_result(historical, canonical, *, required):
    if required and canonical is None:
        raise ValueError("A required canonical comparison is missing")
    domains = {"historical": historical}
    if canonical is not None:
        domains["canonical"] = canonical
    if any(type(report.get("screen_passed")) is not bool for report in domains.values()):
        raise ValueError("Every evaluated domain must have a defined boolean screen result")
    return {"format_version": 2, "domains": domains,
        "screen_passed": all(report["screen_passed"] for report in domains.values()),
        "canonical_required_by_resolution": required,
        "decision_policy": "Every included evaluation domain must pass its unchanged pilot gates; no automatic promotion",
        "promoted": False}


def validate_pool(crops, entries, heldout, rows=None, *, expected_count=UPDATES * BATCH):
    if len(crops) != expected_count or len(entries) != expected_count:
        raise ValueError("The declared update budget and training pool disagree")
    heldout_sources = {crop.source_id for crop in heldout}
    intervals = defaultdict(list)
    scored_samples = 0
    for crop, entry in zip(crops, entries, strict=True):
        _validate_crop(crop)
        if crop.source_id in heldout_sources:
            raise ValueError("Training source overlaps the heldout panel")
        if crop.valid_scored_samples < 9120:
            raise ValueError("A training crop is shorter than the discriminator view")
        if crop.start_frame != crop.context_start_frame + crop.context_frames:
            raise ValueError("Source and context frame accounting disagree")
        window = entry["window"]
        if (window["source_id"], window["start_frame"], window["valid_output_samples48k"]) != (
                crop.source_id, crop.start_frame, crop.valid_scored_samples):
            raise ValueError("Planned data does not match the cached crop")
        start = entry.get("discriminator_start_sample48k")
        if start is not None and (type(start) is not int or not 0 <= start <= crop.valid_scored_samples - 9120):
            raise ValueError("A planned discriminator view exceeds valid scored audio")
        begin, end = crop.start_frame * 1920, crop.start_frame * 1920 + crop.valid_scored_samples
        intervals[("source", crop.source_id)].append((begin, end))
        audio_hash = (rows or {}).get(crop.source_id, {}).get("audio_sha256")
        if audio_hash:
            intervals[("audio_hash", audio_hash)].append((begin, end))
        scored_samples += crop.valid_scored_samples
    for key, spans in intervals.items():
        spans.sort()
        if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
            raise ValueError("Repeated or overlapping scored intervals within an arm: " + repr(key))
    return {"windows": len(crops), "unique_source_ids": len({c.source_id for c in crops}),
            "valid_scored_samples48k": scored_samples, "valid_scored_hours": scored_samples / 48000 / 3600,
            "scored_intervals_nonoverlapping": True, "heldout_source_disjoint": True,
            "context_overlap": "Permitted causal context, excluded from scored intervals",
            "earlier_exposure": "These windows were used by the starting targeted checkpoint; reuse is declared debugging, not a new-data claim."}


def fork_generator_rate(engine, factor):
    if factor not in (1., .25):
        raise ValueError("Only the preregistered current and quarter-rate forks are permitted")
    before = cpu_tree(engine.state_dict())
    rate = engine.recipe.learning_rate * factor
    engine.recipe = replace(engine.recipe, learning_rate=rate)
    engine.config = replace(engine.config, learning_rate=rate)
    for optimizer in engine.optimizer.optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = rate
    after = cpu_tree(engine.state_dict())
    normalized = deepcopy(after)
    normalized["recipe"]["learning_rate"] = before["recipe"]["learning_rate"]
    normalized["config"]["learning_rate"] = before["config"]["learning_rate"]
    for name, optimizer in normalized["optimizer"]["optimizers"].items():
        for index, group in enumerate(optimizer["param_groups"]):
            group["lr"] = before["optimizer"]["optimizers"][name]["param_groups"][index]["lr"]
    if state_fingerprint(normalized) != state_fingerprint(before):
        raise RuntimeError("Rate migration changed something beyond the declared generator rates")
    if engine.learning_rate() != rate:
        raise RuntimeError("The active schedule does not produce the declared fork rate")
    return {"factor": factor, "generator_learning_rate": rate,
        "source_engine_sha256": state_fingerprint(before), "fork_engine_sha256": state_fingerprint(after),
        "only_changed_fields": ["recipe.learning_rate", "config.learning_rate", "generator_optimizer.param_groups[*].lr"],
        "initial_moments_and_ema_preserved": True,
        "evolution_policy": "Generator/discriminator moments, discriminator weights and loss EMA evolve normally; their policies are unchanged."}


def save_checkpoint(path, value, *, expected_bytes):
    reserve = 512 * 1024**2
    if shutil.disk_usage(path.parent).free < expected_bytes * 1.2 + reserve:
        raise RuntimeError("Insufficient space for atomic checkpoint plus 512 MiB reserve")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(cpu_tree(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint(engine, identity, arm, migration, updates):
    return {"format_version": "joint_generator_rate_screen_v1", "engine": engine.state_dict(),
            "rng": _rng_state(), "arm": arm, "generator_updates": updates,
            "parent_sha256": identity["parent_checkpoint_sha256"], "parent_step": PARENT_STEP,
            "experiment_identity_sha256": digest(identity), "migration": migration}


def code_identity(*, canonical=False, fresh_training=False):
    import audiovae_student.recipe_v2 as recipe_module
    import run_corrected_screen as screen_module
    import diagnostic_common as common_module
    import evaluation_audit as evaluation_module
    files = {Path(__file__).resolve(), Path(screen_module.__file__).resolve(), Path(common_module.__file__).resolve(),
             Path(evaluation_module.__file__).resolve()}
    files.update(Path(recipe_module.__file__).parent.glob("*.py"))
    if canonical:
        import canonical_evaluation
        import render_canonical_targets
        files.update((Path(canonical_evaluation.__file__).resolve(), Path(render_canonical_targets.__file__).resolve()))
    if fresh_training:
        import training_overlay
        files.add(Path(training_overlay.__file__).resolve())
    return {str(path): file_sha(path) for path in sorted(files)}


def selected_health_crops(ctx):
    # Fixed panel specified from the existing probes before either arm updates.
    chosen = []
    keys = set()
    for condition in ("encoded_zero", "encoded_fade", "laughter", "whistling", "sindhi", "speech"):
        for crop in ctx.probe_crops:
            meta = ctx.metadata.get(crop.source_id, {})
            description = (crop.source_id + " " + str(meta.get("condition", "")) + " " + str(meta.get("language", ""))).lower()
            if condition in description and crop.source_id not in keys:
                chosen.append(crop); keys.add(crop.source_id); break
    if not any(c.source_id == "encoded_zero" for c in chosen):
        raise ValueError("Health panel is missing encoded zero")
    return chosen


def health(engine, crops, *, device):
    from diagnostic_common import model_metrics
    report = model_metrics(engine.model, crops, device)
    if report["aggregate"]["peak"] > 4:
        raise FloatingPointError("Catastrophic health-panel peak exceeds 4; no further updates")
    return report


def run(ctx, gate_path, out, *, canonical_receipt=None, training_receipt=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity.json").exists():
        raise RuntimeError("Refusing to replay a started experiment automatically")
    if ctx.checkpoints["targeted"]["engine"]["step"] != PARENT_STEP:
        raise ValueError("The prescribed targeted step-8490 checkpoint is required")
    gate_path = Path(gate_path)
    gate = json.loads(gate_path.read_text())
    parent_sha = ctx.expected_hashes["targeted"]
    data_sha = ctx.identity["data_plan_sha256"]
    validate_gate(gate, parent_sha256=parent_sha, target_cache_sha256=ctx.receipt["target_cache_sha256"], data_plan_sha256=data_sha)
    canonical = load_canonical_panel(ctx, gate, canonical_receipt)
    overlay = load_fresh_training(ctx, gate, training_receipt)
    if overlay is not None and canonical is None:
        raise ValueError("Fresh corrected training requires the separate canonical evaluation panel")
    historical_crops = ctx.pools["targeted_generator"]
    crops = overlay["pools"]["targeted_generator"] if overlay is not None else historical_crops
    entries = ctx.data["pools"]["targeted_generator"]
    coverage = validate_pool(crops, entries, ctx.heldout, ctx.data.get("rows"))
    status(out, "verifying_training_tensor_identity")
    historical_crop_identity = _crop_identity(historical_crops)
    if historical_crop_identity != ctx.identity["targets"]["target_crops"]["targeted_generator"]:
        raise ValueError("The original training tensor identity changed")
    crop_identity = _crop_identity(crops) if overlay is not None else historical_crop_identity
    selected = selected_health_crops(ctx)
    parent_bytes = ctx.paths["targeted"].stat().st_size
    if shutil.disk_usage(out).free < 4 * parent_bytes + 1024**3:
        raise RuntimeError("Output storage lacks reserve for both new arms and atomic checkpoint replacement")
    identity = {"format_version": 1, "parent_step": PARENT_STEP, "parent_checkpoint_sha256": parent_sha,
        "parent_path": str(ctx.paths["targeted"]), "target_cache_sha256": ctx.receipt["target_cache_sha256"],
        "data_plan_sha256": data_sha, "training_pool_identity_sha256": digest(crop_identity),
        "historical_training_pool_identity_sha256": digest(historical_crop_identity),
        "training_overlay": overlay["identity"] if overlay is not None else None,
        "training_target_domain": "fresh_corrected_encoder_and_decoder" if overlay is not None else "historical_paired_cache",
        "heldout_identity_sha256": digest(_crop_identity(ctx.heldout)), "heldout_crops": len(ctx.heldout),
        "cache_gate_path": str(gate_path), "cache_gate_sha256": file_sha(gate_path), "cache_resolution": gate,
        "canonical_panel": canonical["identity"] if canonical is not None else None,
        "evaluation_domains": ["historical", "canonical"] if canonical is not None else ["historical"],
        "domain_gate_policy": "Every included domain must independently pass the unchanged preregistered gates",
        "preregistered_gates": GATES, "preregistered_gates_sha256": digest(GATES),
        "updates_per_arm": UPDATES, "batch_size": BATCH, "arms": [{"name": n, "factor": f} for n, f in ARMS],
        "coverage": coverage, "health_updates": list(HEALTH_UPDATES),
        "health_crops": [{"source_id": c.source_id, "start_frame": c.start_frame} for c in selected],
        "storage": {"results_path": str(out.resolve()), "available_bytes_at_start": shutil.disk_usage(out).free,
                    "parent_checkpoint_bytes": parent_bytes, "parent_files_copied": False,
                    "retention": "New artifacts are on the declared mount; no originals are removed. Transfer durable results after comparison."},
        "changes": "Only the joint generator learning rate differs; no new layers, losses, data selection, warmup or normalization calibration.",
        "sources": code_identity(canonical=canonical is not None, fresh_training=overlay is not None)}
    atomic_json(out / "identity.json", identity)
    parent_rng = deepcopy(ctx.checkpoints["targeted"]["rng"])
    initial_comparison = None
    for name, factor in ARMS:
        directory = out / name
        directory.mkdir(exist_ok=False)
        _restore_rng(parent_rng)
        engine = ctx.engine("targeted", device="cuda")
        migration = fork_generator_rate(engine, factor)
        if initial_comparison is None:
            initial_comparison = migration["source_engine_sha256"]
        elif migration["source_engine_sha256"] != initial_comparison:
            raise RuntimeError("Experimental arms have different initial engines")
        bind_views(engine, entries)
        atomic_json(directory / "migration.json", migration)
        status(out, "full_evaluation_before", arm=name)
        before_domains = evaluate_domains(engine, ctx, canonical)
        before = before_domains["historical"]
        atomic_json(directory / "before.json", before)
        if canonical is not None:
            atomic_json(directory / "before-canonical.json", before_domains["canonical"])
        atomic_json(directory / "health-000.json", health(engine, selected, device="cuda"))
        _restore_rng(parent_rng)
        engine.crop_generator.set_state(ctx.checkpoints["targeted"]["engine"]["crop_rng"].cpu())
        started = time.monotonic()
        for i in range(UPDATES):
            batch = crops[i * BATCH:(i + 1) * BATCH]
            atomic_json(directory / "inflight.json", {"update": i + 1, "crops": [(c.source_id, c.start_frame) for c in batch]})
            metrics = engine.train_step(batch)
            if metrics["step"] != PARENT_STEP + i + 1:
                raise RuntimeError("Noncontiguous update clock")
            if metrics["learning_rate"] != migration["generator_learning_rate"]:
                raise RuntimeError("The declared generator learning rate was not used")
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            with (directory / "views.jsonl").open("a") as handle:
                handle.write(json.dumps({"update": i + 1, "views": engine.screen_last_views}, allow_nan=False) + "\n")
            (directory / "inflight.json").unlink()
            if (i + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(directory / "last.pt", checkpoint(engine, identity, name, migration, i + 1), expected_bytes=parent_bytes)
            if i + 1 in HEALTH_UPDATES:
                atomic_json(directory / f"health-{i + 1:03d}.json", health(engine, selected, device="cuda"))
            if (i + 1) % 25 == 0:
                status(out, "training", arm=name, updates=i + 1, seconds=time.monotonic() - started,
                       waveform=metrics["teacher_waveform"], mel=metrics["teacher_mel"])
        elapsed = time.monotonic() - started
        status(out, "full_evaluation_after", arm=name)
        after_domains = evaluate_domains(engine, ctx, canonical)
        after = after_domains["historical"]
        atomic_json(directory / "after.json", after)
        if canonical is not None:
            atomic_json(directory / "after-canonical.json", after_domains["canonical"])
        cpu = deepcopy(engine.model).cpu()
        stream = {}
        check_crops = [next(c for c in ctx.heldout if c.source_id == "encoded_zero"),
                       next(c for c in ctx.heldout if not c.source_id.startswith("encoded_") and c.latents.shape[-1] >= 40)]
        for index, crop in enumerate(check_crops):
            for frames in (2, 4):
                checked = diagnose_streaming(cpu, crop.latents.cpu(), frames_per_chunk=frames)
                checked["passed"] = checked["sample_count_passed"] and checked["max_abs_error"] <= 2e-6
                checked["tolerance"] = 2e-6
                if not checked["passed"]:
                    raise RuntimeError("CPU streaming parity failed")
                stream[f"{index}_{frames}"] = checked
        atomic_json(directory / "streaming.json", stream)
        del cpu
        os.replace(directory / "last.pt", directory / "final.pt")
        atomic_json(directory / "complete.json", {"arm": name, "step": engine.step, "updates": UPDATES,
            "training_seconds_including_small_health_and_saves": elapsed, "checkpoint_sha256": file_sha(directory / "final.pt"),
            "before_summary": before["summary"], "after_summary": after["summary"],
            "canonical_before_summary": before_domains["canonical"]["summary"] if canonical is not None else None,
            "canonical_after_summary": after_domains["canonical"]["summary"] if canonical is not None else None})
        del engine
        gc.collect(); torch.cuda.empty_cache()
    control_before = json.loads((out / "current_rate/before.json").read_text())
    candidate_before = json.loads((out / "quarter_rate/before.json").read_text())
    if digest(control_before) != digest(candidate_before):
        raise RuntimeError("The two arms did not begin with identical measured audio")
    if file_sha(out / "current_rate/views.jsonl") != file_sha(out / "quarter_rate/views.jsonl"):
        raise RuntimeError("Discriminator view schedules differ between arms")
    historical_comparison = compare_reports(control_before, json.loads((out / "current_rate/after.json").read_text()),
                                            json.loads((out / "quarter_rate/after.json").read_text()))
    atomic_json(out / "comparison-historical.json", historical_comparison)
    canonical_comparison = None
    if canonical is not None:
        from canonical_evaluation import compare_canonical_reports
        canonical_control_before = json.loads((out / "current_rate/before-canonical.json").read_text())
        canonical_candidate_before = json.loads((out / "quarter_rate/before-canonical.json").read_text())
        if digest(canonical_control_before) != digest(canonical_candidate_before):
            raise RuntimeError("The two arms did not begin with identical canonical audio")
        canonical_comparison = compare_canonical_reports(canonical_control_before,
            json.loads((out / "current_rate/after-canonical.json").read_text()),
            json.loads((out / "quarter_rate/after-canonical.json").read_text()))
        atomic_json(out / "comparison-canonical.json", canonical_comparison)
        if (file_sha(canonical["identity"]["receipt_path"]) != canonical["identity"]["receipt_sha256"]
                or file_sha(canonical["identity"]["cache_path"]) != canonical["identity"]["cache_sha256"]):
            raise RuntimeError("Canonical receipt/cache files changed during the experiment")
    comparison = combined_domain_result(historical_comparison, canonical_comparison, required=canonical_required(gate))
    if overlay is not None:
        from training_overlay import verify_overlay_files
        comparison["fresh_training_overlay_preservation"] = verify_overlay_files(overlay)
        comparison["training_target_domain"] = "fresh_corrected_encoder_and_decoder"
    atomic_json(out / "comparison.json", comparison)
    original_integrity = ctx.verify_files()
    status(out, "complete", original_integrity=original_integrity, original_run_still_paused=True,
           screen_passed=comparison["screen_passed"],
           promoted=False, next_step="Evaluate preregistered quality gates; neither arm is automatically selected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-resolution", required=True)
    parser.add_argument("--canonical-receipt", help="Separate corrected-domain receipt; mandatory when required by cache resolution")
    parser.add_argument("--training-receipt", help="Pinned fresh corrected training pairs; mandatory when required by resolution")
    parser.add_argument("--output-dir", default="/tmp/fast-audiovae-recovery-20260909/update-results")
    parser.add_argument("--context-module", default="diagnostic_common")
    parser.add_argument("--parent-run-lock", default="/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        with Path(args.parent_run_lock).open("rb") as parent_lock, (out / "runner.lock").open("a") as own_lock:
            fcntl.flock(parent_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(own_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            ctx = importlib.import_module(args.context_module).load_context()
            run(ctx, args.cache_resolution, out, canonical_receipt=args.canonical_receipt, training_receipt=args.training_receipt)
    except Exception as error:
        status(out, "failed", error=repr(error), automatic_replay_permitted=False)
        raise


if __name__ == "__main__":
    main()
