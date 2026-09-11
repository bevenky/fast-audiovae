"""Bounded optimizer-direction comparison. All model updates are discarded.

One shared D update and balanced gradient per batch generates retained and
fresh-generator-optimizer directions. SGD is the plain negative gradient.
Training-only calibration chooses a shared parameter-displacement radius;
the independent problem panel never influences that radius.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from audiovae_student.corrected_calibration import _fixed_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.recipe_v2 import calibration_batch
from audiovae_student.restart_data import digest, file_sha
from audiovae_student.training import _restore_rng
from diagnose_objectives_updates import _capture_replay_state, _restore_replay_state, _preserved_attributes, vector_dot
from gradient_probe import _quiet_mask
from native_chain import QUARTER_SHA, intercept_step, probe_snapshot, restore_quarter, select_probes
from run_corrected_screen import bind_views
from run_update_experiment import load_canonical_panel
from render_canonical_targets import digest as canonical_inventory_digest
from training_overlay import load_training_overlay, verify_overlay_files

METHODS = ("retained", "fresh_generator_state", "plain_sgd")
FRACTIONS = tuple(2. ** -i for i in range(8))
CALIBRATION_BATCHES, COMPARISON_BATCHES, BATCH_SIZE, MIN_QUIET_SOURCES = 3, 4, 32, 4
SELECTION_SEED = "optimizer-causal-comparison-v1"


def rank(value):
    return hashlib.sha256((SELECTION_SEED + "|" + value).encode()).hexdigest()


def _audio_sha(row):
    value = row.get("audio_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("A source is missing its pinned audio SHA256")
    return value


def heldout_hashes(panel, inventory_path):
    inventory = json.loads(Path(inventory_path).read_text())
    computed = canonical_inventory_digest({k: v for k, v in inventory.items() if k != "identity_sha256"})
    if (not inventory.get("ready") or computed != inventory.get("identity_sha256")
            or computed != panel["receipt"]["contract"]["source_inventory_identity_sha256"]):
        raise ValueError("Canonical source inventory is not sealed by the target contract")
    natural = {c.source_id for c in panel["crops"] if not c.source_id.startswith("encoded_")}
    rows = {entry["source_id"]: entry["manifest_row"] for entry in inventory["sources"]
            if entry["source_id"] in natural}
    if set(rows) != natural:
        raise ValueError("Canonical inventory is missing natural recording identities")
    return {_audio_sha(row) for row in rows.values()}, computed


def teacher_quiet_samples(crop):
    teacher = crop.teacher_audio.detach()
    valid = torch.zeros_like(teacher, dtype=torch.bool)
    valid[..., crop.scored_slice] = True  # Exact training mask, no validation-only six-sample exclusion.
    quiet = _quiet_mask(teacher, teacher, valid, QuietAudioConfig())
    return int(quiet.sum())


def require_quiet_contract(config):
    value = config if isinstance(config, dict) else asdict(config)
    if value != asdict(QuietAudioConfig()):
        raise ValueError("Training selection/calibration quiet policy differs from the saved recipe")


def select_batches(crops, rows, heldout_ids, excluded_hashes, *, calibration_batches=CALIBRATION_BATCHES,
                   comparison_batches=COMPARISON_BATCHES, batch_size=BATCH_SIZE, min_quiet=MIN_QUIET_SOURCES):
    """Immutable source/hash split; only teacher targets determine quiet strata."""
    if not 0 < min_quiet <= batch_size:
        raise ValueError("A positive, feasible quiet stratum is required")
    candidates = defaultdict(list)
    for i, crop in enumerate(crops):
        if crop.source_id not in rows:
            raise ValueError("Training source row is missing")
        sha = _audio_sha(rows[crop.source_id])
        if crop.source_id in heldout_ids or sha in excluded_hashes:
            continue
        candidates[crop.source_id].append(i)
    representatives, used_hashes = [], set()
    for sid in sorted(candidates, key=rank):
        sha = _audio_sha(rows[sid])
        if sha in used_hashes:
            continue
        used_hashes.add(sha)
        i = min(candidates[sid], key=lambda j: rank(sid + "|" + str(crops[j].start_frame) + "|" + str(j)))
        representatives.append(i)
    count = calibration_batches + comparison_batches
    needed = count * batch_size
    if len(representatives) < needed:
        raise ValueError("Insufficient unique recording hashes for the declared split")
    quiet_counts = {}
    reserved = []
    for i in representatives:
        quiet_counts[i] = teacher_quiet_samples(crops[i])
        if quiet_counts[i] >= 960:
            reserved.append(i)
            if len(reserved) == count * min_quiet:
                break
    if len(reserved) != count * min_quiet:
        raise ValueError("Insufficient teacher-quiet source coverage; no model-based reselection")
    reserved_set = set(reserved)
    fillers = iter(i for i in representatives if i not in reserved_set)
    batches, used_ids, used_audio = [], set(), set()
    for b in range(count):
        chosen = reserved[b * min_quiet:(b + 1) * min_quiet]
        chosen += [next(fillers) for _ in range(batch_size - min_quiet)]
        # The completed batch order is independently deterministic and has no student-score input.
        chosen.sort(key=lambda i: rank("batch-order|" + str(i)))
        details = []
        for i in chosen:
            crop, row = crops[i], rows[crops[i].source_id]
            sha = _audio_sha(row)
            if crop.source_id in used_ids or sha in used_audio:
                raise RuntimeError("A source or audio hash crosses the immutable batches")
            used_ids.add(crop.source_id); used_audio.add(sha)
            if i not in quiet_counts:
                quiet_counts[i] = teacher_quiet_samples(crop)
            details.append({"pool_index": i, "source_id": crop.source_id, "audio_sha256": sha,
                "start_frame": crop.start_frame, "valid_scored_samples": crop.valid_scored_samples,
                "quiet_samples": quiet_counts[i], "language": row.get("language"),
                "dataset": row.get("dataset"), "condition": row.get("condition")})
        label = "calibration" if b < calibration_batches else "comparison"
        batches.append({"id": label + "_" + str(b if b < calibration_batches else b - calibration_batches),
            "split": label, "indices": chosen, "sources": details,
            "quiet_sources": sum(r["quiet_samples"] >= 960 for r in details),
            "quiet_samples": sum(r["quiet_samples"] for r in details)})
    return {"seed": SELECTION_SEED, "batches": batches, "unique_source_ids": len(used_ids),
        "unique_audio_hashes": len(used_audio), "source_and_hash_disjoint": True,
        "heldout_source_and_hash_disjoint": True, "teacher_only_quiet_stratification": True,
        "minimum_quiet_sources_per_batch": min_quiet,
        "earlier_exposure": "These are previously downloaded/debug-used windows. Only calibration/comparison batches in this diagnostic are mutually source/audio-hash disjoint; this is not a claim of unseen training history."}


@torch.no_grad()
def copy_parameters(model, values):
    for name, p in model.named_parameters():
        p.copy_(values[name].to(p.device))


def generate_directions(engine, crops, *, global_rng, seed=1861):
    """One D update, two G optimizer steps from the identical clipped gradient."""
    original = state_fingerprint(engine.state_dict())
    saved_crop_rng = engine.crop_generator.get_state().clone()
    named = tuple(engine.model.named_parameters())
    parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
    old_grads = [(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters]
    capture = {}
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as guard:
        engine.crop_generator.set_state(saved_crop_rng)
        state = _capture_replay_state(engine)
        try:
            extension = engine.step == engine.recipe.total_steps
            if extension:
                rate = engine.learning_rate()
                engine.recipe = replace(engine.recipe, total_steps=engine.step + 1)
                engine.config = replace(engine.config, total_steps=engine.step + 1)
                if engine.learning_rate() != rate:
                    raise ValueError("Diagnostic-only budget extension changes rate")
            if global_rng is not None:
                _restore_rng(deepcopy(global_rng))
            with intercept_step(engine.optimizer, named, capture):
                metrics = engine.train_step(crops)
            if capture.get("calls") != 1 or engine.discriminator_updates != state["discriminator_updates"] + 1:
                raise RuntimeError("Exactly one native D and retained-G step is required")
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Nonfinite retained update")
            retained_after = {n: p.detach().cpu().clone() for n, p in named}
            retained_delta = tuple(retained_after[n] - state["model"][n] for n, _ in named)
            d_hash = state_fingerprint(engine.discriminators.state_dict())
            d_opt_hash = state_fingerprint(engine.discriminator_optimizer.state_dict())
            balance_hash = state_fingerprint(engine.balancer.state_dict())
            copy_parameters(engine.model, state["model"])
            engine.optimizer.load_state_dict(deepcopy(state["optimizer"]))
            group_hash = state_fingerprint({k: [{a: v for a, v in g.items() if a != "params"}
                for g in optimizer.param_groups] for k, optimizer in engine.optimizer.optimizers.items()})
            for optimizer in engine.optimizer.optimizers.values():
                optimizer.state.clear()
            if any(len(optimizer.state) for optimizer in engine.optimizer.optimizers.values()):
                raise RuntimeError("Fresh generator state is not empty")
            for (_, p), g in zip(named, capture["gradient"], strict=True):
                p.grad = g.to(p.device).clone()
            gradient_hash = state_fingerprint(capture["gradient"])
            if state_fingerprint(tuple(p.grad.detach().cpu() for _, p in named)) != gradient_hash:
                raise RuntimeError("Fresh step does not receive the identical gradient")
            engine.optimizer.step()  # Native implementation, empty G states only. D is not rerun.
            fresh_delta = tuple(p.detach().cpu() - state["model"][n] for n, p in named)
            if (state_fingerprint(engine.discriminators.state_dict()) != d_hash
                    or state_fingerprint(engine.discriminator_optimizer.state_dict()) != d_opt_hash
                    or state_fingerprint(engine.balancer.state_dict()) != balance_hash):
                raise RuntimeError("The fresh-generator counterfactual changed D or loss state")
            if group_hash != state_fingerprint({k: [{a: v for a, v in g.items() if a != "params"}
                for g in optimizer.param_groups] for k, optimizer in engine.optimizer.optimizers.items()}):
                raise RuntimeError("Fresh optimizer changed group options")
            for n, b in engine.model.named_buffers():
                if not torch.equal(b.detach().cpu(), state["model"][n]):
                    raise RuntimeError("A direction-generation step changed model buffers")
            directions = {"retained": retained_delta, "fresh_generator_state": fresh_delta,
                          "plain_sgd": tuple(-g for g in capture["gradient"])}
            norms = {name: math.sqrt(vector_dot(d, d)) for name, d in directions.items()}
            if any(not math.isfinite(n) or n <= 0 for n in norms.values()):
                raise ValueError("Nonfinite or zero direction")
            receipt = {"training_metrics": metrics, "gradient_sha256": gradient_hash,
                "direction_norms": norms, "D_views": deepcopy(getattr(engine, "screen_last_views", [])),
                "source_engine_sha256": original, "generator_step_calls": 2, "discriminator_step_calls": 1,
                "shared_gradient_identical": True, "D_and_balancer_shared": True,
                "fresh_state_scope": "All states of both generator optimizers are empty before one native step; original parameter groups/LR/decay remain. Discriminator moments and its one update are preserved.",
                "plain_sgd_scope": "Negative current clipped gradient only, without momentum, adaptive scaling or weight decay; no SGD optimizer step is executed.",
                "budget_only_extension": extension, "fixed_state": guard}
        finally:
            _restore_replay_state(engine, state)
            for p, (g, saved) in zip(parameters, old_grads, strict=True):
                p.grad = g
                if g is not None:
                    g.copy_(saved)
    if state_fingerprint(engine.state_dict()) != original:
        raise RuntimeError("Direction generation failed exact engine restoration")
    receipt["state_restored"] = True
    return {"directions": directions, "retained_after": retained_after, "receipt": receipt}


@torch.no_grad()
def training_pack(engine, crops):
    require_quiet_contract(engine.config.quiet_audio)
    z, latent_mask = calibration_batch(crops, engine.device)
    length = z.shape[-1] * 1920
    target = torch.cat([F.pad(c.teacher_audio.detach(), (0, length - c.teacher_audio.shape[-1])) for c in crops]).to(engine.device)
    valid = torch.zeros_like(target, dtype=torch.bool)
    for i, crop in enumerate(crops):
        valid[i, :, crop.scored_slice] = True
    quiet = _quiet_mask(target, target, valid, QuietAudioConfig())
    if int(quiet.sum()) < 960:
        raise ValueError("Calibration/comparison batch has no meaningful teacher-quiet samples")
    return {"z": z, "latent_mask": latent_mask, "target": target, "valid": valid, "quiet": quiet, "crops": crops}


@torch.no_grad()
def forward_pack(engine, pack):
    p = engine.model(pack["z"], scored_latent_mask=pack["latent_mask"])
    if p.shape != pack["target"].shape or not bool(torch.isfinite(p).all()):
        raise ValueError("Nonfinite or misaligned batched diagnostic output")
    return p


def baseline_mse_vjps(engine, pack):
    """Exact local MSE parameter derivatives; no .grad or optimizer mutation."""
    with torch.enable_grad():
        prediction = engine.model(pack["z"], scored_latent_mask=pack["latent_mask"])
        params = tuple(engine.model.parameters())
        vectors = {}
        for key, mask in (("valid", pack["valid"]), ("teacher_quiet", pack["quiet"])):
            value = (prediction[mask] - pack["target"][mask]).square().mean()
            values = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
            vectors[key] = tuple(torch.zeros_like(p, device="cpu") if g is None else g.detach().cpu()
                                 for p, g in zip(params, values, strict=True))
            if any(not bool(torch.isfinite(g).all()) for g in vectors[key]):
                raise FloatingPointError("Nonfinite baseline MSE VJP")
    return prediction.detach(), vectors


@torch.no_grad()
def apply_radius(model, baseline, direction, radius, *, metric_vectors=None):
    norm = math.sqrt(vector_dot(direction, direction))
    if norm <= 0 or radius <= 0:
        raise ValueError("A positive direction and radius are required")
    actual_squared = 0.
    terms = {key: 0. for key in metric_vectors or {}}
    for i, ((name, p), d) in enumerate(zip(model.named_parameters(), direction, strict=True)):
        value = (baseline[name].double() + d.double() * (radius / norm)).to(p.dtype)
        displacement = value.double() - baseline[name].double()
        actual_squared += float(displacement.square().sum())
        for key, vectors in (metric_vectors or {}).items():
            terms[key] += float((vectors[i].double() * displacement).sum())
        p.copy_(value.to(p.device))
    actual = math.sqrt(actual_squared)
    return {"requested_parameter_l2": radius, "realized_parameter_l2": actual,
        "relative_rounding_error": abs(actual - radius) / radius,
        "rounding_qualified": abs(actual - radius) / radius <= .01,
        "exact_parameter_first_order_mse_change": terms}


def calibration_guards(baseline, full, half, target, valid, quiet, *, linear_terms):
    answer = {}
    for name, mask in (("valid", valid), ("teacher_quiet", quiet)):
        p0, pf, ph, t = (value[mask].double() for value in (baseline, full, half, target))
        if not p0.numel():
            raise ValueError("Calibration guard region is empty")
        displacement = pf - p0
        linear_displacement = 2 * (ph - p0)
        error = displacement - linear_displacement
        floor = float(32 * torch.finfo(torch.float32).eps * p0.norm()) + 1e-30
        signal = max(float(displacement.norm()), float(linear_displacement.norm()))
        exact_zero = signal == 0
        ratio = float(error.norm()) / max(signal, floor)
        measurable = signal > floor or exact_zero
        residual = p0 - t
        half_linear = float((2 * residual * linear_displacement).mean())
        linear = linear_terms[name]
        if not math.isfinite(linear):
            raise ValueError("An exact finite baseline parameter-MSE derivative is required")
        quadratic = float(displacement.square().mean())
        half_remainder = float((2 * residual * error).mean())
        before_mse, after_mse = float(residual.square().mean()), float((pf - t).square().mean())
        if not math.isclose(after_mse - before_mse, half_linear + quadratic + half_remainder, rel_tol=1e-8, abs_tol=1e-15):
            raise RuntimeError("MSE linear/quadratic accounting failed")
        remainder = after_mse - before_mse - linear - quadratic
        descending = linear < 0
        quadratic_ratio = quadratic / abs(linear) if descending else None
        arithmetic_tolerance = 64 * torch.finfo(torch.float64).eps * max(before_mse, after_mse, 1e-30)
        observed_descent = after_mse - before_mse <= arithmetic_tolerance
        passed = measurable and ratio <= .1 and (not descending or (quadratic_ratio <= .25 and observed_descent))
        answer[name] = {"samples": p0.numel(), "baseline_mse": before_mse, "after_mse": after_mse,
            "actual_mse_change": after_mse - before_mse, "exact_parameter_linear_term": linear,
            "half_step_estimated_linear_term": half_linear,
            "exact_squared_displacement_term": quadratic, "linearization_remainder_term": remainder,
            "direction_classification": "descending" if descending else "non_descending",
            "quadratic_to_absolute_linear_ratio": quadratic_ratio,
            "observed_descent_or_roundoff": observed_descent, "observed_descent_tolerance": arithmetic_tolerance,
            "nonlinear_waveform_ratio": ratio, "displacement_numerically_resolved": measurable,
            "zero_acoustic_displacement": exact_zero, "passed": passed,
            "non_descending_caution": None if descending else "Recorded as non-descending; not relabeled a successful direction merely because the trust-region guard passes."}
    return {"passed": all(row["passed"] for row in answer.values()), "regions": answer}


def calibrate_radius(engine, base_model, calibration):
    initial_radius = min(row["generated"]["receipt"]["direction_norms"]["retained"] for row in calibration)
    report = {"initial_radius": initial_radius, "fractions": list(FRACTIONS), "attempts": [],
        "selection_policy": "Largest declared shared radius qualified on every training calibration batch/direction. No heldout predictions are used.",
        "guards": "Both valid and teacher-quiet waveform nonlinearity <=10%; descending MSE directions require exact squared output displacement <=25% of exact baseline parameter-MSE derivative magnitude AND measured MSE must not increase beyond explicit FP64 arithmetic tolerance. Descent labels use exact VJPs, never half-step slopes; positive/zero directions recorded, not optimized into a pass.",
        "half_probe_minimum_fraction": min(FRACTIONS) / 2,
        "half_probe_scope": "One smaller half-radius point estimates local linearity, never an additional candidate radius."}
    previous_half = {}
    try:
        for fraction in FRACTIONS:
            radius = initial_radius * fraction
            cases = []
            for item in calibration:
                for method in METHODS:
                    key = (item["id"], method)
                    direction = item["generated"]["directions"][method]
                    rounded = apply_radius(engine.model, base_model, direction, radius, metric_vectors=item["metric_vectors"])
                    full = previous_half.pop(key, None)
                    if full is None:
                        full = forward_pack(engine, item["pack"])
                    half_rounding = apply_radius(engine.model, base_model, direction, radius / 2)
                    half = forward_pack(engine, item["pack"])
                    previous_half[key] = half
                    checks = calibration_guards(item["baseline"], full, half, item["pack"]["target"], item["pack"]["valid"], item["pack"]["quiet"],
                        linear_terms=rounded["exact_parameter_first_order_mse_change"])
                    cases.append({"batch": item["id"], "method": method, "rounding": rounded,
                        "half_rounding": half_rounding, **checks,
                        "qualified": checks["passed"] and rounded["rounding_qualified"] and half_rounding["rounding_qualified"]})
            attempt = {"fraction": fraction, "radius": radius, "cases": cases, "qualified": all(c["qualified"] for c in cases)}
            report["attempts"].append(attempt)
            if attempt["qualified"]:
                report.update(selected_radius=radius, selected_fraction=fraction, resolved=True)
                break
        else:
            report.update(selected_radius=None, selected_fraction=None, resolved=False,
                          reason="No declared radius passed all training-only guards; no further tuning or comparison updates")
    finally:
        copy_parameters(engine.model, base_model)
    return report


@torch.no_grad()
def training_metrics(engine, pack, prediction):
    target, valid, quiet = pack["target"], pack["valid"], pack["quiet"]
    def mse(mask):
        return float((prediction[mask].double() - target[mask].double()).square().mean())
    groups = defaultdict(list)
    for i, crop in enumerate(pack["crops"]):
        groups[crop.valid_scored_samples].append(i)
    pairs = [(torch.cat([prediction[i:i + 1, :, pack["crops"][i].scored_slice] for i in indices]),
              torch.cat([target[i:i + 1, :, pack["crops"][i].scored_slice] for i in indices]))
             for indices in groups.values()]
    losses = engine.reconstruction.forward_groups(pairs).losses
    return {"valid_mse": mse(valid), "teacher_quiet_mse": mse(quiet),
        "valid_samples": int(valid.sum()), "quiet_samples": int(quiet.sum()),
        "teacher_waveform": float(losses["teacher_waveform"]), "teacher_mel": float(losses["teacher_mel"])}


def changes(before, after):
    return {key: {"before": value, "after": after[key], "absolute_change": after[key] - value,
                   "relative_change_percent": 100 * (after[key] - value) / value if value else None}
            for key, value in before.items() if isinstance(value, (int, float)) and not key.endswith("samples")}


def compare_batch(engine, base_model, batch_id, crops, generated, probes, radius):
    pack = training_pack(engine, crops)
    copy_parameters(engine.model, base_model)
    baseline_audio = forward_pack(engine, pack)
    baseline_train = training_metrics(engine, pack, baseline_audio)
    baseline_panel, _ = probe_snapshot(engine, probes)
    outcomes = {}
    try:
        for method in (*METHODS, "retained_native_unscaled"):
            if method == "retained_native_unscaled":
                copy_parameters(engine.model, generated["retained_after"])
                radius_report = {"requested_parameter_l2": generated["receipt"]["direction_norms"]["retained"],
                    "realized_parameter_l2": generated["receipt"]["direction_norms"]["retained"],
                    "scope": "Exact retained native parameter endpoint, no normalization"}
            else:
                radius_report = apply_radius(engine.model, base_model, generated["directions"][method], radius)
                if not radius_report["rounding_qualified"]:
                    raise ValueError("Comparison parameter radius was lost to FP32 rounding")
            p = forward_pack(engine, pack)
            after_train = training_metrics(engine, pack, p)
            after_panel, _ = probe_snapshot(engine, probes)
            outcomes[method] = {"radius": radius_report, "training_after": after_train,
                "training_changes": changes(baseline_train, after_train),
                "training_waveform_displacement_rms": float((p[pack["valid"]].double() - baseline_audio[pack["valid"]].double()).square().mean().sqrt()),
                "training_quiet_displacement_rms": float((p[pack["quiet"]].double() - baseline_audio[pack["quiet"]].double()).square().mean().sqrt()),
                "panel_after": after_panel, "panel_changes": changes(baseline_panel["metrics"], after_panel["metrics"])}
    finally:
        copy_parameters(engine.model, base_model)
    return {"batch": batch_id, "direction_generation": generated["receipt"],
        "training_before": baseline_train, "panel_before": baseline_panel, "outcomes": outcomes}


def comparison_summary(results):
    summary = {}
    for method in (*METHODS, "retained_native_unscaled"):
        summary[method] = {}
        for metric in results[0]["panel_before"]["metrics"]:
            values = [r["outcomes"][method]["panel_changes"][metric]["relative_change_percent"] for r in results]
            if any(v is None for v in values):
                raise ValueError("Required baseline panel metric is zero or undefined")
            average = math.fsum(values) / len(values)
            summary[method][metric] = {"batch_changes_percent": values, "mean_percent": average,
                "minimum_percent": min(values), "maximum_percent": max(values),
                "improving_batches": sum(v < 0 for v in values), "batches": len(values)}
    return {"methods": summary, "promoted": False,
        "scope": "Independent single-batch directions from the same checkpoint; not a multi-step optimizer training run or a general audio-quality evaluation.",
        "decision": "Review consistent direction/quiet/peak tradeoffs before any further work. No automatic full-panel run, optimizer replacement, checkpoint promotion or continuation."}


def run(ctx, args):
    from diagnostic_common import atomic_json, status
    started = time.monotonic()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    if file_sha(args.checkpoint) != QUARTER_SHA:
        raise ValueError("Quarter8890 checkpoint hash differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    require_quiet_contract(payload["engine"]["config"]["quiet_audio"])
    overlay = load_training_overlay(ctx, args.training_receipt, required_counts={"targeted_generator": 12800})
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    hashes, inventory_id = heldout_hashes(panel, args.canonical_inventory)
    probes = select_probes(panel["crops"], json.loads(Path(args.selection).read_text()))
    pool = overlay["pools"]["targeted_generator"]
    selection = select_batches(pool, ctx.data["rows"], {c.source_id for c in panel["crops"]}, hashes)
    atomic_json(out / "selection.json", selection)
    engine = ctx.engine("targeted", device="cuda")
    restored = restore_quarter(engine, payload["engine"])
    bind_views(engine, ctx.data["pools"]["targeted_generator"])
    base_model = {n: p.detach().cpu().clone() for n, p in engine.model.named_parameters()}
    inputs = {str(Path(args.checkpoint).resolve()): QUARTER_SHA,
        str(Path(args.canonical_inventory).resolve()): file_sha(args.canonical_inventory),
        str(Path(args.selection).resolve()): file_sha(args.selection)}
    source_files = {Path(__file__).resolve()}
    for name in ("native_chain", "gradient_probe", "diagnose_objectives_updates", "training_overlay",
                 "run_update_experiment", "run_corrected_screen", "canonical_evaluation", "render_canonical_targets", type(ctx).__module__):
        source_files.add(Path(importlib.import_module(name).__file__).resolve())
    import audiovae_student.recipe_v2 as recipe_module
    source_files.update(Path(recipe_module.__file__).parent.glob("*.py"))
    identity = {"version": 1, "checkpoint_sha256": QUARTER_SHA, "restored_engine_sha256": restored,
        "training_overlay": overlay["identity"], "canonical_panel": panel["identity"],
        "canonical_inventory_identity": inventory_id, "selection_sha256": digest(selection),
        "input_file_hashes": inputs, "source_file_hashes": {str(p): file_sha(p) for p in sorted(source_files)},
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "grid": list(FRACTIONS), "calibration_batches": CALIBRATION_BATCHES, "comparison_batches": COMPARISON_BATCHES,
        "batch_size": BATCH_SIZE, "minimum_quiet_sources": MIN_QUIET_SOURCES,
        "parameter_radius_scope": "Same parameter L2 displacement, not equal acoustic effect. Training acoustic displacement and heldout metric changes are reported separately.",
        "fresh_scope": "Reset generator optimizer state only; no discriminator/objective change",
        "retained_updates": 0, "checkpoint_files_written": 0}
    atomic_json(out / "identity.json", identity)
    calibration = []
    for batch in selection["batches"]:
        if batch["split"] != "calibration":
            continue
        status("optimizer_calibration_directions", batch=batch["id"])
        crops = [pool[i] for i in batch["indices"]]
        generated = generate_directions(engine, crops, global_rng=payload["rng"])
        atomic_json(out / (batch["id"] + "-directions.json"), generated["receipt"])
        pack = training_pack(engine, crops)
        with _fixed_engine(engine, 1861, _preserved_attributes(engine)):
            baseline, metric_vectors = baseline_mse_vjps(engine, pack)
        calibration.append({"id": batch["id"], "generated": generated, "pack": pack,
                            "baseline": baseline, "metric_vectors": metric_vectors})
    original = state_fingerprint(engine.state_dict())
    status("optimizer_radius_calibration")
    with _fixed_engine(engine, 1861, _preserved_attributes(engine)):
        radius_report = calibrate_radius(engine, base_model, calibration)
    atomic_json(out / "radius-calibration.json", radius_report)
    if state_fingerprint(engine.state_dict()) != original:
        raise RuntimeError("Calibration changed the saved engine")
    calibration.clear()
    results = []
    if radius_report["resolved"]:
        for batch in selection["batches"]:
            if batch["split"] != "comparison":
                continue
            status("optimizer_comparison_batch", batch=batch["id"])
            crops = [pool[i] for i in batch["indices"]]
            generated = generate_directions(engine, crops, global_rng=payload["rng"])
            with _fixed_engine(engine, 1861, _preserved_attributes(engine)):
                result = compare_batch(engine, base_model, batch["id"], crops, generated, probes, radius_report["selected_radius"])
            results.append(result)
            atomic_json(out / (batch["id"] + ".json"), result)
            if state_fingerprint(engine.state_dict()) != restored:
                raise RuntimeError("Comparison failed exact state restoration")
    answer = {"identity": identity, "selection": selection, "calibration": radius_report,
        "comparisons": results, "summary": comparison_summary(results) if results else None,
        "resolved": radius_report["resolved"], "retained_updates": 0, "promoted": False,
        "checkpoint_files_written": 0, "state_restored": state_fingerprint(engine.state_dict()) == restored,
        "original_files": ctx.verify_files(), "fresh_files": verify_overlay_files(overlay), "seconds": time.monotonic() - started}
    for path, sha in {**inputs, panel["identity"]["receipt_path"]: panel["identity"]["receipt_sha256"],
                      panel["identity"]["cache_path"]: panel["identity"]["cache_sha256"]}.items():
        if file_sha(path) != sha:
            raise RuntimeError("An immutable input changed")
    atomic_json(out / "optimizer-comparison.json", answer)
    status("optimizer_comparison_complete", resolved=answer["resolved"], path=str(out / "optimizer-comparison.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-receipt", "canonical-receipt", "canonical-inventory", "selection", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--context-module", default="diagnostic_common")
    args = p.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(importlib.import_module(args.context_module).load_context(), args)
