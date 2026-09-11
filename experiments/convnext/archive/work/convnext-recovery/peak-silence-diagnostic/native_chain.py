"""Trace one disposable native quarter-rate update, without saving weights.

The raw direction is the negative post-clip parameter gradient. Global norm
clipping preserves its direction. It is not an alternative optimizer trial.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import fcntl
import importlib
import json
import math
from pathlib import Path
import time

import torch

from audiovae_student.corrected_calibration import _fixed_engine
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.restart_data import file_sha
from audiovae_student.training import _restore_rng
from diagnose_objectives_updates import (_capture_replay_state, _restore_replay_state,
    _preserved_attributes, vector_dot)
from gradient_probe import _quiet_mask
from run_corrected_screen import bind_views
from run_update_experiment import load_canonical_panel
from training_overlay import load_training_overlay, verify_overlay_files

QUARTER_SHA = "f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948"
FRESH_RECEIPT_SHA = "5a73b4e601ec43ad81ca841656e35a46d15f03c82f91363df06bd05171afff77"
FRESH_CACHE_SHA = "fd55cf09f3664cf8024d3d6ee9e6a1eca9c3fa1b16e59d2d7d21a340ae8fa1d2"
METRICS = ("peak_excess_mse", "teacher_quiet_residual_mse", "encoded_zero_quiet_mse", "natural_quiet_residual_mse")


def select_probes(crops, selection):
    entries = selection.get("probes") if isinstance(selection, dict) else selection
    if not isinstance(entries, list) or len(entries) != 10:
        raise ValueError("Exactly the ten frozen-diagnostic selections are required")
    keys = [(r["source_id"], r["start_frame"]) for r in entries]
    if len(set(keys)) != 10 or len({k[0] for k in keys}) != 10:
        raise ValueError("Probe sources and crop keys must be unique")
    lookup = {(c.source_id, c.start_frame): c for c in crops}
    if len(lookup) != len(crops) or any(k not in lookup for k in keys):
        raise ValueError("Selected crop is absent from canonical panel")
    if sum(k[0] == "encoded_zero" for k in keys) != 1:
        raise ValueError("Exactly one encoded-zero fixture is required")
    return [lookup[key] for key in keys]


def restore_quarter(engine, saved):
    """Explicit Fusion resume, preserving all experimental state identities."""
    if saved.get("format_version") != "fusion_recipe_v1" or saved.get("step") != 8890:
        raise ValueError("Expected the saved quarter-rate fusion checkpoint at8890")
    if saved.get("fusion_variant") != "control" or saved["recipe"]["learning_rate"] != .00005:
        raise ValueError("Expected the unchanged control architecture at quarter rate")
    source_hash = state_fingerprint(saved)
    state = deepcopy(saved)  # Optimizer loading must never alias checkpoint tensors.
    engine.recipe = type(engine.recipe)(**state["recipe"])
    engine.config = type(engine.config)(**state["config"])
    for key in ("model", "discriminators", "optimizer", "discriminator_optimizer", "balancer"):
        getattr(engine, key).load_state_dict(state[key])
    engine.crop_generator.set_state(state["crop_rng"].cpu())
    for key in ("step", "calibration", "perceptual_start", "discriminator_updates", "fusion_migration"):
        setattr(engine, key, deepcopy(state[key]))
    if state_fingerprint(engine.state_dict()) != source_hash or state_fingerprint(saved) != source_hash:
        raise RuntimeError("Quarter checkpoint restore is not exact")
    if engine.learning_rate() != .00005 or engine.step > engine.recipe.total_steps:
        raise ValueError("Quarter checkpoint schedule is invalid")
    return source_hash


def _masks(crop, device, config):
    teacher = crop.teacher_audio.to(device)
    valid = torch.zeros_like(teacher, dtype=torch.bool)
    start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
    valid[..., start:crop.scored_slice.stop] = True
    # Quiet eligibility uses teacher RMS alone, on the original full tensor grid.
    quiet = _quiet_mask(teacher, teacher, valid, config)
    return teacher, valid, quiet


def probe_snapshot(engine, crops, *, gradients=False):
    config = QuietAudioConfig(**engine.config.quiet_audio) if isinstance(engine.config.quiet_audio, dict) else engine.config.quiet_audio
    bank = [_masks(c, engine.device, config) for c in crops]
    counts = {"peak_excess_mse": sum(int(v.sum()) for _, v, _ in bank),
        "teacher_quiet_residual_mse": sum(int(q.sum()) for _, _, q in bank),
        "encoded_zero_quiet_mse": sum(int(q.sum()) for c, (_, _, q) in zip(crops, bank) if c.source_id == "encoded_zero"),
        "natural_quiet_residual_mse": sum(int(q.sum()) for c, (_, _, q) in zip(crops, bank) if not c.source_id.startswith("encoded_"))}
    if any(n <= 0 for n in counts.values()):
        raise ValueError("The probe panel must include scored, quiet, natural-quiet and encoded-zero samples")
    named = tuple(engine.model.named_parameters())
    parameters = tuple(p for _, p in named)
    vectors = {k: [torch.zeros_like(p, device="cpu") for p in parameters] for k in METRICS} if gradients else {}
    totals, rows = {k: 0. for k in METRICS}, []
    for c, (teacher, valid, quiet) in zip(crops, bank, strict=True):
        with torch.set_grad_enabled(gradients):
            p = engine.model(c.latents.to(engine.device))
            if p.shape != teacher.shape or not bool(torch.isfinite(p).all()):
                raise ValueError("Nonfinite or misaligned probe output")
            excess = (p.abs() - 1).clamp_min(0)
            residual = p - teacher
            sums = {"peak_excess_mse": excess[valid].square().sum(),
                    "teacher_quiet_residual_mse": residual[quiet].square().sum()}
            if c.source_id == "encoded_zero":
                sums["encoded_zero_quiet_mse"] = sums["teacher_quiet_residual_mse"]
            if not c.source_id.startswith("encoded_"):
                sums["natural_quiet_residual_mse"] = sums["teacher_quiet_residual_mse"]
            for key, value in sums.items():
                totals[key] += float(value.detach().double())
                if gradients:
                    values = torch.autograd.grad(value / counts[key], parameters, retain_graph=True, allow_unused=True)
                    for total, g in zip(vectors[key], values, strict=True):
                        if g is not None:
                            if not bool(torch.isfinite(g).all()):
                                raise FloatingPointError("Nonfinite diagnostic VJP")
                            total.add_(g.detach().cpu())
            nq = int(quiet.sum())
            rows.append({"source_id": c.source_id, "start_frame": c.start_frame,
                "samples": int(valid.sum()), "quiet_samples": nq,
                "peak_excess_mse": float(excess[valid].detach().double().square().mean()),
                "teacher_quiet_residual_mse": float(residual[quiet].detach().double().square().mean()) if nq else None,
                "teacher_mse": float(residual[valid].detach().double().square().mean()),
                "waveform_mae": float(residual[valid].detach().double().abs().mean()),
                "peak_abs": float(p[valid].detach().abs().max()),
                "overshoot_samples": int((p[valid].detach().abs() > 1).sum())})
    return {"metrics": {k: totals[k] / counts[k] for k in METRICS}, "denominators": counts, "rows": rows}, vectors


@contextmanager
def intercept_step(bundle, named, captured):
    """Observe post-clip gradients, invoke the original native step once."""
    own = "step" in bundle.__dict__
    previous = bundle.__dict__.get("step")
    original = bundle.step
    def observed(*args, **kwargs):
        if captured.get("calls", 0):
            raise RuntimeError("Only one generator update is permitted")
        values = []
        for _, p in named:
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise ValueError("Missing or nonfinite native generator gradient")
            values.append(p.grad.detach().cpu().clone())
        captured.update(calls=1, gradient=tuple(values))
        return original(*args, **kwargs)
    bundle.step = observed
    try:
        yield
    finally:
        if own:
            bundle.step = previous
        else:
            del bundle.step


def direction_report(metric_vectors, direction):
    norm = math.sqrt(vector_dot(direction, direction))
    report = {}
    for name, metric_gradient in metric_vectors.items():
        metric_norm = math.sqrt(vector_dot(metric_gradient, metric_gradient))
        change = vector_dot(metric_gradient, direction)
        report[name] = {"first_order_change": change,
            "unit_direction_derivative": change / norm if norm else None,
            "direction_cosine_with_metric_gradient": change / (norm * metric_norm) if norm and metric_norm else None,
            "effect": "decrease" if change < 0 else "increase" if change > 0 else "zero_first_derivative"}
    return {"direction_norm": norm, "metrics": report}


def _parameter_family(name):
    # Identical exclusive grouping to the frozen parameter_geometry diagnostic.
    if name.startswith(("stem_norm.", "affine.")) or ".norm." in name:
        return "norm"
    if name.startswith("blocks."):
        return ".".join(name.split(".")[:2])
    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("stem."):
        return "stem"
    if name.startswith(("head.", "activation.")):
        return "head"
    if name.startswith(("output.", "output_filter.")):
        return "output"
    return "other"


def additive_localization(named, metric_vectors, raw, delta, manifest, direction_reports):
    """Partition already-captured first-order dots; no model/optimizer calls."""
    names = [name for name, _ in named]
    entries = [(group["optimizer"], p) for group in manifest for p in group["parameters"]]
    if (len(names) != len(set(names)) or len(entries) != len(names)
            or len({p["name"] for _, p in entries}) != len(entries)
            or {p["name"] for _, p in entries} != set(names)):
        raise ValueError("Optimizer manifest must partition exactly all named parameters")
    routes = {}
    parameters = dict(named)
    for route, entry in entries:
        p = parameters[entry["name"]]
        if entry["shape"] != list(p.shape) or entry["dtype"] != str(p.dtype):
            raise ValueError("Optimizer manifest tensor metadata changed")
        routes[entry["name"]] = route
    assignments = {"optimizer_routes": routes,
        "parameter_families": {name: _parameter_family(name) for name in names},
        "route_by_family": {name: routes[name] + "/" + _parameter_family(name) for name in names}}
    result = {"interpretation": "Additive first-order dot products for exclusive partitions of the same captured direction. These are not additive finite audio effects or optimizer-comparison evidence; optimizer routes own different parameters.",
        "family_policy": "All block LayerNorm and stem/output affine normalization are norm; remaining block parameters are blocks.N; adapter, stem, head, output and other are separate.",
        "no_additional_model_forwards_or_optimizer_steps": True, "directions": {}}
    for label, direction in (("raw_negative_postclip_gradient", raw), ("native_parameter_delta", delta)):
        if len(direction) != len(named) or any(len(v) != len(named) for v in metric_vectors.values()):
            raise ValueError("Gradient vectors do not match named parameter order")
        # Reduce each tensor immediately in FP64, retaining only scalar dots.
        dots = {metric: [vector_dot((g,), (d,)) for g, d in zip(vector, direction, strict=True)]
                for metric, vector in metric_vectors.items()}
        squared_norms = [vector_dot((d,), (d,)) for d in direction]
        partitions = {}
        for partition, mapping in assignments.items():
            groups = {}
            for group in sorted(set(mapping.values())):
                indices = [i for i, name in enumerate(names) if mapping[name] == group]
                groups[group] = {"parameter_names": [names[i] for i in indices],
                    "parameter_tensors": len(indices),
                    "parameter_elements": sum(named[i][1].numel() for i in indices),
                    "direction_norm": math.sqrt(math.fsum(squared_norms[i] for i in indices)),
                    "first_order_metric_change": {metric: math.fsum(values[i] for i in indices)
                                                  for metric, values in dots.items()}}
            checks = {}
            for metric in metric_vectors:
                values = [group["first_order_metric_change"][metric] for group in groups.values()]
                observed = math.fsum(values)
                expected = direction_reports[label]["metrics"][metric]["first_order_change"]
                tolerance = 1e-14 + 1e-10 * math.fsum(abs(v) for v in values)
                if abs(observed - expected) > tolerance:
                    raise RuntimeError("Exclusive first-order partitions do not sum to the total")
                checks[metric] = {"sum": observed, "total": expected,
                    "absolute_error": abs(observed - expected), "tolerance": tolerance, "passed": True}
            partitions[partition] = {"groups": groups, "additivity_checks": checks}
        result["directions"][label] = partitions
    return result


def adamw_analytic_decomposition(named, current_gradients, delta, before_parameters, optimizer, metric_vectors):
    """Read the actual post-step AdamW state; never execute an optimizer.

    Counterfactual directions hold the actual second moment fixed. They are
    algebraic attribution devices, not alternative training trajectories.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("This decomposition is only for the actual AdamW route")
    names = {id(p): name for name, p in named}
    index = {name: i for i, (name, _) in enumerate(named)}
    labels = ("raw_gradient", "first_moment_without_preconditioner",
        "current_gradient_with_actual_divisor", "native_adaptive_term",
        "decoupled_weight_decay", "reconstructed_adamw", "observed_adamw", "rounding_residual")
    accumulated = {label: {"squared_norm": [], "dots": {m: [] for m in metric_vectors}} for label in labels}
    covered, groups, steps = [], [], set()
    maximum_error = maximum_relative_bound = squared_error = actual_squared = 0.
    elements = 0
    for group_index, group in enumerate(optimizer.param_groups):
        if group.get("maximize", False):
            raise ValueError("The retained descent recipe must not maximize")
        lr, wd, eps = float(group["lr"]), float(group["weight_decay"]), float(group["eps"])
        beta1, beta2 = map(float, group["betas"])
        if not (lr > 0 and wd >= 0 and eps > 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Invalid saved AdamW hyperparameters")
        declared_names = group.get("param_names")
        actual_names = [names.get(id(p)) for p in group["params"]]
        if None in actual_names or declared_names != actual_names:
            raise ValueError("AdamW parameter-name order does not match the model")
        groups.append({"index": group_index, "learning_rate": lr, "weight_decay": wd,
            "epsilon": eps, "betas": [beta1, beta2], "amsgrad": bool(group.get("amsgrad", False)),
            "parameter_names": actual_names})
        for name, p in zip(actual_names, group["params"], strict=True):
            if name in covered or p.dtype != torch.float32:
                raise ValueError("AdamW decomposition requires unique FP32 parameters")
            covered.append(name)
            i = index[name]
            saved = optimizer.state[p]
            step = float(saved["step"])
            if step < 1 or step != int(step):
                raise ValueError("Invalid actual post-update AdamW step")
            steps.add(int(step))
            g = current_gradients[i].detach().cpu().double()
            m = saved["exp_avg"].detach().cpu().double() / (1 - beta1 ** step)
            variance_name = "max_exp_avg_sq" if group.get("amsgrad", False) else "exp_avg_sq"
            variance = saved[variance_name].detach().cpu().double() / (1 - beta2 ** step)
            if bool((variance < 0).any()):
                raise ValueError("Negative AdamW variance")
            divisor = variance.sqrt() + eps
            prior = before_parameters[name].detach().cpu().double()
            parts = {"raw_gradient": -lr * g,
                "first_moment_without_preconditioner": -lr * m,
                "current_gradient_with_actual_divisor": -lr * g / divisor,
                "native_adaptive_term": -lr * m / divisor,
                "decoupled_weight_decay": -lr * wd * prior,
                "observed_adamw": delta[i].detach().cpu().double()}
            parts["reconstructed_adamw"] = parts["native_adaptive_term"] + parts["decoupled_weight_decay"]
            parts["rounding_residual"] = parts["observed_adamw"] - parts["reconstructed_adamw"]
            residual = parts["rounding_residual"]
            # Conservative FP32 operation-rounding envelope. The analytic
            # expression uses FP64 on actual FP32 moments; native CUDA uses
            # FP32 moment/bias arithmetic and rounds the final parameter add.
            bound = 8 * torch.finfo(torch.float32).eps * (prior.abs()
                + parts["native_adaptive_term"].abs() + parts["decoupled_weight_decay"].abs())
            bound = bound.clamp_min(torch.finfo(torch.float32).tiny)
            if bool((residual.abs() > bound).any()):
                raise RuntimeError("AdamW algebra does not reconstruct the observed delta within the FP32 rounding envelope")
            maximum_error = max(maximum_error, float(residual.abs().max()))
            maximum_relative_bound = max(maximum_relative_bound, float((residual.abs() / bound).max()))
            squared_error += float(residual.square().sum())
            actual_squared += float(parts["observed_adamw"].square().sum())
            elements += residual.numel()
            for label, direction in parts.items():
                if not bool(torch.isfinite(direction).all()):
                    raise FloatingPointError("Nonfinite AdamW analytic direction")
                accumulated[label]["squared_norm"].append(float(direction.square().sum()))
                for metric, vectors in metric_vectors.items():
                    accumulated[label]["dots"][metric].append(float((vectors[i].double() * direction).sum()))
    if not covered:
        raise ValueError("The AdamW route is empty")
    metric_norms = {metric: math.sqrt(math.fsum(float(vectors[index[name]].double().square().sum())
                                              for name in covered)) for metric, vectors in metric_vectors.items()}
    result = {}
    for label, values in accumulated.items():
        norm = math.sqrt(math.fsum(values["squared_norm"]))
        metric_reports = {}
        for metric, dots in values["dots"].items():
            dot = math.fsum(dots)
            metric_norm = metric_norms[metric]
            metric_reports[metric] = {"first_order_change": dot,
                "unit_direction_derivative": dot / norm if norm else None,
                "direction_cosine_with_metric_gradient": dot / (norm * metric_norm) if norm and metric_norm else None,
                "effect": "decrease" if dot < 0 else "increase" if dot > 0 else "zero_first_derivative"}
        result[label] = {"direction_norm": norm, "metrics": metric_reports}
    for metric in metric_vectors:
        observed = result["observed_adamw"]["metrics"][metric]["first_order_change"]
        restored = math.fsum(result[k]["metrics"][metric]["first_order_change"]
                             for k in ("native_adaptive_term", "decoupled_weight_decay", "rounding_residual"))
        if not math.isclose(observed, restored, rel_tol=1e-10, abs_tol=1e-14):
            raise RuntimeError("AdamW decomposition metric dots do not sum to the observed route")
    return {"directions": result, "groups": groups, "actual_post_update_steps": sorted(steps),
        "parameters": len(covered), "parameter_elements": elements,
        "reconstruction": {"passed": True, "maximum_absolute_delta_error": maximum_error,
            "rms_delta_error": math.sqrt(squared_error / elements),
            "relative_l2_delta_error": math.sqrt(squared_error / actual_squared) if actual_squared else None,
            "maximum_fraction_of_fp32_envelope": maximum_relative_bound,
            "tolerance_policy": "Per element: 8*FP32_epsilon*(abs(prior_parameter)+abs(adaptive_delta)+abs(decay_delta)), floored at FP32_tiny. Rounding residual and its metric dots are retained."},
        "formula": "mhat=actual_poststep_exp_avg/(1-beta1**step); divisor=sqrt(actual_poststep_exp_avg_sq/(1-beta2**step))+eps; native_adaptive=-lr*mhat/divisor; decay=-lr*wd*prior_parameter",
        "counterfactual_scope": "Analytical directions on only the same AdamW parameters, using actual post-step states and actual group rates. Removing a term is not a tested optimizer or a recommended reset. The fixed divisor already includes the current gradient observation.",
        "normalization": "All formulas use actual group learning rates; unit-L2 direction derivatives permit direction comparison despite different norm scales.",
        "additional_optimizer_steps": 0, "additional_model_forwards": 0}


def native_chain_once(engine, training, probes, *, global_rng=None, seed=1861):
    original_hash = state_fingerprint(engine.state_dict())
    saved_crop_rng = engine.crop_generator.get_state().clone()
    named = tuple(engine.model.named_parameters())
    parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
    old_grads = [(p.grad, p.grad.detach().clone() if p.grad is not None else None) for p in parameters]
    captured = {}
    started = time.monotonic()
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as guard:
        engine.crop_generator.set_state(saved_crop_rng)
        state = _capture_replay_state(engine)
        try:
            extension = None
            if engine.step == engine.recipe.total_steps:
                previous_rate = engine.learning_rate()
                extension = {"recipe_total_steps_before": engine.recipe.total_steps,
                    "config_total_steps_before": engine.config.total_steps,
                    "total_steps_during_disposable_probe": engine.step + 1,
                    "scope": "One diagnostic-only budget slot; both saved configurations restored afterward"}
                engine.recipe = replace(engine.recipe, total_steps=engine.step + 1)
                engine.config = replace(engine.config, total_steps=engine.step + 1)
                if engine.learning_rate() != previous_rate:
                    raise ValueError("A budget-only extension changed the learning rate")
            before, metric_vectors = probe_snapshot(engine, probes, gradients=True)
            engine.crop_generator.set_state(saved_crop_rng)
            if global_rng is not None:
                _restore_rng(deepcopy(global_rng))
            with intercept_step(engine.optimizer, named, captured):
                training_metrics = engine.train_step(training)
            if captured.get("calls") != 1 or engine.step != state["step"] + 1:
                raise RuntimeError("Expected exactly one native generator update")
            if engine.discriminator_updates != state["discriminator_updates"] + 1:
                raise RuntimeError("Expected exactly one native discriminator update")
            if not all(math.isfinite(v) for v in training_metrics.values()):
                raise FloatingPointError("Nonfinite native training metrics")
            changed_buffers = [n for n, v in engine.model.named_buffers()
                               if not torch.equal(v.detach().cpu(), state["model"][n])]
            if changed_buffers:
                raise RuntimeError("Native update changed frozen model buffers")
            delta = tuple(p.detach().cpu() - state["model"][n] for n, p in named)
            raw = tuple(-g for g in captured["gradient"])
            raw_report = direction_report(metric_vectors, raw)
            native_report = direction_report(metric_vectors, delta)
            localization = additive_localization(named, metric_vectors, raw, delta,
                engine.optimizer.group_manifest, {"raw_negative_postclip_gradient": raw_report,
                                                "native_parameter_delta": native_report})
            adamw_parts = adamw_analytic_decomposition(named, captured["gradient"], delta,
                state["model"], engine.optimizer.optimizers["adamw"], metric_vectors)
            after, _ = probe_snapshot(engine, probes)
            if before["denominators"] != after["denominators"]:
                raise RuntimeError("Teacher-defined diagnostic masks changed")
            changes = {k: after["metrics"][k] - before["metrics"][k] for k in METRICS}
            for k in METRICS:
                native_report["metrics"][k].update(actual_change=changes[k],
                    actual_relative_change_percent=100 * changes[k] / before["metrics"][k] if before["metrics"][k] else None,
                    actual_minus_first_order=changes[k] - native_report["metrics"][k]["first_order_change"])
            denom = raw_report["direction_norm"] * native_report["direction_norm"]
            answer = {"before": before, "after": after, "raw_negative_postclip_gradient": raw_report,
                "native_parameter_delta": native_report,
                "additive_first_order_localization": localization,
                "adamw_analytic_decomposition": adamw_parts,
                "raw_vs_native_cosine": vector_dot(raw, delta) / denom if denom else None,
                "training_metrics": training_metrics, "D_views": deepcopy(getattr(engine, "screen_last_views", [])),
                "generator_updates": captured["calls"], "discriminator_updates": 1,
                "disposable_budget_extension": extension,
                "model_buffers_unchanged": True, "fixed_state": guard,
                "metric_aggregation": "Sample-pooled full-scale excess MSE; sample-pooled teacher-window quiet residual MSE; encoded-zero and natural-only quiet MSE separately. Natural excludes encoded_ fixtures. Per-crop values also retained.",
                "gradient_space": "Raw negative parameter gradient observed after global norm clipping; direction is unchanged by this scalar clip. Includes actual updated D and current-batch balancer observation.",
                "native_space": "Actual quarter-rate native Muon/AdamW displacement, including saved moments, weight decay and preconditioning; no optimizer substitution.",
                "interpretation": "Signs identify local shared-parameter/batch direction, native transformation and finite-step effects for this single batch only. This does not assign causality to one optimizer family or prove long-run effects.",
                "rng_policy": "Saved checkpoint crop RNG; checkpoint global RNG restored before native step" if global_rng is not None else "Saved crop RNG and diagnostic seeded global RNG",
                "checkpoint_files_written": 0, "retained_updates": 0, "seconds": time.monotonic() - started}
        finally:
            _restore_replay_state(engine, state)
            for p, (gradient, saved) in zip(parameters, old_grads, strict=True):
                p.grad = gradient
                if gradient is not None:
                    gradient.copy_(saved)
    if state_fingerprint(engine.state_dict()) != original_hash:
        raise RuntimeError("Native-chain diagnostic failed to restore the exact engine")
    answer["state_restored"] = True
    return answer


def run(ctx, args):
    from diagnostic_common import atomic_json, status
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != QUARTER_SHA or file_sha(args.training_receipt) != FRESH_RECEIPT_SHA:
        raise ValueError("Quarter checkpoint or fresh32 receipt does not match the pinned input")
    overlay = load_training_overlay(ctx, args.training_receipt, required_counts={"gradient_calibration": 32})
    if overlay["identity"]["cache_sha256"] != FRESH_CACHE_SHA:
        raise ValueError("Fresh32 reference cache differs")
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, args.canonical_receipt)
    selection = json.loads(Path(args.selection).read_text())
    probes = select_probes(panel["crops"], selection)
    training = overlay["pools"]["gradient_calibration"]
    if {c.source_id for c in training} & {c.source_id for c in probes}:
        raise ValueError("Training and diagnostic probe sources overlap")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "rng" not in payload:
        raise ValueError("The quarter checkpoint must preserve global RNG")
    engine = ctx.engine("targeted", device="cuda")
    restored_hash = restore_quarter(engine, payload["engine"])
    bind_views(engine, ctx.data["pools"]["gradient_calibration"])
    import gradient_probe, diagnose_objectives_updates, training_overlay, run_corrected_screen
    import run_update_experiment, canonical_evaluation, render_canonical_targets, evaluation_audit
    import audiovae_student.recipe_v2 as recipe_module
    files = {Path(__file__).resolve(), Path(importlib.import_module(type(ctx).__module__).__file__).resolve()}
    files.update(Path(m.__file__).resolve() for m in (gradient_probe, diagnose_objectives_updates, training_overlay,
        run_corrected_screen, run_update_experiment, canonical_evaluation, render_canonical_targets, evaluation_audit))
    files.update(Path(recipe_module.__file__).parent.glob("*.py"))
    identity = {"format_version": 1, "checkpoint_path": str(checkpoint), "checkpoint_sha256": QUARTER_SHA,
        "restored_engine_sha256": restored_hash, "step": 8890, "training_overlay": overlay["identity"],
        "canonical_panel": panel["identity"], "selection_path": str(Path(args.selection).resolve()),
        "selection_sha256": file_sha(args.selection), "probes": [{"source_id": c.source_id, "start_frame": c.start_frame} for c in probes],
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "runtime": {"device": str(engine.device), "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "source_hashes": {str(p): file_sha(p) for p in sorted(files)},
        "scope": "One disposable B32 native D-then-G update at saved8890 quarter rate; no retained model changes or checkpoint writes"}
    atomic_json(out / "identity.json", identity)
    status("native_chain_start", step=8890, probes=len(probes))
    report = native_chain_once(engine, training, probes, global_rng=payload["rng"])
    report["identity"] = identity
    report["original_files"] = ctx.verify_files()
    report["fresh_files"] = verify_overlay_files(overlay)
    for path, expected in ((checkpoint, QUARTER_SHA), (panel["identity"]["receipt_path"], panel["identity"]["receipt_sha256"]),
                           (panel["identity"]["cache_path"], panel["identity"]["cache_sha256"]), (args.selection, identity["selection_sha256"])):
        if file_sha(path) != expected:
            raise RuntimeError("An immutable input file changed")
    atomic_json(out / "native-chain.json", report)
    status("native_chain_complete", path=str(out / "native-chain.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--canonical-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context-module", default="diagnostic_common")
    args = parser.parse_args()
    lock = Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(importlib.import_module(args.context_module).load_context(), args)
