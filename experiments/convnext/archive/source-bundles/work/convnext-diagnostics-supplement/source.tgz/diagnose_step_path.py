"""Measure a frozen line segment along one exact disposable native update.

This is not a learning-rate sweep. The optimizer is executed once at its saved
learning rate; subsequent probes interpolate parameters along that fixed delta.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import time

import torch

from audiovae_student.corrected_calibration import _fixed_engine
from audiovae_student.objective_comparison import state_fingerprint
from diagnose_objectives_updates import (_capture_replay_state, _restore_replay_state,
    _preserved_attributes, _probe_snapshot, vector_dot)


FRACTIONS = (0., .1, .25, .5, 1.)


@torch.no_grad()
def set_parameter_fraction(model, before, after, fraction, active_names=None):
    if not 0 <= fraction <= 1:
        raise ValueError("The diagnostic stays within the fixed update segment")
    for name, parameter in model.named_parameters():
        if before[name].shape != parameter.shape or after[name].shape != parameter.shape:
            raise ValueError("Parameter endpoint shape mismatch")
        if fraction == 0 or (active_names is not None and name not in active_names):
            value = before[name]
        elif fraction == 1:
            value = after[name]
        else:
            value = torch.lerp(before[name].double(), after[name].double(), fraction).to(parameter.dtype)
        parameter.copy_(value.to(parameter.device))


def mean_metrics(rows):
    names = ("teacher_mse", "quiet_residual_mse", "teacher_relative_peak_excess_energy", "waveform_mae")
    return {name: sum(row[name] for row in rows) / len(rows) for name in names}


def one_step_path(engine, training_crops, probe_crops, *, seed=1861):
    report = {"fractions": [], "seed": seed,
        "method": "parameter interpolation along one fixed native D-then-G delta",
        "native_training_step_calls": 1, "retained_updates": 0,
        "interpretation": "Not a Muon/AdamW learning-rate sweep: optimizer moments, clipping and weight decay are computed once at the saved rate.",
        "metric_aggregation": "equal-crop means on the same fixed probe panel, historical six-sample interior exclusion",
        "interior_rounding": "FP64 interpolation of FP32 endpoints, cast to original parameter dtype; exact endpoint copies"}
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as identity:
        state = _capture_replay_state(engine)
        identity_before = state_fingerprint(state)
        parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
        prior_grads = [(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters]
        named = tuple(engine.model.named_parameters())
        try:
            baseline_rows, diagnostic_gradients = _probe_snapshot(engine, probe_crops, with_gradient=True)
            baseline = mean_metrics(baseline_rows)
            # Preserve the exact diagnostic batch RNG after the baseline probe.
            engine.crop_generator.set_state(state["crop_rng"].clone())
            extended = engine.step >= engine.recipe.total_steps
            if extended:
                previous_lr = engine.learning_rate()
                engine.recipe = replace(engine.recipe, total_steps=engine.step + 1)
                engine.config = replace(engine.config, total_steps=engine.step + 1)
                if engine.learning_rate() != previous_lr:
                    raise ValueError("Budget-only extension changes the current learning rate")
            started = time.monotonic()
            train_metrics = engine.train_step(training_crops)
            changed_buffers = [name for name, buffer in engine.model.named_buffers()
                               if not torch.equal(buffer.detach().cpu(), state["model"][name])]
            if changed_buffers:
                raise RuntimeError("Native step changed model buffers: " + ", ".join(changed_buffers))
            after = {name: parameter.detach().cpu().clone() for name, parameter in named}
            delta = tuple(after[name] - state["model"][name] for name, _ in named)
            predicted_full = {name: vector_dot(gradient, delta) for name, gradient in diagnostic_gradients.items()}
            report.update(fixed_state=identity, baseline_rows=baseline_rows, baseline=baseline,
                training_metrics=train_metrics, disposable_budget_extension=extended,
                model_buffers_unchanged_after_native_step=True,
                fixed_delta_norm=vector_dot(delta, delta)**.5, first_order_full_delta_metric_change=predicted_full)
            for fraction in FRACTIONS:
                set_parameter_fraction(engine.model, state["model"], after, fraction)
                rows, _ = _probe_snapshot(engine, probe_crops)
                observed = mean_metrics(rows)
                changes = {name: observed[name] - baseline[name] for name in observed}
                first = {name: value * fraction for name, value in predicted_full.items()}
                report["fractions"].append({"fraction": fraction, "metrics": observed,
                    "absolute_change": changes,
                    "relative_change_percent": {name: 100 * change / baseline[name] if baseline[name] else None
                                                for name, change in changes.items()},
                    "first_order_change": first,
                    "actual_minus_first_order": {name: changes[name] - value for name, value in first.items()},
                    "rows": rows})
            if any(value != 0 for value in report["fractions"][0]["absolute_change"].values()):
                raise RuntimeError("The zero-displacement endpoint changed baseline audio")
            all_names = {name for name, _ in named}
            projection = {name for name in all_names if name.startswith("output.")}
            late = {name for name in all_names if name.startswith(("blocks.8.", "blocks.9."))}
            remaining = all_names - projection - late
            routes = {parameter["name"]: group["optimizer"] for group in engine.optimizer.group_manifest
                      for parameter in group["parameters"]}
            if set(routes) != all_names:
                raise RuntimeError("Optimizer routing does not cover exactly all student parameters")
            groups = {"output_projection": projection, "late_blocks_8_9": late,
                "remaining_parameters": remaining,
                "muon_parameters": {name for name in all_names if routes[name] == "muon"},
                "adamw_parameters": {name for name in all_names if routes[name] == "adamw"}}
            report["delta_localization"] = {}
            for label, active_names in groups.items():
                set_parameter_fraction(engine.model, state["model"], after, 1., active_names)
                rows, _ = _probe_snapshot(engine, probe_crops)
                observed = mean_metrics(rows)
                changes = {name: observed[name] - baseline[name] for name in observed}
                partial_delta = tuple(value if name in active_names else torch.zeros_like(value)
                                      for (name, _), value in zip(named, delta))
                first = {name: vector_dot(gradient, partial_delta) for name, gradient in diagnostic_gradients.items()}
                report["delta_localization"][label] = {
                    "parameter_names": sorted(active_names), "parameter_tensor_count": len(active_names),
                    "delta_norm": vector_dot(partial_delta, partial_delta)**.5,
                    "metrics": observed, "absolute_change": changes,
                    "relative_change_percent": {name: 100 * change / baseline[name] if baseline[name] else None
                                                for name, change in changes.items()},
                    "first_order_change": first,
                    "actual_minus_first_order": {name: changes[name] - value for name, value in first.items()},
                    "rows": rows}
            report["localization_interpretation"] = "Counterfactual audio effects of subsets of the same cached update, not additive responsibility percentages. The first three groups partition parameters; Muon and AdamW are a second partition."
            report["seconds"] = time.monotonic() - started
        finally:
            _restore_replay_state(engine, state)
            for parameter, (gradient, saved) in zip(parameters, prior_grads):
                parameter.grad = gradient
                if gradient is not None:
                    gradient.copy_(saved)
        if state_fingerprint(_capture_replay_state(engine)) != identity_before:
            raise RuntimeError("Step-path diagnostic changed retained engine state")
        report["state_restored"] = True
    return report


def run(ctx):
    from diagnostic_common import atomic_json, status
    from run_corrected_screen import bind_views
    answer = {"format_version": 1, "checkpoint_files_written": 0, "checkpoints": {}}
    previous_path = ctx.out / "objectives-updates.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    for name in ("parent", "targeted", "complex"):
        status("native_step_path", checkpoint=name)
        engine = ctx.engine(name, device="cuda")
        bind_views(engine, ctx.data["pools"]["gradient_calibration"])
        report = one_step_path(engine, tuple(ctx.pools["gradient_calibration"][:32]), tuple(ctx.probe_crops))
        if previous is not None:
            old = previous["checkpoints"][name]["native_updates"]["variants"]["exact_training_step"]["probe_after"]
            current = report["fractions"][-1]["rows"]
            if [(r["source_id"], r["start_frame"]) for r in old] != [(r["source_id"], r["start_frame"]) for r in current]:
                raise RuntimeError("The native-step endpoint comparison uses different probes")
            keys = ("teacher_mse", "quiet_residual_mse", "teacher_relative_peak_excess_energy", "waveform_mae", "peak_abs")
            mismatches = [{"source_id": a["source_id"], "metric": key, "previous": b[key], "current": a[key]}
                          for a, b in zip(current, old) for key in keys
                          if not math.isclose(a[key], b[key], rel_tol=1e-5, abs_tol=1e-12)]
            mismatches += [{"source_id": a["source_id"], "metric": "overshoot_samples",
                            "previous": b["overshoot_samples"], "current": a["overshoot_samples"]}
                           for a, b in zip(current, old) if a["overshoot_samples"] != b["overshoot_samples"]]
            report["previous_native_endpoint_comparison"] = {
                "source": str(previous_path), "matched": not mismatches, "mismatches": mismatches,
                "relative_tolerance": 1e-5, "absolute_tolerance": 1e-12,
                "scope": "Per-crop recorded metrics and exact overshoot counts; previous report does not retain sample waveforms."}
        else:
            report["previous_native_endpoint_comparison"] = {"matched": None, "reason": "Prior report unavailable"}
        answer["checkpoints"][name] = report
        atomic_json(ctx.out / ("native-step-path-" + name + ".json"), report)
        if previous is not None and not report["previous_native_endpoint_comparison"]["matched"]:
            raise RuntimeError("Parameter-path endpoint disagrees with the earlier native step; see saved mismatch report")
        del engine
        torch.cuda.empty_cache()
    atomic_json(ctx.out / "native-step-path.json", answer)
    status("native_step_path_complete")
    return answer


if __name__ == "__main__":
    from diagnostic_common import load_context
    run(load_context())
