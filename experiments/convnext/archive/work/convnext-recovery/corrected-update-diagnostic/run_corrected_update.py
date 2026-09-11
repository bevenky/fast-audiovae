"""Narrow corrected-domain loss and update-size diagnostic; no retained update."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import importlib
import math
from pathlib import Path

import torch

from audiovae_student.corrected_calibration import _fixed_engine
from audiovae_student.objective_comparison import state_fingerprint
from diagnose_objectives_updates import (_capture_replay_state, _restore_replay_state,
    _preserved_attributes, _probe_snapshot, objective_geometry, vector_dot)
from diagnose_step_path import one_step_path, mean_metrics
from run_corrected_screen import bind_views
from run_update_experiment import fork_generator_rate, load_canonical_panel
from training_overlay import load_training_overlay, verify_overlay_files


def source_identity(ctx):
    from audiovae_student.restart_data import file_sha
    import audiovae_student.recipe_v2 as recipe_module
    modules = [importlib.import_module(name) for name in
        ("diagnose_objectives_updates", "diagnose_step_path", "run_corrected_screen", "run_update_experiment",
         "training_overlay", "canonical_evaluation", "render_canonical_targets", "evaluation_audit", type(ctx).__module__)]
    files = {Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules)}
    files.update(Path(recipe_module.__file__).parent.glob("*.py"))
    return {str(path): file_sha(path) for path in sorted(files)}


def canonical_probes(ctx, panel):
    lookup = {(crop.source_id, crop.start_frame): crop for crop in panel["crops"]}
    if len(lookup) != len(panel["crops"]):
        raise ValueError("Duplicate canonical crop keys")
    probes = [lookup[(crop.source_id, crop.start_frame)] for crop in ctx.probe_crops]
    if len(probes) != len(ctx.probe_crops):
        raise ValueError("Canonical probe panel is incomplete")
    return probes


def native_rate_once(engine, training_crops, probes, *, factor, seed=1861):
    """Actual native D-then-G step; restore the exact input engine before exit."""
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as guard:
        state = _capture_replay_state(engine)
        before_hash = state_fingerprint(state)
        all_parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
        old_grads = [(p.grad, p.grad.detach().clone() if p.grad is not None else None) for p in all_parameters]
        try:
            migration = fork_generator_rate(engine, factor)
            baseline_rows, _ = _probe_snapshot(engine, probes)
            baseline = mean_metrics(baseline_rows)
            engine.crop_generator.set_state(state["crop_rng"].clone())
            metrics = engine.train_step(training_crops)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite native-step metrics")
            buffers = [name for name, value in engine.model.named_buffers()
                       if not torch.equal(value.detach().cpu(), state["model"][name])]
            if buffers:
                raise RuntimeError("Native step changed fixed model buffers")
            rows, _ = _probe_snapshot(engine, probes)
            observed = mean_metrics(rows)
            delta = tuple(parameter.detach().cpu() - state["model"][name]
                          for name, parameter in engine.model.named_parameters())
            changes = {name: observed[name] - baseline[name] for name in observed}
            result = {"factor": factor, "migration": migration, "baseline": baseline,
                "baseline_rows": baseline_rows, "after": observed, "after_rows": rows,
                "absolute_change": changes,
                "relative_change_percent": {name: value / baseline[name] * 100 if baseline[name] else None
                                            for name, value in changes.items()},
                "training_metrics": metrics, "parameter_delta_norm": math.sqrt(vector_dot(delta, delta)),
                "D_views": deepcopy(getattr(engine, "screen_last_views", [])), "fixed_state": guard,
                "interpretation": "Actual joint generator-rate fork; same starting moments/EMA and native discriminator step, all state discarded/restored"}
        finally:
            _restore_replay_state(engine, state)
            for parameter, (gradient, saved) in zip(all_parameters, old_grads):
                parameter.grad = gradient
                if gradient is not None:
                    gradient.copy_(saved)
        if state_fingerprint(_capture_replay_state(engine)) != before_hash:
            raise RuntimeError("Native rate diagnostic did not restore the original engine")
    result.update(state_restored=True, retained_updates=0, disposable_generator_updates=1)
    return result


def run(ctx, training_receipt, canonical_receipt, out):
    from diagnostic_common import atomic_json, status
    from audiovae_student.restart_data import file_sha
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity.json").exists():
        raise FileExistsError("Use a fresh versioned diagnostic directory")
    overlay = load_training_overlay(ctx, training_receipt, required_counts={"gradient_calibration": 32})
    panel = load_canonical_panel(ctx, {"requires_canonical_evaluation": True}, canonical_receipt)
    train = overlay["pools"]["gradient_calibration"]
    probes = canonical_probes(ctx, panel)
    if {c.source_id for c in train} & {c.source_id for c in panel["crops"]}:
        raise ValueError("Diagnostic training and canonical validation sources overlap")
    identity = {"format_version": 1, "checkpoint": "targeted", "step": 8490,
        "checkpoint_sha256": ctx.expected_hashes["targeted"], "training_overlay": overlay["identity"],
        "canonical_panel": panel["identity"], "probes": [{"source_id": c.source_id, "start_frame": c.start_frame} for c in probes],
        "torch": str(torch.__version__), "cudnn": torch.backends.cudnn.version(),
        "source_sha256": file_sha(__file__), "sources": source_identity(ctx), "retained_updates": 0,
        "scope": "Five-point teacher-output interpolation, one fixed native delta path, actual current/quarter native steps; no full routing or leave-one-loss-out sweep"}
    atomic_json(out / "identity.json", identity)
    engine = ctx.engine("targeted")
    if engine.step != 8490:
        raise ValueError("Targeted step8490 required")
    bind_views(engine, ctx.data["pools"]["gradient_calibration"])
    initial = state_fingerprint(engine.state_dict())
    answer = {"identity": identity}
    status("corrected_loss_geometry", crops=len(probes))
    objective_crops = [crop for crop in probes if crop.valid_scored_samples >= engine.recipe.adversarial_samples]
    answer["objective_geometry"] = objective_geometry(engine, objective_crops)
    atomic_json(out / "loss-geometry.json", answer["objective_geometry"])
    status("corrected_native_delta_path")
    answer["fixed_native_delta_path"] = one_step_path(engine, train, probes)
    atomic_json(out / "native-step-path.json", answer["fixed_native_delta_path"])
    answer["native_rates"] = {}
    for name, factor in (("current", 1.), ("quarter", .25)):
        status("corrected_actual_native_rate", name=name)
        answer["native_rates"][name] = native_rate_once(engine, train, probes, factor=factor)
        atomic_json(out / ("native-rate-" + name + ".json"), answer["native_rates"][name])
    if answer["native_rates"]["current"]["D_views"] != answer["native_rates"]["quarter"]["D_views"]:
        raise RuntimeError("Actual rate diagnostics use different discriminator views")
    if state_fingerprint(engine.state_dict()) != initial:
        raise RuntimeError("Corrected diagnostic changed retained engine state")
    answer["state_restored"] = True
    answer["original_files"] = ctx.verify_files()
    answer["fresh_overlay_files"] = verify_overlay_files(overlay)
    if file_sha(panel["identity"]["receipt_path"]) != panel["identity"]["receipt_sha256"] or file_sha(panel["identity"]["cache_path"]) != panel["identity"]["cache_sha256"]:
        raise RuntimeError("Canonical panel files changed")
    answer["continuation_decision"] = "Review corrected results under the already authorized conditional400-step plan; this diagnostic does not launch continuation"
    atomic_json(out / "corrected-update-diagnostic.json", answer)
    del engine
    gc.collect(); torch.cuda.empty_cache()
    status("corrected_update_diagnostic_complete", path=str(out / "corrected-update-diagnostic.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--canonical-receipt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context-module", default="diagnostic_common")
    args = parser.parse_args()
    run(importlib.import_module(args.context_module).load_context(), args.training_receipt, args.canonical_receipt, args.out)
