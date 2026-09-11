"""Extend the existing diagnostic ridge grid without altering any model."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch

from diagnostic_common import load_context
from diagnose_features import (
    atomic_json, block_probe_metrics, collect_features, crop_key, fit_readout,
    probe_statistics, quality_from_cache, select_probe_crops,
)


GRID = (.01, .03, .1, .3, 1., 3., 10.)


def run(ctx):
    base = Path(ctx.outPath) / "features"
    out = Path(ctx.outPath) / "ridge-extension"
    out.mkdir(parents=True, exist_ok=True)
    previous_split = json.loads((base / "probe-split.json").read_text())
    selected, partition, split_report = select_probe_crops(
        ctx.pools["targeted_generator"], ctx.data["rows"], ctx.heldout)
    if partition != previous_split["source_partition"]:
        raise ValueError("The original source partition changed")
    if [list(crop_key(c)) for c in selected] != previous_split["windows"]:
        raise ValueError("The original fitting windows changed")
    started = time.monotonic()
    summary = {"format_version": 1, "relative_ridge_grid": GRID,
        "source_split": split_report, "parameter_updates": 0,
        "only_change": "Extend the same RMS-scaled diagnostic ridge family beyond original grid boundary",
        "arms": {}}
    for name in ("parent", "targeted", "complex"):
        engine = ctx.engine(name, device="cuda")
        original = json.loads((base / (name + ".json")).read_text())
        if engine.step != original["step"] or engine.model.config.to_dict() != original["model_config"]:
            raise ValueError("Checkpoint identity differs from the original probe")
        print(json.dumps({"ridge_extension": name, "phase": "collect"}), flush=True)
        training = collect_features(engine, selected, full=False)
        stats = probe_statistics(training, partition, "fit", engine.device)
        if stats["count"] != original["fit_features"]["blocks"]:
            raise ValueError("Complete fitting block selection changed")
        report = {"step": engine.step, "readouts": {}, "original_quality": original["original_quality"]["groups"]}
        heldout = None
        for intercept in (False, True):
            label = "affine" if intercept else "same_dimension_bias_free"
            grid, best = [], None
            for ridge in GRID:
                coefficient, bias, details = fit_readout(stats, ridge=ridge, intercept=intercept)
                tuning = block_probe_metrics(training, partition, "tune", coefficient, bias)
                if ridge == .01:
                    expected = next(c["tune"]["mse"] for c in original["readouts"][label]["grid"] if c["relative_ridge"] == .01)
                    if abs(tuning["mse"] - expected) > max(1e-12, abs(expected) * 1e-5):
                        raise ValueError("Original ridge=.01 tuning measurement did not reproduce")
                grid.append({**details, "tune": tuning})
                if best is None or tuning["mse"] < best[0]:
                    best = (tuning["mse"], coefficient, bias, details)
            _, coefficient, bias, details = best
            if heldout is None:
                heldout = collect_features(engine, ctx.heldout, full=True)
            quality = quality_from_cache(engine, heldout, ctx.metadata, coefficient=coefficient, bias=bias)
            report["readouts"][label] = {"grid": grid, "selected": details,
                "selection": "Lowest fitting-source-disjoint tuning MSE; never heldout",
                "selected_hits_upper_grid_boundary": details["relative_ridge"] == GRID[-1],
                "fit": block_probe_metrics(training, partition, "fit", coefficient, bias),
                "tune": block_probe_metrics(training, partition, "tune", coefficient, bias),
                "original_selected_blocks": original["original_selected_blocks"],
                "heldout": quality}
        atomic_json(out / (name + ".json"), report)
        summary["arms"][name] = {"step": engine.step,
            "original_quality": report["original_quality"],
            "readouts": {label: {key:value for key,value in value.items() if key != "heldout"}
                | {"heldout_groups": value["heldout"]["groups"]} for label,value in report["readouts"].items()}}
        del engine, training, stats, heldout, report, original, quality, coefficient, bias, best
        gc.collect(); torch.cuda.empty_cache()
    summary["seconds"] = time.monotonic() - started
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run(load_context())
