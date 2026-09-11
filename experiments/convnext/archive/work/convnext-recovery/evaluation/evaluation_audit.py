"""Fixed-panel recovery reporting; no training or checkpoint writes.

build_evaluation invokes the existing evaluator exactly once. All other helpers
operate solely on its JSON report. Pilot margins are engineering screening
limits, not audibility thresholds or proof of final perceptual equivalence.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

FIXTURES = frozenset({"encoded_zero", "encoded_quiet_noise", "encoded_fade"})
GATES = {
    "natural_quiet_ratio_vs_control_max": .90,
    "natural_quiet_ratio_vs_baseline_max": 1.0,
    "encoded_zero_steady_ratio_vs_control_max": 1.0,
    "encoded_zero_steady_ratio_vs_baseline_max": 1.0,
    "natural_mae_and_mel_ratio_max": 1.01,
    "speech_expressive_mae_and_mel_ratio_max": 1.02,
    "event_language_mae_and_mel_ratio_max": 1.05,
    "minimum_sources_for_group_gate": 2,
    "maximum_peak_ratio_max": 1.01,
    "peak_excess_energy_ratio_max": 1.10,
}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _quiet(rows, *, steady_start=None):
    windows = [w for r in rows for w in r["quiet_windows"]["windows"]
               if w["is_quiet"] and (steady_start is None or w["start_sample"] >= steady_start)]
    samples = sum(w["valid_samples"] for w in windows)
    return {"quiet_windows": len(windows), "quiet_samples": samples,
            "quiet_failed_windows": sum(w["passed"] is False for w in windows),
            "quiet_residual_rms": math.sqrt(sum(w["residual_rms"] ** 2 * w["valid_samples"]
                                                for w in windows) / samples) if samples else None,
            "quiet_student_rms": math.sqrt(sum(w["student_rms"] ** 2 * w["valid_samples"]
                                               for w in windows) / samples) if samples else None,
            "quiet_teacher_rms": math.sqrt(sum(w["teacher_rms"] ** 2 * w["valid_samples"]
                                               for w in windows) / samples) if samples else None}


def _aggregate(rows):
    samples = sum(r["samples"] for r in rows)
    active = [r for r in rows if r["teacher_rms"] >= .001 and r["waveform_cosine_defined"]]
    return {"crops": len(rows), "sources": len({r["source_id"] for r in rows}),
            "samples": samples,
            "raw_mae": sum(r["waveform_raw_mae"] * r["samples"] for r in rows) / samples,
            "mel": sum(r["teacher_mel"] for r in rows) / len(rows),
            "nonquiet_cosine_mean": sum(r["waveform_cosine"] for r in active) / len(active) if active else None,
            "nonquiet_crops": len(active),
            "maximum_peak": max(r["student_peak_abs"] for r in rows),
            "peak_excess_energy": (sum(r["peak_excess_energy"] * r["samples"] for r in rows) / samples
                                   if all("peak_excess_energy" in r for r in rows) else None),
            "scored_overshoot_samples": sum(r["student_overshoot_samples"] for r in rows),
            "overshoot_crops": sum(r["student_overshoot_samples"] > 0 for r in rows),
            "overshoot_sources": len({r["source_id"] for r in rows if r["student_overshoot_samples"]}),
            "near_saturation_samples": sum(r["student_near_saturation_samples"] for r in rows),
            **_quiet(rows)}


def _contract(report):
    """Bind target-derived masks and source accounting, not student values."""
    rows = []
    for r in report["rows"]:
        windows = [{k: w[k] for k in ("window_index", "start_sample", "stop_sample", "valid_samples",
                                      "teacher_rms", "is_quiet", "residual_limit", "output_rms_limit")}
                   for w in r["quiet_windows"]["windows"]]
        rows.append({k: r[k] for k in ("source_id", "start_frame", "context_start_frame",
                                      "metadata", "cohort", "samples", "original_scored_samples",
                                      "context_excluded_samples", "absolute_scored_start_sample",
                                      "teacher_rms", "teacher_peak_abs")}
                    | {"teacher_quiet_windows": windows})
    rows.sort(key=lambda r: (r["source_id"], r["start_frame"]))
    return {"criterion_config": report["criterion_config"], "quiet_config": report["quiet_config"],
            "mask_policy": report["mask_policy"], "rows": rows}


def decorate_evaluation(report):
    result = deepcopy(report)
    rows = result["rows"]
    keys = [(r["source_id"], r["start_frame"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source/start-frame evaluation crop")
    if len(rows) != 285 or len({r["source_id"] for r in rows}) != 147:
        raise ValueError("Recovery requires the complete sealed 285-crop / 147-source panel")
    fixture_rows = [r for r in rows if r["source_id"] in FIXTURES]
    if len(fixture_rows) != 3 or {r["source_id"] for r in fixture_rows} != FIXTURES:
        raise ValueError("Missing or repeated encoded fixture")
    if any((r["metadata"].get("dataset") == "synthetic_fixture") != (r["source_id"] in FIXTURES) for r in rows):
        raise ValueError("Synthetic classification differs from the sealed fixture identities")
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        natural = r["source_id"] not in FIXTURES
        groups["natural" if natural else "synthetic"].append(r)
        groups["source/" + r["source_id"]].append(r)
        if natural:
            groups["cohort/" + r["cohort"]].append(r)
            for field in ("language", "condition"):
                value = str(r["metadata"].get(field) or "unverified").strip().casefold().replace(" ", "_")
                groups[field + "/" + value].append(r)
    metrics = {k: _aggregate(v) for k, v in groups.items()}
    source_metrics = {k.removeprefix("source/"): v for k, v in metrics.items() if k.startswith("source/")}
    metrics["natural"]["source_equal_mean_raw_mae"] = sum(
        v["raw_mae"] for k, v in source_metrics.items() if k not in FIXTURES) / 144
    metrics["natural"]["source_equal_mean_mel"] = sum(
        v["mel"] for k, v in source_metrics.items() if k not in FIXTURES) / 144
    if (metrics["all"]["samples"] != 26206830 or metrics["natural"]["samples"] != 25342830
            or metrics["natural"]["crops"] != 282 or metrics["natural"]["sources"] != 144
            or metrics["all"]["quiet_windows"] != 4073 or metrics["natural"]["quiet_windows"] != 3434
            or metrics["synthetic"]["quiet_windows"] != 639
            or sum(r["context_excluded_samples"] for r in rows) != 618):
        raise ValueError("Recovery panel counts, quiet masks, or context exclusions changed")
    zero = [r for r in fixture_rows if r["source_id"] == "encoded_zero"]
    if zero[0]["context_start_frame"] != 0 or zero[0]["samples"] != 288000:
        raise ValueError("Encoded-zero fixture must retain its complete six-second trajectory")
    steady = _quiet(zero, steady_start=96000)
    if steady["quiet_samples"] != 192000 or steady["quiet_windows"] != 200:
        raise ValueError("Encoded-zero steady slice must be exactly seconds 2 through 6")
    result["recovery_metrics"] = metrics
    result["encoded_zero_steady_2_to_6_seconds"] = steady
    result["evaluation_contract_sha256"] = _json_hash(_contract(report))
    result["recovery_interpretation"] = {
        "mae": "sample-weighted raw waveform MAE",
        "mel": "unchanged legacy diagnostic criterion; equal crop mean",
        "quiet": "teacher-defined 20ms windows; residual power pooled by valid samples, then square root",
        "peak_counts": "scored sample observations; overlapping crops are not deduplicated physical events",
        "group_gates": "small-cohort screening only, not language or expression quality certification",
        "unknown_groups_retained": True,
    }
    json.dumps(result, allow_nan=False)
    return result


def build_evaluation(engine, heldout, metadata):
    """Collect peak energy during the evaluator's original forwards, without replay."""
    import torch
    from audiovae_student.fusion_evaluation import evaluate_fusion
    heldout = tuple(heldout)
    peak_rows = []

    def capture_peak_energy(module, inputs, output):
        index = len(peak_rows)
        if index >= len(heldout):
            raise ValueError("Evaluator performed an unexpected extra model forward")
        crop = heldout[index]
        excluded = 6 if crop.context_start_frame > 0 else 0
        selected = output[..., crop.scored_slice.start + excluded:crop.scored_slice.stop].detach().double()
        target = crop.teacher_audio[..., crop.scored_slice.start + excluded:crop.scored_slice.stop]
        if not bool(torch.isfinite(selected).all()) or float(target.abs().max()) > 1:
            raise ValueError("Nonfinite prediction or changed teacher amplitude contract")
        peak_rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                          "peak_excess_energy": float((selected.abs() - 1).clamp_min(0).square().mean())})

    handle = engine.model.register_forward_hook(capture_peak_energy)
    try:
        report = evaluate_fusion(engine, heldout, metadata)
    finally:
        handle.remove()
    if len(peak_rows) != len(report["rows"]):
        raise ValueError("Peak capture missed an evaluation crop")
    for row, peak in zip(report["rows"], peak_rows):
        if (row["source_id"], row["start_frame"]) != (peak["source_id"], peak["start_frame"]):
            raise ValueError("Peak-energy capture does not match evaluator ordering")
        row["peak_excess_energy"] = peak["peak_excess_energy"]
    report["peak_excess_energy_definition"] = "sample mean of max(abs(student)-1,0)^2; all teacher samples <=1"
    return decorate_evaluation(report)


def _ratio(value, baseline):
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return None
    return value / baseline


def compare_reports(before, control, candidate):
    reports = [r if "recovery_metrics" in r else decorate_evaluation(r)
               for r in (before, control, candidate)]
    before, control, candidate = reports
    if len({r["evaluation_contract_sha256"] for r in reports}) != 1:
        raise ValueError("Before/control/candidate target, grouping, masks or criterion differ")
    if before["evaluated_step"] != 8490 or any(r["evaluated_step"] != 8890 for r in reports[1:]):
        raise ValueError("The registered comparison is step8490 plus exactly400 generator updates")
    checks = []

    def check(name, value, limit, *, details=None, strict=False):
        checks.append({"name": name, "value": value, "maximum": limit,
                       "comparison": "<" if strict else "<=",
                       "passed": value is not None and math.isfinite(value)
                       and (value < limit if strict else value <= limit),
                       "details": details})

    c = candidate["recovery_metrics"]
    for label, reference in (("baseline", before), ("control", control)):
        r = reference["recovery_metrics"]
        limit = GATES["natural_quiet_ratio_vs_" + label + "_max"]
        check("natural_quiet_vs_" + label, _ratio(c["natural"]["quiet_residual_rms"], r["natural"]["quiet_residual_rms"]), limit,
              strict=label == "baseline")
        check("encoded_zero_steady_vs_" + label,
              _ratio(candidate["encoded_zero_steady_2_to_6_seconds"]["quiet_residual_rms"],
                     reference["encoded_zero_steady_2_to_6_seconds"]["quiet_residual_rms"]),
              GATES["encoded_zero_steady_ratio_vs_" + label + "_max"], strict=True)
        for metric in ("raw_mae", "mel"):
            check("natural_" + metric + "_vs_" + label, _ratio(c["natural"][metric], r["natural"][metric]), GATES["natural_mae_and_mel_ratio_max"])
    baseline = before["recovery_metrics"]
    for group in ("cohort/speech", "cohort/expressive"):
        if group not in c:
            raise ValueError("Required natural speech/expressive cohort missing")
        for metric in ("raw_mae", "mel"):
            check(group + "/" + metric, _ratio(c[group][metric], baseline[group][metric]), GATES["speech_expressive_mae_and_mel_ratio_max"])
    group_changes = []
    for group in sorted(c):
        if not group.startswith(("language/", "condition/")):
            continue
        relative = {metric: _ratio(c[group][metric], baseline[group][metric]) for metric in ("raw_mae", "mel")}
        enough = c[group]["sources"] >= GATES["minimum_sources_for_group_gate"]
        group_changes.append({"group": group, "sources": c[group]["sources"], "crops": c[group]["crops"],
                              "gated": enough, "ratios_vs_baseline": relative})
        if enough:
            for metric, value in relative.items():
                check(group + "/" + metric, value, GATES["event_language_mae_and_mel_ratio_max"])
    check("natural_maximum_peak_vs_baseline", _ratio(c["natural"]["maximum_peak"], baseline["natural"]["maximum_peak"]), GATES["maximum_peak_ratio_max"])
    check("natural_peak_excess_energy_vs_baseline", _ratio(c["natural"]["peak_excess_energy"], baseline["natural"]["peak_excess_energy"]), GATES["peak_excess_energy_ratio_max"])
    source_changes = []
    for group in sorted(c):
        if not group.startswith("source/") or group.removeprefix("source/") in FIXTURES:
            continue
        source_changes.append({"source_id": group.removeprefix("source/"), "crops": c[group]["crops"],
            "samples": c[group]["samples"], "ratios_vs_baseline": {
                metric: _ratio(c[group][metric], baseline[group][metric])
                for metric in ("raw_mae", "mel", "quiet_residual_rms", "maximum_peak", "peak_excess_energy")}})
    return {"format_version": 1, "gates": GATES, "gates_sha256": _json_hash(GATES),
            "evaluation_contract_sha256": before["evaluation_contract_sha256"], "checks": checks,
            "screen_passed": all(x["passed"] for x in checks), "group_changes": group_changes,
            "source_changes": source_changes,
            "undefined_ratio_policy": "None; never silently passes a required gate",
            "meaning": "Bounded pilot continuation screen; not final quality equivalence, no automatic deployment",
            "retained_checkpoints_unchanged_by_reporting": True}
