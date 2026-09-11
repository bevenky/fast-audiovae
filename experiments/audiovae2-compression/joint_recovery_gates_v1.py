"""Pure-Python review gates for a fixed teacher/student development panel.

These are conservative engineering review thresholds, not audibility limits,
statistical confidence tests, or instructions to freeze model parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

VERSION = "audiovae2_joint_recovery_review_gates_v1"


@dataclass(frozen=True)
class ReviewPolicy:
    expected_sources: int = 96
    error_regression_fraction: float = 0.05
    quiet_regression_fraction: float = 0.10
    quiet_regression_absolute: float = 1e-5
    source_waveform_regression_absolute: float = 1e-4
    level_error_regression_db: float = 0.5
    minimum_region_samples: int = 960
    error_progress_fraction: float = 0.002
    correlation_progress_absolute: float = 0.0001
    minimum_stall_reviews: int = 4
    bounded_peak_tolerance: float = 1e-6

    def __post_init__(self):
        for key in ("expected_sources", "minimum_region_samples", "minimum_stall_reviews"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key, value in asdict(self).items():
            if key not in ("expected_sources", "minimum_region_samples", "minimum_stall_reviews"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be positive and finite")


def _number(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its valid range")
    return float(value)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _check_finite_tree(value, path="report"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a nonfinite value")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{path}[{index}]")


def _level(student_energy, teacher_energy):
    if teacher_energy == 0:
        if student_energy != 0:
            raise ValueError("Active student energy exists without teacher-active samples")
        return {"rms_ratio": None, "absolute_error_db": None}
    if student_energy == 0:
        return {"rms_ratio": 0.0, "absolute_error_db": None}
    ratio = math.sqrt(student_energy / teacher_energy)
    return {"rms_ratio": ratio, "absolute_error_db": abs(20.0 * math.log10(ratio))}


def _snapshot(report, policy):
    _check_finite_tree(report)
    aggregate, rows = report["aggregate"], report["rows"]
    overview = report["overview_window_metrics"]
    windows = overview["by_source"]
    if len(rows) != policy.expected_sources or aggregate["sources"] != len(rows):
        raise ValueError("The fixed development source count changed")
    ids = [row["source_id"] for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Development source identities are invalid or duplicated")
    if set(ids) != set(windows):
        raise ValueError("Window observations do not match development sources")
    if _count(aggregate["overshoot_samples"], "overshoot_samples") != 0:
        raise ValueError("Bounded decoder has full-scale overshoot samples")
    peak = _number(aggregate["peak_abs_max"], "peak_abs_max", minimum=0)
    if peak > 1.0 + policy.bounded_peak_tolerance:
        raise ValueError("Bounded decoder peak invariant failed")
    current = {k: _number(aggregate[k], k, minimum=0) for k in
               ("mae", "mel", "group_mse", "quiet_residual_rms_mean")}
    correlation = _number(aggregate["nonquiet_cosine_mean"], "nonquiet_cosine_mean")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("Correlation is outside its valid range")
    current["correlation"] = correlation
    sources = {}
    totals = {k: 0 for k in ("samples", "quiet_windows", "quiet_failed", "near_windows", "near_failed")}
    teacher_energy = student_energy = 0.0
    for row in rows:
        source_id = row["source_id"]
        observed = windows[source_id]
        n = _count(row["samples"], "source samples")
        qn = _count(row["quiet_samples"], "quiet samples")
        an = _count(row["nonquiet_samples"], "active samples")
        qw = _count(row["quiet_windows"], "quiet windows")
        qf = _count(row["quiet_failed"], "quiet failures")
        nw = _count(observed["near_silence_windows"], "near-silence windows")
        nf = _count(observed["near_silence_failed_windows"], "near-silence failures")
        if (not n or qn + an != n or qf > qw or nf > nw or nw > qw
                or nf > qf or observed["valid_samples"] != n
                or observed["quiet_windows"] != qw or observed["quiet_failed_windows"] != qf):
            raise ValueError(f"Invalid sample/window accounting for {source_id}")
        if _count(row["overshoot"], "source overshoots") != 0:
            raise ValueError(f"Bounded output overshoot for {source_id}")
        te = _number(observed["active_teacher_energy"], "active teacher energy", minimum=0)
        pe = _number(observed["active_student_energy"], "active student energy", minimum=0)
        if (an == 0) != (te == 0):
            raise ValueError(f"Active energy and sample count disagree for {source_id}")
        qe = _number(row["quiet_error_sum"], "quiet error energy", minimum=0)
        if qn == 0 and qe != 0:
            raise ValueError("Quiet residual energy exists without quiet samples")
        sources[source_id] = {
            "mae": _number(row["mae"], "source mae", minimum=0),
            "mel": _number(row["mel"], "source mel", minimum=0),
            "quiet_rms": math.sqrt(qe / qn) if qn else None,
            "samples": n, "quiet_samples": qn, "active_samples": an,
            "quiet_windows": qw, "quiet_failed": qf, "near_windows": nw, "near_failed": nf,
            "teacher_active_energy": te, "level": _level(pe, te),
        }
        for key, value in (("samples", n), ("quiet_windows", qw), ("quiet_failed", qf),
                           ("near_windows", nw), ("near_failed", nf)):
            totals[key] += value
        teacher_energy += te
        student_energy += pe
    expected = (aggregate["samples"], aggregate["quiet_windows"], aggregate["quiet_failed_windows"],
                overview["near_silence_windows"], overview["near_silence_failed_windows"])
    if tuple(totals.values()) != expected:
        raise ValueError("Aggregate sample/window counts differ from their source totals")
    current.update(quiet_failed_windows=totals["quiet_failed"], near_silence_failed_windows=totals["near_failed"],
                   active_level=_level(student_energy, teacher_energy), near_samples=None, near_rms=None)
    for source in sources.values():
        source.update(near_samples=None, near_rms=None)
    recovery = report.get("recovery_window_metrics")
    if recovery is not None:
        if set(recovery["by_source"]) != set(sources):
            raise ValueError("Near-silence observations do not match development sources")
        near_total = 0
        near_error = 0.0
        for source_id, source in sources.items():
            observed = recovery["by_source"][source_id]
            n = _count(observed["near_samples"], "near-silence samples")
            error = _number(observed["near_error_sum"], "near-silence error energy", minimum=0)
            rms = math.sqrt(error/n) if n else None
            if n > source["quiet_samples"] or (not n and error != 0):
                raise ValueError("Near-silence samples/error disagree with quiet region")
            recorded = observed["near_residual_rms"]
            if ((rms is None and recorded is not None) or (rms is not None and
                    not math.isclose(_number(recorded, "near-silence RMS", minimum=0), rms, rel_tol=1e-10, abs_tol=1e-12))):
                raise ValueError("Near-silence RMS differs from saved energy/count")
            source.update(near_samples=n, near_rms=rms)
            near_total += n
            near_error += error
        if (recovery["near_samples"] != near_total or not math.isclose(
                _number(recovery["near_error_sum"], "aggregate near error", minimum=0), near_error, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Near-silence aggregate differs from source totals")
        rms = math.sqrt(near_error/near_total) if near_total else None
        recorded = recovery["near_residual_rms"]
        if ((rms is None and recorded is not None) or (rms is not None and
                not math.isclose(_number(recorded, "aggregate near RMS", minimum=0), rms, rel_tol=1e-10, abs_tol=1e-12))):
            raise ValueError("Aggregate near-silence RMS differs from energy/count")
        current.update(near_samples=near_total, near_rms=rms)
    return {"metrics": current, "sources": sources}


def _same_panel(reference, current):
    if set(reference["sources"]) != set(current["sources"]):
        raise ValueError("Development source identities changed between reviews")
    for source_id, before in reference["sources"].items():
        after = current["sources"][source_id]
        for key in ("samples", "quiet_samples", "active_samples", "quiet_windows", "near_windows"):
            if before[key] != after[key]:
                raise ValueError(f"Teacher-defined region coverage changed: {source_id}/{key}")
        if before["near_samples"] is not None and after["near_samples"] is not None and before["near_samples"] != after["near_samples"]:
            raise ValueError(f"Teacher-defined near-silence sample coverage changed: {source_id}")
        if not math.isclose(before["teacher_active_energy"], after["teacher_active_energy"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Teacher active energy changed: {source_id}")


def _material(current, previous, fraction, absolute=0.0):
    delta = current - previous
    return delta > 0 and delta >= fraction * previous and delta >= absolute


def _level_worse(current, previous, threshold):
    if current["rms_ratio"] is None or previous["rms_ratio"] is None:
        return False
    if current["rms_ratio"] == 0:
        return previous["rms_ratio"] != 0
    if previous["rms_ratio"] == 0:
        return False
    return current["absolute_error_db"] - previous["absolute_error_db"] >= threshold


def _regressions(current, references, policy):
    flags = {}
    def add(metric, scope, source_id, reference_name, before, after):
        key = (metric, scope, source_id)
        row = flags.setdefault(key, {"metric": metric, "scope": scope, "source_id": source_id, "references": []})
        row["references"].append({"name": reference_name, "before": before, "after": after})
    for label, reference in references:
        now, old = current["metrics"], reference["metrics"]
        for metric in ("mae", "mel"):
            if _material(now[metric], old[metric], policy.error_regression_fraction):
                add(metric, "aggregate", None, label, old[metric], now[metric])
        key = "quiet_residual_rms_mean"
        if _material(now[key], old[key], policy.quiet_regression_fraction, policy.quiet_regression_absolute):
            add("quiet_rms", "aggregate", None, label, old[key], now[key])
        if (now["near_rms"] is not None and old["near_rms"] is not None
                and now["near_samples"] >= policy.minimum_region_samples and _material(
                    now["near_rms"], old["near_rms"], policy.quiet_regression_fraction, policy.quiet_regression_absolute)):
            add("near_rms", "aggregate", None, label, old["near_rms"], now["near_rms"])
        if _level_worse(now["active_level"], old["active_level"], policy.level_error_regression_db):
            add("active_level", "aggregate", None, label, old["active_level"], now["active_level"])
        for source_id, after in current["sources"].items():
            before = reference["sources"][source_id]
            if _material(after["mae"], before["mae"], policy.error_regression_fraction, policy.source_waveform_regression_absolute):
                add("mae", "source", source_id, label, before["mae"], after["mae"])
            if after["quiet_samples"] >= policy.minimum_region_samples and _material(
                    after["quiet_rms"], before["quiet_rms"], policy.quiet_regression_fraction, policy.quiet_regression_absolute):
                add("quiet_rms", "source", source_id, label, before["quiet_rms"], after["quiet_rms"])
            if (after["near_rms"] is not None and before["near_rms"] is not None
                    and after["near_samples"] >= policy.minimum_region_samples and _material(
                        after["near_rms"], before["near_rms"], policy.quiet_regression_fraction, policy.quiet_regression_absolute)):
                add("near_rms", "source", source_id, label, before["near_rms"], after["near_rms"])
            if after["active_samples"] >= policy.minimum_region_samples and _level_worse(
                    after["level"], before["level"], policy.level_error_regression_db):
                add("active_level", "source", source_id, label, before["level"], after["level"])
    return flags


def _interval(before, after, policy):
    a, b = before["metrics"], after["metrics"]
    def improvement(old, new):
        return (old - new) / old if old > 0 else (0.0 if new == 0 else None)
    mae = improvement(a["mae"], b["mae"])
    mel = improvement(a["mel"], b["mel"])
    correlation = b["correlation"] - a["correlation"]
    progressed = ((mae is not None and mae >= policy.error_progress_fraction)
                  or (mel is not None and mel >= policy.error_progress_fraction)
                  or correlation >= policy.correlation_progress_absolute)
    return {"mae_improvement_fraction": mae, "mel_improvement_fraction": mel,
            "correlation_improvement_absolute": correlation, "stalled": not progressed}


def classify_review(history_reports, policy=ReviewPolicy()):
    """Classify initial report plus fixed-cadence completed review reports.

    Only the last two completed intervals determine persistence. The runner owns
    the cadence, source ledger, maximum update budget and any later A/B decision.
    """
    result = {"version": VERSION, "action": "continue", "review_index": max(len(history_reports)-1, 0),
              "flags": [], "alerts": [], "material_regressions": [], "repeated_regressions": [],
              "thresholds": asdict(policy), "automatic_freezing": False,
              "interpretation": "Heuristic review gates, not audibility or statistical qualification. Pause requires diagnosis; it never freezes parameters."}
    try:
        if not history_reports:
            raise ValueError("An initial fixed-panel report is required")
        snapshots = [_snapshot(report, policy) for report in history_reports]
        for snapshot in snapshots[1:]:
            _same_panel(snapshots[0], snapshot)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result.update(action="pause_for_diagnosis", flags=[{"kind": "invalid_or_invariant_failure", "detail": str(exc)}])
        return result
    result["current_metrics"] = snapshots[-1]["metrics"]
    for source_id, source in snapshots[-1]["sources"].items():
        if source["active_samples"] and source["level"]["rms_ratio"] == 0:
            result["alerts"].append({"kind": "active_output_collapsed", "source_id": source_id,
                                     "detail": "Student active energy is zero; logarithmic level error is undefined/infinite, not a nonfinite model tensor."})
    if len(snapshots) == 1:
        result["stall"] = {"two_intervals_stalled": False, "eligible": False,
                            "minimum_reviews": policy.minimum_stall_reviews, "intervals": []}
        return result
    def regressions_at(index):
        refs = [("start", snapshots[0])]
        if index > 1:
            refs.append(("previous", snapshots[index-1]))
        return _regressions(snapshots[index], refs, policy)
    latest = regressions_at(len(snapshots)-1)
    repeated = set(latest) & set(regressions_at(len(snapshots)-2)) if len(snapshots) >= 3 else set()
    result["material_regressions"] = list(latest.values())
    result["repeated_regressions"] = [latest[key] for key in sorted(repeated, key=str)]
    if repeated:
        result["flags"].append({"kind": "repeated_material_regression", "affected": result["repeated_regressions"]})
    intervals = [_interval(snapshots[i-1], snapshots[i], policy) for i in range(max(1, len(snapshots)-2), len(snapshots))]
    eligible = len(snapshots)-1 >= policy.minimum_stall_reviews
    stalled = eligible and len(intervals) == 2 and all(row["stalled"] for row in intervals)
    result["stall"] = {"two_intervals_stalled": stalled, "eligible": eligible,
                       "minimum_reviews": policy.minimum_stall_reviews, "intervals": intervals}
    if stalled:
        result["flags"].append({"kind": "two_interval_progress_stall", "detail": "Waveform/mel/correlation progress is below review thresholds twice. This does not prove convergence or rule out quiet-region progress."})
    for metric in ("quiet_failed_windows", "near_silence_failed_windows"):
        old, new = snapshots[-2]["metrics"][metric], snapshots[-1]["metrics"][metric]
        if new > old:
            result["alerts"].append({"kind": "count_only_alert", "metric": metric, "before": old, "after": new,
                                     "detail": "Inspect continuous residual/output levels; threshold crossings alone do not trigger freezing or a pause."})
    if result["flags"]:
        result["action"] = "pause_for_diagnosis"
    return result
