"""Derived reporting only. Never supplies gradients or changes training state."""
from __future__ import annotations

import math
from statistics import mean, median


LOSSES = ("teacher_waveform", "teacher_mel", "feature_matching", "adversarial")


def training_scalars(row):
    """Make reconstruction naming and component-norm semantics explicit."""
    values = {}
    for key in LOSSES + ("discriminator",):
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value):
            values["loss/" + key] = value
    if all(k in row for k in ("teacher_waveform", "teacher_mel")):
        values["loss/reconstruction_only"] = row["teacher_waveform"] + row["teacher_mel"]
    norms = {key: row.get(key + "/scaled_norm", 0.) for key in LOSSES}
    if any(not math.isfinite(v) or v < 0 for v in norms.values()):
        raise ValueError("Invalid component gradient norm")
    total = sum(norms.values())
    for key in LOSSES:
        if total:
            values["balance/component_norm_fraction/" + key] = norms[key] / total
        if key + "/target_share" in row:
            values["balance/scheduled_share/" + key] = row[key + "/target_share"]
    for key in ("unique_scored_hours", "discriminator_updates", "generator_gradient_norm",
                "discriminator_gradient_norm", "seconds_in_training_loop"):
        if key in row:
            values["progress/" + key] = row[key]
    if "gradient_norm" in row:
        values["progress/generator_gradient_norm"] = row["gradient_norm"]
    for key in ("student_peak_abs", "student_full_scale_samples", "teacher_peak_abs",
                "teacher_full_scale_samples", "scored_samples"):
        name = "training_signal/" + key
        if name in row:
            if row[name] < 0:
                raise ValueError("Negative training signal statistic")
            values[name] = row[name]
    if any(not math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite reported training scalar")
    return values


def signal_summary(rows):
    """Keep absolute errors visible when a strict pass fraction stays at zero."""
    if not rows or any(not {"samples", "student_peak_abs", "student_clipped_samples"} <= r.keys() for r in rows):
        raise ValueError("Peak reporting requires complete per-crop signal checks")
    quiet = [w for r in rows for w in r.get("quiet_windows", {}).get("windows", [])
             if w.get("is_quiet", False)]
    residuals = [w["residual_rms"] for w in quiet if "residual_rms" in w]
    phases = [r["quiet_phase480"]["residual_template_rms"] for r in rows
              if "residual_template_rms" in r.get("quiet_phase480", {})]
    values = {"scored_samples": sum(r["samples"] for r in rows),
              "peak_abs_max": max((r.get("student_peak_abs", 0.) for r in rows), default=0.),
              "samples_at_or_above_full_scale": sum(r.get("student_clipped_samples", 0) for r in rows),
              "crops_at_or_above_full_scale": sum(r.get("student_clipped_samples", 0) > 0 for r in rows)}
    if residuals:
        values.update(quiet_residual_rms_mean=mean(residuals),
                      quiet_residual_rms_max=max(residuals), quiet_residual_rms_p50=median(residuals))
    if phases:
        values.update(phase480_residual_rms_p50=median(phases), phase480_residual_rms_max=max(phases))
    return values


def evaluation_scalars(evaluation):
    values = {}
    keep = {"teacher_waveform_mean", "teacher_mel_mean", "nonquiet_cosine_mean", "nonquiet_cosine_min",
            "nonquiet_fraction_passing_099", "quiet_fraction_passing", "quiet_window_count",
            "quiet_failed_count", "gain_db_mean", "waveform_cosine"}
    for group, metrics in evaluation.get("groups", {}).items():
        for key, value in metrics.items():
            if key in keep and isinstance(value, (int, float)) and math.isfinite(value):
                values[f"quality/{group}/{key}"] = value
    for key, value in signal_summary(evaluation["rows"]).items():
        values["quality/all/" + key] = value
    values["quality/all/target_cosine"] = .99
    values["quality/all/full_scale_excursions_absent"] = int(
        values["quality/all/peak_abs_max"] <= 1.)
    # This augments acceptance reporting; it never unlocks or blocks GAN training.
    values["quality/all/reconstruction_and_peak_checks_pass"] = int(
        bool(evaluation.get("gate", {}).get("passed"))
        and values["quality/all/full_scale_excursions_absent"])
    return values
