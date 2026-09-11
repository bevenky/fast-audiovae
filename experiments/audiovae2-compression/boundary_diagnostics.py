"""Measurement-only teacher/student decoder boundary comparisons.

These diagnostics do not add training objectives or gradients. Internal stages
2 and 3 have changed coordinates/capacity: their selected-channel comparisons
are partial observations, not full teacher-feature reconstruction scores.
"""
from __future__ import annotations

import math

import torch

from audiovae_student.quiet_audio import quiet_window_metrics
from group_model import teacher_trace


STAGE_RATES = {
    "stage1_output": 200,
    "stage2_output": 1200,
    "stage3_output": 6000,
    "stage4_output": 12000,
    "stage5_output": 24000,
    "stage6_output": 48000,
    "pre_tanh": 48000,
    "waveform": 48000,
}
REGIONS = ("all", "quiet", "active")
SUM_KEYS = ("error_square_sum", "error_absolute_sum", "teacher_square_sum", "student_square_sum", "dot_sum",
            "valid_audio_samples", "weighted_feature_elements", "valid_feature_cells", "partial_feature_cells")


@torch.no_grad()
def region_masks(teacher_waveform, valid):
    if teacher_waveform.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Teacher waveform and boolean validity mask must have identical shapes")
    if teacher_waveform.ndim != 3 or teacher_waveform.shape[:2] != (1, 1):
        raise ValueError("Boundary diagnostics require a singleton mono waveform")
    # Classification is entirely teacher-derived. Preserve the original time
    # grid and exclude context/holes/tails from the per-window RMS calculation.
    panel = quiet_window_metrics(teacher_waveform, teacher_waveform, valid)
    quiet = torch.zeros_like(valid)
    for row in panel["windows"]:
        if row["is_quiet"]:
            start, stop = row["start_sample"], row["stop_sample"]
            quiet[..., start:stop] = valid[..., start:stop]
    return {"all": valid, "quiet": quiet, "active": valid & ~quiet}, panel["config"]


@torch.no_grad()
def weighted_statistics(student, teacher, audio_mask, rate_hz):
    if rate_hz not in STAGE_RATES.values() or 48000 % rate_hz:
        raise ValueError("Unsupported integer feature-to-audio rate")
    factor = 48000 // rate_hz
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[0] != 1:
        raise ValueError("Compared boundary tensors must have identical [1, channels, frames] shapes")
    if audio_mask.shape != (1, 1, student.shape[-1] * factor) or audio_mask.dtype != torch.bool:
        raise ValueError("Audio validity does not align with this boundary's temporal cells")
    if student.device != teacher.device or student.device != audio_mask.device:
        raise ValueError("Boundary tensors and mask must share a device")
    weights = audio_mask.reshape(1, 1, student.shape[-1], factor).sum(-1)
    valid_cells = weights > 0
    p = student.detach().double().masked_fill(~valid_cells, 0)
    t = teacher.detach().double().masked_fill(~valid_cells, 0)
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError("Scored boundary features must be finite")
    w = weights.double()
    error = p-t
    samples = int(weights.sum())
    channels = student.shape[1]
    return {
        "error_square_sum": float((error.square()*w).sum()),
        "error_absolute_sum": float((error.abs()*w).sum()),
        "teacher_square_sum": float((t.square()*w).sum()),
        "student_square_sum": float((p.square()*w).sum()),
        "dot_sum": float((p*t*w).sum()),
        "valid_audio_samples": samples,
        "weighted_feature_elements": samples*channels,
        "valid_feature_cells": int(valid_cells.sum())*channels,
        "partial_feature_cells": int(((weights>0)&(weights<factor)).sum())*channels,
        "max_abs_error": float(error.abs().max()) if samples else None,
    }


def summarize_statistics(stats):
    count = stats["weighted_feature_elements"]
    result = {key: stats[key] for key in ("valid_audio_samples", "weighted_feature_elements", "valid_feature_cells", "partial_feature_cells")}
    result["max_abs_error"] = stats["max_abs_error"]
    if not count:
        result.update({key: None for key in ("mse", "rmse", "mae", "nrmse", "cosine", "teacher_rms", "student_rms")})
        return result
    target_energy, prediction_energy = stats["teacher_square_sum"], stats["student_square_sum"]
    result.update({
        "mse": stats["error_square_sum"]/count,
        "rmse": math.sqrt(stats["error_square_sum"]/count),
        "mae": stats["error_absolute_sum"]/count,
        "teacher_rms": math.sqrt(target_energy/count),
        "student_rms": math.sqrt(prediction_energy/count),
        "nrmse": math.sqrt(stats["error_square_sum"]/target_energy) if target_energy>0 else None,
        "cosine": stats["dot_sum"]/math.sqrt(target_energy*prediction_energy) if target_energy>0 and prediction_energy>0 else None,
    })
    return result


def merge_statistics(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("At least one source is required to aggregate boundary diagnostics")
    merged = {key: sum(row[key] for row in rows) for key in SUM_KEYS}
    maxima = [row["max_abs_error"] for row in rows if row["max_abs_error"] is not None]
    merged["max_abs_error"] = max(maxima) if maxima else None
    return merged


@torch.no_grad()
def evaluate_boundaries(teacher, student, crops, selection, batch_fn, teacher_forward_fn):
    """Measure fixed sources one at a time using the caller's teacher backend.

    ``batch_fn([crop])`` returns z, cached_teacher_audio, valid, spans.
    ``teacher_forward_fn(teacher, z)`` returns the complete teacher_trace mapping.
    The student consumes the same original latents and its actual own-prefix
    activations; internal stages are not individually teacher-forced.
    """
    crops = list(crops)
    if not crops or len({crop["source_id"] for crop in crops}) != len(crops):
        raise ValueError("A nonempty fixed source-unique boundary panel is required")
    named = ("stage2_indices", "stage3_indices")
    if any(list(selection[name]) != student.selections[name] for name in named):
        raise ValueError("Diagnostic channel selections differ from the student initialization")
    stats_by_stage = {stage: {region: [] for region in REGIONS} for stage in STAGE_RATES}
    sources = []
    shapes = {}
    config = None
    for crop in crops:
        z, target_wave, valid, _ = batch_fn([crop])
        if z.shape[0] != 1:
            raise ValueError("Boundary diagnostics cannot use a batched teacher call")
        teacher_values = teacher_forward_fn(teacher, z)
        student_values = teacher_trace(student.decoder, z, output_sample_rate=student.output_sample_rate)
        if teacher_values["waveform"].shape != target_wave.shape or target_wave.shape != valid.shape:
            raise ValueError("Cached waveform, live waveform and score mask do not align")
        masks, config = region_masks(target_wave, valid)
        source = {"source_id": crop["source_id"], "stages": {}}
        for stage, rate in STAGE_RATES.items():
            t, p = teacher_values[stage], student_values[stage]
            original_channels = t.shape[1]
            if stage in ("stage2_output", "stage3_output"):
                name = "stage2_indices" if stage == "stage2_output" else "stage3_indices"
                index = torch.tensor(selection[name], device=t.device, dtype=torch.long)
                t = t.index_select(1, index)
            shapes[stage] = {"teacher_channels": original_channels, "compared_channels": t.shape[1], "rate_hz": rate}
            source["stages"][stage] = {}
            for region in REGIONS:
                stats = weighted_statistics(p, t, masks[region], rate)
                stats_by_stage[stage][region].append(stats)
                source["stages"][stage][region] = summarize_statistics(stats)
        sources.append(source)
    aggregate = {stage: {region: summarize_statistics(merge_statistics(rows)) for region, rows in regions.items()}
                 for stage, regions in stats_by_stage.items()}
    comparisons = {}
    for stage, shape in shapes.items():
        subset = stage in ("stage2_output", "stage3_output")
        comparisons[stage] = {**shape,
            "scope": "selected_teacher_coordinates_only" if subset else "shared_full_boundary",
            "interpretation": (
                "Partial internal-coordinate diagnostic. Narrowed internal representations may redistribute information; this is not full teacher-feature fidelity and does not train corresponding layers."
                if subset else "Complete shared coordinates, compared after the student's actual preceding stages."),
        }
    return {"version": "audiovae2_boundary_diagnostics_v1", "measurement_only": True,
            "batch_size": 1, "sources": len(sources), "source_ids": [row["source_id"] for row in sources],
            "stage_order": list(STAGE_RATES), "regions": list(REGIONS), "quiet_config": config,
            "comparisons": comparisons, "aggregate": aggregate, "per_source": sources,
            "weighting": "Each feature frame is weighted by the number of scored 48k waveform samples in its aligned time cell; partial final cells are included. Aggregate errors and energies pool these weights across sources.",
            "normalization": "NRMSE is sqrt(pooled squared error / pooled teacher energy), with null when teacher energy is zero. Cosine is null for a zero-energy side. No normalization or boundary penalty enters training.",
            "counts": "weighted_feature_elements counts channel-by-valid-audio-sample equivalents, not the number of unique feature cells."}
