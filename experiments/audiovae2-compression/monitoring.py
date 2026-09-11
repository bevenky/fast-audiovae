"""Named teacher/student objectives and validation groups for TensorBoard."""
from __future__ import annotations

import math
import re


TARGET_CORRELATION = 0.99

LOSS_SPECIFICATION = {
    "framework": "PyTorch; TensorBoard for monitoring",
    "teacher": "Frozen original AudioVAE2 encoder and decoder, verified checkpoint bytes",
    "student": "Stages 2–4 internal widths 256/128/128, all nine residual units; rest frozen",
    "objectives": {
        "waveform": "Native48k aligned teacher waveform L1, pooled by valid samples",
        "mel": "Teacher magnitude mel linear+log errors at256/512/1024/2048/4096 sample windows, pooled by valid spectral elements",
        "feature": "Raw MSE at the entire stage2–4 group's128-channel output; partial feature cells weighted by valid waveform samples",
    },
    "coefficients": "Fixed after training-only pooled parameter-gradient calibration and disposable AdamW update checks",
    "active_from_update": 1,
    "effective_batch_sources": 3,
    "forward_batch_sources": 1,
    "gradient_accumulation": 3,
    "teacher_comparison": "Same frozen64-channel posterior-mean latents and causal history; no random latent sampling",
    "no_latent_or_kl_loss": "Encoder and latent coordinates are unchanged by construction",
    "correlation_target": TARGET_CORRELATION,
    "correlation_scope": "Waveform cosine over teacher-active20ms windows at native48k, report each source plus aggregate",
    "quality_interpretation": "0.99 correlation is a reconstruction target, not99percent perceptual accuracy",
    "additional_checks": ["quiet absolute residual and amplitude", "transitions", "peak magnitude and transient fidelity",
                          "language and expressive groups", "paired source-referenced audio scores", "blind listening", "causal streaming parity"],
    "adversarial_and_discriminator_feature_matching": "Conditional later fine-tuning if texture deficits remain; step-based warmup, never gated on0.99 correlation",
    "teacher_feature_matching": "The direct group-output feature objective is active from step1 and is distinct from discriminator feature matching",
    "boundary_diagnostics": "Stages1–6, pre-tanh and waveform at fixed validation milestones; stages2/3 compare selected teacher coordinates only and do not add training penalties",
}


def _tag(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def _scalar(writer, tag, value, step):
    if value is not None and isinstance(value, (int,float)) and math.isfinite(value):
        writer.add_scalar(tag, value, step)


def log_training(writer, record, coefficients, learning_rate, step):
    names = {"waveform":"teacher_waveform", "mel":"teacher_mel", "feature":"teacher_group_mse"}
    calculated = sum(coefficients[k]*record[k] for k in names)
    if not math.isclose(record["total"],calculated,rel_tol=2e-5,abs_tol=1e-8):
        raise ValueError("Dashboard total excludes a weighted training objective")
    for key,label in names.items():
        _scalar(writer,"loss/"+label,record[key],step)
        _scalar(writer,"loss_weighted/"+label,coefficients[key]*record[key],step)
        _scalar(writer,"configuration/weight_"+key,coefficients[key],step)
    _scalar(writer,"loss/total",record["total"],step)
    _scalar(writer,"train/total",record["total"],step)
    for key in ("unique_sources","audio_hours"):
        _scalar(writer,"data/"+key,record[key],step)
    _scalar(writer,"optimization/learning_rate",learning_rate,step)
    _scalar(writer,"performance/elapsed_seconds",record["elapsed_seconds"],step)
    _scalar(writer,"performance/seconds_per_update_so_far",record["elapsed_seconds"]/step,step)
    for key in ("gradient_norm","gpu_memory_gib","step_seconds"):
        if key in record:_scalar(writer,"optimization/"+key,record[key],step)


def _summarize(rows):
    samples=sum(r["samples"] for r in rows)
    quiet_samples=sum(r["quiet_samples"] for r in rows)
    cos=[r["cosine"] for r in rows if r["cosine"] is not None]
    return {
        "sources":len(rows),
        "waveform_mae":sum(r["mae"]*r["samples"] for r in rows)/samples,
        "waveform_residual_rms":math.sqrt(sum(r["mse"]*r["samples"] for r in rows)/samples),
        "nonquiet_cosine_mean":sum(cos)/len(cos) if cos else None,
        "nonquiet_cosine_min":min(cos) if cos else None,
        "nonquiet_sources_below_099":sum(c<TARGET_CORRELATION for c in cos),
        "nonquiet_sources_scored":len(cos),
        "quiet_residual_rms":math.sqrt(sum(r["quiet_error_sum"] for r in rows)/quiet_samples) if quiet_samples else None,
        "quiet_failed_windows":sum(r["quiet_failed"] for r in rows),
        "quiet_windows":sum(r["quiet_windows"] for r in rows),
        "peak_abs_max":max(r["peak"] for r in rows),
        "overshoot_samples":sum(r["overshoot"] for r in rows),
    }


def log_validation(writer, report, step, source_metadata):
    for key,value in report["aggregate"].items():_scalar(writer,"quality/all/"+key,value,step)
    _scalar(writer,"target/nonquiet_waveform_correlation",TARGET_CORRELATION,step)
    groups={"all":report["rows"]}
    for row in report["rows"]:
        metadata=source_metadata.get(row["source_id"],{})
        labels=metadata.get("verified_source_labels",[])
        if isinstance(labels,dict):labels=list(labels)
        if isinstance(labels,str):labels=[labels]
        names=["language/"+_tag(metadata.get("normalized_language","unknown"))]
        if labels:names.append("expressive/all")
        names.extend("expressive/"+_tag(label) for label in labels)
        role=metadata.get("selection_kind")
        if role:names.append("selection/"+_tag(role))
        for name in names:groups.setdefault(name,[]).append(row)
    summaries={name:_summarize(rows) for name,rows in groups.items()}
    for name,metrics in summaries.items():
        prefix="quality/"+name if name=="all" else "quality_groups/"+name
        for key,value in metrics.items():_scalar(writer,prefix+"/"+key,value,step)
    return summaries


def log_boundaries(writer, report, step):
    """Write the explicitly labelled diagnostic boundary aggregates."""
    for name, regions in report["aggregate"].items():
        label = name + "_selected_coordinates" if name in ("stage2_output", "stage3_output") else name
        for region, metrics in regions.items():
            for key, value in metrics.items():
                tag = "/".join(("quality_boundaries", _tag(label), _tag(region), _tag(key)))
                _scalar(writer, tag, value, step)
