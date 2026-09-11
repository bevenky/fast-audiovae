"""Matched, read-only quality diagnostics for decoder fusion candidates.

All arms use the same retained samples and the engine's unchanged diagnostic
criterion. Six samples at interior crop starts are excluded to accommodate the
output-filter candidate's extra history beyond the archived 29 latent frames.
This is an explicit evaluation-context exclusion, never a streaming loss.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from statistics import median
from typing import Mapping

import torch
from torch.nn import functional as F

from .batching import _validate_crop
from .cache import DECODER_HOP
from .objective_comparison import state_fingerprint
from .quiet_audio import QuietAudioConfig, quiet_window_metrics
from .training import _restore_rng, _rng_state


_MASK_POLICY = {
    "format_version": 1,
    "interior_prefix_exclusion_samples": 6,
    "condition": "context_start_frame > 0",
    "reason": "shared_archived_context_limit_for_seven_tap_output_filter",
    "applies_to": "all_comparison_arms_and_all_quality_metrics",
    "genuine_utterance_start_exclusion_samples": 0,
    "changes_output_or_streaming_sample_count": False,
}


@contextmanager
def _preserved_evaluation(*modules):
    """Restore modes/RNG and fail with restored tensors if forward mutates state."""
    rng = _rng_state()
    modes = {child: child.training for module in modules for child in module.modules()}
    snapshots = [(module, {name: value.detach().clone() for name, value in module.state_dict().items()})
                 for module in modules]
    changed = []
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        try:
            for module, saved in snapshots:
                current = module.state_dict()
                if current.keys() != saved.keys() or any(
                        not torch.equal(current[name], value) for name, value in saved.items()):
                    changed.append(type(module).__name__)
                    module.load_state_dict(saved, strict=True)
        finally:
            for module, mode in modes.items():
                module.training = mode
            _restore_rng(rng)
        if changed:
            raise RuntimeError("Evaluation mutated module tensors; restored original state: " + ", ".join(changed))


def _high_frequency(prediction, teacher):
    size, hop = 2048, 512
    window = torch.hann_window(size, device=prediction.device, dtype=torch.float32)
    frequencies = torch.fft.rfftfreq(size, d=1 / 48000, device=prediction.device)
    selected = (frequencies >= 8000) & (frequencies <= 24000)

    def spectrum(audio):
        return torch.stft(audio[0, 0].float(), n_fft=size, hop_length=hop,
            win_length=size, window=window, center=False, normalized=False,
            onesided=True, return_complex=True)[selected] / window.sum()

    p, t = spectrum(prediction), spectrum(teacher)
    pm, tm = p.abs(), t.abs()
    return {
        "low_hz": 8000, "high_hz": 24000, "n_fft": size, "hop": hop,
        "normalization": "complex_spectrum_divided_by_hann_window_sum",
        "frames": p.shape[-1], "bins": p.shape[0], "elements": p.numel(),
        "magnitude_mae": float((pm - tm).abs().mean()),
        "log_magnitude_mae": float((pm.clamp_min(1e-7).log() - tm.clamp_min(1e-7).log()).abs().mean()),
        "log_epsilon": 1e-7,
        "complex_residual_rms": float((p - t).abs().square().mean().sqrt()),
        "teacher_magnitude_mean": float(tm.mean()),
    }


def _phase_480(prediction, teacher, mask, quiet_config):
    length = prediction.numel() // 960 * 960
    complete = mask.reshape(-1)[:length].reshape(-1, 960).all(1)
    # Remove invalid history/tails before doing any arithmetic on them.
    p = prediction.reshape(-1)[:length].double().reshape(-1, 960)[complete]
    t = teacher.reshape(-1)[:length].double().reshape(-1, 960)[complete]
    quiet = t.square().mean(1).sqrt() <= quiet_config.quiet_teacher_rms_max
    count = int(quiet.sum())
    result = {"quiet_complete_20ms_windows": count, "minimum_windows": 8,
              "alignment": "original_crop_grid_complete_valid_windows_only",
              "includes_dc": True, "residual_template_rms": None,
              "student_template_rms": None, "teacher_template_rms": None,
              "student_explained_ac_fraction": None}
    if count >= 8:
        p, t = p[quiet].reshape(-1, 480), t[quiet].reshape(-1, 480)
        pt, tt = p.mean(0), t.mean(0)
        result.update(residual_template_rms=float((pt - tt).square().mean().sqrt()),
            student_template_rms=float(pt.square().mean().sqrt()),
            teacher_template_rms=float(tt.square().mean().sqrt()),
            student_explained_ac_fraction=float((pt - p.mean()).square().mean()
                / (p - p.mean()).square().mean().clamp_min(1e-24)))
    return result


def _cohort(meta):
    explicit = meta.get("group") or meta.get("kind") or meta.get("type")
    if explicit in {"speech", "expressive", "silence"}:
        return explicit
    condition = meta.get("condition") or meta.get("event")
    if isinstance(condition, str):
        condition = condition.lower().strip().replace(" ", "_")
        if condition in {"speech", "silence", "encoded_zero", "digital_silence"}:
            return "speech" if condition == "speech" else "silence"
        if condition in {"laughter", "laughing", "giggling", "screaming", "whistling",
                "crying", "shouting", "yell", "yelling", "whisper", "whispering",
                "breath", "breathing", "emotional_nonverbal"}:
            return "expressive"
    return "unverified"


def _summarize(rows, quiet_config):
    count, samples = len(rows), sum(row["samples"] for row in rows)
    quiet = [window for row in rows for window in row["quiet_windows"]["windows"] if window["is_quiet"]]
    active = [row for row in rows if row["teacher_rms"] >= quiet_config.quiet_teacher_rms_max]
    phase = [row["quiet_phase480"]["residual_template_rms"] for row in rows
             if row["quiet_phase480"]["residual_template_rms"] is not None]
    quiet_samples = sum(window["valid_samples"] for window in quiet)
    hf_elements = sum(row["high_frequency"]["elements"] for row in rows)
    return {
        "crops": count, "sources": len({row["source_id"] for row in rows}),
        "scored_samples": samples,
        "original_scored_samples": sum(row["original_scored_samples"] for row in rows),
        "context_excluded_samples": sum(row["context_excluded_samples"] for row in rows),
        "raw_mae_sample_weighted": sum(row["waveform_raw_mae"] * row["samples"] for row in rows) / samples,
        "teacher_waveform_mean": sum(row["teacher_waveform"] for row in rows) / count,
        "teacher_mel_mean": sum(row["teacher_mel"] for row in rows) / count,
        "diagnostic_loss_reduction": "equal_crop_mean",
        "nonquiet_crops": len(active),
        "nonquiet_cosine_mean": sum(row["waveform_cosine"] for row in active) / len(active) if active else None,
        "nonquiet_cosine_min": min((row["waveform_cosine"] for row in active), default=None),
        "nonquiet_fraction_passing_099": sum(row["waveform_cosine"] >= .99 for row in active) / len(active) if active else None,
        "peak_abs_max": max(row["student_peak_abs"] for row in rows),
        "teacher_peak_abs_max": max(row["teacher_peak_abs"] for row in rows),
        "student_overshoot_samples": sum(row["student_overshoot_samples"] for row in rows),
        "teacher_overshoot_samples": sum(row["teacher_overshoot_samples"] for row in rows),
        "overshoot_crops": sum(row["student_overshoot_samples"] > 0 for row in rows),
        "student_near_saturation_samples": sum(row["student_near_saturation_samples"] for row in rows),
        "teacher_near_saturation_samples": sum(row["teacher_near_saturation_samples"] for row in rows),
        "quiet_window_count": len(quiet), "quiet_valid_samples": quiet_samples,
        "quiet_failed_count": sum(window["passed"] is False for window in quiet),
        "quiet_fraction_passing": sum(window["passed"] for window in quiet) / len(quiet) if quiet else None,
        "quiet_residual_rms_mean": sum(window["residual_rms"] for window in quiet) / len(quiet) if quiet else None,
        "quiet_residual_rms_pooled": math.sqrt(sum(window["residual_rms"] ** 2 * window["valid_samples"]
                                                  for window in quiet) / quiet_samples) if quiet_samples else None,
        "quiet_phase480_residual_template_rms_median": median(phase) if phase else None,
        "high_frequency_magnitude_mae": sum(row["high_frequency"]["magnitude_mae"] * row["high_frequency"]["elements"]
                                             for row in rows) / hf_elements,
        "high_frequency_log_magnitude_mae": sum(row["high_frequency"]["log_magnitude_mae"] * row["high_frequency"]["elements"]
                                                 for row in rows) / hf_elements,
        "high_frequency_complex_residual_rms": math.sqrt(sum(row["high_frequency"]["complex_residual_rms"] ** 2
            * row["high_frequency"]["elements"] for row in rows) / hf_elements),
    }


def evaluate_fusion(engine, crops, metadata: Mapping[str, dict]) -> dict:
    """Evaluate every arm on identical valid slices, without altering training.

    ``metadata`` maps source IDs to optional dataset/language/condition/group
    fields. Unknown categories remain unverified. The engine criterion is the
    original common diagnostic, never the arm-specific training reconstruction.
    """
    crops = tuple(crops)
    if not crops or not isinstance(metadata, Mapping):
        raise ValueError("Fusion evaluation requires nonempty crops and source metadata")
    for crop in crops:
        _validate_crop(crop)
        if (type(crop.context_start_frame) is not int or crop.context_start_frame < 0
                or type(crop.start_frame) is not int
                or crop.start_frame != crop.context_start_frame + crop.context_frames):
            raise ValueError("Crop start and actual causal context disagree")
        if crop.context_start_frame > 0 and crop.context_frames < 29:
            raise ValueError("Interior evaluation requires at least the archived 29 context frames")
        if not isinstance(metadata.get(crop.source_id, {}), Mapping):
            raise ValueError("Source metadata must contain mappings")
        excluded = 6 if crop.context_start_frame > 0 else 0
        if crop.valid_scored_samples - excluded < max(2048, *engine.criterion.config.fft_sizes):
            raise ValueError("Retained scored region is too short for common evaluation FFTs")
    quiet_config = getattr(engine.config, "quiet_audio", QuietAudioConfig())
    rows = []
    with _preserved_evaluation(engine.model, engine.criterion):
        fingerprint = state_fingerprint(engine.model.state_dict())
        for crop in crops:
            latents = crop.latents.detach().to(engine.device)
            if not bool(torch.isfinite(latents).all()):
                raise ValueError("Evaluation latents must be finite")
            prediction = engine.model(latents)
            expected_shape = (1, 1, latents.shape[-1] * DECODER_HOP)
            if tuple(prediction.shape) != expected_shape:
                raise ValueError("Decoder output sample count changed during evaluation")
            target = crop.teacher_audio.detach().to(engine.device)
            excluded = 6 if crop.context_start_frame > 0 else 0
            start, stop = crop.scored_slice.start + excluded, crop.scored_slice.stop
            mask = torch.zeros_like(prediction, dtype=torch.bool)
            mask[..., start:stop] = True
            p, t = prediction[..., start:stop], target[..., start:stop]
            if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(t).all()):
                raise ValueError("Retained evaluation samples must be finite")
            # Every waveform, spectral, peak and quiet metric derives from this
            # same mask. Never merge old evaluate_crops results with new slices.
            losses = engine.criterion(p, t)
            rms, prms = t.square().mean().sqrt(), p.square().mean().sqrt()
            error = (p - t).abs().mean()
            meta = dict(metadata.get(crop.source_id, {}))
            row = {"source_id": crop.source_id, "start_frame": crop.start_frame,
                "context_start_frame": crop.context_start_frame,
                "evaluated_step": engine.step, "model_state_sha256": fingerprint,
                "fixed_statistics": True, "metadata": meta, "cohort": _cohort(meta),
                "original_scored_samples": crop.valid_scored_samples,
                "context_excluded_samples": excluded, "samples": stop - start,
                "scored_start_sample_in_crop": start, "scored_stop_sample_in_crop": stop,
                "absolute_scored_start_sample": crop.context_start_frame * DECODER_HOP + start,
                **{name: float(value) for name, value in losses.items()},
                "waveform_raw_mae": float(error), "teacher_rms": float(rms), "student_rms": float(prms),
                "waveform_to_silence_error_ratio": float(error / t.abs().mean().clamp_min(1e-8)),
                "rms_db_error": float(20 * torch.log10(prms.clamp_min(1e-8) / rms.clamp_min(1e-8))),
                "waveform_cosine": float(F.cosine_similarity(p.flatten(), t.flatten(), dim=0, eps=1e-12)),
                "waveform_cosine_defined": bool(p.square().sum() > 0 and t.square().sum() > 0),
                "student_peak_abs": float(p.abs().max()), "teacher_peak_abs": float(t.abs().max()),
                "student_overshoot_samples": int((p.abs() > 1).sum()),
                "teacher_overshoot_samples": int((t.abs() > 1).sum()),
                "student_full_scale_samples": int((p.abs() >= 1).sum()),
                "student_near_saturation_samples": int((p.abs() >= .999).sum()),
                "teacher_near_saturation_samples": int((t.abs() >= .999).sum()),
                "near_saturation_threshold": .999,
                "quiet_windows": quiet_window_metrics(prediction, target, mask, config=quiet_config),
                "quiet_phase480": _phase_480(prediction, target, mask, quiet_config),
                "high_frequency": _high_frequency(p, t)}
            rows.append(row)
    members = defaultdict(list)
    for row in rows:
        members["all"].append(row)
        members["group/" + row["cohort"]].append(row)
        for name in ("dataset", "language", "condition"):
            value = row["metadata"].get(name)
            members[f"{name}/{value if isinstance(value, str) and value else 'unverified'}"].append(row)
    groups = {name: _summarize(items, quiet_config) for name, items in members.items()}
    result = {"format_version": 1, "evaluated_step": engine.step,
        "model_state_sha256": fingerprint, "criterion_config": asdict(engine.criterion.config),
        "mask_policy": dict(_MASK_POLICY), "quiet_config": asdict(quiet_config),
        "rows": rows, "summary": groups["all"], "groups": groups,
        "model_tensors_preserved": True, "module_modes_preserved": True,
        "global_rng_preserved": True, "optimizer_updates": 0}
    # Refuse an apparently successful report containing NaN/Inf or opaque metadata.
    json.dumps(result, allow_nan=False)
    return result


def diagnose_streaming(model, latents, frames_per_chunk=2) -> dict:
    """Compare untrimmed batch/stream samples; no latency measurement or alignment."""
    if type(frames_per_chunk) is not int or frames_per_chunk < 1:
        raise ValueError("frames_per_chunk must be a positive integer")
    if (latents.ndim != 3 or latents.shape[:2] != (1, 64)
            or not latents.is_floating_point() or not bool(torch.isfinite(latents).all())):
        raise ValueError("Streaming diagnostic requires finite [1,64,frames] latents")
    with _preserved_evaluation(model):
        batch = model(latents.detach())
        stream = model.stream()
        chunks, counts = [], []
        try:
            for start in range(0, latents.shape[-1], frames_per_chunk):
                item = latents[..., start:start + frames_per_chunk].detach()
                output = stream.decode_chunk(item)
                if output.ndim != 3 or output.shape[:2] != (1, 1) or not bool(torch.isfinite(output).all()):
                    raise ValueError("Streaming diagnostic received malformed or nonfinite chunk")
                chunks.append(output)
                counts.append({"input_frames": item.shape[-1], "expected_samples": item.shape[-1] * DECODER_HOP,
                               "output_samples": output.shape[-1]})
            tail = stream.flush()
            if tail.ndim != 3 or tail.shape[:2] != (1, 1) or not bool(torch.isfinite(tail).all()):
                raise ValueError("Streaming diagnostic received malformed or nonfinite flush")
            chunks.append(tail)
            streamed = torch.cat(chunks, dim=-1)
        finally:
            stream.close()
        expected = latents.shape[-1] * DECODER_HOP
        if batch.ndim != 3 or batch.shape[:2] != (1, 1) or not bool(torch.isfinite(batch).all()):
            raise ValueError("Streaming diagnostic received malformed or nonfinite batch output")
        count_passed = (batch.shape[-1] == expected == streamed.shape[-1] and tail.shape[-1] == 0
            and all(row["expected_samples"] == row["output_samples"] for row in counts))
        delta = batch - streamed if count_passed else None
        return {"format_version": 1, "frames_per_chunk": frames_per_chunk,
            "input_frames": latents.shape[-1], "expected_samples": expected,
            "batch_samples": batch.shape[-1], "stream_samples": streamed.shape[-1],
            "flush_samples": tail.shape[-1], "sample_count_passed": count_passed,
            "chunks": counts,
            "max_abs_error": float(delta.abs().max()) if delta is not None and delta.numel() else (0. if count_passed else None),
            "rms_error": float(delta.square().mean().sqrt()) if delta is not None and delta.numel() else (0. if count_passed else None),
            "comparison": "same_untrimmed_samples_no_shift_rescale_or_context_exclusion",
            "model_tensors_preserved": True, "module_modes_preserved": True, "global_rng_preserved": True,
            "rtf_measured": False}
