"""Training-only source splitting and exact block-head sufficient statistics.

No model/teacher calls, optimizer steps, checkpoint writes, or heldout scoring.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import torch


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _audio_sha(row):
    value = row.get("audio_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Every training source needs a pinned lowercase audio SHA256")
    return value


def select_training_sources(crops, rows, heldout_ids, excluded_audio_hashes, *,
                            fit_count=2048, validation_count=256, seed=20260909):
    """One crop per distinct source AND audio hash; no student-score selection.

    The source is ranked before its crop, and both use domain-separated hashes.
    Existing recording aliases are deduplicated before assigning either split.
    A fixed seed only determines ordering; availability failure is explicit.
    """
    if any(type(v) is not int or v <= 0 for v in (fit_count, validation_count)):
        raise ValueError("Positive integer split counts are required")
    if type(seed) is not int:
        raise ValueError("The selection seed must be an integer")
    heldout_ids, excluded_audio_hashes = set(heldout_ids), set(excluded_audio_hashes)
    def rank(domain, value):
        return hashlib.sha256(f"{seed}|{domain}|{value}".encode()).hexdigest()
    candidates = defaultdict(list)
    hashes = {}
    for index, crop in enumerate(crops):
        if crop.source_id not in rows:
            raise ValueError("Training source is missing its immutable row")
        sha = _audio_sha(rows[crop.source_id])
        hashes[crop.source_id] = sha
        if crop.source_id in heldout_ids or sha in excluded_audio_hashes:
            continue
        if crop.valid_scored_samples <= 0:
            raise ValueError("Training crop has no valid scored audio")
        candidates[crop.source_id].append(index)
    representatives, used_hashes = [], set()
    for sid in sorted(candidates, key=lambda value: (rank("source", value), value)):
        if hashes[sid] in used_hashes:
            continue
        used_hashes.add(hashes[sid])
        indices = candidates[sid]
        index = min(indices, key=lambda i: (rank("crop", f"{sid}|{crops[i].start_frame}|{i}"), i))
        representatives.append(index)
    needed = fit_count + validation_count
    if len(representatives) < needed:
        raise ValueError(f"Need {needed} unique source/audio-hash representatives, found {len(representatives)}")
    selected = representatives[:needed]
    splits = {}
    for split, indices in (("fit", selected[:fit_count]), ("validation", selected[fit_count:])):
        details = []
        for index in indices:
            crop, row = crops[index], rows[crops[index].source_id]
            details.append({"pool_index": index, "source_id": crop.source_id,
                "audio_sha256": hashes[crop.source_id], "start_frame": crop.start_frame,
                "context_start_frame": crop.context_start_frame,
                "valid_scored_samples": crop.valid_scored_samples,
                "language": row.get("language"), "condition": row.get("condition"),
                "dataset": row.get("dataset")})
        splits[split] = {"indices": indices, "sources": details,
                        "source_count": len(details), "audio_hash_count": len({r["audio_sha256"] for r in details})}
    all_rows = [row for split in splits.values() for row in split["sources"]]
    if len({r["source_id"] for r in all_rows}) != needed or len({r["audio_sha256"] for r in all_rows}) != needed:
        raise RuntimeError("Source/hash disjointness failed")
    result = {"format_version": 1, "seed": seed, "splits": splits,
        "eligible_source_count": len(candidates), "eligible_unique_audio_hash_count": len(representatives),
        "source_and_audio_hash_disjoint": True, "heldout_source_and_audio_hash_disjoint": True,
        "selection_inputs": "Pinned source identity and deterministic ranking only; no model output, target level, or quality score",
        "earlier_exposure": "Previously used debugging audio may reappear; no source repeats within or across these fit/validation splits"}
    result["identity_sha256"] = _digest(result)
    return result


def complete_scored_frames(crop, features, *, block_samples=480, target_override=None):
    """Return H[N,C], T[N,480] for complete blocks inside the fixed score mask.

    The canonical six-sample interior exclusion is applied before rounding the
    fit interval inward to whole 480-sample synthesis blocks. No feature from
    outside that scored interval enters the sufficient statistics. This is a
    fit-only mask; final canonical evaluation keeps its original full mask.
    """
    if block_samples != 480:
        raise ValueError("This experiment fixes the native 480-sample head")
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("Expected singleton pre-output features [1,C,T]")
    target = crop.teacher_audio if target_override is None else target_override
    if target.shape != crop.teacher_audio.shape:
        raise ValueError("Target override must retain the exact full cached waveform geometry")
    if target.ndim != 3 or target.shape[:2] != (1, 1) or target.shape[-1] != features.shape[-1] * 480:
        raise ValueError("Pre-output frames and complete cached teacher waveform disagree")
    if not features.is_floating_point() or not target.is_floating_point():
        raise TypeError("Floating pre-output features and teacher waveform are required")
    start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
    stop = crop.scored_slice.stop
    if not 0 <= start < stop <= target.shape[-1]:
        raise ValueError("Invalid scored waveform interval")
    first, last = (start + 479) // 480, stop // 480
    if first >= last:
        raise ValueError("No complete scored synthesis block remains")
    h = features[0, :, first:last].transpose(0, 1)
    t = target[0, 0, first * 480:last * 480].reshape(last - first, 480).to(features.device)
    if not bool(torch.isfinite(h).all() and torch.isfinite(t).all()):
        raise ValueError("Nonfinite feature or teacher target")
    receipt = {"source_id": crop.source_id, "start_frame": crop.start_frame,
        "score_start_sample": start, "score_stop_sample": stop,
        "first_internal_frame": first, "stop_internal_frame": last,
        "fit_frames": last - first, "fit_samples": (last - first) * 480,
        "full_score_samples": stop - start, "fit_edge_excluded_samples": stop - start - (last - first) * 480,
        "fit_target": "cached_post_tanh_teacher" if target_override is None else "explicit_override_same_geometry"}
    return h, t, receipt


@torch.no_grad()
def validation_masks(crop, *, device, quiet_config=None):
    """Freeze valid and teacher-only 20 ms quiet masks before candidate scoring.

    Uses the existing quiet implementation on the original full waveform grid.
    A future pre-tanh regression target must never change these post-tanh
    validation masks or the waveform target used for quality comparisons.
    """
    from audiovae_student.quiet_audio import QuietAudioConfig, _window_values
    default = QuietAudioConfig()
    config = default if quiet_config is None else quiet_config
    if isinstance(config, dict):
        config = QuietAudioConfig(**config)
    if config != default or config.window_samples != 960:
        raise ValueError("This experiment preserves the fixed existing 20 ms quiet contract")
    teacher = crop.teacher_audio.detach().to(device)
    start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
    stop = crop.scored_slice.stop
    if not 0 <= start < stop <= teacher.shape[-1] or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Invalid canonical validation waveform geometry")
    valid = torch.zeros_like(teacher, dtype=torch.bool)
    valid[..., start:stop] = True
    _, _, _, _, windows = _window_values(teacher, teacher, valid, config)
    quiet = windows.repeat_interleave(config.window_samples, dim=-1).unsqueeze(1)[..., :teacher.shape[-1]] & valid
    return teacher, valid, quiet


def select_regularization(validation_rows, baseline):
    """Pick minimum post-waveform MSE among the four predeclared eligible fits.

    MAE and teacher-quiet MSE must each remain within 1% of the unchanged
    baseline on the separate training-validation split. A failed screen yields
    no selection, not a best failed model. Thresholds are fixed in this helper.
    """
    rows = list(validation_rows)
    if len(rows) != 4 or sorted(row["lambda"] for row in rows) != [.001, .01, .1, 1.]:
        raise ValueError("Exactly the four declared regularization candidates are required")
    for result in [baseline, *rows]:
        if any(not isinstance(result.get(key), (float, int)) or not math.isfinite(result[key]) or result[key] < 0
               for key in ("mse", "mae", "quiet_mse")):
            raise ValueError("Finite nonnegative waveform/quiet validation metrics are required")
        if any(type(result.get(key)) is not int or result[key] <= 0 for key in ("valid_samples", "quiet_samples")):
            raise ValueError("Positive valid and quiet sample counts are required")
        if any(result[key] != baseline[key] for key in ("valid_samples", "quiet_samples")):
            raise ValueError("Candidate validation sample masks/counts differ")
    outcomes = []
    for row in rows:
        passed_mae = row["mae"] <= baseline["mae"] * 1.01
        passed_quiet = row["quiet_mse"] <= baseline["quiet_mse"] * 1.01
        outcomes.append({**row, "mae_qualified": passed_mae, "quiet_qualified": passed_quiet,
                         "qualified": passed_mae and passed_quiet})
    qualified = [row for row in outcomes if row["qualified"]]
    # The largest regularization wins an exact equal-MSE tie, declared here.
    chosen = min(qualified, key=lambda row: (row["mse"], -row["lambda"])) if qualified else None
    return {"selected_lambda": chosen["lambda"] if chosen else None,
            "selected": chosen, "candidates": outcomes,
            "baseline": baseline, "max_mae_ratio": 1.01, "max_quiet_mse_ratio": 1.01,
            "selection": "Lowest training-validation post-waveform MSE among qualified candidates; largest lambda breaks exact ties",
            "heldout_used": False, "no_qualified_candidate": chosen is None}


class ReadoutStatistics:
    """FP64 sample-pooled Gram/cross moments without retaining feature arrays."""

    def __init__(self, weight):
        if weight.ndim == 3 and weight.shape[-1] == 1:
            weight = weight[..., 0]
        if weight.ndim != 2 or weight.shape[0] != 480 or not bool(torch.isfinite(weight).all()):
            raise ValueError("Expected finite existing output weights [480,C] or [480,C,1]")
        self.weight = weight.detach().to(dtype=torch.float64).clone()
        width = weight.shape[1]
        self.gram_sum = self.weight.new_zeros(width, width)
        self.cross_target_sum = self.weight.new_zeros(width, 480)
        self.target_square_sum = self.weight.new_zeros(())
        self.frames = 0
        self.crops = 0
        self._seen = set()

    @torch.no_grad()
    def add(self, h, target, *, source_id):
        if source_id in self._seen:
            raise ValueError("A source was repeated in fitting statistics")
        if h.ndim != 2 or h.shape[1] != self.weight.shape[1] or target.shape != (h.shape[0], 480) or h.shape[0] <= 0:
            raise ValueError("Unexpected feature/target matrix shape")
        h = h.detach().to(device=self.weight.device, dtype=torch.float64)
        target = target.detach().to(device=self.weight.device, dtype=torch.float64)
        if not bool(torch.isfinite(h).all() and torch.isfinite(target).all()):
            raise ValueError("Nonfinite sufficient-statistics input")
        self.gram_sum.addmm_(h.T, h)
        self.cross_target_sum.addmm_(h.T, target)
        self.target_square_sum.add_(target.square().sum())
        self.frames += h.shape[0]
        self.crops += 1
        self._seen.add(source_id)

    @torch.no_grad()
    def finalize(self):
        if not self.frames:
            raise ValueError("No fit frames accumulated")
        a = self.gram_sum / self.frames
        b = (self.cross_target_sum - self.gram_sum @ self.weight.T) / self.frames
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError("Nonfinite sufficient statistics")
        return {"A": a, "B": b, "frames": self.frames, "samples": self.frames * 480,
                "crops": self.crops, "feature_width": self.weight.shape[1],
                "trace_scale": float(a.trace() / a.shape[0]),
                "target_mean_square": float(self.target_square_sum / (self.frames * 480)),
                "definition": "A=mean(h*hT), B=mean(h*(teacher-W*h)T), FP64; equal weight per complete 480-sample block"}
