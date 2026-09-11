"""Isolated training candidate for a repeating waveform-head residual in pauses.

This module is not integrated into any training recipe. It penalizes the mean
480-sample phase template of the student-minus-teacher residual over at least
eight complete, valid, teacher-quiet 20 ms windows per example. It does not mute,
filter, subtract a template at inference, or penalize matching teacher room tone.
Thresholds are provisional engineering settings, not audibility guarantees.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class QuietPhaseConfig:
    period_samples: int = 480
    window_samples: int = 960
    minimum_windows: int = 8
    quiet_teacher_rms_max: float = 1e-3
    loss_rms_floor: float = 1e-3

    def __post_init__(self):
        if type(self.period_samples) is not int or self.period_samples < 1:
            raise ValueError("period_samples must be a positive integer")
        if type(self.window_samples) is not int or self.window_samples != 2 * self.period_samples:
            raise ValueError("window_samples must contain exactly two complete periods")
        if type(self.minimum_windows) is not int or self.minimum_windows < 8:
            raise ValueError("minimum_windows must be at least eight complete windows")
        for name in ("quiet_teacher_rms_max", "loss_rms_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def _phase_values(student: Tensor, teacher: Tensor, valid_mask: Tensor | None,
                  config: QuietPhaseConfig):
    if (student.ndim != 3 or student.shape[0] < 1 or student.shape[1] != 1
            or student.shape[-1] < 1 or teacher.shape != student.shape):
        raise ValueError("Student and teacher require identical nonempty [batch, 1, samples] shapes")
    if (not student.is_floating_point() or not teacher.is_floating_point()
            or student.device != teacher.device):
        raise ValueError("Student and teacher must be floating-point tensors on the same device")
    if valid_mask is None:
        valid_mask = torch.ones_like(student, dtype=torch.bool)
    elif valid_mask.dtype != torch.bool or valid_mask.shape != student.shape:
        raise ValueError("valid_mask must be bool with exactly the audio shape")
    valid_mask = valid_mask.detach().to(student.device)
    target = teacher.detach()
    if not bool(torch.isfinite(student.masked_select(valid_mask)).all()):
        raise ValueError("Scored student samples must be finite")
    if not bool(torch.isfinite(target.masked_select(valid_mask)).all()):
        raise ValueError("Scored teacher samples must be finite")
    dtype = torch.promote_types(torch.promote_types(student.dtype, teacher.dtype), torch.float32)
    # Mask before arithmetic so NaN/Inf in excluded context or padding cannot
    # enter the forward value or backward path. Original grid alignment remains.
    prediction = student.masked_fill(~valid_mask, 0).to(dtype)
    target = target.masked_fill(~valid_mask, 0).to(dtype)
    padding = (-student.shape[-1]) % config.window_samples
    shape = (student.shape[0], -1, config.window_samples)
    prediction = F.pad(prediction, (0, padding)).reshape(shape)
    target = F.pad(target, (0, padding)).reshape(shape)
    mask = F.pad(valid_mask, (0, padding)).reshape(shape)
    complete = mask.all(dim=-1)
    teacher_window_rms = torch.linalg.vector_norm(target, dim=-1) / math.sqrt(config.window_samples)
    quiet = complete & (teacher_window_rms <= config.quiet_teacher_rms_max)
    quiet_counts = quiet.sum(dim=-1)
    active = quiet_counts >= config.minimum_windows
    eligible = quiet & active.unsqueeze(-1)
    eligible_counts = eligible.sum(dim=-1)
    selected_target = target.masked_fill(~eligible.unsqueeze(-1), 0)
    selected_prediction = prediction.masked_fill(~eligible.unsqueeze(-1), 0)
    # Each complete 20 ms window contains two aligned waveform-head periods.
    # Summing their residual templates does not concatenate disjoint timelines.
    residual = (selected_prediction - selected_target).reshape(
        student.shape[0], -1, 2, config.period_samples)
    template = residual.sum(dim=(1, 2)) / (eligible_counts * 2).clamp_min(1).unsqueeze(-1)
    template_rms = torch.linalg.vector_norm(template, dim=-1) / math.sqrt(config.period_samples)
    teacher_rms = torch.linalg.vector_norm(selected_target.flatten(1), dim=-1) / (
        eligible_counts * config.window_samples).clamp_min(1).to(dtype).sqrt()
    normalized = template_rms / teacher_rms.clamp_min(config.loss_rms_floor)
    # vector_norm has a finite zero subgradient. The connected zero also covers
    # wholly masked or nonquiet inputs without dividing by an empty selection.
    loss = normalized.masked_fill(~active, 0).sum() / active.sum().clamp_min(1)
    return loss, complete, quiet_counts, eligible_counts, active, teacher_rms, template_rms, normalized


def quiet_phase_loss(student: Tensor, teacher: Tensor, valid_mask: Tensor | None = None,
                     *, config: QuietPhaseConfig = QuietPhaseConfig()) -> Tensor:
    """Normalized RMS of the mean residual phase template; student-only gradients.

    Reduce over qualifying windows within each example, then average active
    examples equally. Partial windows, context, holes, loud teacher windows and
    examples with fewer than eight qualifying windows contribute no gradient.
    Matching teacher periodic audio costs zero. This is not a general noise loss:
    nonrepeating residuals can cancel, which is the intended targeted behavior.
    """
    return _phase_values(student, teacher, valid_mask, config)[0]


@torch.no_grad()
def quiet_phase_metrics(student: Tensor, teacher: Tensor, valid_mask: Tensor | None = None,
                        *, config: QuietPhaseConfig = QuietPhaseConfig()) -> dict:
    """JSON-safe eligibility and template evidence, with no inferred quality pass."""
    loss, complete, quiet, eligible, active, teacher_rms, template_rms, normalized = _phase_values(
        student, teacher, valid_mask, config)
    matrices = [value.cpu().tolist() for value in
                (complete.sum(dim=-1), quiet, eligible, active, teacher_rms, template_rms, normalized)]
    examples = []
    for i in range(student.shape[0]):
        full, quiet_count, count, is_active, target_rms, residual_rms, normalized_rms = (
            values[i] for values in matrices)
        examples.append({"batch_index": i, "complete_valid_windows": full,
                         "teacher_quiet_complete_windows": quiet_count,
                         "eligible_windows": count, "eligible_head_blocks": count * 2,
                         "active": is_active, "teacher_rms": target_rms if is_active else None,
                         "residual_template_rms": residual_rms if is_active else None,
                         "normalized_template_rms": normalized_rms if is_active else None})
    return {"config": asdict(config),
            "alignment": "Original tensor grid; complete valid windows only; no joining across holes",
            "loss": float(loss.cpu()), "active_examples": int(active.sum().cpu()),
            "complete_valid_windows": int(complete.sum().cpu()),
            "teacher_quiet_complete_windows": int(quiet.sum().cpu()),
            "eligible_windows": int(eligible.sum().cpu()), "examples": examples,
            "scope": "Training-only phase-residual candidate; no acceptance threshold or inference operation"}
