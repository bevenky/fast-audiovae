"""Teacher-conditioned checks for silence and low-level audio.

Training and acceptance use this module only through explicit opt-in controls.
Thresholds are provisional engineering checks, not calibrated audibility
limits. Residual supervision preserves the teacher's low-level signal instead
of muting quiet windows. A zeroed or phase-inverted breath remains an error.

Windows follow the original tensor-time grid. Masks exclude context, holes and
padding without concatenating disjoint samples. Every valid partial-tail sample
participates. Loss reduction is the mean over quiet windows within each example,
then the mean over examples with at least one quiet window. Examples with no
quiet window do not dilute the loss.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class QuietAudioConfig:
    window_samples: int = 960  # 20 ms at the decoder's 48 kHz sample rate.
    quiet_teacher_rms_max: float = 1e-3
    loss_rms_floor: float = 1e-3
    residual_relative_limit: float = math.sqrt(0.02)
    absolute_rms_floor: float = 1e-5
    amplitude_db_max: float = 1.0

    def __post_init__(self):
        if type(self.window_samples) is not int or self.window_samples < 1:
            raise ValueError("window_samples must be a positive integer")
        for name in ("quiet_teacher_rms_max", "loss_rms_floor", "residual_relative_limit",
                     "absolute_rms_floor", "amplitude_db_max"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        try:
            finite_amplitude = math.isfinite(10.0 ** (self.amplitude_db_max / 20))
        except OverflowError:
            finite_amplitude = False
        if not finite_amplitude:
            raise ValueError("amplitude_db_max must produce a finite amplitude ratio")


def _window_values(student: Tensor, teacher: Tensor, valid_mask: Tensor | None,
                   config: QuietAudioConfig):
    if (student.ndim != 3 or student.shape[0] < 1 or student.shape[1] != 1
            or student.shape[-1] < 1 or teacher.shape != student.shape):
        raise ValueError("Student and teacher require identical nonempty [batch, 1, samples] shapes")
    if not student.is_floating_point() or not teacher.is_floating_point() or student.device != teacher.device:
        raise ValueError("Student and teacher must be floating-point tensors on the same device")
    if valid_mask is None:
        valid_mask = torch.ones_like(student, dtype=torch.bool)
    elif valid_mask.shape != student.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with exactly the audio shape")
    valid_mask = valid_mask.detach().to(student.device)
    if not bool(torch.isfinite(student.masked_select(valid_mask)).all()):
        raise ValueError("Scored student samples must be finite")
    if not bool(torch.isfinite(teacher.detach().masked_select(valid_mask)).all()):
        raise ValueError("Scored teacher samples must be finite")
    # Mask before any arithmetic so excluded NaN/Inf context cannot leak into
    # values or backward. Accumulate half/bfloat audio in at least float32.
    dtype = torch.promote_types(torch.promote_types(student.dtype, teacher.dtype), torch.float32)
    predicted = student.masked_fill(~valid_mask, 0).to(dtype)
    target = teacher.detach().masked_fill(~valid_mask, 0).to(dtype)
    padding = (-student.shape[-1]) % config.window_samples
    shape = (student.shape[0], -1, config.window_samples)
    predicted = F.pad(predicted, (0, padding)).reshape(shape)
    target = F.pad(target, (0, padding)).reshape(shape)
    mask = F.pad(valid_mask, (0, padding)).reshape(shape)
    counts = mask.sum(dim=-1)
    denominator = counts.clamp_min(1).to(dtype).sqrt()
    # vector_norm has a finite zero subgradient, unlike a naive sqrt(mean(x*x)).
    teacher_rms = torch.linalg.vector_norm(target, dim=-1) / denominator
    student_rms = torch.linalg.vector_norm(predicted, dim=-1) / denominator
    residual_rms = torch.linalg.vector_norm(predicted - target, dim=-1) / denominator
    quiet = (counts > 0) & (teacher_rms <= config.quiet_teacher_rms_max)
    return counts, teacher_rms, student_rms, residual_rms, quiet


def quiet_residual_loss(student: Tensor, teacher: Tensor, valid_mask: Tensor | None = None,
                        *, config: QuietAudioConfig = QuietAudioConfig()) -> Tensor:
    """Mean normalized quiet-window residual RMS, differentiable only to student.

    Teacher RMS determines window eligibility and normalization with a detached
    floor of 1e-3 by default. The floor caps gain on near-silence. An empty quiet
    selection returns a connected zero with finite zero student gradients.
    """
    counts, teacher_rms, _, residual_rms, quiet = _window_values(student, teacher, valid_mask, config)
    normalized = residual_rms / teacher_rms.clamp_min(config.loss_rms_floor)
    per_example_counts = quiet.sum(dim=-1)
    per_example = normalized.masked_fill(~quiet, 0).sum(dim=-1) / per_example_counts.clamp_min(1)
    active_examples = per_example_counts > 0
    return per_example.masked_fill(~active_examples, 0).sum() / active_examples.sum().clamp_min(1)


@torch.no_grad()
def quiet_window_metrics(student: Tensor, teacher: Tensor, valid_mask: Tensor | None = None,
                         *, config: QuietAudioConfig = QuietAudioConfig()) -> dict:
    """Return JSON-compatible per-window evidence, including quiet failures.

    Quiet acceptance requires both a bounded residual from the original teacher
    and a bounded output RMS. Exact silence needs no cosine calculation. When
    no quiet windows exist, ``quiet_passed`` is None rather than a quality pass.
    """
    counts, teacher_rms, student_rms, residual_rms, quiet = _window_values(student, teacher, valid_mask, config)
    residual_limits = (teacher_rms * config.residual_relative_limit).clamp_min(config.absolute_rms_floor)
    output_limits = (teacher_rms * 10.0 ** (config.amplitude_db_max / 20)).clamp_min(config.absolute_rms_floor)
    accepted = (residual_rms <= residual_limits) & (student_rms <= output_limits)
    matrices = [value.detach().cpu().tolist() for value in
                (counts, teacher_rms, student_rms, residual_rms, quiet, residual_limits, output_limits, accepted)]
    rows, examples = [], []
    for batch_index in range(student.shape[0]):
        example_rows = []
        for window_index in range(counts.shape[1]):
            count, trms, srms, rrms, is_quiet, rlimit, olimit, passed = (
                matrix[batch_index][window_index] for matrix in matrices)
            if not count:
                continue
            row = {"batch_index": batch_index, "window_index": window_index,
                   "start_sample": window_index * config.window_samples,
                   "stop_sample": min((window_index + 1) * config.window_samples, student.shape[-1]),
                   "valid_samples": count, "teacher_rms": trms, "student_rms": srms,
                   "residual_rms": rrms, "is_quiet": is_quiet, "residual_limit": rlimit,
                   "output_rms_limit": olimit, "passed": passed if is_quiet else None}
            rows.append(row)
            example_rows.append(row)
        quiet_count = sum(row["is_quiet"] for row in example_rows)
        failed_count = sum(row["passed"] is False for row in example_rows)
        examples.append({"batch_index": batch_index,
                         "valid_samples": sum(row["valid_samples"] for row in example_rows),
                         "window_count": len(example_rows), "quiet_window_count": quiet_count,
                         "quiet_failed_count": failed_count,
                         "quiet_passed": failed_count == 0 if quiet_count else None})
    quiet_count = sum(row["is_quiet"] for row in rows)
    failed_count = sum(row["passed"] is False for row in rows)
    return {"config": asdict(config), "sample_rate": 48000,
            "window_alignment": "original_tensor_time_grid_with_invalid_samples_excluded",
            "valid_samples": sum(row["valid_samples"] for row in rows), "window_count": len(rows),
            "quiet_window_count": quiet_count, "quiet_failed_count": failed_count,
            "quiet_passed": failed_count == 0 if quiet_count else None,
            "examples": examples, "windows": rows}
