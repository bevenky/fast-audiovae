"""Differentiable reconstruction losses for decoder warmup.

The fullband target is the frozen teacher's 48 kHz waveform. A 16 kHz
recording supervises only matching frequencies up to its Nyquist frequency.
These are starting loss weights for preflight, not validated quality settings.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class WarmupLossConfig:
    teacher_fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    reference_fft_sizes_16k: tuple[int, ...] = (256, 512, 1024)
    teacher_spectral_weight: float = 15.0
    teacher_waveform_weight: float = 1.0
    reference_spectral_weight: float = 45.0
    log_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("teacher_fft_sizes", "reference_fft_sizes_16k"):
            sizes = getattr(self, name)
            if not sizes or any(not isinstance(n, int) or n < 8 or n % 4 for n in sizes):
                raise ValueError(f"{name} must contain FFT sizes >= 8 divisible by four")
        weights = (self.teacher_spectral_weight, self.teacher_waveform_weight,
                   self.reference_spectral_weight)
        if any(not torch.isfinite(torch.tensor(w)) or w < 0 for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.teacher_spectral_weight + self.teacher_waveform_weight <= 0:
            raise ValueError("At least one fullband teacher loss must be enabled")
        if not 0 < self.log_epsilon < 1:
            raise ValueError("log_epsilon must be between zero and one")


def _check_waveform(name: str, waveform: Tensor) -> None:
    if waveform.ndim != 3 or waveform.shape[1] != 1 or waveform.shape[0] < 1:
        raise ValueError(f"{name} must have shape [batch, 1, samples]")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError(f"{name} must contain finite floating-point audio")


def _magnitude(waveform: Tensor, n_fft: int) -> Tensor:
    if waveform.shape[-1] < n_fft:
        raise ValueError(f"Audio has {waveform.shape[-1]} samples but FFT needs {n_fft}")
    # Normalize by the window sum so equal physical amplitudes are comparable
    # across FFT sizes and across the 48/16 kHz reference views.
    window = torch.hann_window(n_fft, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(waveform[:, 0], n_fft=n_fft, hop_length=n_fft // 4,
                          window=window, center=False, return_complex=True)
    return spectrum.abs() / window.sum()


class WarmupReconstructionLoss(nn.Module):
    """Return a scalar total and its unweighted components.

    Teacher and original-reference tensors are detached defensively. Student
    STFTs remain differentiable. There is no best-lag alignment, rescaling,
    truncation, loudness normalization, or padding inside this loss.
    """

    def __init__(self, config: WarmupLossConfig = WarmupLossConfig()) -> None:
        super().__init__()
        self.config = config

    def forward(self, student_audio: Tensor, teacher_audio: Tensor,
                reference_audio_16k: Tensor | None = None) -> dict[str, Tensor]:
        _check_waveform("student_audio", student_audio)
        _check_waveform("teacher_audio", teacher_audio)
        if student_audio.shape != teacher_audio.shape:
            raise ValueError("Student and teacher must have identical, aligned shapes")
        if student_audio.device != teacher_audio.device:
            raise ValueError("Student and teacher must be on the same device")

        # STFT/log reductions are FP32 even when a future caller uses AMP.
        with torch.autocast(device_type=student_audio.device.type, enabled=False):
            student = student_audio.float()
            teacher = teacher_audio.detach().float()
            eps = self.config.log_epsilon
            spectral_terms = []
            for size in self.config.teacher_fft_sizes:
                prediction = _magnitude(student, size)
                target = _magnitude(teacher, size)
                spectral_terms.append((prediction.clamp_min(eps).log()
                                       - target.clamp_min(eps).log()).abs().mean())
            teacher_spectral = torch.stack(spectral_terms).mean()
            teacher_waveform = (student - teacher).abs().mean()
            reference_spectral = student.new_zeros(())

            if reference_audio_16k is not None:
                _check_waveform("reference_audio_16k", reference_audio_16k)
                reference = reference_audio_16k.detach().float()
                if reference.device != student.device or reference.shape[:2] != student.shape[:2]:
                    raise ValueError("Original reference must match student batch and device")
                if reference.shape[-1] * 3 != student.shape[-1]:
                    raise ValueError("48 kHz student length must be exactly 3x the 16 kHz reference")
                reference_terms = []
                for size in self.config.reference_fft_sizes_16k:
                    # These windows have identical duration and frequency-bin
                    # spacing. Discard student bins above the true 8 kHz limit.
                    prediction = _magnitude(student, size * 3)[:, :size // 2 + 1]
                    target = _magnitude(reference, size)
                    reference_terms.append((prediction.clamp_min(eps).log()
                                            - target.clamp_min(eps).log()).abs().mean())
                reference_spectral = torch.stack(reference_terms).mean()

            total = (self.config.teacher_spectral_weight * teacher_spectral
                     + self.config.teacher_waveform_weight * teacher_waveform
                     + self.config.reference_spectral_weight * reference_spectral)
        return {"total": total, "teacher_spectral": teacher_spectral,
                "teacher_waveform": teacher_waveform,
                "reference_spectral": reference_spectral}
