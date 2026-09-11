"""Teacher-only reconstruction with explicitly defined magnitude mel analysis.

All reductions average elements within an example, then resolutions/examples.
Hann STFTs are unnormalized, uncentered and one-sided. The triangular filters
use both Slaney's frequency scale and area normalization. Waveforms themselves
are never rescaled or realigned. Call only on the valid scored region.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DistillationLossConfig:
    sample_rate: int = 48000
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    mel_bands: tuple[int, ...] = (64, 128, 128)
    log_epsilon: float = 1e-5
    waveform_rms_floor: float = 1e-3
    mel_linear_weight: float = 1.0
    mel_log_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fft_sizes", tuple(self.fft_sizes))
        object.__setattr__(self, "mel_bands", tuple(self.mel_bands))
        if self.sample_rate != 48000:
            raise ValueError("Distillation targets must be full-band 48 kHz")
        if not self.fft_sizes or len(self.fft_sizes) != len(self.mel_bands):
            raise ValueError("Each FFT size needs a mel-band count")
        if any(type(n) is not int or n < 8 or n % 4 for n in self.fft_sizes):
            raise ValueError("FFT sizes must be integers >= 8 divisible by four")
        if any(type(n) is not int or n < 1 for n in self.mel_bands):
            raise ValueError("Mel-band counts must be positive integers")
        for name in ("log_epsilon", "waveform_rms_floor"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        weights = (self.mel_linear_weight, self.mel_log_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Mel weights must be finite, nonnegative and not both zero")


def slaney_mel_filterbank(n_fft: int, n_mels: int, sample_rate: int = 48000) -> Tensor:
    """Return [mel, frequency] Slaney triangles, computed in float64 then FP32."""
    if n_fft < 8 or n_mels < 1 or sample_rate <= 2000:
        raise ValueError("Invalid filterbank dimensions/sample rate")
    linear_step = 200.0 / 3
    log_step = math.log(6.4) / 27
    max_hz = sample_rate / 2
    max_mel = 15 + math.log(max_hz / 1000) / log_step
    mel = torch.linspace(0, max_mel, n_mels + 2, dtype=torch.float64)
    hz = torch.where(mel < 15, linear_step * mel,
                     1000 * torch.exp(log_step * (mel - 15)))
    hz[0], hz[-1] = 0.0, max_hz
    frequencies = torch.linspace(0, max_hz, n_fft // 2 + 1, dtype=torch.float64)
    rising = (frequencies[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    falling = (hz[2:, None] - frequencies[None]) / (hz[2:] - hz[1:-1])[:, None]
    filters = torch.minimum(rising, falling).clamp_min(0)
    filters *= (2 / (hz[2:] - hz[:-2]))[:, None]
    if bool((filters.sum(dim=1) == 0).any()):
        raise ValueError("An empty mel filter requires fewer bands or a larger FFT")
    return filters.float()


def _check_audio(name: str, audio: Tensor) -> None:
    if audio.ndim != 3 or audio.shape[0] == 0 or audio.shape[1] != 1 or audio.shape[2] == 0:
        raise ValueError(f"{name} must be nonempty [batch, 1, samples] audio")
    if not audio.is_floating_point() or not bool(torch.isfinite(audio).all()):
        raise ValueError(f"{name} must contain finite floating-point audio")


class DistillationReconstructionLoss(nn.Module):
    """Teacher mel plus normalized waveform loss; reference16k is not supervised.

    ``total`` is the unbalanced sum for diagnostics. The trainer balances the
    two scalar ``teacher_waveform`` and ``teacher_mel`` branches explicitly.
    Linear and log mel have declared weights, not the former coefficient 45.
    """

    def __init__(self, config: DistillationLossConfig = DistillationLossConfig()) -> None:
        super().__init__()
        self.config = config
        for i, (size, bands) in enumerate(zip(config.fft_sizes, config.mel_bands)):
            self.register_buffer(f"window_{i}", torch.hann_window(size, periodic=True))
            self.register_buffer(f"mel_{i}", slaney_mel_filterbank(size, bands, config.sample_rate))

    def forward(self, student: Tensor, teacher: Tensor,
                reference16k: Tensor | None = None) -> dict[str, Tensor]:
        _check_audio("student", student)
        _check_audio("teacher", teacher)
        if student.shape != teacher.shape or student.device != teacher.device:
            raise ValueError("Student and teacher must have identical aligned shapes and devices")
        if student.shape[-1] < max(self.config.fft_sizes):
            raise ValueError("The actual scored region is shorter than the largest FFT")
        # reference16k is deliberately ignored. It remains evaluation data only.
        with torch.autocast(device_type=student.device.type, enabled=False):
            prediction, target = student.float(), teacher.detach().float()
            raw_per_item = (prediction - target).abs().mean(dim=(1, 2))
            target_rms = target.square().mean(dim=(1, 2)).sqrt()
            waveform = (raw_per_item / target_rms.clamp_min(self.config.waveform_rms_floor)).mean()
            linear_terms, log_terms = [], []
            for i, size in enumerate(self.config.fft_sizes):
                window = getattr(self, f"window_{i}").to(prediction)
                bank = getattr(self, f"mel_{i}").to(prediction)
                def mel(audio: Tensor) -> Tensor:
                    spectrum = torch.stft(audio[:, 0], n_fft=size, hop_length=size // 4,
                                          win_length=size, window=window, center=False,
                                          normalized=False, onesided=True, return_complex=True)
                    return torch.matmul(bank, spectrum.abs())
                p, t = mel(prediction), mel(target)
                linear_terms.append((p - t).abs().mean(dim=(1, 2)))
                log_terms.append((p.clamp_min(self.config.log_epsilon).log()
                                  - t.clamp_min(self.config.log_epsilon).log()).abs().mean(dim=(1, 2)))
            linear = torch.stack(linear_terms).mean()
            logarithmic = torch.stack(log_terms).mean()
            spectral = self.config.mel_linear_weight * linear + self.config.mel_log_weight * logarithmic
            rms_error = (prediction.square().mean(dim=(1, 2)).sqrt() - target_rms).abs().mean()
        return {"total": waveform + spectral, "teacher_waveform_raw": raw_per_item.mean(),
                "teacher_waveform": waveform, "teacher_mel": spectral,
                "teacher_mel_linear": linear, "teacher_mel_log": logarithmic,
                "teacher_rms_error": rms_error}
