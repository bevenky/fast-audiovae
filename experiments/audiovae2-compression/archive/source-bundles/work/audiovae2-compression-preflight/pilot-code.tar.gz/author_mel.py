"""Original AudioVAE/DAC author mel objective, explicitly adapted to 48 kHz.

The confirmed pre-V2 recipe uses seven summed log10 mel distances. This module
reproduces DAC/audiotools defaults: periodic Hann, centered reflect-padded STFT,
Slaney mel filters, magnitude power one and no linear-magnitude loss. It is not
a claim that the complete checkpoint-specific AudioVAE2 recipe is published.

Sources:
https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845
https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/loss.py
https://github.com/descriptinc/audiotools/blob/348ebf2034ce24e2a91a553e3171cb00c0c71678/audiotools/core/audio_signal.py
https://github.com/librosa/librosa/blob/f808bac0812469049fd8167c87f45813b7d57b32/librosa/filters.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AuthorMelConfig:
    format_version: int = 1
    sample_rate: int = 48000
    fft_sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
    mel_bands: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
    log_epsilon: float = 1e-5

    def __post_init__(self):
        object.__setattr__(self, "fft_sizes", tuple(self.fft_sizes))
        object.__setattr__(self, "mel_bands", tuple(self.mel_bands))
        if self.format_version != 1 or type(self.format_version) is not int:
            raise ValueError("Unsupported author mel objective version")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if not self.fft_sizes or len(self.fft_sizes) != len(self.mel_bands):
            raise ValueError("Provide one mel band count for each nonempty FFT scale")
        if any(type(n) is not int or n < 4 or n % 4 for n in self.fft_sizes):
            raise ValueError("FFT sizes must be positive multiples of four")
        if any(type(n) is not int or n < 1 for n in self.mel_bands):
            raise ValueError("Mel band counts must be positive integers")
        if not math.isfinite(self.log_epsilon) or self.log_epsilon <= 0:
            raise ValueError("The magnitude floor must be finite and positive")


def periodic_hann(size: int) -> Tensor:
    """SciPy get_window('hann', size, fftbins=True), converted to FP32.

    Use its float64 generalized-cosine construction before conversion, rather
    than constructing the window directly in FP32 with torch.hann_window.
    """
    if type(size) is not int or size < 1:
        raise ValueError("Window length must be a positive integer")
    if size == 1:
        return torch.ones(1, dtype=torch.float32)
    phase = np.linspace(-np.pi, np.pi, size + 1)
    return torch.from_numpy((0.5 + 0.5*np.cos(phase))[:-1].astype(np.float32))


def _hz_to_mel(frequency):
    frequency = np.asarray(frequency, dtype=np.float64)
    mel = frequency/(200.0/3)
    mask = frequency >= 1000.0
    mel[mask] = 15.0 + np.log(frequency[mask]/1000.0)/(np.log(6.4)/27.0)
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    frequency = (200.0/3)*mel
    mask = mel >= 15.0
    frequency[mask] = 1000.0*np.exp((np.log(6.4)/27.0)*(mel[mask]-15.0))
    return frequency


def slaney_mel_bank(size: int, bands: int, sample_rate: int) -> Tensor:
    """librosa.filters.mel defaults, without importing its audio dependencies.

    Follow the upstream dtype order: assign triangular slopes into FP32 first,
    then multiply by float64 Slaney area normalization in-place. Empty filters
    are retained as zero rows, as in librosa; they are disclosed in provenance.
    """
    if any(type(value) is not int or value < 1 for value in (size, bands, sample_rate)):
        raise ValueError("FFT size, band count and sample rate must be positive integers")
    fft_frequencies = np.fft.rfftfreq(size, d=1.0/sample_rate)
    mel_limits = _hz_to_mel(np.array([0.0, sample_rate/2]))
    mel_frequencies = _mel_to_hz(np.linspace(mel_limits[0], mel_limits[1], bands+2))
    differences = np.diff(mel_frequencies)
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    weights = np.zeros((bands, 1+size//2), dtype=np.float32)
    for index in range(bands):
        lower = -ramps[index]/differences[index]
        upper = ramps[index+2]/differences[index+1]
        weights[index] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0/(mel_frequencies[2:]-mel_frequencies[:-2]))[:, np.newaxis]
    return torch.from_numpy(weights)


def center_reflect(audio: Tensor, padding: int) -> Tensor:
    """Native STFT center-reflection samples with deterministic backward.

    Reflection excludes each endpoint. Slice/flip/cat expresses the identical
    map without CUDA reflection_pad1d_backward, whose atomic reduction is
    rejected by torch.use_deterministic_algorithms(True).
    """
    if not isinstance(audio, Tensor) or audio.ndim < 1:
        raise ValueError("Reflection requires an audio tensor with a sample axis")
    if type(padding) is not int or padding < 0 or padding >= audio.shape[-1]:
        raise ValueError("Reflection padding must be nonnegative and shorter than the audio")
    if padding == 0:
        return audio
    return torch.cat((audio[..., 1:padding+1].flip(-1), audio,
                      audio[..., -padding-1:-1].flip(-1)), dim=-1)


@dataclass(frozen=True)
class AuthorMelTerms:
    config: AuthorMelConfig
    log_sums: tuple[Tensor, ...]
    mel_element_counts: tuple[int, ...]
    sample_count: int
    example_count: int


@dataclass(frozen=True)
class AuthorMelResult:
    losses: dict[str, Tensor]
    counts: dict[str, int | tuple[int, ...]]


class AuthorMelLoss(nn.Module):
    def __init__(self, config: AuthorMelConfig = AuthorMelConfig()):
        super().__init__()
        if not isinstance(config, AuthorMelConfig):
            raise TypeError("AuthorMelConfig is required")
        self.config = config
        for index, (size, bands) in enumerate(zip(config.fft_sizes, config.mel_bands)):
            self.register_buffer(f"window_{index}", periodic_hann(size))
            self.register_buffer(f"mel_{index}", slaney_mel_bank(size, bands, config.sample_rate))

    def element_counts(self, length: int, batch_size: int = 1) -> tuple[int, ...]:
        if type(length) is not int or length <= max(self.config.fft_sizes)//2:
            raise ValueError("A valid crop must be longer than half the largest FFT for native reflect padding")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return tuple(batch_size*bands*(length//(size//4)+1)
                     for size, bands in zip(self.config.fft_sizes, self.config.mel_bands))

    @property
    def provenance(self) -> dict:
        return {
            "config": asdict(self.config),
            "definition": "Sum of seven per-resolution means of absolute differences between log10 floored magnitude mel spectra",
            "source_scope": "Confirmed original AudioVAE/DAC recipe, evaluated at 48 kHz; not a verified complete AudioVAE2 training recipe",
            "center": True, "pad_mode": "reflect", "match_stride": False,
            "center_implementation": "Exact endpoint-excluding slice/flip/cat reflection, then center=False STFT; deterministic CUDA backward",
            "window": "Periodic Hann from float64 SciPy-equivalent construction, cast to FP32",
            "hop": "FFT size divided by four", "normalized_stft": False,
            "magnitude_power": 1, "linear_magnitude_weight": 0,
            "log_base": 10, "resolution_reduction": "sum",
            "mel": {"scale": "Slaney", "norm": "slaney", "htk": False,
                    "fmin": 0, "fmax": self.config.sample_rate/2, "dtype": "float32"},
            "empty_mel_rows": [torch.nonzero(getattr(self, f"mel_{i}").abs().sum(1)==0).flatten().cpu().tolist()
                               for i in range(len(self.config.fft_sizes))],
            "pooling": "Pool valid independent-crop mel elements per scale; never concatenate sources or mask invalid samples to zero",
            "crop_edges": "Reflect only the supplied contiguous scored crop; no extra context or match-stride frame removal",
        }

    def _mel(self, audio: Tensor, index: int) -> Tensor:
        size = self.config.fft_sizes[index]
        window = getattr(self, f"window_{index}").to(audio)
        bank = getattr(self, f"mel_{index}").to(audio)
        centered = center_reflect(audio[:,0], size//2)
        spectrum = torch.stft(centered, n_fft=size, hop_length=size//4, win_length=size,
                              window=window, center=False, normalized=False,
                              onesided=True, return_complex=True)
        # Match AudioSignal's time-by-frequency matrix orientation.
        return (spectrum.abs().transpose(1,2) @ bank.T).transpose(1,2)

    def group_terms(self, student: Tensor, teacher: Tensor) -> AuthorMelTerms:
        for name, value in (("student",student),("teacher",teacher)):
            if not isinstance(value,Tensor) or value.ndim != 3 or value.shape[0]<1 or value.shape[1]!=1:
                raise ValueError(f"{name} must be nonempty [batch,1,valid_samples] audio")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite real audio")
        if student.shape != teacher.shape or student.device != teacher.device:
            raise ValueError("Teacher and student must have identical aligned shapes and devices")
        counts = self.element_counts(student.shape[-1], student.shape[0])
        with torch.autocast(device_type=student.device.type, enabled=False):
            prediction, target = student.float(), teacher.detach().float()
            sums = []
            for index in range(len(self.config.fft_sizes)):
                p = self._mel(prediction,index).clamp_min(self.config.log_epsilon).log10()
                t = self._mel(target,index).clamp_min(self.config.log_epsilon).log10()
                if p.numel() != counts[index]:
                    raise RuntimeError("Native centered-STFT frame count differs from its denominator")
                sums.append((p-t).abs().sum())
        return AuthorMelTerms(self.config,tuple(sums),counts,student.numel(),student.shape[0])

    def _terms(self, terms: AuthorMelTerms | Iterable[AuthorMelTerms]) -> tuple[AuthorMelTerms, ...]:
        values = (terms,) if isinstance(terms,AuthorMelTerms) else tuple(terms)
        if not values or any(not isinstance(t,AuthorMelTerms) or t.config!=self.config for t in values):
            raise ValueError("Matching nonempty author mel terms are required")
        device = values[0].log_sums[0].device
        if any(len(t.log_sums)!=len(self.config.fft_sizes) or len(t.mel_element_counts)!=len(self.config.fft_sizes)
               or any(v.device!=device for v in t.log_sums) for t in values):
            raise ValueError("Term scale counts or devices differ")
        return values

    def loss_from_terms(self, terms: AuthorMelTerms | Iterable[AuthorMelTerms],
                        mel_element_counts: tuple[int, ...] | None = None) -> Tensor:
        values = self._terms(terms)
        local_counts = tuple(sum(t.mel_element_counts[i] for t in values) for i in range(len(self.config.fft_sizes)))
        counts = local_counts if mel_element_counts is None else tuple(mel_element_counts)
        if len(counts)!=len(local_counts) or any(type(n) is not int or n<local for n,local in zip(counts,local_counts)):
            raise ValueError("Fixed per-scale denominators must contain the contributing local elements")
        scales = [torch.stack([t.log_sums[i] for t in values]).sum()/counts[i] for i in range(len(counts))]
        # Native DAC adds per-scale means; averaging these seven values would
        # silently divide the objective and its gradients by seven.
        loss = scales[0]
        for value in scales[1:]:
            loss = loss+value
        return loss

    def aggregate(self, terms: AuthorMelTerms | Iterable[AuthorMelTerms]) -> AuthorMelResult:
        values = self._terms(terms)
        counts = tuple(sum(t.mel_element_counts[i] for t in values) for i in range(len(self.config.fft_sizes)))
        loss = self.loss_from_terms(values,counts)
        return AuthorMelResult({"teacher_mel":loss,"total":loss}, {
            "valid_samples":sum(t.sample_count for t in values), "examples":sum(t.example_count for t in values),
            "mel_elements_by_resolution":counts,
            "mel_frames_by_resolution":tuple(n//bands for n,bands in zip(counts,self.config.mel_bands)),
        })

    def forward(self, student: Tensor, teacher: Tensor) -> AuthorMelResult:
        return self.aggregate(self.group_terms(student,teacher))
