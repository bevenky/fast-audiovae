"""Reconstruction whose examples contribute by valid samples and mel frames.

Pass already sliced, aligned scored regions. Batching or splitting examples
into equal-length groups does not change their weights. Do not concatenate
independent recordings along time: that would create artificial STFT frames.
Legacy diagnostics retain their per-example reductions and are detached.
"""

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .losses_distillation import DistillationLossConfig, _check_audio, slaney_mel_filterbank


@dataclass(frozen=True)
class ReconstructionV2Config:
    format_version: int = 2
    sample_rate: int = 48000
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    mel_bands: tuple[int, ...] = (64, 128, 128)
    log_epsilon: float = 1e-5
    mel_linear_weight: float = 1.0
    mel_log_weight: float = 1.0

    def __post_init__(self):
        if type(self.format_version) is not int or self.format_version != 2:
            raise ValueError("ReconstructionV2 requires objective format version 2")
        object.__setattr__(self, "fft_sizes", tuple(self.fft_sizes))
        object.__setattr__(self, "mel_bands", tuple(self.mel_bands))
        self.legacy_config()  # Shared spectral configuration validation.

    def legacy_config(self) -> DistillationLossConfig:
        """Configuration for comparable diagnostics, never the new gradient."""
        fields = asdict(self)
        fields.pop("format_version")
        return DistillationLossConfig(**fields)


@dataclass(frozen=True)
class ReconstructionV2Terms:
    config: ReconstructionV2Config
    waveform_absolute_sum: Tensor
    sample_count: int
    example_count: int
    mel_linear_sums: tuple[Tensor, ...]
    mel_log_sums: tuple[Tensor, ...]
    mel_element_counts: tuple[int, ...]
    # Each is a sum of per-example means, for historical diagnostics only.
    legacy_sums: dict[str, Tensor]


@dataclass(frozen=True)
class ReconstructionV2Result:
    losses: dict[str, Tensor]
    legacy_diagnostics: dict[str, Tensor]
    counts: dict[str, int | tuple[int, ...]]


class ReconstructionV2(nn.Module):
    """Raw waveform MAE plus magnitude mel losses with global valid counts.

    ``teacher_waveform`` is sum(abs(prediction - teacher)) / valid_samples.
    Mel terms pool valid time-frequency elements separately per resolution,
    then average resolutions. No signal-dependent waveform weighting is used.
    ``total`` is an unweighted diagnostic sum; recipe coefficients belong to
    the training engine. ``legacy_diagnostics`` must never drive backward.
    """

    def __init__(self, config: ReconstructionV2Config = ReconstructionV2Config()):
        super().__init__()
        if not isinstance(config, ReconstructionV2Config):
            raise TypeError("A ReconstructionV2Config is required")
        self.config = config
        for i, (size, bands) in enumerate(zip(config.fft_sizes, config.mel_bands)):
            self.register_buffer(f"window_{i}", torch.hann_window(size, periodic=True))
            self.register_buffer(f"mel_{i}", slaney_mel_filterbank(size, bands, config.sample_rate))

    def group_terms(self, student: Tensor, teacher: Tensor) -> ReconstructionV2Terms:
        """Compute sufficient statistics for one unpadded equal-length group.

        A group contains [B, 1, valid_samples] audio. Score masks, context and
        padded tails must be removed before calling. This requirement lets the
        STFT use only fully valid windows, including for short final crops.
        """
        _check_audio("student", student)
        _check_audio("teacher", teacher)
        if student.shape != teacher.shape or student.device != teacher.device:
            raise ValueError("Student and teacher need identical aligned shapes and devices")
        if student.shape[-1] < max(self.config.fft_sizes):
            raise ValueError("Every valid scored region must accommodate the largest FFT")
        with torch.autocast(device_type=student.device.type, enabled=False):
            prediction, target = student.float(), teacher.detach().float()
            absolute = (prediction - target).abs()
            raw_per_item = absolute.mean(dim=(1, 2))
            target_rms = target.square().mean(dim=(1, 2)).sqrt()
            linear_sums, log_sums, counts = [], [], []
            legacy_linear, legacy_log = [], []
            for i, size in enumerate(self.config.fft_sizes):
                window = getattr(self, f"window_{i}").to(prediction)
                bank = getattr(self, f"mel_{i}").to(prediction)

                def mel(audio):
                    spectrum = torch.stft(audio[:, 0], n_fft=size, hop_length=size // 4,
                        win_length=size, window=window, center=False, normalized=False,
                        onesided=True, return_complex=True)
                    return torch.matmul(bank, spectrum.abs())

                p, t = mel(prediction), mel(target)
                linear = (p - t).abs()
                logarithmic = (p.clamp_min(self.config.log_epsilon).log()
                               - t.clamp_min(self.config.log_epsilon).log()).abs()
                linear_sums.append(linear.sum())
                log_sums.append(logarithmic.sum())
                counts.append(linear.numel())
                legacy_linear.append(linear.detach().mean(dim=(1, 2)))
                legacy_log.append(logarithmic.detach().mean(dim=(1, 2)))
            # Diagnostics use the original default floor but cannot introduce
            # that inverse-RMS derivative into the new waveform objective.
            legacy = {
                "teacher_waveform_raw": raw_per_item.detach().sum(),
                "teacher_waveform": (raw_per_item.detach() / target_rms.clamp_min(
                    self.config.legacy_config().waveform_rms_floor)).sum(),
                "teacher_mel_linear": torch.stack(legacy_linear).mean(0).sum(),
                "teacher_mel_log": torch.stack(legacy_log).mean(0).sum(),
                "teacher_rms_error": (prediction.detach().square().mean(dim=(1, 2)).sqrt()
                                      - target_rms).abs().sum(),
            }
        return ReconstructionV2Terms(self.config, absolute.sum(), absolute.numel(),
            prediction.shape[0], tuple(linear_sums), tuple(log_sums), tuple(counts), legacy)

    def aggregate(self, groups: Iterable[ReconstructionV2Terms]) -> ReconstructionV2Result:
        """Combine groups by counts, never by an average of group means.

        This can combine independently computed minibatch partitions without
        changing example or STFT boundaries. Splitting an audio clip in time
        changes available STFT windows, so temporal splits are not equivalent.
        """
        groups = tuple(groups)
        if not groups:
            raise ValueError("At least one valid reconstruction group is required")
        if any(not isinstance(g, ReconstructionV2Terms) or g.config != self.config for g in groups):
            raise ValueError("Every group must use the same reconstruction configuration")
        device = groups[0].waveform_absolute_sum.device
        if any(g.waveform_absolute_sum.device != device for g in groups):
            raise ValueError("Reconstruction groups must use one device")
        count = sum(g.sample_count for g in groups)
        examples = sum(g.example_count for g in groups)
        if count <= 0 or examples <= 0:
            raise ValueError("Reconstruction groups require positive valid counts")
        waveform = torch.stack([g.waveform_absolute_sum for g in groups]).sum() / count
        counts = tuple(sum(g.mel_element_counts[i] for g in groups)
                       for i in range(len(self.config.fft_sizes)))
        linear = torch.stack([torch.stack([g.mel_linear_sums[i] for g in groups]).sum() / counts[i]
                              for i in range(len(counts))]).mean()
        logarithmic = torch.stack([torch.stack([g.mel_log_sums[i] for g in groups]).sum() / counts[i]
                                   for i in range(len(counts))]).mean()
        spectral = self.config.mel_linear_weight * linear + self.config.mel_log_weight * logarithmic
        losses = {"teacher_waveform": waveform, "teacher_mel": spectral,
                  "teacher_mel_linear": linear, "teacher_mel_log": logarithmic,
                  "total": waveform + spectral}
        legacy = {name: torch.stack([g.legacy_sums[name] for g in groups]).sum().detach() / examples
                  for name in groups[0].legacy_sums}
        legacy["teacher_mel"] = (self.config.mel_linear_weight * legacy["teacher_mel_linear"]
                                 + self.config.mel_log_weight * legacy["teacher_mel_log"])
        legacy["total"] = legacy["teacher_waveform"] + legacy["teacher_mel"]
        return ReconstructionV2Result(losses, legacy, {"valid_samples": count, "examples": examples,
            "mel_elements_by_resolution": counts,
            "mel_frames_by_resolution": tuple(c // b for c, b in zip(counts, self.config.mel_bands))})

    def forward_groups(self, pairs: Iterable[tuple[Tensor, Tensor]]) -> ReconstructionV2Result:
        """Score valid prediction/teacher slices, optionally bucketed by length."""
        return self.aggregate(self.group_terms(prediction, target) for prediction, target in pairs)

    def forward(self, student: Tensor, teacher: Tensor) -> ReconstructionV2Result:
        return self.forward_groups(((student, teacher),))
