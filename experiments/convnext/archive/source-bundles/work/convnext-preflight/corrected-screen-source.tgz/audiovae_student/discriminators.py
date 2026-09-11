"""Training-only lightweight MPD/MRD and explicitly averaged LS-GAN losses.

Paper dimensions and +/-1 targets are retained. Weight normalization,
LeakyReLU(0.1), padding, log floor and per-head/feature-element means are our
declared choices. Inputs must already be aligned, valid scored crops.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


@dataclass(frozen=True)
class DiscriminatorConfig:
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    mpd_channels: tuple[int, ...] = (16, 64, 256, 512, 512)
    fft_sizes: tuple[int, ...] = (512, 1024, 2048)
    mrd_channels: int = 16
    log_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("periods", "mpd_channels", "fft_sizes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.periods or any(type(p) is not int or p < 2 for p in self.periods):
            raise ValueError("MPD periods must be integers >= 2")
        if len(set(self.periods)) != len(self.periods):
            raise ValueError("MPD periods must be distinct")
        if len(self.mpd_channels) != 5 or any(type(c) is not int or c < 1 for c in self.mpd_channels):
            raise ValueError("MPD requires five positive hidden widths")
        if not self.fft_sizes or any(type(n) is not int or n < 8 or n % 4 for n in self.fft_sizes):
            raise ValueError("MRD FFT sizes must be >= 8 and divisible by four")
        if type(self.mrd_channels) is not int or self.mrd_channels < 1:
            raise ValueError("MRD width must be positive")
        if not math.isfinite(self.log_epsilon) or not 0 < self.log_epsilon < 1:
            raise ValueError("Log epsilon must be finite and between zero and one")


@dataclass(frozen=True)
class DiscriminatorOutput:
    logits: Tensor
    features: tuple[Tensor, ...]


def _conv(in_channels: int, out_channels: int, kernel: tuple[int, int],
          stride: tuple[int, int] = (1, 1)) -> nn.Module:
    return weight_norm(nn.Conv2d(in_channels, out_channels, kernel, stride,
                                padding=(kernel[0] // 2, kernel[1] // 2)))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList([_conv(a, b, (5, 1), (3 if i < 4 else 1, 1))
                                    for i, (a, b) in enumerate(zip((1, *channels[:-1]), channels))])
        self.output = _conv(channels[-1], 1, (3, 1))

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        pad = (-audio.shape[-1]) % self.period
        # Reflection only satisfies the period reshape. It is never scored as
        # additional source exposure and is identical for the aligned pair.
        if pad:
            if audio.shape[-1] <= pad:
                raise ValueError("Valid crop is too short for MPD reflection")
            # CUDA reflection_pad1d backward has no deterministic kernel.
            # This is the identical right reflection, with an explicit
            # slice/flip backward and no atomic scatter accumulation.
            audio = torch.cat((audio, audio[..., -pad - 1:-1].flip(-1)), dim=-1)
        x = audio.reshape(audio.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return DiscriminatorOutput(self.output(x), tuple(features))


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft: int, channels: int, epsilon: float) -> None:
        super().__init__()
        self.n_fft, self.epsilon = n_fft, epsilon
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        strides = ((1, 1), (2, 1), (2, 1), (2, 1), (1, 1))
        self.layers = nn.ModuleList([_conv(1 if i == 0 else channels, channels, (5, 5), stride)
                                    for i, stride in enumerate(strides)])
        self.output = _conv(channels, 1, (3, 3))

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        if audio.shape[-1] < self.n_fft:
            raise ValueError("Actual scored crop is shorter than MRD FFT")
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spectrum = torch.stft(audio[:, 0].float(), n_fft=self.n_fft,
                                  hop_length=self.n_fft // 4, win_length=self.n_fft,
                                  window=self.window.float().to(audio.device), center=False,
                                  normalized=False, onesided=True, return_complex=True)
            # Axes are [batch, channel, frequency, time]; stride (2,1) reduces
            # frequency. There is no DC removal or peak/gain normalization.
            x = spectrum.abs().clamp_min(self.epsilon).log().unsqueeze(1)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return DiscriminatorOutput(self.output(x), tuple(features))


class AudioDiscriminators(nn.Module):
    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()) -> None:
        super().__init__()
        self.config = config
        self.periods = nn.ModuleList([PeriodDiscriminator(p, config.mpd_channels) for p in config.periods])
        self.resolutions = nn.ModuleList([ResolutionDiscriminator(n, config.mrd_channels, config.log_epsilon)
                                          for n in config.fft_sizes])

    def forward(self, audio: Tensor) -> tuple[DiscriminatorOutput, ...]:
        if audio.ndim != 3 or audio.shape[0] < 1 or audio.shape[1] != 1:
            raise ValueError("Discriminators require [batch, 1, valid samples]")
        if not audio.is_floating_point() or not bool(torch.isfinite(audio).all()):
            raise ValueError("Discriminator input must be finite floating-point audio")
        if audio.shape[-1] < max(max(self.config.fft_sizes), max(self.config.periods)):
            raise ValueError("Valid crop is too short for the configured discriminators")
        return tuple(d(audio) for d in (*self.periods, *self.resolutions))


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    parameters = tuple(module.parameters())
    flags = tuple(p.requires_grad for p in parameters)
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def _validate_pair(prediction: Tensor, teacher: Tensor) -> None:
    if prediction.shape != teacher.shape or prediction.device != teacher.device:
        raise ValueError("Adversarial waveforms must have identical aligned shapes and devices")


def _example_weights(weights: Tensor | None, prediction: Tensor) -> Tensor | None:
    if weights is None:
        return None
    if (not isinstance(weights, Tensor) or weights.ndim != 1
            or weights.shape[0] != prediction.shape[0] or weights.is_complex()):
        raise ValueError("Example weights must be a real 1D tensor matching the audio batch")
    # Normalize in double precision to avoid overflow when summing integer
    # sample counts or large finite weights. Weights never receive gradients.
    weights = weights.detach().to(device=prediction.device, dtype=torch.float64)
    total = weights.sum()
    if (not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all())
            or not bool(torch.isfinite(total)) or not bool(total > 0)):
        raise ValueError("Example weights must be finite and positive with a finite positive sum")
    return weights / total


def _weighted_mean(value: Tensor, weights: Tensor) -> Tensor:
    per_example = value.reshape(value.shape[0], -1).mean(dim=1)
    return (per_example * weights.to(per_example)).sum()


def generator_losses(discriminators: AudioDiscriminators, prediction: Tensor,
                     teacher: Tensor, *, example_weights: Tensor | None = None) -> dict[str, Tensor]:
    """Freeze D, retaining its audio derivative and optional example weights.

    With weights, each layer/head averages elements per example, then applies
    the normalized detached weights. The default retains legacy arithmetic.
    """
    _validate_pair(prediction, teacher)
    weights = _example_weights(example_weights, prediction)
    with frozen_parameters(discriminators):
        fake = discriminators(prediction)
        with torch.no_grad():
            real = discriminators(teacher.detach())
        if weights is None:
            adversarial = torch.stack([(head.logits - 1).square().mean() for head in fake]).mean()
            # First average elements in each feature, then layers within a head,
            # then heads. All current heads have the same five hidden layers.
            matching = torch.stack([torch.stack([(f - r.detach()).abs().mean()
                                                 for f, r in zip(fh.features, rh.features)]).mean()
                                    for fh, rh in zip(fake, real)]).mean()
        else:
            adversarial = torch.stack([_weighted_mean((head.logits - 1).square(), weights)
                                       for head in fake]).mean()
            matching = torch.stack([torch.stack([_weighted_mean((f - r.detach()).abs(), weights)
                                                 for f, r in zip(fh.features, rh.features)]).mean()
                                    for fh, rh in zip(fake, real)]).mean()
    return {"adversarial": adversarial, "feature_matching": matching}


def discriminator_loss(discriminators: AudioDiscriminators, prediction: Tensor,
                       teacher: Tensor, *, example_weights: Tensor | None = None) -> Tensor:
    """Use +1/-1 labels and identical optional weights for each real/fake pair."""
    _validate_pair(prediction, teacher)
    weights = _example_weights(example_weights, prediction)
    real = discriminators(teacher.detach())
    fake = discriminators(prediction.detach())
    if weights is None:
        return torch.stack([(r.logits - 1).square().mean() + (f.logits + 1).square().mean()
                            for r, f in zip(real, fake)]).mean()
    return torch.stack([_weighted_mean((r.logits - 1).square(), weights)
                        + _weighted_mean((f.logits + 1).square(), weights)
                        for r, f in zip(real, fake)]).mean()
