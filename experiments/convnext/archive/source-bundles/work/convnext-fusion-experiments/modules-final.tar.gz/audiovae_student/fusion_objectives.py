"""Isolated training-only candidates for AudioVAE decoder distillation.

Short-time reconstruction keeps the current objective and extends its scale
set. Complex spectral discrimination retains raw amplitude and the existing
GAN losses. Neither candidate changes the deployed decoder. These are declared
adaptations of audio-codec techniques, not a reproduced AudioVAE2 recipe.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .discriminators import (AudioDiscriminators, DiscriminatorConfig,
                             DiscriminatorOutput, PeriodDiscriminator)
from .losses_distillation import _check_audio
from .reconstruction_v2 import ReconstructionV2, ReconstructionV2Config


SHORT_TIME_RECONSTRUCTION_CONFIG = ReconstructionV2Config(
    fft_sizes=(256, 512, 1024, 2048, 4096),
    mel_bands=(16, 32, 64, 128, 128),
)


class ShortTimeReconstructionV2(ReconstructionV2):
    """Add 256/512 windows while keeping current reductions and return keys.

    Each resolution pools its valid elements, then all five resolutions are
    averaged equally. Consequently the new scales jointly receive 2/5 of the
    spectral average, not two additional full-strength loss branches. Linear
    and log magnitude retain their original weights. Waveform MAE is unchanged.
    Pass already sliced valid regions, exactly as for ReconstructionV2.
    """

    def __init__(self):
        super().__init__(SHORT_TIME_RECONSTRUCTION_CONFIG)

    @property
    def objective_identity(self) -> dict:
        return {
            "candidate": "short_time_reconstruction_v1",
            "config": asdict(self.config),
            "resolution_reduction": "equal_mean_after_global_valid_element_pooling",
            "new_resolution_share": 2 / 5,
            "waveform_reduction": "global_valid_sample_mean",
        }


DEFAULT_FREQUENCY_BANDS = ((0., .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.))


def _band_slices(n_fft: int, bands: tuple[tuple[float, float], ...]) -> tuple[slice, ...]:
    bins = n_fft // 2 + 1
    result = tuple(slice(int(low * bins), int(high * bins)) for low, high in bands)
    if any(band.stop <= band.start for band in result):
        raise ValueError("Every frequency band must contain at least one FFT bin")
    return result


@dataclass(frozen=True)
class ComplexDiscriminatorConfig(DiscriminatorConfig):
    """Checkpoint-visible complex MRD adaptation; MPD settings are unchanged.

    log_epsilon is inherited for configuration compatibility but is not applied
    to the complex representation. No log, DC subtraction, amplitude division
    or artificial padding is used to construct MRD inputs.
    """

    mrd_channels: int = 32
    bands: tuple[tuple[float, float], ...] = DEFAULT_FREQUENCY_BANDS
    format_version: int = 1
    representation: str = "complex_real_imaginary"
    amplitude_preprocessing: str = "none"
    stft_padding: str = "valid_no_center"
    feature_reduction: str = "equal_band_layer_mean"

    def __post_init__(self):
        super().__post_init__()
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("Unsupported complex discriminator format version")
        choices = {
            "representation": "complex_real_imaginary",
            "amplitude_preprocessing": "none",
            "stft_padding": "valid_no_center",
            "feature_reduction": "equal_band_layer_mean",
        }
        if any(getattr(self, name) != value for name, value in choices.items()):
            raise ValueError("Complex discriminator representation/reduction policy changed")
        try:
            bands = tuple(tuple(pair) for pair in self.bands)
        except TypeError as exc:
            raise ValueError("Frequency bands require numeric low/high pairs") from exc
        if (not bands or any(len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in pair) for pair in bands)):
            raise ValueError("Frequency bands require finite numeric low/high pairs")
        bands = tuple((float(low), float(high)) for low, high in bands)
        if (bands[0][0] != 0. or bands[-1][1] != 1.
                or any(not 0 <= low < high <= 1 for low, high in bands)
                or any(left[1] != right[0] for left, right in zip(bands, bands[1:]))):
            raise ValueError("Frequency bands must form one contiguous partition of [0, 1]")
        object.__setattr__(self, "bands", bands)
        for n_fft in self.fft_sizes:
            _band_slices(n_fft, bands)


def _complex_conv(in_channels, out_channels, kernel, stride=(1, 1)):
    return weight_norm(nn.Conv2d(in_channels, out_channels, kernel, stride,
                                padding=(kernel[0] // 2, kernel[1] // 2)))


class ComplexBandResolutionDiscriminator(nn.Module):
    """One raw complex STFT head with independent frequency-band feature stacks.

    Axes are [batch, real/imaginary channel, frequency, time]. Each stack has
    five hidden layers; its three strided layers reduce frequency only. Hidden
    features flatten band then layer, so the unchanged generator_losses helper
    averages bands/layers equally. Joined final band features feed one output
    layer. Reflection at period boundaries remains confined to the unchanged
    MPD implementation.
    """

    def __init__(self, n_fft: int, channels: int = 32,
                 bands=DEFAULT_FREQUENCY_BANDS):
        super().__init__()
        config = ComplexDiscriminatorConfig(fft_sizes=(n_fft,), mrd_channels=channels,
                                            bands=bands)
        self.n_fft, self.bands = n_fft, config.bands
        self.band_slices = _band_slices(n_fft, self.bands)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        kernels = ((9, 3), (9, 3), (9, 3), (9, 3), (3, 3))
        strides = ((1, 1), (2, 1), (2, 1), (2, 1), (1, 1))
        self.band_layers = nn.ModuleList([
            nn.ModuleList([_complex_conv(2 if i == 0 else channels, channels, kernel, stride)
                           for i, (kernel, stride) in enumerate(zip(kernels, strides))])
            for _ in self.bands
        ])
        self.output = _complex_conv(channels, 1, (3, 3))

    def spectrogram_bands(self, audio: Tensor) -> tuple[Tensor, ...]:
        """Expose the unchanged-amplitude complex representation for diagnostics."""
        _check_audio("Complex discriminator audio", audio)
        if audio.shape[-1] < self.n_fft:
            raise ValueError("Actual scored crop is shorter than complex MRD FFT")
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spectrum = torch.stft(audio[:, 0].float(), n_fft=self.n_fft,
                hop_length=self.n_fft // 4, win_length=self.n_fft,
                window=self.window.float().to(audio.device), center=False,
                normalized=False, onesided=True, return_complex=True)
            features = torch.stack((spectrum.real, spectrum.imag), dim=1)
        return tuple(features[:, :, band, :] for band in self.band_slices)

    def forward(self, audio: Tensor) -> DiscriminatorOutput:
        features, final_bands = [], []
        for band, layers in zip(self.spectrogram_bands(audio), self.band_layers):
            for layer in layers:
                band = F.leaky_relu(layer(band), .1)
                features.append(band)
            final_bands.append(band)
        return DiscriminatorOutput(self.output(torch.cat(final_bands, dim=2)), tuple(features))


class ComplexAudioDiscriminators(AudioDiscriminators):
    """Existing MPD bank plus fresh complex MRDs, using existing GAN loss APIs."""

    def __init__(self, config: ComplexDiscriminatorConfig = ComplexDiscriminatorConfig()):
        nn.Module.__init__(self)
        if not isinstance(config, ComplexDiscriminatorConfig):
            raise TypeError("A ComplexDiscriminatorConfig is required")
        self.config = config
        self.periods = nn.ModuleList([PeriodDiscriminator(p, config.mpd_channels)
                                     for p in config.periods])
        self.resolutions = nn.ModuleList([
            ComplexBandResolutionDiscriminator(size, config.mrd_channels, config.bands)
            for size in config.fft_sizes
        ])

    @classmethod
    def from_existing(cls, existing: AudioDiscriminators,
                      config: ComplexDiscriminatorConfig | None = None):
        """Copy trained MPDs without aliasing; initialize only replacement MRDs.

        This does not migrate any optimizer state or warm up new parameters.
        The caller must record the changed configuration and decide which
        discriminator parameters participate in preparation/training. Existing
        modules, optimizer state and train/eval flags are never mutated.
        """
        if type(existing) is not AudioDiscriminators:
            raise TypeError("Replacement requires the original AudioDiscriminators bank")
        if config is None:
            config = ComplexDiscriminatorConfig(periods=existing.config.periods,
                mpd_channels=existing.config.mpd_channels, fft_sizes=existing.config.fft_sizes)
        if not isinstance(config, ComplexDiscriminatorConfig):
            raise TypeError("A ComplexDiscriminatorConfig is required")
        if (config.periods != existing.config.periods
                or config.mpd_channels != existing.config.mpd_channels):
            raise ValueError("Replacement must preserve all existing MPD periods and widths")
        parameters = tuple(existing.parameters())
        if (not parameters or any(p.dtype != torch.float32 for p in parameters)
                or len({p.device for p in parameters}) != 1):
            raise ValueError("Existing discriminator must use one device and FP32 parameters")
        # Avoid even temporary random MPD initialization: the candidate's RNG
        # consumption belongs solely to the new spectral heads.
        replacement = cls.__new__(cls)
        nn.Module.__init__(replacement)
        replacement.config = config
        replacement.training = existing.training
        # Keep registration/parameter order identical to the ordinary
        # constructor so an explicitly migrated optimizer can resume safely.
        replacement.periods = deepcopy(existing.periods)
        replacement.resolutions = nn.ModuleList([
            ComplexBandResolutionDiscriminator(size, config.mrd_channels, config.bands)
            for size in config.fft_sizes
        ]).to(parameters[0].device)
        replacement.resolutions.train(existing.training)
        return replacement
