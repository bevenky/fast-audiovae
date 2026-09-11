"""Isolated decoder-head and startup ablations for a saved student.

These variants are training experiments, not qualified production exports.
The original StudentConfig and existing parameter names stay intact; callers
must separately bind ``fusion_config.to_dict()`` into checkpoint identity.
Construction copies the supplied decoder and never mutates its weights,
normalization statistics, streaming state, or random-number generators.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import CausalConv, DecoderState, StudentDecoder


@dataclass(frozen=True)
class FusionArchitectureConfig:
    terminal_tanh: bool = False
    causal_output_filter: bool = False
    zero_startup_padding: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


class _ZeroStartupConv(CausalConv):
    """The original convolution with zero, rather than replicated, history."""

    @classmethod
    def from_conv(cls, source: CausalConv) -> _ZeroStartupConv:
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.history_size = source.history_size
        result.conv = deepcopy(source.conv)
        result.training = source.training
        return result

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.history_size, 0)))

    def step(self, x: Tensor, history: Tensor, started: Tensor) -> tuple[Tensor, Tensor]:
        previous = torch.where(started, history, torch.zeros_like(history))
        joined = torch.cat((previous, x), dim=-1)
        return self.conv(joined), joined[..., -self.history_size:]


class _IdentityOutputFilter(nn.Module):
    """Seven causal waveform taps, initially the identity, with no bias."""

    history_size = 6

    def __init__(self, reference: Tensor):
        super().__init__()
        # Conv1d's ordinary initialization consumes CPU randomness. Restore it
        # so construction cannot perturb matched data/optimizer experiments.
        # Allocate on CPU explicitly, then move without drawing device RNG.
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv1d(1, 1, 7, bias=False, device="cpu")
        self.conv.to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[..., -1] = 1

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.history_size, 0)))

    def step(self, x: Tensor, history: Tensor, started: Tensor) -> tuple[Tensor, Tensor]:
        previous = torch.where(started, history, torch.zeros_like(history))
        joined = torch.cat((previous, x), dim=-1)
        return self.conv(joined), joined[..., -self.history_size:]


class FusionStudentDecoder(StudentDecoder):
    """Copy a trained student while selecting independent architecture tests.

    Existing parameters retain their exact names, values, dtype and device.
    Only a selected output filter adds trainable parameters. It operates before
    optional tanh and stores six waveform samples in the functional state.
    Zero startup changes every existing temporal layer's boundary convention.

    Fixed calibrated BatchNorm buffers are preserved. Normalization folding is
    deliberately unavailable: zero startup needs an affine-boundary correction,
    and all variant exports need a separately qualified architecture manifest.
    """

    def __init__(self, decoder: StudentDecoder, fusion_config: FusionArchitectureConfig):
        if type(decoder) is not StudentDecoder:
            raise ValueError("Expected an original StudentDecoder, not an already wrapped variant")
        if not isinstance(fusion_config, FusionArchitectureConfig):
            raise ValueError("Expected FusionArchitectureConfig")
        if fusion_config.zero_startup_padding and decoder.config.normalization_mode == "folded":
            raise ValueError("Zero startup requires an unfolded decoder and unchanged normalization buffers")
        nn.Module.__init__(self)
        self.config = deepcopy(decoder.config)
        self.fusion_config = fusion_config
        self.training = decoder.training
        # Copy once to retain relationships between modules while sharing no
        # storage with the source. Do not construct fresh base layers or call
        # initialize(), which would change learned weights and global RNG.
        copied = deepcopy(decoder)
        for name in ("adapter", "stem", "blocks", "stem_norm", "affine", "head", "activation", "output"):
            self.add_module(name, getattr(copied, name))
        if fusion_config.zero_startup_padding:
            self.stem = _ZeroStartupConv.from_conv(self.stem)
            for block in self.blocks:
                block.depthwise = _ZeroStartupConv.from_conv(block.depthwise)
            self.head = _ZeroStartupConv.from_conv(self.head)
        self.output_filter = (_IdentityOutputFilter(self.output.weight)
                              if fusion_config.causal_output_filter else None)
        if self.output_filter is not None:
            self.output_filter.training = self.training

    @classmethod
    def from_decoder(cls, decoder: StudentDecoder,
                     fusion_config: FusionArchitectureConfig | None = None) -> FusionStudentDecoder:
        return cls(decoder, fusion_config or FusionArchitectureConfig())

    @property
    def normalization_layout(self) -> str:
        base = super().normalization_layout
        c = self.fusion_config
        # Prevent histories crossing between distinct startup/head functions.
        return f"{base}|fusion:t{int(c.terminal_tanh)}f{int(c.causal_output_filter)}z{int(c.zero_startup_padding)}"

    @property
    def required_context_latent_frames(self) -> int:
        """Real history needed before an interior scored training region.

        The extra waveform filter can depend on samples from the previous
        latent frame. With the standard body, its first six scored samples
        therefore require 30 preceding raw latents rather than 29. Keep the
        scored intervals unchanged and include this extra unscored context.
        """
        history_samples = self.config.history_frames * self.samples_per_internal_frame
        if self.output_filter is not None:
            history_samples += self.output_filter.history_size
        return (history_samples + self.samples_per_latent - 1) // self.samples_per_latent

    def state_shapes(self, batch_size: int = 1) -> tuple[tuple[int, int, int], ...]:
        shapes = super().state_shapes(batch_size)
        if self.output_filter is not None:
            shapes += ((batch_size, 1, self.output_filter.history_size),)
        return shapes

    def _forward_nonempty(self, latents: Tensor, scored_latent_mask: Tensor | None = None) -> Tensor:
        audio = super()._forward_nonempty(latents, scored_latent_mask)
        if self.output_filter is not None:
            audio = self.output_filter(audio)
        return torch.tanh(audio) if self.fusion_config.terminal_tanh else audio

    def _step_nonempty(self, latents: Tensor, state: DecoderState) -> tuple[Tensor, DecoderState]:
        base_state = state
        if self.output_filter is not None:
            base_state = DecoderState(state.started, state.histories[:-1], state.normalization_layout)
        audio, result = super()._step_nonempty(latents, base_state)
        if self.output_filter is not None:
            audio, history = self.output_filter.step(audio, state.histories[-1], state.started)
            result = DecoderState(result.started, result.histories + (history,), self.normalization_layout)
        if self.fusion_config.terminal_tanh:
            audio = torch.tanh(audio)
        return audio, result

    def fold_normalization(self) -> StudentDecoder:
        raise ValueError("Fusion variants are not export-qualified; normalization folding is intentionally disabled")
