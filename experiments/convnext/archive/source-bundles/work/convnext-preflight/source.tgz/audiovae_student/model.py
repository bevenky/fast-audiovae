"""Freshly initialized causal decoder for unscaled AudioVAE2 posterior means.

All learned temporal processing happens at 100 Hz. There are no imported
parameters or hidden mutable buffers in the model. Training uses ``forward``;
streaming carries explicit, bounded convolution histories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StudentConfig:
    hidden_channels: int = 512
    expansion_channels: int = 2048
    head_channels: int = 2048
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)
    layer_norm_eps: float = 1e-6
    layer_scale_init: float = 1e-6
    prelu_init: float = 0.25

    def __post_init__(self):
        object.__setattr__(self, "dilations", tuple(self.dilations))
        for name in ("hidden_channels", "expansion_channels", "head_channels"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.dilations or any(type(d) is not int or d < 1 for d in self.dilations):
            raise ValueError("dilations must contain positive integers")
        import math
        for name in ("layer_norm_eps", "layer_scale_init", "prelu_init"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.layer_norm_eps <= 0 or self.layer_scale_init < 0:
            raise ValueError("Invalid normalization epsilon or layer scale")

    @property
    def history_frames(self) -> int:
        """Internal 100 Hz frames of past context, excluding the current frame."""
        return 6 + 6 * sum(self.dilations) + 2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dilations"] = list(self.dilations)
        return result


@dataclass(frozen=True)
class DecoderState:
    started: Tensor
    histories: tuple[Tensor, ...]


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, *, dilation=1, groups=1, bias=True):
        super().__init__()
        self.history_size = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.history_size, 0), mode="replicate"))

    def step(self, x: Tensor, history: Tensor, started: Tensor) -> tuple[Tensor, Tensor]:
        # The first internal frame initializes each layer independently. Padding
        # the raw latent once would give different deeper-layer boundary values.
        previous = torch.where(started, history, x[..., :1].expand_as(history))
        joined = torch.cat((previous, x), dim=-1)
        return self.conv(joined), joined[..., -self.history_size:]


class TemporalBlock(nn.Module):
    def __init__(self, config: StudentConfig, dilation: int):
        super().__init__()
        width = config.hidden_channels
        self.depthwise = CausalConv(width, width, 7, dilation=dilation, groups=width)
        self.norm = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.expand = nn.Linear(width, config.expansion_channels)
        self.project = nn.Linear(config.expansion_channels, width)
        self.scale = nn.Parameter(torch.full((width,), config.layer_scale_init))

    def mix(self, residual: Tensor, temporal: Tensor) -> Tensor:
        value = self.norm(temporal.transpose(1, 2))
        value = self.project(F.gelu(self.expand(value), approximate="none"))
        return residual + (value * self.scale).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.mix(x, self.depthwise(x))


class ChannelAffine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.bias


class StudentDecoder(nn.Module):
    latent_channels = 64
    latent_rate = 25
    sample_rate = 48000
    samples_per_latent = 1920
    phases = 4
    samples_per_internal_frame = 480

    def __init__(self, config: StudentConfig | None = None):
        super().__init__()
        self.config = config or StudentConfig()
        width = self.config.hidden_channels
        self.adapter = nn.Conv1d(64, 64 * self.phases, 1)
        self.stem = CausalConv(64, width, 7)
        self.blocks = nn.ModuleList(TemporalBlock(self.config, d) for d in self.config.dilations)
        self.affine = ChannelAffine(width)
        self.head = CausalConv(width, self.config.head_channels, 3)
        self.activation = nn.PReLU(num_parameters=1, init=self.config.prelu_init)
        self.output = nn.Conv1d(self.config.head_channels, 480, 1, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _phase_frames(self, latents: Tensor) -> Tensor:
        value = self.adapter(latents)
        # Channel ordering is [channel, phase], then [time, phase].
        return value.reshape(value.shape[0], 64, 4, value.shape[-1]).permute(0, 1, 3, 2).reshape(value.shape[0], 64, -1)

    @staticmethod
    def _waveform(value: Tensor) -> Tensor:
        # Each internal frame predicts 480 contiguous waveform samples.
        return value.transpose(1, 2).reshape(value.shape[0], 1, -1)

    def _forward_nonempty(self, latents: Tensor) -> Tensor:
        x = self.stem(self._phase_frames(latents))
        for block in self.blocks:
            x = block(x)
        return self._waveform(self.output(self.activation(self.head(self.affine(x)))))

    def _validate_latents(self, latents):
        if not isinstance(latents, Tensor) or latents.ndim != 3 or latents.shape[1] != 64 or latents.shape[0] < 1:
            raise ValueError("Expected latents shaped [B,64,T] with positive B")
        if not latents.is_floating_point():
            raise ValueError("Latents must have floating-point dtype")
        if latents.device != self.adapter.weight.device:
            raise ValueError("Latents and decoder must use the same device")

    def forward(self, latents: Tensor) -> Tensor:
        self._validate_latents(latents)
        if latents.shape[-1] == 0:
            return latents.new_empty((latents.shape[0], 1, 0))
        return self._forward_nonempty(latents)

    @property
    def temporal_layers(self) -> tuple[CausalConv, ...]:
        return (self.stem, *(b.depthwise for b in self.blocks), self.head)

    def state_shapes(self, batch_size: int = 1) -> tuple[tuple[int, int, int], ...]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        return tuple((batch_size, layer.conv.in_channels, layer.history_size) for layer in self.temporal_layers)

    def initial_state(self, batch_size: int = 1) -> DecoderState:
        weight = self.adapter.weight
        return DecoderState(torch.zeros(1, device=weight.device, dtype=torch.bool),
                            tuple(weight.new_zeros(shape) for shape in self.state_shapes(batch_size)))

    def _step_nonempty(self, latents: Tensor, state: DecoderState) -> tuple[Tensor, DecoderState]:
        histories = []
        x, history = self.stem.step(self._phase_frames(latents), state.histories[0], state.started)
        histories.append(history)
        for index, block in enumerate(self.blocks, 1):
            temporal, history = block.depthwise.step(x, state.histories[index], state.started)
            histories.append(history)
            x = block.mix(x, temporal)
        x, history = self.head.step(self.affine(x), state.histories[-1], state.started)
        histories.append(history)
        audio = self._waveform(self.output(self.activation(x)))
        return audio, DecoderState(torch.ones_like(state.started), tuple(histories))

    def forward_stream(self, latents: Tensor, state: DecoderState) -> tuple[Tensor, DecoderState]:
        """Differentiable functional step, also used to construct the export.

        History views can retain the current computation graph/storage. Use
        ``stream()`` for inference with detached, compact persistent buffers.
        """
        self._validate_latents(latents)
        shapes = self.state_shapes(latents.shape[0])
        if (not isinstance(state, DecoderState) or state.started.shape != (1,)
                or state.started.dtype != torch.bool or state.started.device != latents.device
                or len(state.histories) != len(shapes)):
            raise ValueError("Invalid decoder state")
        if any(tuple(h.shape) != shape or h.device != latents.device or h.dtype != latents.dtype
               for h, shape in zip(state.histories, shapes)):
            raise ValueError("State shape, dtype or device does not match latents")
        if latents.shape[-1] == 0:
            return latents.new_empty((latents.shape[0], 1, 0)), state
        return self._step_nonempty(latents, state)

    def stream(self) -> DecoderStream:
        return DecoderStream(self)


class DecoderStream:
    """Single-utterance inference state; separate streams share only weights.

    Calls on one stream must be ordered. Empty chunks do not advance state.
    The raw latent API emits 1920 samples/frame and has no delayed flush tail.
    """

    def __init__(self, model: StudentDecoder):
        if model.training:
            raise ValueError("Call decoder.eval() before opening an inference stream")
        self.model = model
        self.state = model.initial_state()
        self.frames_decoded = 0
        self.closed = False

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Stream is closed")
        if self.model.training:
            raise RuntimeError("Decoder must remain in evaluation mode during streaming")

    @torch.inference_mode()
    def decode_chunk(self, latents: Tensor) -> Tensor:
        self._check_open()
        self.model._validate_latents(latents)
        if latents.shape[0] != 1 or not torch.isfinite(latents).all():
            raise ValueError("Stream requires finite latents shaped [1,64,T]")
        audio, candidate = self.model.forward_stream(latents, self.state)
        if not torch.isfinite(audio).all() or any(not torch.isfinite(h).all() for h in candidate.histories):
            raise RuntimeError("Decoder produced non-finite output or state")
        # Clone to avoid retaining a full chunk's storage through a history view.
        self.state = DecoderState(candidate.started.clone(), tuple(h.clone() for h in candidate.histories))
        self.frames_decoded += latents.shape[-1]
        return audio

    def reset(self):
        self._check_open()
        self.state = self.model.initial_state()
        self.frames_decoded = 0

    def flush(self) -> Tensor:
        self._check_open()
        return self.model.adapter.weight.new_empty((1, 1, 0))

    def close(self):
        self.closed = True
        self.state = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()


def architecture_summary(config: StudentConfig | None = None) -> dict[str, Any]:
    """Analytical convolution/linear work; excludes nonlinear and memory costs."""
    c = config or StudentConfig()
    width = c.hidden_channels
    per_second = 25 * 64 * 256
    per_second += 100 * (7 * 64 * width + len(c.dilations) * (7 * width + 2 * width * c.expansion_channels))
    per_second += 100 * (3 * width * c.head_channels + c.head_channels * 480)
    state_elements = 64 * 6 + width * 6 * sum(c.dilations) + width * 2
    return {"config": c.to_dict(), "initialization": "fresh", "latent_channels": 64,
            "latent_rate": 25, "sample_rate": 48000, "samples_per_latent": 1920,
            "internal_rate": 100, "history_internal_frames": c.history_frames,
            "history_latent_frames": (c.history_frames + 3) // 4,
            "history_seconds": c.history_frames / 100,
            "state_fp32_bytes_per_stream": state_elements * 4 + 1,
            "conv_linear_macs_per_audio_second": per_second}
