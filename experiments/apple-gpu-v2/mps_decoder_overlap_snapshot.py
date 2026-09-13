"""Explicit FP32 PyTorch AudioVAE2 decoder, with caller-owned causal history.

Only the pinned original ONNX export is accepted. This module does not select
devices automatically, download checkpoints, alter CPU kernels, or interpret
arbitrary ONNX graphs. ``device='cpu'`` exists for internal reference checks;
the public MPS adapter must request MPS and must never retry on the CPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import assets


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _buffer(module, name, value):
    array = np.asarray(value)
    _require(array.dtype == np.float32 and np.isfinite(array).all(),
             "Finite literal FP32 coefficients required: " + name)
    # Own the storage; neither external ONNX bytes nor caller arrays may alias it.
    tensor = torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))
    module.register_buffer(name, tensor, persistent=True)


class _Snake(nn.Module):
    def __init__(self, alpha, reciprocal):
        super().__init__()
        _require(np.asarray(alpha).ndim == np.asarray(reciprocal).ndim == 1
                 and np.asarray(alpha).shape == np.asarray(reciprocal).shape,
                 "Snake channel coefficients differ")
        _buffer(self, "alpha", np.asarray(alpha).reshape(1, -1, 1))
        _buffer(self, "reciprocal", np.asarray(reciprocal).reshape(1, -1, 1))

    def forward(self, x):
        # Preserve the ONNX order and its stored reciprocal, not 1/alpha.
        scaled = self.alpha * x
        sine = torch.sin(scaled)
        squared = sine * sine
        correction = self.reciprocal * squared
        return x + correction


class _Pointwise(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        _require(np.asarray(weight).ndim == 3 and np.asarray(weight).shape[2] == 1,
                 "Pointwise convolution must have kernel length one")
        _buffer(self, "weight", weight)
        _buffer(self, "bias", bias)

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias)


class _CausalConv1d(nn.Module):
    def __init__(self, weight, bias, *, dilation=1, groups=1):
        super().__init__()
        shape = np.asarray(weight).shape
        _require(len(shape) == 3 and shape[2] == 7 and dilation in (1, 3, 9)
                 and type(groups) is int and groups > 0, "Unsupported causal convolution")
        self.channels = shape[1] * groups
        self.halo = 6 * dilation
        self.dilation, self.groups = dilation, groups
        _buffer(self, "weight", weight)
        _buffer(self, "bias", bias)

    def decode(self, x, history):
        joined = torch.cat((history, x), dim=-1)
        if x.shape[-1] == 0:
            return x.new_empty((x.shape[0], self.weight.shape[0], 0)), history.clone()
        y = F.conv1d(joined, self.weight, self.bias, dilation=self.dilation, groups=self.groups)
        # clone prevents a tiny history from retaining a whole packet allocation.
        return y, joined[..., -self.halo:].clone()


class _CausalTranspose1d(nn.Module):
    def __init__(self, weight, bias, *, stride):
        super().__init__()
        shape = np.asarray(weight).shape
        _require(len(shape) == 3 and stride in (2, 5, 6, 8) and shape[2] == 2 * stride,
                 "Expected the original two-overlap transpose convolution")
        self.channels, self.output_channels, self.stride = shape[0], shape[1], stride
        _buffer(self, "weight", weight)
        _buffer(self, "bias", bias)

    def decode(self, x, history):
        if x.shape[-1] == 0:
            return x.new_empty((x.shape[0], self.weight.shape[1], 0)), history.clone()
        # The final stride is the last input's bias-free projected overlap.
        # Reusing it avoids projecting that input again on the next packet.
        # Keep the incoming history and all caller-owned tensors untouched.
        full = F.conv_transpose1d(x, self.weight, None, stride=self.stride)
        end = x.shape[-1] * self.stride
        next_history = full[..., end:].clone()
        y = torch.cat((full[..., :self.stride] + history,
                       full[..., self.stride:end]), dim=-1)
        y = y + self.bias.reshape(1, -1, 1)
        return y.contiguous(), next_history


class _Residual(nn.Module):
    def __init__(self, before, depthwise, after, pointwise):
        super().__init__()
        self.before, self.depthwise = before, depthwise
        self.after, self.pointwise = after, pointwise

    def decode(self, x, history):
        branch, next_history = self.depthwise.decode(self.before(x), history)
        branch = self.pointwise(self.after(branch))
        return x + branch, next_history


class _Stage(nn.Module):
    def __init__(self, scale, offset, snake, transpose, residuals):
        super().__init__()
        _buffer(self, "scale", np.asarray(scale).reshape(1, -1, 1))
        _buffer(self, "offset", np.asarray(offset).reshape(1, -1, 1))
        self.snake, self.transpose = snake, transpose
        self.residuals = nn.ModuleList(residuals)

    def decode(self, x, states, output, prefix):
        # The six affine operations survive ONNX export separately from Conv.
        x = x * self.scale
        x = x + self.offset
        key = prefix + ".transpose.history"
        x, output[key] = self.transpose.decode(self.snake(x), states[key])
        for index, residual in enumerate(self.residuals):
            key = prefix + f".residual{index}.history"
            x, output[key] = residual.decode(x, states[key])
        return x


class MPSModel(nn.Module):
    """Frozen decoder with an explicit, bounded per-stream state dictionary.

    ``decode(z, {})`` starts a stream. Later calls pass the returned dictionary.
    Neither the dictionary nor its tensors are modified, including on failure.
    The six transpose histories hold bias-free projected output overlaps,
    with channels and stride kept as separate axes.
    """
    sample_rate = 48000
    hop_samples = 1920
    latent_channels = 64
    rates = (8, 6, 5, 2, 2, 2)

    def __init__(self, stem, pointwise, stages, final_snake, final_conv):
        super().__init__()
        self.stem, self.pointwise = stem, pointwise
        self.stages = nn.ModuleList(stages)
        self.final_snake, self.final_conv = final_snake, final_conv
        self.state_shapes = {"stem.history": (1, stem.channels, stem.halo)}
        for i, stage in enumerate(stages):
            self.state_shapes[f"stage{i}.transpose.history"] = (
                1, stage.transpose.output_channels, stage.transpose.stride)
            for j, residual in enumerate(stage.residuals):
                dw = residual.depthwise
                self.state_shapes[f"stage{i}.residual{j}.history"] = (1, dw.channels, dw.halo)
        self.state_shapes["final.history"] = (1, final_conv.channels, final_conv.halo)
        self.state_bytes = sum(int(np.prod(shape)) * 4 for shape in self.state_shapes.values())
        self.eval()
        self.requires_grad_(False)

    @property
    def device(self):
        return self.stem.weight.device

    @classmethod
    def from_onnx(cls, path, device="mps"):
        target = torch.device(device)
        _require(target.type in ("mps", "cpu") and target.index in (None, 0),
                 "Only explicit MPS or internal CPU reference devices are supported")
        if target.type == "mps":
            _require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
                     "MPS CPU fallback must be disabled before importing PyTorch")
            _require(os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "1",
                     "MPS fast math is outside the FP32 reference contract")
            _require(torch.backends.mps.is_available(), "MPS is unavailable; no CPU fallback is permitted")
        # This verifies BOTH the graph and external data before any coefficient
        # is materialized. Imports alone perform no downloads or model loads.
        source = assets.verify_model(Path(path))
        import onnx
        from .graph.elementwise import Graph

        exported = onnx.load(source, load_external_data=False)
        graph = Graph(exported, source.parent)
        _require(len(graph.nodes) == 402 and len(graph.initializers) == 216
                 and graph.inputs == {"z"} and graph.outputs == {"audio"},
                 "Original decoder graph contract changed")
        convolutions = [n for n in graph.nodes if n.op_type == "Conv"]
        transposes = [n for n in graph.nodes if n.op_type == "ConvTranspose"]
        snakes, rejected = graph.snake_matches()
        _require(len(convolutions) == 39 and len(transposes) == 6 and len(snakes) == 43 and not rejected,
                 "Original convolution/Snake inventory changed")
        snake_position = 0

        def coefficient(name, shape):
            value = graph.constant(name)
            _require(value is not None and value.dtype == np.float32 and value.shape == shape
                     and np.isfinite(value).all(), "Coefficient shape/type changed: " + name)
            return value

        def conv(name, expected_shape, *, dilation=1, groups=1, transpose=False, stride=1):
            candidates = [n for n in graph.nodes if len(n.input) == 3 and n.input[1] == name + ".weight"]
            _require(len(candidates) == 1, "Expected exactly one weight consumer: " + name)
            node = candidates[0]
            expected = {"group": groups, "pads": [0, 0] if transpose or expected_shape[2] == 1 else [6*dilation, 0],
                        "auto_pad": b"NOTSET", "strides": [stride], "dilations": [dilation]}
            if transpose:
                expected["output_padding"] = [0]
            _require(node.op_type == ("ConvTranspose" if transpose else "Conv")
                     and graph.attrs(node) == expected and node.input[2] == name + ".bias",
                     "Original convolution attributes changed: " + name)
            weight = coefficient(name + ".weight", expected_shape)
            bias = coefficient(name + ".bias", (expected_shape[1] if transpose else expected_shape[0],))
            if transpose:
                return _CausalTranspose1d(weight, bias, stride=stride)
            if expected_shape[2] == 1:
                return _Pointwise(weight, bias)
            return _CausalConv1d(weight, bias, dilation=dilation, groups=groups)

        def snake(channels):
            nonlocal snake_position
            record = snakes[snake_position]
            snake_position += 1
            _require(record["channels"] == channels, "Snake channel order changed")
            return _Snake(record["alpha"], record["reciprocal"])

        root = "audio_vae.decoder.model."
        stem = conv(root + "0", (64, 1, 7), groups=64)
        pointwise = conv(root + "1", (2048, 64, 1))
        stages = []
        channels = 2048
        for i, stride in enumerate(cls.rates):
            prefix = root + str(i + 2) + ".block."
            out_channels = channels // 2
            scale_name = "unsqueeze" if i == 0 else f"unsqueeze_{2*i}"
            offset_name = f"unsqueeze_{2*i+1}"
            scale = coefficient(scale_name, (1, channels, 1))
            offset = coefficient(offset_name, (1, channels, 1))
            before = snake(channels)
            transpose = conv(prefix + "1", (channels, out_channels, 2*stride), transpose=True, stride=stride)
            residuals = []
            for j, dilation in enumerate((1, 3, 9)):
                residual = prefix + str(j + 2) + ".block."
                pre = snake(out_channels)
                dw = conv(residual + "1", (out_channels, 1, 7), dilation=dilation, groups=out_channels)
                post = snake(out_channels)
                pw = conv(residual + "3", (out_channels, out_channels, 1))
                residuals.append(_Residual(pre, dw, post, pw))
            stages.append(_Stage(scale, offset, before, transpose, residuals))
            channels = out_channels
        last_snake = snake(32)
        last_conv = conv(root + "9", (1, 32, 7))
        _require(snake_position == 43, "Not all Snake coefficients were consumed")
        model = cls(stem, pointwise, stages, last_snake, last_conv)
        _require(len(model.state_shapes) == 26 and model.state_bytes == 683264,
                 "Causal history inventory changed")
        model.source_identity = {"graph_sha256": assets.MODEL_FILES["audio_vae_decoder.onnx"],
                                 "external_data_sha256": assets.MODEL_FILES["audio_vae_decoder.onnx.data"],
                                 "coefficient_source": "literal_verified_onnx_initializers",
                                 "precision": "float32", "transpose_state": "bias_free_projected_overlap"}
        return model.to(device=target, dtype=torch.float32).eval()

    def _checked_states(self, z, states):
        _require(isinstance(states, dict), "States must be a dictionary")
        if not states:
            return {name: z.new_zeros(shape) for name, shape in self.state_shapes.items()}
        _require(set(states) == set(self.state_shapes), "Partial or unknown state dictionary")
        for name, expected in self.state_shapes.items():
            value = states[name]
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                     and value.device == z.device and value.layout == torch.strided
                     and tuple(value.shape) == expected and value.is_contiguous(),
                     "State shape/device/FP32/contiguity mismatch: " + name)
        return states

    @torch.inference_mode()
    def decode(self, z, states):
        _require(isinstance(z, torch.Tensor) and z.dtype == torch.float32 and z.ndim == 3
                 and tuple(z.shape[:2]) == (1, self.latent_channels) and z.device == self.device
                 and z.layout == torch.strided and z.is_contiguous(),
                 "Expected contiguous FP32 [1,64,L] on the model's explicit device")
        _require(self.stem.weight.dtype == torch.float32, "Model coefficients must remain FP32")
        history = self._checked_states(z, states)
        if z.shape[-1] == 0:
            return z.new_empty((1, 1, 0)), {name: value.clone() for name, value in history.items()}
        output = {}
        # A caller's autocast context must not silently select half precision.
        with torch.autocast(device_type=z.device.type, enabled=False):
            x, output["stem.history"] = self.stem.decode(z, history["stem.history"])
            x = self.pointwise(x)
            for i, stage in enumerate(self.stages):
                x = stage.decode(x, history, output, f"stage{i}")
            x, output["final.history"] = self.final_conv.decode(self.final_snake(x), history["final.history"])
            audio = torch.tanh(x)
        _require(tuple(audio.shape) == (1, 1, z.shape[-1] * self.hop_samples),
                 "Decoder output sample count differs from the original architecture")
        return audio, output

    def forward(self, z):
        return self.decode(z, {})[0]
