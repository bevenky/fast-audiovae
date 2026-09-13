"""Two fixed streaming shapes compiled for Apple GPU execution.

Other chunk sizes use the same model eagerly on MPS. The cache is bounded:
variable user packet sizes cannot create an unlimited set of compiled graphs.
"""
from __future__ import annotations

import torch
from .mps_decoder import _require


class TensorStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.names = tuple(model.state_shapes)

    def forward(self, z, *state):
        model = self.model
        history = dict(zip(self.names, state))
        output = {}
        x, output["stem.history"] = model.stem.decode(z, history["stem.history"])
        x = model.pointwise(x)
        for index, stage in enumerate(model.stages):
            x = stage.decode(x, history, output, f"stage{index}")
        x, output["final.history"] = model.final_conv.decode(model.final_snake(x), history["final.history"])
        return (torch.tanh(x), *(output[name].contiguous() for name in self.names))


class CompiledModel:
    """Called under GPUDecoder's lock, including during preparation."""
    compiled_lengths = (1, 2)

    def __init__(self, model):
        self.model = model
        self.state_shapes, self.state_bytes = model.state_shapes, model.state_bytes
        self.sample_rate = model.sample_rate
        self.hop_samples = model.hop_samples
        self.latent_channels = model.latent_channels
        self.names = tuple(model.state_shapes)
        self.step = TensorStep(model)
        self.compiled = {}

    def decode(self, z, state):
        self.model._checked_input(z)
        history = self.model._checked_states(z, state)
        length = z.shape[-1]
        if length not in self.compiled_lengths:
            return self.model.decode(z, state)
        operation = self.compiled.get(length)
        if operation is None:
            operation = torch.compile(self.step, backend="inductor", fullgraph=True, dynamic=False)
        result = operation(z, *(history[name] for name in self.names))
        _require(len(result) == len(self.names) + 1, "Compiled GPU history inventory changed")
        # A failed compile/call must not populate the cache. Do not fall back to
        # another backend when a requested GPU graph cannot be executed.
        self.compiled[length] = operation
        # Inductor can elide an in-graph contiguous operation and return a view.
        return result[0], {name: value.contiguous() for name, value in zip(self.names, result[1:])}
