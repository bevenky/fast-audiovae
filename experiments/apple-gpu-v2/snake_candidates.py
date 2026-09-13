"""Experiment-only FP32 Snake callables; import performs no GPU work.

Use ``candidate = make_metal_snake()`` then ``candidate(x, alpha, reciprocal)``.
For an isolated model, bind ``types.MethodType(make_forward(candidate), block)``
to each existing Snake block's ``forward``. The original registered buffers are
used directly. No class, package, coefficient, or history is modified here.

The Metal creator compiles once. Calls allocate a new output and enqueue one
thread per element on the current PyTorch MPS stream. They do not synchronize,
copy to CPU, or perform value checks; the harness must time completed work and
validate results. Noncontiguous positive/zero-stride x views are read directly.
Coefficients must be contiguous [C] or [1,C,1], exactly as stored by the model.

Launch a fresh process with PYTORCH_MPS_FAST_MATH=0 and
PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch. PyTorch2.14 compile_shader
accepts source only: OperationUtils.mm:789-815 selects Safe/Precise math from
that environment (cached at first shader compilation). Source pragmas also
request safe arithmetic/no contraction; sine explicitly uses precise::sin.
The fixed threadgroup is min(256, pipeline maximum), capped by element count
for tiny arrays. This is a declared configuration, not an autotuned winner.

Primary API/source:
https://github.com/pytorch/pytorch/blob/v2.14.0/torch/mps/__init__.py
https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/mps/Module.cpp#L430-L505
https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/OperationUtils.mm#L789-L815
"""
from __future__ import annotations

import hashlib
import os

import torch


METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

kernel void snake_fp32(
    device const float* x [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* reciprocal [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant long& frames [[buffer(4)]],
    constant long& channel_stride [[buffer(5)]],
    constant long& time_stride [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  const long channel = long(index) / frames;
  const long time = long(index) % frames;
  const float value = x[channel * channel_stride + time * time_stride];
  const float scaled = alpha[channel] * value;
  const float sine = precise::sin(scaled);
  const float squared = sine * sine;
  const float correction = reciprocal[channel] * squared;
  y[index] = value + correction;
}
"""

SHADER_SHA256 = hashlib.sha256(METAL_SOURCE.encode()).hexdigest()


def snake_tensor(x, alpha, reciprocal):
    """The original tensor-only expression, with no Python metadata checks."""
    scaled = alpha * x
    sine = torch.sin(scaled)
    squared = sine * sine
    correction = reciprocal * squared
    return x + correction


def _environment():
    if torch.__version__.split("+")[0] != "2.14.0":
        raise RuntimeError("This experiment requires the reviewed PyTorch2.14.0 API")
    for name in ("PYTORCH_MPS_FAST_MATH", "PYTORCH_ENABLE_MPS_FALLBACK"):
        if os.environ.get(name, "0") != "0":
            raise RuntimeError(name + " must be 0 before importing torch")


def _validate(x, alpha, reciprocal):
    tensors = (x, alpha, reciprocal)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("Snake accepts three tensors")
    if any(t.dtype != torch.float32 or t.layout != torch.strided for t in tensors):
        raise TypeError("Snake requires strided FP32 tensors")
    if x.device.type != "mps" or any(t.device != x.device for t in tensors):
        raise ValueError("All Snake tensors must be on the same MPS device")
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1:
        raise ValueError("Snake input must have shape [1,C,T], C positive")
    channels, frames = x.shape[1:]
    if x.numel() > 2**32 - 1 or any(s < 0 for s in x.stride()):
        raise ValueError("Snake input exceeds the shader indexing contract")
    for t in (alpha, reciprocal):
        if tuple(t.shape) not in ((channels,), (1, channels, 1)) or not t.is_contiguous():
            raise ValueError("Stored coefficients must be contiguous [C] or [1,C,1]")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("This experiment is inference-only and has no autograd kernel")
    return channels, frames


def make_metal_snake():
    """Compile the explicit precise-FP32 shader; call only under a GPU lease."""
    _environment()
    library = torch.mps.compile_shader(METAL_SOURCE)
    kernel = library.snake_fp32
    group_size = min(256, kernel.max_threads_per_threadgroup)
    if group_size < 1:
        raise RuntimeError("Invalid Metal pipeline threadgroup limit")

    def snake(x, alpha, reciprocal):
        channels, frames = _validate(x, alpha, reciprocal)
        output = torch.empty((1, channels, frames), device=x.device, dtype=torch.float32)
        if frames:
            # Tensor binding includes storage_offset. Scalar integers are int64
            # in torch2.14's pybind interface, matching Metal constant long&.
            kernel(x, alpha, reciprocal, output, frames, x.stride(1), x.stride(2),
                   threads=x.numel(), group_size=min(group_size, x.numel()))
        return output

    # Keep the library and pipeline alive for asynchronous, stream-ordered work.
    snake.library = library
    snake.metadata = dict(backend="metal", shader_sha256=SHADER_SHA256,
                          group_size=group_size, precise_sin=True,
                          fast_math=False, contraction=False, allocation_in_call=True)
    return snake


def make_compiled_snake(*, dynamic=True):
    """Create a tensor-only Inductor callable, compiled lazily on first call.

    Compiler fusion may change rounding. This is a separate candidate and must
    pass numerical gates; neither compile success nor parity is assumed here.
    Metadata checks remain outside the captured function. No eager fallback is
    caught or substituted if fullgraph compilation fails.
    """
    _environment()
    compiled = torch.compile(snake_tensor, backend="inductor", fullgraph=True,
                             dynamic=dynamic)

    def snake(x, alpha, reciprocal):
        channels, _ = _validate(x, alpha, reciprocal)
        return compiled(x, alpha.reshape(1, channels, 1), reciprocal.reshape(1, channels, 1))

    snake.metadata = dict(backend="inductor", fullgraph=True, dynamic=dynamic,
                          fast_math=False, contraction="compiler-controlled")
    return snake


def make_forward(candidate):
    """Return an unbound ``forward(self, x)`` for per-instance harness binding."""
    def forward(self, x):
        return candidate(x, self.alpha, self.reciprocal)
    return forward


def apply_snake_candidate(model, backend):
    """Bind one candidate to the 43 existing Snake instances; return model.

    Compilation occurs before any instance is changed. Use a fresh experiment
    model for each backend; this helper deliberately refuses repeated patching.
    """
    from types import MethodType
    from fast_audiovae.mps_decoder import _Snake

    blocks = [block for block in model.modules() if isinstance(block, _Snake)]
    if len(blocks) != 43:
        raise ValueError("Expected exactly43 original Snake blocks")
    if any("forward" in block.__dict__ for block in blocks):
        raise ValueError("Snake forward already replaced; use a fresh model")
    if backend == "metal":
        candidate = make_metal_snake()
    elif backend == "compile":
        candidate = make_compiled_snake(dynamic=True)
    else:
        raise ValueError("Snake backend must be 'metal' or 'compile'")
    forward = make_forward(candidate)
    for block in blocks:
        block.forward = MethodType(forward, block)
    model.snake_candidate_metadata = dict(candidate.metadata, instances=len(blocks))
    return model
