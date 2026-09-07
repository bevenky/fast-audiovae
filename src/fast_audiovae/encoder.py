"""Small, inference-only preparation for the upstream AudioVAE2 encoder."""
import types

import torch
from torch.nn.utils.weight_norm import WeightNorm


_PREPARED = "_fast_audiovae_encoder_preparation"
_MISSING = object()


def _check_model(encoder):
    for module in encoder.modules():
        if module.training:
            raise ValueError("The encoder and all its layers must be in eval mode")
    for tensor in encoder.parameters():
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("Encoder parameters must be CPU float32")
    for tensor in encoder.buffers():
        if tensor.device.type != "cpu":
            raise ValueError("Encoder parameters and buffers must be on CPU")
        if (tensor.is_floating_point() or tensor.is_complex()) and tensor.dtype != torch.float32:
            raise ValueError("Floating-point encoder parameters and buffers must be float32")


def _check_call(encoder, x):
    if torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("Use no_grad or inference_mode with CPU autocast disabled")
    _check_model(encoder)
    if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype != torch.float32:
        raise ValueError("Encoder input must be a CPU float32 tensor")
    if x.ndim != 3 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError("Encoder input must have nonempty B,1,T shape")


class EncoderPreparation:
    """Prepared encoder handle. Restoring forwards leaves weights folded."""

    def __init__(self, encoder, folded, pointwise):
        self.encoder = encoder
        self.folded_layers = tuple(folded)
        self.pointwise_layers = tuple(pointwise)
        self._originals = []

    def _set_forward(self, module, forward):
        self._originals.append((module, module.__dict__.get("forward", _MISSING)))
        module.forward = types.MethodType(forward, module)

    def restore(self):
        """Restore the original forward methods; weight normalization stays folded.

        This does not reload a checkpoint or recreate weight-normalization hooks.
        Restoring a handle more than once is harmless.
        """
        for module, original in reversed(self._originals):
            if original is _MISSING:
                del module.forward
            else:
                module.forward = original
        self._originals.clear()
        if getattr(self.encoder, _PREPARED, None) is self:
            delattr(self.encoder, _PREPARED)


def prepare_encoder(encoder):
    """Prepare an existing upstream CausalEncoder for CPU FP32 inference.

    Call this on ``vae.encoder.eval()``. Legacy weight normalization is folded
    once. Matching kernel-size-one causal convolutions skip redundant zero-pad
    copies. Original Snake and depthwise forward methods remain untouched.
    Use the encoder through its normal call interface inside ``torch.no_grad()``
    or ``torch.inference_mode()``. Input padding and posterior selection remain
    the responsibility of the upstream VAE. No models or native libraries are
    loaded, and no process-wide thread settings are changed.

    Only the supported CausalEncoder structure is accepted. Preparation is not
    a training API: ``handle.restore()`` restores forwards but leaves the folded
    weights in place. Reload the original checkpoint for training.
    """
    if not isinstance(encoder, torch.nn.Module) or not any(
        cls.__name__ == "CausalEncoder" for cls in type(encoder).__mro__
    ):
        raise ValueError("Pass the upstream CausalEncoder, not the full VAE or decoder")
    if hasattr(encoder, _PREPARED):
        raise RuntimeError("This encoder is already prepared; restore it first")
    _check_model(encoder)
    folded, pointwise = [], []
    # Validate every candidate before changing weights or methods.
    for name, module in encoder.named_modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            raise ValueError("Parametrized weights are unsupported; use upstream legacy weight norm")
        hooks = [h for h in module._forward_pre_hooks.values()
                 if isinstance(h, WeightNorm) and h.name == "weight"]
        if hooks or hasattr(module, "weight_g") or hasattr(module, "weight_v"):
            if len(hooks) != 1 or not all(hasattr(module, attr) for attr in ("weight_g", "weight_v")):
                raise ValueError("Unsupported weight-normalization state")
            folded.append((name, module))
        if (type(module).__name__ == "CausalConv1d"
                and isinstance(module, torch.nn.Conv1d)
                and module.kernel_size == (1,)
                and getattr(module, "_CausalConv1d__padding", None) == 0
                and getattr(module, "_CausalConv1d__output_padding", None) == 0):
            if "forward" in module.__dict__:
                raise ValueError("Restore existing causal-convolution patches before preparation")
            pointwise.append((name, module))
    with torch.no_grad():
        for _, module in folded:
            torch.nn.utils.remove_weight_norm(module)
    handle = EncoderPreparation(encoder, [name for name, _ in folded],
                                [name for name, _ in pointwise])
    original_forward = encoder.forward

    def forward(self, x, *args, **kwargs):
        _check_call(self, x)
        return original_forward(x, *args, **kwargs)

    def pointwise_forward(self, x):
        # The parent guard checks inference mode and all parameters. Keep the
        # upstream Conv1d backend and its accumulation order exactly as before.
        return torch.nn.Conv1d.forward(self, x)

    handle._set_forward(encoder, forward)
    for _, module in pointwise:
        handle._set_forward(module, pointwise_forward)
    setattr(encoder, _PREPARED, handle)
    return handle
