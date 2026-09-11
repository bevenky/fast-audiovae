"""Experimental output projection and diagnostics, outside production code.

The projection has no parameters, temporal state, or lookahead. For finite
teacher samples in [-1, 1], it cannot increase pointwise absolute/squared error.
That statement does not guarantee spectral or perceptual quality.
"""
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def bounded_waveform(raw: Tensor) -> Tensor:
    """Project floating point samples into [-1, 1], preserving the interior.

    Callers must reject nonfinite model outputs separately. There is deliberately
    no reduction/device synchronization in this inference postprocessor.
    """
    if not raw.is_floating_point():
        raise TypeError("Audio samples must be floating point")
    return torch.clamp(raw, min=-1.0, max=1.0)


@dataclass(frozen=True)
class BoundedDecoderView:
    """Evaluation view of an existing decoder; owns no model or stream state.

    Configure and inspect the original decoder directly. This view forwards its
    batch and functional streaming calls and projects only their waveform output.
    The exact original next-state object is returned unchanged.
    """

    decoder: Any

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return bounded_waveform(self.decoder(*args, **kwargs))

    def initial_state(self, *args, **kwargs):
        return self.decoder.initial_state(*args, **kwargs)

    def state_shapes(self, *args, **kwargs):
        return self.decoder.state_shapes(*args, **kwargs)

    def forward_stream(self, *args, **kwargs):
        audio, state = self.decoder.forward_stream(*args, **kwargs)
        return bounded_waveform(audio), state


def _error_summary(error: Tensor, mask: Tensor) -> dict:
    selected = error[mask]
    count = selected.numel()
    if not count:
        return {"count": 0, "absolute_error_sum": 0.0,
                "squared_error_sum": 0.0, "mae": None, "rmse": None,
                "max_abs_error": None}
    absolute = selected.abs()
    squared_sum = float(selected.square().sum())
    absolute_sum = float(absolute.sum())
    return {"count": count, "absolute_error_sum": absolute_sum,
            "squared_error_sum": squared_sum, "mae": absolute_sum / count,
            "rmse": (squared_sum / count) ** 0.5,
            "max_abs_error": float(absolute.max())}


def transient_errors(prediction: Tensor, teacher: Tensor, valid: Tensor) -> dict:
    """Score first/second waveform differences using only valid neighbors.

    Inputs and the Boolean mask have the same shape, with time in the last
    dimension. No adjacency spans rows, invalid gaps, or crop boundaries.
    Errors use FP64 sample increments, not derivatives scaled by sample rate.
    This returns error magnitudes and denominators, not event counts.
    """
    if prediction.shape != teacher.shape or prediction.shape != valid.shape:
        raise ValueError("Prediction, teacher and mask must have identical shapes")
    if prediction.ndim < 1 or valid.dtype != torch.bool:
        raise ValueError("A temporal dimension and Boolean mask are required")
    if not prediction.is_floating_point() or not teacher.is_floating_point():
        raise TypeError("Waveforms must be floating point")
    if prediction.device != teacher.device or prediction.device != valid.device:
        raise ValueError("Inputs must share a device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Nonfinite waveforms cannot be scored")
    residual = prediction.double() - teacher.double()
    first = residual[..., 1:] - residual[..., :-1]
    first_valid = valid[..., 1:] & valid[..., :-1]
    second = first[..., 1:] - first[..., :-1]
    second_valid = first_valid[..., 1:] & first_valid[..., :-1]
    return {"difference_scaling": "FP64 sample increments; no sample-rate factor",
            "first_difference": _error_summary(first, first_valid),
            "second_difference": _error_summary(second, second_valid)}


def capture_teacher_pre_tanh(teacher, crop, quiet_config) -> dict:
    """Capture the real teacher readout for a singleton cached latent crop.

    Full crop context is retained. Canonical comparison uses the same scored
    mask as natural_layers/native_chain, including their six-sample exclusion
    when context starts after the original source boundary. The caller verifies
    the pinned teacher state hash around the complete experiment; each capture
    also enforces immutable state versions, modes and existing gradients.

    Returned tensors remain on the teacher device. No inverse tanh, encoder
    inference, target inversion, teacher parameter change, or optimizer is used.
    """
    from audiovae_student.quiet_audio import _window_values
    from audiovae_student.teacher import SOURCE_SHA256, CHECKPOINT_SHA256

    if (teacher.provenance.get("source_sha256") != SOURCE_SHA256
            or teacher.provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Unexpected teacher source or checkpoint provenance")
    if (crop.latents.ndim != 3 or crop.latents.shape[:2] != (1, 64)
            or crop.latents.dtype != torch.float32 or crop.latents.shape[-1] < 1):
        raise ValueError("Expected nonempty singleton FP32 raw 64-channel latents")
    layers = teacher.model.decoder.model
    if (len(layers) < 2 or not isinstance(layers[-1], torch.nn.Tanh)
            or not isinstance(layers[-2], torch.nn.Conv1d)):
        raise ValueError("Expected the teacher's final convolution followed by tanh")
    expected_shape = (1, 1, crop.latents.shape[-1] * 1920)
    target = crop.teacher_audio.detach().to(teacher.device)
    if tuple(target.shape) != expected_shape or target.dtype != torch.float32:
        raise ValueError("Canonical waveform and latent context do not align")
    start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
    stop = crop.scored_slice.stop
    if not 0 <= start < stop <= expected_shape[-1]:
        raise ValueError("Scored crop must contain valid samples after context exclusion")
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[..., start:stop] = True
    with torch.no_grad():
        _, _, _, _, windows = _window_values(target, target, valid, quiet_config)
        quiet = windows.repeat_interleave(quiet_config.window_samples, dim=-1).unsqueeze(1)
        quiet = quiet[..., :target.shape[-1]] & valid
    z = crop.latents.detach().to(teacher.device).clone()
    model = teacher.model
    modes = tuple((m, m.training) for m in model.modules())
    versions = tuple((k, v._version) for k, v in model.state_dict().items())
    flags = tuple(p.requires_grad for p in model.parameters())
    gradients = tuple((p.grad, p.grad._version if p.grad is not None else None)
                      for p in model.parameters())
    observed = []
    handle = None
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=teacher.device.type, enabled=False):
            ordinary = teacher.decode(z)

            def capture(_module, _inputs, output):
                if not isinstance(output, Tensor):
                    raise RuntimeError("Final teacher convolution did not return a tensor")
                observed.append(output.detach().clone())

            handle = layers[-2].register_forward_hook(capture)
            hooked = teacher.decode(z)
            handle.remove()
            handle = None
            if len(observed) != 1:
                raise RuntimeError("Expected one final-convolution call per singleton decode")
            pre = observed[0]
            if any(tuple(t.shape) != expected_shape for t in (ordinary, hooked, pre)):
                raise RuntimeError("Teacher pre/post-activation waveform shape mismatch")
            if any(t.dtype != torch.float32 or t.device != teacher.device
                   or not bool(torch.isfinite(t).all()) for t in (ordinary, hooked, pre)):
                raise RuntimeError("Teacher capture violated the finite FP32/device contract")
            if not torch.equal(ordinary, hooked):
                raise RuntimeError("Hooked teacher differs from ordinary singleton execution")
            if not torch.equal(torch.tanh(pre), ordinary):
                raise RuntimeError("Captured final convolution plus tanh is not the native output")
            if not torch.equal(ordinary[valid], target[valid]):
                raise RuntimeError("Teacher scored output differs from the canonical target")
            result = {"pre_tanh": pre, "post_tanh": ordinary.detach(),
                "target": target, "valid": valid, "quiet": quiet,
                "receipt": {"source_id": crop.source_id, "start_frame": crop.start_frame,
                    "context_start_frame": crop.context_start_frame,
                    "full_context_latent_shape": list(z.shape),
                    "full_context_waveform_shape": list(expected_shape),
                    "scored_samples": int(valid.sum()), "quiet_samples": int(quiet.sum()),
                    "scored_start_sample": start, "scored_stop_sample": stop,
                    "native_vs_hooked_bitwise": True, "tanh_capture_vs_native_bitwise": True,
                    "scored_native_vs_canonical_bitwise": True,
                    "singleton_decoder_forwards": 2, "encoder_forwards": 0,
                    "inverse_tanh_used": False}}
    finally:
        if handle is not None:
            handle.remove()
        for module, mode in modes:
            module.training = mode
        if versions != tuple((k, v._version) for k, v in model.state_dict().items()):
            raise RuntimeError("Teacher parameter or buffer changed during capture")
        if flags != tuple(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Teacher gradient flags changed during capture")
        if any(p.grad is not grad or (grad is not None and grad._version != version)
               for p, (grad, version) in zip(model.parameters(), gradients)):
            raise RuntimeError("Existing teacher gradients changed during capture")
    return result
