"""Fixed-weight, training-only population calibration for the two decoder norms.

The caller supplies the same ordered examples twice, as a reusable iterable or
a factory. False mask positions do not enter moments, but real causal context
still enters convolutions. No waveform head, optimizer or gradient is needed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import Tensor

from .model import MaskedBatchNorm, StudentDecoder


CalibrationBatch = tuple[Tensor, Tensor]


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _hash_tensors(values) -> str:
    result = hashlib.sha256()
    for name, value in values:
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        result.update(len(descriptor).to_bytes(8, "big"))
        result.update(descriptor)
        result.update(_tensor_bytes(value))
    return result.hexdigest()


class _Moments:
    """Chan/Welford merges with exact sample counts and FP64 accumulation."""

    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def add(self, values: Tensor, mask: Tensor):
        # Select before arithmetic, so excluded nonfinite padding cannot poison
        # moments. Nonfinite context that reaches a scored feature is rejected.
        selected = values.transpose(1, 2)[mask].to(torch.float64)
        count = selected.shape[0]
        if count == 0:
            return
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("Calibration has nonfinite scored features")
        mean = selected.mean(dim=0)
        m2 = (selected - mean).square().sum(dim=0)
        if self.count == 0:
            self.mean, self.m2 = mean, m2
        else:
            total = self.count + count
            delta = mean - self.mean
            self.m2 = self.m2 + m2 + delta.square() * (self.count * count / total)
            self.mean = self.mean + delta * (count / total)
        self.count += count

    def population(self, minimum: int) -> tuple[Tensor, Tensor]:
        if self.count < minimum:
            raise ValueError(f"Calibration needs at least {minimum} scored internal frames; got {self.count}")
        variance = self.m2 / self.count
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(variance).all()):
            raise FloatingPointError("Calibration moments are not finite")
        return self.mean, variance


def calibrate_normalization(
    model: StudentDecoder,
    batches: Iterable[CalibrationBatch] | Callable[[], Iterable[CalibrationBatch]],
    *,
    provenance: Mapping,
    minimum_internal_frames: int = 2,
) -> dict:
    """Replace running moments sequentially, freeze both norms, and return evidence.

    ``provenance['split']`` must be ``'train'``. The caller is responsible for
    binding that declaration to its training manifest; tensors carry no split
    identity. Latents and boolean ``[B,T]`` masks must be on the model device.
    Input examples and masks must repeat identically between passes, although
    their batch partition may change. Neither a random sampler nor a one-shot
    iterator is an appropriate input.

    Population variances divide by N, not N-1. EMA batch counters reset to zero
    because these statistics were accumulated directly. Existing parameter
    gradients, all training flags and all parameter values are preserved.
    Failure restores every original parameter and buffer.
    """
    if not isinstance(model, StudentDecoder) or not all(
        isinstance(norm, MaskedBatchNorm) for norm in (model.stem_norm, model.affine)
    ):
        raise ValueError("Calibration requires the student's two unfurled masked BatchNorm layers")
    if type(minimum_internal_frames) is not int or minimum_internal_frames < 2:
        raise ValueError("minimum_internal_frames must be an integer >= 2")
    if not isinstance(provenance, Mapping) or provenance.get("split") != "train":
        raise ValueError("Calibration requires explicit training-only provenance: split='train'")
    provenance_json = json.dumps(dict(provenance), sort_keys=True, allow_nan=False, separators=(",", ":"))
    provenance_copy = json.loads(provenance_json)
    if not callable(batches) and iter(batches) is batches:
        raise ValueError("Calibration needs a repeatable iterable or iterator factory")
    parameters = tuple(model.named_parameters())
    if any(p.dtype not in (torch.float32, torch.float64) for _, p in parameters):
        raise ValueError("Calibration requires full-precision model parameters")
    if any(not bool(torch.isfinite(p).all()) for _, p in parameters):
        raise FloatingPointError("Calibration parameters must be finite")
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    parameter_sha = _hash_tensors(parameters)
    before_sha = _hash_tensors(initial.items())
    versions = {name: p._version for name, p in parameters}

    def verify_weights():
        if any(p._version != versions[name] for name, p in parameters):
            raise RuntimeError("Model parameters changed during normalization calibration")

    def collect(stage: str):
        moments = _Moments()
        inputs = hashlib.sha256()
        examples = batch_count = selected_latents = 0
        iterator = iter(batches() if callable(batches) else batches)
        for batch in iterator:
            verify_weights()
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Calibration batches must contain (latents, scored_latent_mask)")
            latents, mask = batch
            model._validate_latents(latents)
            if latents.shape[-1] == 0 or latents.dtype != model.stem.conv.weight.dtype:
                raise ValueError("Calibration latents must be nonempty and match model precision")
            if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
                    or mask.shape != (latents.shape[0], latents.shape[-1]) or mask.device != latents.device):
                raise ValueError("Calibration mask must be boolean [B,T] on the latent device")
            # Per-example hashes make the replay check independent of batch
            # partition while retaining context, padding, order and dtype.
            for index in range(latents.shape[0]):
                inputs.update(bytes.fromhex(_hash_tensors((("latents", latents[index]), ("mask", mask[index])))))
            expanded_mask = mask.repeat_interleave(model.phases, dim=-1)
            features = model.stem(model._phase_frames(latents.detach()))
            if stage == "final":
                features = model.stem_norm.fixed(features)
                for block in model.blocks:
                    features = block(features)
            moments.add(features, expanded_mask)
            verify_weights()
            selected_latents += int(mask.sum())
            examples += latents.shape[0]
            batch_count += 1
        verify_weights()
        mean, variance = moments.population(minimum_internal_frames)
        return mean, variance, {
            "internal_frame_count": moments.count, "selected_latent_count": selected_latents,
            "examples": examples, "batches": batch_count, "inputs_sha256": inputs.hexdigest(),
            "fp64_moments_sha256": _hash_tensors((("mean", mean), ("population_variance", variance))),
        }

    def publish(norm: MaskedBatchNorm, mean: Tensor, variance: Tensor):
        mean, variance = mean.to(norm.running_mean), variance.to(norm.running_var)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
            raise FloatingPointError("Calibration moments overflow their destination precision")
        norm.running_mean.copy_(mean)
        norm.running_var.copy_(variance)
        norm.num_batches_tracked.zero_()
        norm.freeze_statistics()

    try:
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=model.stem.conv.weight.device.type, enabled=False):
            stem_mean, stem_variance, stem_report = collect("stem")
            publish(model.stem_norm, stem_mean, stem_variance)
            final_mean, final_variance, final_report = collect("final")
            replay_keys = ("internal_frame_count", "selected_latent_count", "examples", "inputs_sha256")
            if any(stem_report[key] != final_report[key] for key in replay_keys):
                raise ValueError("Calibration passes must replay exactly the same training examples and masks")
            publish(model.affine, final_mean, final_variance)
        if _hash_tensors(model.named_parameters()) != parameter_sha:
            raise RuntimeError("Model parameters changed during normalization calibration")
        return {
            "format_version": 1, "method": "sequential_fp64_sample_weighted_population_moments",
            "order": ["stem", "final_under_calibrated_frozen_stem"],
            "provenance": provenance_copy,
            "provenance_sha256": hashlib.sha256(provenance_json.encode()).hexdigest(),
            "parameter_sha256": parameter_sha, "parameters_unchanged": True,
            "model_state_before_sha256": before_sha,
            "model_state_after_sha256": _hash_tensors(model.state_dict().items()),
            "minimum_internal_frames": minimum_internal_frames,
            "stem": stem_report, "final": final_report,
            "variance_divisor": "population_N", "ema_batch_counters": "reset_to_zero",
            "statistics_frozen": True,
        }
    except BaseException:
        model.load_state_dict(initial, strict=True)
        raise
    finally:
        for module, training in modes:
            module.training = training
