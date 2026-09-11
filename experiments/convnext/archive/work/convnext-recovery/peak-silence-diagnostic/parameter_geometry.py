"""Read-only VJP diagnosis of shared-parameter coupling at one frozen crop.

Compares -metric_audio_gradient @ loss_audio_gradient with
-(J.T @ metric_audio_gradient) @ (J.T @ loss_audio_gradient).
The latter is the first derivative for plain parameter gradient descent, NOT
for the retained Muon/AdamW optimizer or a real minibatch training update.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from audiovae_student.batching import _validate_crop
from audiovae_student.discriminators import generator_losses
from gradient_probe import (FrozenWaveformGradientProbe, NAMES, VIEW_SAMPLES,
                            _fresh_balancer, _metric_gradients, _quiet_mask)


def _family(name):
    if name.startswith(("stem_norm.", "affine.")) or ".norm." in name:
        return "norm"
    if name.startswith("blocks."):
        return ".".join(name.split(".")[:2])
    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("stem."):
        return "stem"
    if name.startswith(("head.", "activation.")):
        return "head"
    if name.startswith(("output.", "output_filter.")):
        return "output"
    return "other"


def _vjp(output, parameters, direction):
    gradients = torch.autograd.grad(output, parameters, grad_outputs=direction.detach(),
        retain_graph=True, create_graph=False, allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Nonfinite model VJP")
    return tuple(None if g is None else g.detach() for g in gradients)


def _dots(names, metric_grads, loss_grads):
    grouped = {}
    for name, m, h in zip(names, metric_grads, loss_grads):
        if m is None and h is None:
            continue
        reference = m if m is not None else h
        family = _family(name)
        if family not in grouped:
            grouped[family] = torch.zeros(3, dtype=torch.float64, device=reference.device)
        # Reduce each parameter immediately; retain only three scalars/family.
        if m is not None:
            grouped[family][1] += m.double().square().sum()
        if h is not None:
            grouped[family][2] += h.double().square().sum()
        if m is not None and h is not None:
            grouped[family][0] += (m.double() * h.double()).sum()
    families = {}
    for name, values in grouped.items():
        dot, m2, h2 = values.cpu().tolist()
        families[name] = {"directional_derivative": -dot,
            "metric_parameter_gradient_l2": math.sqrt(m2),
            "loss_parameter_gradient_l2": math.sqrt(h2),
            "gradient_cosine": dot / math.sqrt(m2 * h2) if m2 and h2 else None}
    derivative = sum(v["directional_derivative"] for v in families.values())
    mnorm = math.sqrt(sum(v["metric_parameter_gradient_l2"] ** 2 for v in families.values()))
    hnorm = math.sqrt(sum(v["loss_parameter_gradient_l2"] ** 2 for v in families.values()))
    absolute = sum(abs(v["directional_derivative"]) for v in families.values())
    return {"directional_derivative": derivative, "metric_parameter_gradient_l2": mnorm,
        "loss_parameter_gradient_l2": hnorm,
        "gradient_cosine": -derivative / (mnorm * hnorm) if mnorm and hnorm else None,
        "unit_parameter_direction_derivative": derivative / hnorm if hnorm else None,
        "family_absolute_directional_sum": absolute,
        "family_cancellation_fraction": 1 - abs(derivative) / absolute if absolute else None,
        "families": families}


def _linearity(expected, actual):
    error2 = reference2 = None
    maximum_error = maximum_reference = 0.0
    for a, b in zip(expected, actual):
        if a is None and b is None:
            continue
        reference = a if a is not None else b
        a = torch.zeros_like(reference) if a is None else a
        b = torch.zeros_like(reference) if b is None else b
        delta = (a - b).double()
        e, r = delta.square().sum(), a.double().square().sum()
        error2 = e if error2 is None else error2 + e
        reference2 = r if reference2 is None else reference2 + r
        maximum_error = max(maximum_error, float(delta.abs().max()))
        maximum_reference = max(maximum_reference, float(a.abs().max()))
    error = math.sqrt(float(error2)) if error2 is not None else 0.0
    norm = math.sqrt(float(reference2)) if reference2 is not None else 0.0
    # Algebra is exact; FP32 VJPs need a reported numerical verification bound.
    passed = maximum_error <= 2e-6 + 2e-4 * maximum_reference
    result = {"max_absolute_error": maximum_error, "reference_max_abs": maximum_reference,
        "error_l2": error, "relative_l2_error": error / norm if norm else None,
        "bound": "max_abs_error <= 2e-6 + 2e-4*reference_max_abs", "passed": passed}
    if not passed:
        raise RuntimeError("VJP linearity check failed: " + str(result))
    return result


def _difference(a, b):
    return tuple(None if x is None and y is None else -y if x is None else x.clone()
                 if y is None else x - y for x, y in zip(a, b))


def _effect(value):
    return "decrease" if value < 0 else "increase" if value > 0 else "zero_first_derivative"


@torch.inference_mode(False)
def probe_parameter_geometry(model, crop, engine_state, device="cuda", *, loss_probe=None,
                             expected_prediction=None, include_per_loss=False):
    """One eval forward; sequential VJPs; no .backward(), .grad writes or updates.

    Optional expected_prediction must exactly match this differentiable forward.
    The caller supplies the matching frozen checkpoint model on the given device.
    Default is <=7 model VJPs (both nonzero metrics); optional per-loss adds <=8.
    """
    _validate_crop(crop)
    device = torch.device(device)
    named = tuple(model.named_parameters())
    if not named or any(p.device != device or p.dtype != torch.float32 for _, p in named):
        # Resolve unindexed CUDA to the current device before rejecting it.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if not named or any(p.device != device or p.dtype != torch.float32 for _, p in named):
            raise ValueError("Supply the existing FP32 model on the diagnostic device")
    names, parameters = tuple(n for n, _ in named), tuple(p for _, p in named)
    flags = tuple(p.requires_grad for p in parameters)
    modes = tuple((m, m.training) for m in model.modules())
    previous_grads = tuple((p.grad, p.grad._version if p.grad is not None else None) for p in parameters)
    state_versions = tuple((name, tensor._version) for name, tensor in model.state_dict().items())
    if loss_probe is None:
        loss_probe = FrozenWaveformGradientProbe(engine_state, device=device)
    if loss_probe.step != engine_state["step"] or loss_probe.balancer_state != engine_state["balancer"]:
        raise ValueError("Loss probe belongs to a different checkpoint or balance history")
    report = None
    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(True)
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            latents = crop.latents.detach().to(device).clone()
            teacher = crop.teacher_audio.detach().to(device).clone()
            prediction = model(latents)
            if prediction.shape != teacher.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError("Prediction geometry/nonfinite output differs")
            expected = None
            if expected_prediction is not None:
                if expected_prediction.shape != prediction.shape:
                    raise ValueError("Expected detached prediction has a different shape")
                expected = expected_prediction.detach().to(device)
                if not torch.equal(prediction.detach(), expected):
                    raise RuntimeError("Differentiable prediction differs from cached evaluation by max "
                                       + str(float((prediction.detach() - expected).abs().max())))
            p0 = prediction.detach()
            start = crop.scored_slice.start + (6 if crop.context_start_frame > 0 else 0)
            stop = crop.scored_slice.stop
            if stop - start < max(VIEW_SAMPLES, max(loss_probe.reconstruction_config.fft_sizes)):
                raise ValueError("Valid score region does not fit the diagnostic losses")
            valid = torch.zeros_like(p0, dtype=torch.bool); valid[..., start:stop] = True
            quiet = _quiet_mask(p0, teacher, valid, loss_probe.quiet_config)
            peak = (p0.abs() > 1) & valid
            metrics = _metric_gradients(p0, teacher, valid, quiet)
            selection = ((p0.abs() - 1).clamp_min(0).masked_fill(~valid, -1) if bool(peak.any())
                         else (p0 - teacher).square().masked_fill(~quiet if bool(quiet.any()) else ~valid, -1))
            center = int(selection.flatten().argmax())
            offset = max(start, min(center - VIEW_SAMPLES // 2, stop - VIEW_SAMPLES))
            result = loss_probe.reconstruction(prediction[..., start:stop], teacher[..., start:stop])
            losses = {name: result.losses[name] for name in NAMES[:2]}
            losses.update(generator_losses(loss_probe.discriminators,
                prediction[..., offset:offset + VIEW_SAMPLES], teacher[..., offset:offset + VIEW_SAMPLES]))
            loss_values = {name: float(value.detach()) for name, value in losses.items()}
            balancer = _fresh_balancer(loss_probe.balancer_state)
            balanced = balancer.combine(losses, prediction, valid_mask=valid)
            g = balanced.gradient.detach()
            individual = ({name: torch.autograd.grad(losses[name], prediction, retain_graph=True)[0]
                .detach().masked_fill(~valid, 0) for name in NAMES} if include_per_loss else {})
            del losses, result
            h = _vjp(prediction, parameters, g)
            rows = {}
            for metric_name, (value, metric_audio_gradient) in metrics.items():
                if metric_audio_gradient is None or not bool(metric_audio_gradient.any()):
                    rows[metric_name] = {"value": value, "defined": metric_audio_gradient is not None,
                        "reason": "no quiet samples" if metric_audio_gradient is None else "zero metric gradient",
                        "parameter_directional_derivative": None if metric_audio_gradient is None else 0.0}
                    continue
                m = _vjp(prediction, parameters, metric_audio_gradient)
                full = _dots(names, m, h)
                direct = -float((metric_audio_gradient.double() * g.double()).sum())
                region = peak if metric_name == "peak_excess_mse" else quiet
                local_g, rest_g = g.masked_fill(~region, 0), g.masked_fill(region, 0)
                local_h = _vjp(prediction, parameters, local_g)
                local = _dots(names, m, local_h)
                expected_rest = _difference(h, local_h)
                del local_h
                rest_h = _vjp(prediction, parameters, rest_g)
                rest = _dots(names, m, rest_h)
                linearity = _linearity(expected_rest, rest_h)
                del expected_rest, rest_h
                derivative_sum = local["directional_derivative"] + rest["directional_derivative"]
                absolute = abs(local["directional_derivative"]) + abs(rest["directional_derivative"])
                row = {"value": value, "defined": True, "metric_region_samples": int(region.sum()),
                    "direct_waveform_directional_derivative": direct, "direct_waveform_effect": _effect(direct),
                    "parameter_gradient_direction": full,
                    "parameter_effect": _effect(full["directional_derivative"]),
                    "local_metric_region_loss_gradient": local, "remaining_loss_gradient": rest,
                    "local_vs_rest_cancellation_fraction": 1 - abs(derivative_sum) / absolute if absolute else None,
                    "decomposed_directional_derivative_sum": derivative_sum,
                    "decomposition_directional_absolute_error": abs(derivative_sum - full["directional_derivative"]),
                    "vjp_linearity": linearity, "per_loss": {}}
                for name, audio_gradient in individual.items():
                    scale = balanced.metrics.get(name + "/scale", 0.0)
                    term_h = _vjp(prediction, parameters, audio_gradient)
                    term = _dots(names, m, term_h)
                    del term_h
                    row["per_loss"][name] = {"raw_parameter_gradient_direction": term,
                        "balance_scale": scale,
                        "balanced_parameter_directional_derivative": term["directional_derivative"] * scale,
                        "raw_waveform_directional_derivative": -float((metric_audio_gradient.double()
                                                                       * audio_gradient.double()).sum())}
                rows[metric_name] = row
                del m
            del h
            report = {"format_version": 1, "engine_step": engine_state["step"],
                "source_id": crop.source_id, "start_frame": crop.start_frame,
                "diagnostic": "frozen_eval_model_plain_parameter_gradient_geometry",
                "interpretation": "Negative derivatives improve infinitesimally. Parameter derivative is -<J^T metric_gradient,J^T loss_gradient>, not Muon/AdamW, clipping, weight decay, a finite step, or the actual minibatch update. D is frozen before any new D update.",
                "family_policy": "Exclusive groups; all block LayerNorm and stem/output affine normalization belong to norm; other block parameters belong to blocks.N",
                "checkpoint_prediction_reproduced_exactly": True if expected is not None else None,
                "prediction_comparison_status": "exact" if expected is not None else "caller did not supply a cached prediction",
                "valid_start_in_crop": start, "valid_stop_in_crop": stop,
                "quiet_samples": int(quiet.sum()), "peak_samples": int(peak.sum()),
                "event_sample_in_crop": center, "event_absolute_sample": crop.context_start_frame * 1920 + center,
                "perceptual_start_in_crop": offset, "perceptual_samples": VIEW_SAMPLES,
                "loss_values": loss_values, "balancer_metrics": balanced.metrics,
                "copied_balancer_updates_before": loss_probe.balancer_state["updates"],
                "copied_balancer_updates_after": balancer.updates,
                "metrics": rows, "model_parameter_count": sum(p.numel() for p in parameters),
                "parameter_tensors": len(parameters), "include_per_loss": include_per_loss,
                "student_or_optimizer_updates": 0}
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        if state_versions != tuple((n, t._version) for n, t in model.state_dict().items()):
            raise RuntimeError("Model parameters or buffers changed")
        if any(p.grad is not old or (old is not None and old._version != version)
               for p, (old, version) in zip(parameters, previous_grads)):
            raise RuntimeError("Existing parameter .grad fields changed")
    report["parameter_buffer_versions_preserved"] = True
    report["module_modes_requires_grad_and_grad_fields_restored"] = True
    return report


def self_test():
    """Exact small linear model shows output improvement/parameter interference."""
    theta = torch.tensor([1.0], requires_grad=True)
    # The two output samples share one parameter with opposite Jacobian signs.
    output = torch.stack((theta[0], -2 * theta[0]))
    metric_audio_gradient, g = torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])
    m, h = _vjp(output, (theta,), metric_audio_gradient), _vjp(output, (theta,), g)
    direct = -float(metric_audio_gradient @ g)
    full = _dots(("head.weight",), m, h)
    assert direct == -1 and full["directional_derivative"] == 1
    local_h = _vjp(output, (theta,), torch.tensor([1.0, 0.0]))
    rest_h = _vjp(output, (theta,), torch.tensor([0.0, 1.0]))
    assert _dots(("head.weight",), m, local_h)["directional_derivative"] == -1
    assert _dots(("head.weight",), m, rest_h)["directional_derivative"] == 2
    assert _linearity(_difference(h, local_h), rest_h)["max_absolute_error"] == 0
    assert theta.grad is None and float(theta.detach()) == 1
    assert _family("blocks.4.norm.weight") == "norm" and _family("blocks.4.project.weight") == "blocks.4"
    print("PASS: exact CPU linear VJP fixture, opposite direct/parameter signs, local/rest interference, linearity, no parameter or .grad update")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args(); self_test()
