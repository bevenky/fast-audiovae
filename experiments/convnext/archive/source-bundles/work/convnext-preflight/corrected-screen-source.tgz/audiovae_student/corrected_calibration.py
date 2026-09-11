"""Fixed-weight calibration and audits of the existing output-gradient balancer.

Only explicitly selected EMA fields are replaced. This is an experimental
objective migration, not an exact resume and not normalization calibration.
No model/optimizer update or teacher inference is performed. Targets are the
unaltered frozen-teacher waveforms already carried by TrainingCrop objects.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch

from .discriminators import generator_losses
from .distillation_training import _aligned_quiet_target
from .objective_comparison import state_fingerprint
from .quiet_audio import QuietAudioConfig, _window_values
from .recipe_v2 import scored_batch_v2
from .training import _restore_rng, _rng_state


def _fixed_state(engine):
    return {"model": state_fingerprint(engine.model.state_dict()),
            "discriminators": state_fingerprint(engine.discriminators.state_dict()),
            "optimizer": state_fingerprint(engine.optimizer.state_dict()),
            "discriminator_optimizer": state_fingerprint(engine.discriminator_optimizer.state_dict()),
            "calibration": state_fingerprint(engine.calibration),
            "step": engine.step, "discriminator_updates": engine.discriminator_updates,
            "perceptual_start": engine.perceptual_start}


@contextmanager
def _fixed_engine(engine, seed, preserve_attributes=()):
    if engine.calibration is None or not all(bool(getattr(engine.model, name).statistics_frozen)
                                            for name in ("stem_norm", "affine")):
        raise ValueError("Fixed-weight gradient calibration requires calibrated frozen normalization")
    if type(seed) is not int or seed < 0:
        raise ValueError("Calibration seed must be a nonnegative integer")
    if engine.balancer.mel_cap is not None:
        raise ValueError("Calibrate the uncapped recipe; a mel-cap experiment needs its own audit")
    if (len(set(preserve_attributes)) != len(preserve_attributes)
            or any(not isinstance(name, str) or not hasattr(engine, name) for name in preserve_attributes)):
        raise ValueError("Preserved runtime attributes must be unique existing engine attribute names")
    extra = {name: (getattr(engine, name), deepcopy(getattr(engine, name))) for name in preserve_attributes}
    before = _fixed_state(engine)
    rng = deepcopy(_rng_state())
    crop_rng = engine.crop_generator.get_state().clone()
    modules = tuple(dict.fromkeys((*engine.model.modules(), *engine.discriminators.modules())))
    modes = tuple(module.training for module in modules)
    buffers = [(value, value.detach().clone()) for module in (engine.model, engine.discriminators)
               for value in module.buffers()]
    parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
    flags = tuple(parameter.requires_grad for parameter in parameters)
    # Existing .grad values are neither zeroed nor populated by this audit.
    gradients = tuple((parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
                      for parameter in parameters)
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        engine.crop_generator.manual_seed(seed)
        # Match the ordinary train forward. The required frozen running
        # statistics prevent model-buffer updates; current D has no BN/dropout.
        engine.model.train()
        engine.discriminators.train()
        yield before
        if _fixed_state(engine) != before:
            raise RuntimeError("Fixed-weight gradient audit changed a model, optimizer or training clock")
        for parameter, (gradient, saved) in zip(parameters, gradients):
            if parameter.grad is not gradient or (saved is not None and not torch.equal(gradient, saved)):
                raise RuntimeError("Output-gradient audit changed a parameter gradient")
    finally:
        # Restore runtime flags/RNG even if a crop or loss fails. Buffers are
        # also restored before propagating an invariant failure.
        with torch.no_grad():
            for value, saved in buffers:
                value.copy_(saved)
        for module, mode in zip(modules, modes):
            module.training = mode
        for parameter, flag, (gradient, saved) in zip(parameters, flags, gradients):
            parameter.requires_grad_(flag)
            parameter.grad = gradient
            if gradient is not None:
                gradient.copy_(saved)
        engine.crop_generator.set_state(crop_rng)
        for name, (original, saved) in extra.items():
            if isinstance(original, dict):
                original.clear()
                original.update(saved)
                setattr(engine, name, original)
            elif isinstance(original, list):
                original[:] = saved
                setattr(engine, name, original)
            else:
                setattr(engine, name, saved)
        _restore_rng(rng)


class _DetachedStudent:
    """Expose exactly the student output as a leaf, avoiding G parameter grads."""
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            prediction = self.model(*args, **kwargs)
        return prediction.detach().requires_grad_(True)


def _scheduled_weights(engine):
    fraction = (min(1., (engine.step - engine.perceptual_start + 1) / engine.recipe.perceptual_ramp_steps)
                if engine.perceptual_start is not None else 0.)
    return {"teacher_waveform": .5 - .2 * fraction, "teacher_mel": .5 - .1 * fraction,
            "feature_matching": .2 * fraction, "adversarial": .1 * fraction}


def _fixed_batch_losses(engine, crops):
    if not crops or any(c.valid_scored_samples < engine.recipe.adversarial_samples for c in crops):
        raise ValueError("Every calibration crop must support the ordinary adversarial sample count")
    batch, _ = scored_batch_v2(_DetachedStudent(engine.model), crops, engine.reconstruction, engine.device)
    losses = {name: batch.losses[name] for name in ("teacher_waveform", "teacher_mel")}
    if engine.perceptual_start is not None:
        predicted, target, count = engine._perceptual_audio(batch)
        if count != len(crops):
            raise RuntimeError("A calibration adversarial example was silently dropped")
        example_weights = predicted.new_tensor([crop.valid_scored_samples for crop in crops])
        losses.update(generator_losses(engine.discriminators, predicted, target,
                                       example_weights=example_weights))
    return batch, losses


def _regions(batch, quiet_config):
    target = _aligned_quiet_target(batch)
    _, _, _, _, quiet = _window_values(batch.prediction, target, batch.score_mask, quiet_config)
    quiet_mask = quiet.repeat_interleave(quiet_config.window_samples, dim=-1)
    quiet_mask = quiet_mask[:, None, :batch.prediction.shape[-1]] & batch.score_mask
    return target, {"all": batch.score_mask, "quiet": quiet_mask,
                    "active": batch.score_mask & ~quiet_mask}


def _measure_batch(engine, crops, quiet_config):
    batch, losses = _fixed_batch_losses(engine, crops)
    weights = _scheduled_weights(engine)
    gradients = {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        loss = losses[name]
        if loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Calibration requires finite differentiable scalar losses")
        gradient, = torch.autograd.grad(loss, batch.prediction, retain_graph=True)
        gradient = gradient.detach().float().masked_fill(~batch.score_mask, 0)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Calibration encountered nonfinite audio gradients")
        gradients[name] = gradient
    target, masks = _regions(batch, quiet_config)
    residual = (batch.prediction.detach().float() - target.float()).masked_fill(~batch.score_mask, 0)
    row = {"examples": len(crops), "scored_samples": int(batch.score_mask.sum()),
           "losses": {name: float(loss.detach()) for name, loss in losses.items()}, "regions": {}}
    for region, mask in masks.items():
        selected = {name: gradient.masked_fill(~mask, 0) for name, gradient in gradients.items()}
        error = residual.masked_fill(~mask, 0)
        # The balancer uses mean per-example L2 norms. Inner products below
        # use the entire selected batch and only diagnose output directions.
        norms = {name: float(value.flatten(1).norm(dim=1).mean()) for name, value in selected.items()}
        energy = {name: float(value.double().square().sum()) for name, value in selected.items()}
        error_energy = float(error.double().square().sum())
        dot_error = {name: float((value.double() * error.double()).sum()) for name, value in selected.items()}
        dots = {a: {b: float((ga.double() * gb.double()).sum()) for b, gb in selected.items()}
                for a, ga in selected.items()}
        row["regions"][region] = {"samples": int(mask.sum()), "raw_norms": norms,
            "gradient_energy": energy, "error_energy": error_energy,
            "gradient_dot_residual": dot_error, "gradient_inner_products": dots}
    return row


def _annotate(row, state, weights):
    result = deepcopy(row)
    config = state["config"]
    scales, averages, desired_scales, saturated = {}, {}, {}, {}
    for name, norm in row["regions"]["all"]["raw_norms"].items():
        ema = state["ema"][name]
        mean = ema["total"] / ema["weight"] if ema["weight"] else norm
        desired = (weights[name] / sum(weights.values()) * config["total_norm"] / mean
                   if norm > config["epsilon"] and mean > config["epsilon"] else 0.)
        scales[name] = (min(config["max_scale"], max(config["min_scale"], desired)) if desired else 0.)
        averages[name], desired_scales[name] = mean, desired
        saturated[name] = scales[name] != desired
    result["scales"] = scales
    result["ema_norms"] = averages
    result["requested_scales"] = desired_scales
    result["scale_saturated"] = saturated
    result["target_shares"] = {name: value / sum(weights.values()) for name, value in weights.items()}
    scaled = {name: scales[name] * value for name, value in row["regions"]["all"]["raw_norms"].items()}
    total = sum(scaled.values())
    result["scaled_norms"] = scaled
    result["achieved_presum_shares"] = {name: value / total if total else 0. for name, value in scaled.items()}
    for region, values in result["regions"].items():
        energy = values["gradient_energy"]
        error_energy = values["error_energy"]
        values["cosine_to_residual"] = {
            name: values["gradient_dot_residual"][name] / math.sqrt(value * error_energy)
            if value > 0 and error_energy > 0 else None for name, value in energy.items()}
        combined_energy = sum(scales[a] * scales[b] * dot for a, pairs in values["gradient_inner_products"].items()
                              for b, dot in pairs.items())
        # Numerical cancellation can leave a tiny negative FP64 scalar.
        combined_energy = max(0., combined_energy)
        combined_dot = sum(scales[name] * dot for name, dot in values["gradient_dot_residual"].items())
        values["combined_gradient_energy"] = combined_energy
        values["combined_cosine_to_residual"] = (combined_dot / math.sqrt(combined_energy * error_energy)
            if combined_energy > 0 and error_energy > 0 else None)
    total_energy = result["regions"]["all"]["combined_gradient_energy"]
    result["combined_quiet_energy_share"] = (result["regions"]["quiet"]["combined_gradient_energy"] / total_energy
                                              if total_energy else None)
    return result


def audit_output_gradients(engine, crops: Sequence, *, seed: int = 1307,
                           quiet_config: QuietAudioConfig = QuietAudioConfig(),
                           preserve_attributes: Sequence[str] = ()) -> dict:
    """Read-only audit using the current EMA without a model or EMA update.

    Positive cosine to residual means a negative-gradient audio-space step
    would locally reduce squared teacher error. This is not a parameter-space
    prediction. Quiet masks follow the teacher's 20 ms RMS and scored mask.
    """
    state = deepcopy(engine.balancer.state_dict())
    with _fixed_engine(engine, seed, preserve_attributes) as before:
        row = _measure_batch(engine, tuple(crops), quiet_config)
        result = _annotate(row, state, _scheduled_weights(engine))
    if engine.balancer.state_dict() != state:
        raise RuntimeError("Read-only gradient audit changed the balancer")
    return {"format_version": 1, "method": "fixed_weight_output_gradient_audit", "seed": seed,
            "fixed_state": before, "quiet_config": vars(quiet_config), "measurement": result,
            "parameter_updates": 0, "optimizer_updates": 0,
            "semantics": "Current-EMA pre-sum audio gradients, not parameter-update shares"}


def calibrate_changed_losses(engine, batches: Iterable[Sequence], *,
                             loss_names=("feature_matching", "adversarial"),
                             provenance: dict, seed: int = 1307,
                             quiet_config: QuietAudioConfig = QuietAudioConfig(),
                             preserve_attributes: Sequence[str] = ()) -> dict:
    """Reset only named loss EMAs and seed them from fixed-weight observations.

    Uses the existing balancer's exact {total, weight} debiased-EMA schema and
    decay, starting selected accumulators at zero. Training clocks, all other
    EMA entries, model normalization and Muon/AdamW moments are preserved.
    Calibration observes the current D; no discriminator warmup happens here.
    Reuse across calibration/audit is allowed, but the runner must document it
    separately from optimizer-update audio exposure and keep validation out.
    """
    if not isinstance(provenance, dict) or provenance.get("split") != "train":
        raise ValueError("Gradient calibration needs explicit training-only provenance")
    names = tuple(loss_names)
    previous = deepcopy(engine.balancer.state_dict())
    weights = _scheduled_weights(engine)
    if not names or len(set(names)) != len(names) or any(name not in previous["ema"] for name in names):
        raise ValueError("Declare unique existing loss names to calibrate")
    if any(weights[name] <= 0 for name in names):
        raise ValueError("Cannot calibrate an inactive scheduled loss")
    candidate = deepcopy(previous)
    for name in names:
        candidate["ema"][name] = {"total": 0., "weight": 0.}
    rows = []
    with _fixed_engine(engine, seed, preserve_attributes) as before:
        for crops in batches:
            row = _measure_batch(engine, tuple(crops), quiet_config)
            rows.append(row)
            for name in names:
                norm = row["regions"]["all"]["raw_norms"][name]
                if norm <= engine.balancer.config.epsilon:
                    continue
                ema = candidate["ema"][name]
                ema["total"] = engine.balancer.config.ema_decay * ema["total"] + norm
                ema["weight"] = engine.balancer.config.ema_decay * ema["weight"] + 1.
        if not rows or any(candidate["ema"][name]["weight"] == 0 for name in names):
            raise ValueError("Every selected loss requires at least one finite nonzero gradient observation")
    # Commit once, only after all frozen-state and finite-gradient checks pass.
    engine.balancer.load_state_dict(candidate)
    statistics = {}
    for name in weights:
        observed = [row["regions"]["all"]["raw_norms"][name] for row in rows
                    if name in row["regions"]["all"]["raw_norms"]]
        if observed:
            ema = candidate["ema"][name]
            statistics[name] = {"observations": len(observed), "observed_mean_raw_norm": sum(observed) / len(observed),
                "observed_min_raw_norm": min(observed), "observed_max_raw_norm": max(observed),
                "positive_observations": sum(value > engine.balancer.config.epsilon for value in observed),
                "post_calibration_ema_norm": ema["total"] / ema["weight"] if ema["weight"] else None,
                "recalibrated": name in names}
    return {"format_version": 1, "method": "fixed_weight_selected_loss_ema_reset", "seed": seed,
        "provenance": deepcopy(provenance), "calibrated_loss_names": list(names),
        "batches": len(rows), "examples": sum(row["examples"] for row in rows),
        "scored_samples": sum(row["scored_samples"] for row in rows),
        "fixed_state": before, "balancer_before": previous, "balancer_after": candidate,
        "loss_statistics": statistics,
        "balancer_training_updates_preserved": candidate["updates"] == previous["updates"],
        "parameter_updates": 0, "optimizer_updates": 0, "quiet_config": vars(quiet_config),
        "gradient_scale_semantics": "Fixed before/final EMA on the same calibration observations; no extra EMA update",
        "before": [_annotate(row, previous, weights) for row in rows],
        "after": [_annotate(row, candidate, weights) for row in rows]}
