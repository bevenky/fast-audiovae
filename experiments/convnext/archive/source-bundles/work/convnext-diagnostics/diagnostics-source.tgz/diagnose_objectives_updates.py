"""Frozen-checkpoint objective geometry and disposable native update diagnostics.

No checkpoint is written. Counterfactual updates affect a disposable engine only;
its complete state is restored and fingerprinted before returning. Leave-one-out
effects are not additive attributions through nonlinear adaptive optimizers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import math
import time

import torch

from audiovae_student.corrected_calibration import _fixed_engine, _regions, _scheduled_weights
from audiovae_student.discriminators import generator_losses
from audiovae_student.distillation_training import ScoredBatch, _aligned_quiet_target
from audiovae_student.gradient_balancer import GradientBalancer
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.recipe_v2 import scored_batch_v2

LOSSES = ("teacher_waveform", "teacher_mel", "feature_matching", "adversarial")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


def _preserved_attributes(engine):
    return tuple(name for name in ("screen_last_views", "screen_view_entries") if hasattr(engine, name))


def _losses_for_prediction(engine, source_batch, prediction, *, crop_rng):
    """Same target, sample weighting, normalization and view RNG for every input."""
    predictions, targets, groups = [], [], defaultdict(list)
    for index, crop in enumerate(source_batch.crops):
        predictions.append(prediction[index:index + 1, :, crop.scored_slice])
        targets.append(source_batch.targets[index])
        groups[crop.valid_scored_samples].append(index)
    terms = engine.reconstruction.forward_groups([
        (torch.cat([predictions[index] for index in indices]),
         torch.cat([targets[index] for index in indices])) for indices in groups.values()])
    batch = ScoredBatch(prediction, source_batch.score_mask, source_batch.crops,
                        tuple(predictions), tuple(targets), terms.losses)
    losses = {name: terms.losses[name] for name in LOSSES[:2]}
    engine.crop_generator.set_state(crop_rng.clone())
    predicted, target, count = engine._perceptual_audio(batch)
    if count != len(batch.crops):
        raise ValueError("Every diagnostic crop must contain a complete discriminator view")
    weights = predicted.new_tensor([crop.valid_scored_samples for crop in batch.crops])
    losses.update(generator_losses(engine.discriminators, predicted, target, example_weights=weights))
    return batch, losses


def current_ema_scales(engine, gradients):
    """Read the current EMA, without adding this diagnostic as an observation."""
    weights = _scheduled_weights(engine)
    state = engine.balancer.state_dict()
    scales = {}
    for name, gradient in gradients.items():
        norm = float(gradient.float().flatten(1).norm(dim=1).mean())
        history = state["ema"][name]
        average = history["total"] / history["weight"] if history["weight"] else norm
        scale = weights[name] / sum(weights.values()) * engine.balancer.config.total_norm / average if average > engine.balancer.config.epsilon else 0.
        scales[name] = min(engine.balancer.config.max_scale, max(engine.balancer.config.min_scale, scale)) if scale else 0.
    return scales


def audio_gradients(losses, prediction, mask):
    answer = {}
    for name in LOSSES:
        gradient, = torch.autograd.grad(losses[name], prediction, retain_graph=True)
        answer[name] = gradient.detach().float().masked_fill(~mask, 0)
        if not bool(torch.isfinite(answer[name]).all()):
            raise FloatingPointError(f"Nonfinite {name} gradient")
    return answer


def _inner(first, second):
    return float((first.double() * second.double()).sum())


def _cosine(first, second):
    scale = math.sqrt(_inner(first, first) * _inner(second, second))
    return _inner(first, second) / scale if scale else None


def _regional_audio_summary(gradients, scales, masks, direction, residual):
    result = {}
    combined = sum(gradients[name] * scales[name] for name in LOSSES)
    for region, mask in masks.items():
        correction = direction.masked_fill(~mask, 0)
        error = residual.masked_fill(~mask, 0)
        row = {"samples": int(mask.sum()), "losses": {}}
        for name, full in {**gradients, "combined": combined}.items():
            gradient = full.masked_fill(~mask, 0)
            row["losses"][name] = {"norm": math.sqrt(_inner(gradient, gradient)),
                "gradient_cosine_to_residual": _cosine(gradient, error),
                "directional_derivative_toward_teacher": _inner(gradient, correction)}
        result[region] = row
    return result


def objective_geometry(engine, crops, *, seed=1861):
    """Freeze every trained state; interpolate outputs, never model weights."""
    report = {"method": "fixed_output_interpolation", "alphas": [],
              "scales_policy": "current checkpoint EMA, fixed at original student for every alpha",
              "warning": "Loss value alone does not establish a conflicting gradient; GAN loss need not vanish at the teacher."}
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as identity:
        with torch.no_grad():
            original, _ = scored_batch_v2(engine.model, crops, engine.reconstruction, engine.device)
        target, masks = _regions(original, QuietAudioConfig())
        masks["teacher_high_peak"] = original.score_mask & (target.abs() >= .8)
        masks["original_student_overshoot"] = original.score_mask & (original.prediction.abs() > 1.)
        start = original.prediction.detach()
        direction = (target - start).masked_fill(~original.score_mask, 0)
        crop_rng = engine.crop_generator.get_state().clone()
        scales = None
        for alpha in (0., .5, .9, .99, 1.):
            # torch.lerp at alpha1 is not relied upon for teacher identity.
            prediction = (target.clone() if alpha == 1 else start + alpha * direction).detach().requires_grad_(True)
            batch, losses = _losses_for_prediction(engine, original, prediction, crop_rng=crop_rng)
            gradients = audio_gradients(losses, prediction, batch.score_mask)
            if scales is None:
                scales = current_ema_scales(engine, gradients)
            point = {"alpha": alpha, "loss_values": {k: float(v.detach()) for k, v in losses.items()},
                "regions": _regional_audio_summary(gradients, scales, masks, direction,
                                                     prediction.detach() - target),
                "weighted_loss": sum(scales[name] * float(losses[name].detach()) for name in LOSSES)}
            if hasattr(engine, "screen_last_views"):
                point["discriminator_views"] = deepcopy(engine.screen_last_views)
            if alpha == 1:
                point["teacher_identity_max_abs"] = float((prediction.detach() - target).abs().max())
                point["nonadversarial_exact_zero_losses"] = {name: bool(losses[name].detach() == 0) for name in LOSSES[:3]}
                point["nonadversarial_max_abs_gradients"] = {name: float(gradients[name].abs().max()) for name in LOSSES[:3]}
            report["alphas"].append(point)
        report.update(fixed_state=identity, scales=scales, optimizer_updates=0, parameter_updates=0,
                      crops=[{"source_id": c.source_id, "start_frame": c.start_frame} for c in crops])
    return report


def parameter_group(name):
    pieces = name.split(".")
    return ".".join(pieces[:2]) if pieces[0] == "blocks" else pieces[0]


def _parameter_gradient(prediction, parameters, gradient, *, retain_graph=True):
    values = torch.autograd.grad(prediction, parameters, grad_outputs=gradient.to(prediction),
                                 retain_graph=retain_graph, allow_unused=True)
    return tuple(torch.zeros_like(p) if value is None else value.detach() for p, value in zip(parameters, values))


def vector_dot(first, second):
    return sum(_inner(a, b) for a, b in zip(first, second))


def vector_summary(named, vectors, optimizer_routes=None):
    groups = defaultdict(list)
    for index, (name, _) in enumerate(named):
        groups[parameter_group(name)].append(index)
        if optimizer_routes is not None:
            groups["optimizer/" + optimizer_routes[name]].append(index)
    groups["all"] = list(range(len(named)))
    answer = {}
    for group, indices in groups.items():
        norms = {key: math.sqrt(sum(_inner(values[i], values[i]) for i in indices)) for key, values in vectors.items()}
        dots = {a: {b: sum(_inner(ga[i], gb[i]) for i in indices) for b, gb in vectors.items()}
                for a, ga in vectors.items()}
        answer[group] = {"norms": norms, "inner_products": dots,
            "cosines": {a: {b: dot / (norms[a] * norms[b]) if norms[a] * norms[b] else None
                              for b, dot in pairs.items()} for a, pairs in dots.items()}}
    return answer


def diagnostic_values(batch):
    """Diagnostic quantities only; none is inserted into the training objective."""
    target, masks = _regions(batch, QuietAudioConfig())
    prediction, valid = batch.prediction, batch.score_mask
    residual = prediction - target
    def mean_mask(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1)
    # Fixed teacher-relative full-scale envelope. Using a dynamic selected-peak
    # mean would change the denominator when one peak crosses the threshold.
    excess = (prediction.abs() - target.abs().clamp_min(1.)).clamp_min(0)
    return {"teacher_mse": mean_mask(residual.square(), valid),
        "quiet_residual_mse": mean_mask(residual.square(), masks["quiet"]),
        "teacher_relative_peak_excess_energy": mean_mask(excess.square(), valid)}, masks


def routing_diagnostic(engine, crops, *, seed=1861):
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as identity:
        batch, _ = scored_batch_v2(engine.model, crops, engine.reconstruction, engine.device)
        crop_rng = engine.crop_generator.get_state().clone()
        batch, losses = _losses_for_prediction(engine, batch, batch.prediction, crop_rng=crop_rng)
        gradients = audio_gradients(losses, batch.prediction, batch.score_mask)
        scales = current_ema_scales(engine, gradients)
        named = tuple(engine.model.named_parameters())
        routes = {parameter["name"]: group["optimizer"] for group in engine.optimizer.group_manifest
                  for parameter in group["parameters"]}
        parameters = tuple(p for _, p in named)
        vectors = {name: _parameter_gradient(batch.prediction, parameters, gradient * scales[name])
                   for name, gradient in gradients.items()}
        combined = sum(gradients[name] * scales[name] for name in LOSSES)
        metrics, masks = diagnostic_values(batch)
        vectors["combined_quiet_origin"] = _parameter_gradient(batch.prediction, parameters, combined.masked_fill(~masks["quiet"], 0))
        vectors["combined_active_origin"] = _parameter_gradient(batch.prediction, parameters, combined.masked_fill(~masks["active"], 0))
        all_gradient = tuple(sum(vectors[name][i] for name in LOSSES) for i in range(len(parameters)))
        vectors["combined"] = all_gradient
        direct = _parameter_gradient(batch.prediction, parameters, combined)
        linearity_error = max(float((a - b).abs().max()) for a, b in zip(all_gradient, direct))
        metric_gradients = {name: tuple(torch.zeros_like(p) if v is None else v.detach() for p, v in zip(parameters,
            torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True))) for name, value in metrics.items()}
        descent = {metric: {loss: -vector_dot(gradient, values) for loss, values in vectors.items()}
                   for metric, gradient in metric_gradients.items()}
        report = {"fixed_state": identity, "parameter_updates": 0, "optimizer_updates": 0,
            "scales": scales, "module_routing": vector_summary(named, vectors, routes),
            "metric_values": {k: float(v.detach()) for k, v in metrics.items()},
            "unit_sgd_first_order_metric_change": descent,
            "negative_change_means": "locally improving diagnostic metric under plain negative-gradient descent, not the native optimizer",
            "gradient_linearity_max_abs_error": linearity_error}
    return report


def _capture_replay_state(engine):
    return {"model": _cpu_tree(engine.model.state_dict()), "discriminators": _cpu_tree(engine.discriminators.state_dict()),
            "optimizer": _cpu_tree(engine.optimizer.state_dict()), "discriminator_optimizer": _cpu_tree(engine.discriminator_optimizer.state_dict()),
            "balancer": deepcopy(engine.balancer.state_dict()), "step": engine.step,
            "discriminator_updates": engine.discriminator_updates, "crop_rng": engine.crop_generator.get_state().clone(),
            "recipe": asdict(engine.recipe), "config": asdict(engine.config)}


def _restore_replay_state(engine, state):
    engine.model.load_state_dict(state["model"], strict=True)
    engine.discriminators.load_state_dict(state["discriminators"], strict=True)
    engine.optimizer.load_state_dict(deepcopy(state["optimizer"]))
    engine.discriminator_optimizer.load_state_dict(deepcopy(state["discriminator_optimizer"]))
    engine.balancer.load_state_dict(deepcopy(state["balancer"]))
    engine.step, engine.discriminator_updates = state["step"], state["discriminator_updates"]
    engine.crop_generator.set_state(state["crop_rng"].clone())
    engine.recipe = type(engine.recipe)(**deepcopy(state["recipe"]))
    engine.config = type(engine.config)(**deepcopy(state["config"]))
    engine.optimizer.zero_grad()
    engine.discriminator_optimizer.zero_grad(set_to_none=True)


def _probe_snapshot(engine, crops, *, with_gradient=False):
    """Small sequential probe set, preventing a padded long-crop GPU batch."""
    named = tuple(engine.model.named_parameters())
    parameters = tuple(p for _, p in named)
    rows, grads = [], {}
    for index, crop in enumerate(crops):
        with torch.set_grad_enabled(with_gradient):
            batch, _ = scored_batch_v2(engine.model, [crop], engine.reconstruction, engine.device)
            # Historical heldout policy, separate from unchanged training masks.
            if crop.context_start_frame > 0:
                batch.score_mask[..., crop.scored_slice.start:crop.scored_slice.start + 6] = False
            values, masks = diagnostic_values(batch)
            target = _aligned_quiet_target(batch)
            predicted = batch.prediction.detach()
            valid = batch.score_mask
            rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                "samples": int(valid.sum()), "quiet_samples": int(masks["quiet"].sum()),
                **{name: float(value.detach()) for name, value in values.items()},
                "waveform_mae": float((predicted - target).abs()[valid].mean()),
                "peak_abs": float(predicted[valid].abs().max()),
                "overshoot_samples": int(((predicted.abs() > 1) & valid).sum())})
            if with_gradient:
                for name, value in values.items():
                    values_grad = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                    # Store aggregate equal-probe gradients only, not many full copies.
                    if name not in grads:
                        grads[name] = [torch.zeros_like(p, device="cpu") for p in parameters]
                    for total, value_grad in zip(grads[name], values_grad):
                        if value_grad is not None:
                            total.add_(value_grad.detach().cpu(), alpha=1 / len(crops))
    return rows, {name: tuple(values) for name, values in grads.items()}


def native_replays(engine, training_crops, probe_crops, *, seed=1861):
    """Disposable native optimizer updates; restore every trained state on exit."""
    result = {"training_batch_split": "train gradient_calibration, first 32 crops",
        "semantics": "Leave-one-loss-out effects are nonadditive counterfactuals. All clones start with identical optimizer moments and parameters.",
        "probe_aggregation": "equal crop mean for first-order metric derivative, per-crop observed metrics retained; historical six-sample exclusion when context begins inside a recording",
        "rng_policy": "fixed diagnostic seed and supplied batch, not a claim to replay the historical next sampled training batch",
        "variants": {}}
    with _fixed_engine(engine, seed, _preserved_attributes(engine)) as identity:
        state = _capture_replay_state(engine)
        all_parameters = tuple((*engine.model.parameters(), *engine.discriminators.parameters()))
        original_grads = [(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in all_parameters]
        baseline_hash = state_fingerprint(state)
        baseline, diagnostic_gradients = _probe_snapshot(engine, probe_crops, with_gradient=True)
        result["probe_before"] = baseline
        named = tuple(engine.model.named_parameters())
        routes = {parameter["name"]: group["optimizer"] for group in engine.optimizer.group_manifest
                  for parameter in group["parameters"]}
        parameters = tuple(p for _, p in named)
        batch, _ = scored_batch_v2(engine.model, training_crops, engine.reconstruction, engine.device)
        batch, losses = _losses_for_prediction(engine, batch, batch.prediction, crop_rng=state["crop_rng"])
        raw = audio_gradients(losses, batch.prediction, batch.score_mask)
        # Match the next training observation's EMA scale using a throwaway balancer.
        virtual_balancer = deepcopy(engine.balancer)
        virtual_balancer.set_weights(_scheduled_weights(engine))
        balanced = virtual_balancer.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        scales = {name: balanced.metrics[name + "/scale"] for name in LOSSES}
        per_loss = {name: tuple(value.cpu() for value in _parameter_gradient(batch.prediction, parameters, raw[name] * scales[name])) for name in LOSSES}
        result["fixed_D_scales"] = scales
        del batch, losses, raw, balanced, virtual_balancer
        try:
            variants = ["all", *["without_" + name for name in LOSSES], "zero_current_gradient", "exact_training_step"]
            for variant in variants:
                _restore_replay_state(engine, state)
                start = time.monotonic()
                if variant == "exact_training_step":
                    extended = engine.step >= engine.recipe.total_steps
                    if extended:
                        # A disposable clock-budget extension permits this one diagnostic only.
                        before_lr = engine.learning_rate()
                        engine.recipe = replace(engine.recipe, total_steps=engine.step + 1)
                        engine.config = replace(engine.config, total_steps=engine.step + 1)
                        if engine.learning_rate() != before_lr:
                            raise RuntimeError("Budget-only extension changed the current learning rate")
                    train_metrics = engine.train_step(training_crops)
                    clip_norm = train_metrics["gradient_norm"]
                else:
                    active = [name for name in LOSSES if variant != "without_" + name] if variant != "zero_current_gradient" else []
                    for index, parameter in enumerate(parameters):
                        parameter.grad = sum((per_loss[name][index].to(parameter) for name in active), torch.zeros_like(parameter))
                    for optimizer in engine.optimizer.optimizers.values():
                        for group in optimizer.param_groups:
                            group["lr"] = engine.learning_rate()
                    clip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, engine.config.gradient_clip, error_if_nonfinite=True))
                    engine.optimizer.step()
                    train_metrics, extended = None, False
                delta = tuple(parameter.detach().cpu() - state["model"][name] for name, parameter in named)
                after, _ = _probe_snapshot(engine, probe_crops)
                observed = {metric: sum(a[metric] - b[metric] for a, b in zip(after, baseline)) / len(baseline)
                            for metric in diagnostic_gradients}
                result["variants"][variant] = {
                    "discriminator_policy": "one native discriminator step before generator loss" if variant == "exact_training_step" else "fixed discriminator",
                    "balancer_policy": "ordinary next EMA observation" if variant == "exact_training_step" else "scales from identical all-loss virtual next EMA observation, no leave-out renormalization",
                    "disposable_budget_extension": extended,
                    "global_preclip_gradient_norm": float(clip_norm),
                    "global_clip_multiplier": min(1., engine.config.gradient_clip / (float(clip_norm) + 1e-6)),
                    "delta_norm": math.sqrt(vector_dot(delta, delta)),
                    "parameter_group_delta_norms": {group: values["norms"]["delta"]
                        for group, values in vector_summary(named, {"delta": delta}, routes).items()},
                    "first_order_probe_metric_change": {name: vector_dot(gradient, delta) for name, gradient in diagnostic_gradients.items()},
                    "observed_probe_metric_change": observed, "probe_after": after,
                    "training_metrics": train_metrics, "seconds": time.monotonic() - start}
        finally:
            _restore_replay_state(engine, state)
            for parameter, (gradient, saved) in zip(all_parameters, original_grads):
                parameter.grad = gradient
                if gradient is not None:
                    gradient.copy_(saved)
        if state_fingerprint(_capture_replay_state(engine)) != baseline_hash:
            raise RuntimeError("Disposable update replay failed to restore checkpoint state")
        result.update(fixed_state=identity, retained_parameter_updates=0, retained_optimizer_updates=0,
                      disposable_generator_updates=len(result["variants"]), state_restored=True,
                      first_order_method="parameter diagnostic gradient dot native parameter delta, not a full waveform JVP",
                      peak_diagnostic="mean_valid(relu(abs(student)-max(1,abs(teacher)))**2), fixed denominator")
    return result


def run(ctx, names=("parent", "targeted", "complex")):
    from diagnostic_common import atomic_json, status
    from run_corrected_screen import bind_views
    output = {"format_version": 1, "checkpoints": {}, "checkpoint_files_written": 0}
    update_crops = tuple(ctx.pools["gradient_calibration"][:32])
    probes = tuple(ctx.probe_crops)
    # Every objective probe must support the 190 ms discriminator view.
    objective_crops = tuple(c for c in probes if c.valid_scored_samples >= 9120)
    for name in names:
        engine = ctx.engine(name, device="cuda")
        bind_views(engine, ctx.data["pools"]["gradient_calibration"])
        status("objective_geometry", checkpoint=name, probe_crops=len(objective_crops))
        entry = {"objective_geometry": objective_geometry(engine, objective_crops)}
        status("parameter_gradient_routing", checkpoint=name)
        entry["routing"] = routing_diagnostic(engine, objective_crops)
        status("native_optimizer_disposable_replays", checkpoint=name)
        entry["native_updates"] = native_replays(engine, update_crops, probes)
        output["checkpoints"][name] = entry
        atomic_json(ctx.out / ("objectives-updates-" + name + ".json"), entry)
        del engine
        torch.cuda.empty_cache()
        status("objectives_updates_checkpoint_complete", checkpoint=name)
    atomic_json(ctx.out / "objectives-updates.json", output)
    return output


if __name__ == "__main__":
    from diagnostic_common import load_context
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["parent", "targeted", "complex"])
    args = parser.parse_args()
    run(load_context(), tuple(args.checkpoints))
