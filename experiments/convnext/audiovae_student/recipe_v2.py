"""Complete frozen-teacher recipe with scheduled perceptual training.

This is a new experiment format, not an exact continuation of the old
correlation-gated reconstruction runs. Final acceptance stays independent.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import hashlib
import json

import torch
from torch.nn import functional as F

from .batching import _validate_crop
from .cache import DECODER_HOP
from .distillation_training import (DistillationEngine, DistillationTrainingConfig,
    ScoredBatch, _finite_parameters)
from .discriminators import discriminator_loss, generator_losses
from .normalization_calibration import calibrate_normalization
from .reconstruction_v2 import ReconstructionV2


def _norm_fingerprint(state):
    digest = hashlib.sha256()
    for module in ("stem_norm", "affine"):
        for name in ("running_mean", "running_var", "num_batches_tracked", "statistics_frozen"):
            key = module + "." + name
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _report_hash(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RecipeV2Config:
    total_steps: int = 10000
    reconstruction_warmup_steps: int = 500
    perceptual_ramp_steps: int = 500
    learning_rate_warmup_steps: int = 50
    learning_rate: float = 2e-4
    discriminator_learning_rate: float = 2e-4
    adversarial_samples: int = 9120
    optimizer: str = "muon_adamw"
    seed: int = 47

    def __post_init__(self):
        for name in ("total_steps", "reconstruction_warmup_steps", "perceptual_ramp_steps",
                     "learning_rate_warmup_steps", "adversarial_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if not self.learning_rate_warmup_steps <= self.reconstruction_warmup_steps < self.total_steps:
            raise ValueError("Warmup must precede the scheduled perceptual stage")
        for name in ("learning_rate", "discriminator_learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Learning rates must be finite and positive")


def calibration_batch(crops, device):
    """Preserve real history, exclude unscored context and incomplete latents."""
    longest = max(c.latents.shape[-1] for c in crops)
    latents = torch.cat([F.pad(c.latents.detach(), (0, longest - c.latents.shape[-1])) for c in crops]).to(device)
    mask = torch.zeros((len(crops), longest), dtype=torch.bool, device=device)
    for i, crop in enumerate(crops):
        _validate_crop(crop)
        mask[i, crop.context_frames:crop.context_frames + crop.valid_scored_samples // DECODER_HOP] = True
    return latents, mask


def scored_batch_v2(model, crops, criterion, device):
    crops = tuple(crops)
    if not crops:
        raise ValueError("A nonempty scored batch is required")
    if any(c.valid_scored_samples < max(criterion.config.fft_sizes) for c in crops):
        raise ValueError("Scored audio must accommodate every reconstruction FFT")
    latents, latent_mask = calibration_batch(crops, device)
    prediction = model(latents, scored_latent_mask=latent_mask)
    if prediction.shape != (len(crops), 1, latents.shape[-1] * DECODER_HOP):
        raise ValueError("Decoder changed latent/sample accounting")
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    predictions, targets, buckets = [], [], defaultdict(list)
    for i, crop in enumerate(crops):
        mask[i, :, crop.scored_slice] = True
        predictions.append(prediction[i:i + 1, :, crop.scored_slice])
        targets.append(crop.teacher_audio[..., crop.scored_slice].detach().to(device))
        buckets[crop.valid_scored_samples].append(i)
    result = criterion.forward_groups([(torch.cat([predictions[i] for i in indices]),
        torch.cat([targets[i] for i in indices])) for indices in buckets.values()])
    batch = ScoredBatch(prediction, mask, crops, tuple(predictions), tuple(targets), result.losses)
    return batch, result


class RecipeV2Engine(DistillationEngine):
    def __init__(self, model, *, recipe=RecipeV2Config(), discriminators=None):
        if model.config.adapter_mode != "raw_repeat_phase_bias":
            raise ValueError("Recipe v2 requires the identity-preserving four-phase adapter")
        config = DistillationTrainingConfig(total_steps=recipe.total_steps,
            warmup_steps=recipe.learning_rate_warmup_steps,
            freeze_normalization_step=recipe.reconstruction_warmup_steps,
            perceptual_ramp_steps=recipe.perceptual_ramp_steps,
            learning_rate=recipe.learning_rate,
            discriminator_learning_rate=recipe.discriminator_learning_rate,
            reconstruction_waveform_share=.5, learning_rate_schedule="constant_after_warmup",
            adversarial_samples=recipe.adversarial_samples, optimizer=recipe.optimizer,
            crop_seed=recipe.seed, quiet_window_gate=True)
        super().__init__(model, config=config, discriminators=discriminators)
        self.recipe = recipe
        self.reconstruction = ReconstructionV2().to(self.device)
        self.calibration = None
        self.discriminator_updates = 0
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminators.parameters(),
            lr=recipe.discriminator_learning_rate, betas=(.8, .9), weight_decay=0)

    @property
    def calibration_due(self):
        return self.step == self.recipe.reconstruction_warmup_steps and self.calibration is None

    def calibrate(self, batches, *, provenance):
        if not self.calibration_due:
            raise ValueError("Calibration belongs at the fixed warmup boundary, exactly once")
        report = calibrate_normalization(self.model, batches, provenance=provenance)
        self.calibration = {"completed_step": self.step, "report": report,
            "report_sha256": _report_hash(report),
            "fixed_buffers_sha256": _norm_fingerprint(self.model.state_dict())}
        self.perceptual_start = self.step
        return deepcopy(self.calibration)

    def enable_perceptual(self, gate):
        raise ValueError("Recipe v2 uses the declared step schedule, not a quality entry gate")

    def fork_reconstruction(self, config):
        raise ValueError("Recipe v2 cannot import an old reconstruction-stage objective")

    def set_reconstruction_objective(self, **kwargs):
        raise ValueError("Recipe v2 objective is fixed in its versioned schedule")

    def gradient_probe_names(self):
        return (self.model.adapter_parameter_name,
                f"blocks.{len(self.model.blocks) // 2}.project.weight", "output.weight")

    def train_step(self, crops):
        if self.step >= self.recipe.total_steps:
            raise ValueError("Declared update budget exhausted")
        if self.step >= self.recipe.reconstruction_warmup_steps and self.calibration is None:
            raise ValueError("Fixed-weight normalization calibration is required at the warmup boundary")
        if any(c.valid_scored_samples < self.recipe.adversarial_samples for c in crops):
            raise ValueError("Every planned crop must support the same adversarial sample count")
        self.model.train()
        lr = self.learning_rate()
        for optimizer in self.optimizer.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = lr
        self.optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        batch, result = scored_batch_v2(self.model, crops, self.reconstruction, self.device)
        losses = {key: batch.losses[key] for key in ("teacher_waveform", "teacher_mel")}
        fraction, disc_loss, disc_norm = 0., None, None
        if self.perceptual_start is not None:
            fraction = min(1., (self.step - self.perceptual_start + 1) / self.recipe.perceptual_ramp_steps)
            predicted, target, count = self._perceptual_audio(batch)
            if count != len(crops):
                raise RuntimeError("An adversarial example was silently dropped")
            example_weights = predicted.new_tensor([crop.valid_scored_samples for crop in crops])
            self.discriminators.train()
            disc_loss = discriminator_loss(self.discriminators, predicted, target,
                                           example_weights=example_weights)
            if not bool(torch.isfinite(disc_loss)):
                raise FloatingPointError("Nonfinite discriminator loss")
            disc_loss.backward()
            disc_norm = torch.nn.utils.clip_grad_norm_(self.discriminators.parameters(),
                self.config.gradient_clip, error_if_nonfinite=True)
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            losses.update(generator_losses(self.discriminators, predicted, target,
                                           example_weights=example_weights))
            self.discriminator_updates += 1
        self.balancer.set_weights({"teacher_waveform": .5 - .2 * fraction,
            "teacher_mel": .5 - .1 * fraction,
            "feature_matching": .2 * fraction, "adversarial": .1 * fraction})
        balanced = self.balancer.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        batch.prediction.backward(balanced.gradient)
        parameters = tuple(self.model.parameters())
        if any(p.grad is None for p in parameters):
            raise FloatingPointError("An intended student parameter has no gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        if (not _finite_parameters((*parameters, *self.model.buffers())) or
                not _finite_parameters((*self.discriminators.parameters(), *self.discriminators.buffers()))):
            raise FloatingPointError("Nonfinite trained parameters or statistics")
        if any(p.grad is not None for p in self.discriminators.parameters()):
            raise RuntimeError("Generator backward accumulated discriminator gradients")
        self.step += 1
        metrics = {key: float(value.detach()) for key, value in {**batch.losses, **losses}.items()}
        metrics.update({"legacy/" + key: float(value) for key, value in result.legacy_diagnostics.items()})
        metrics.update(balanced.metrics)
        metrics.update(step=self.step, learning_rate=lr, gradient_norm=float(norm),
            examples=len(crops), scored_samples=sum(c.valid_scored_samples for c in crops),
            perceptual_fraction=fraction, discriminator_updates=self.discriminator_updates,
            fixed_statistics=float(self.calibration is not None))
        if disc_loss is not None:
            metrics.update(discriminator=float(disc_loss.detach()), discriminator_gradient_norm=float(disc_norm))
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite training metrics")
        return metrics

    def state_dict(self):
        if self.calibration and _norm_fingerprint(self.model.state_dict()) != self.calibration["fixed_buffers_sha256"]:
            raise RuntimeError("Calibrated normalization statistics changed during training")
        state = super().state_dict()
        state.update(format_version="recipe_v2", recipe=asdict(self.recipe),
            reconstruction_v2=asdict(self.reconstruction.config),
            calibration=deepcopy(self.calibration), discriminator_updates=self.discriminator_updates)
        return state

    def load_state_dict(self, state):
        if (state.get("format_version") != "recipe_v2" or state.get("recipe") != asdict(self.recipe)
                or state.get("config") != asdict(self.config)
                or state.get("model_config") != self.model.config.to_dict()
                or state.get("loss_config") != asdict(self.criterion.config)
                or state.get("reconstruction_v2") != asdict(self.reconstruction.config)
                or state.get("discriminator_config") != asdict(self.discriminators.config)
                or state.get("gate") is not None):
            raise ValueError("Recipe, model or objective changed; this is not an exact resume")
        step, calibration, start = state.get("step"), state.get("calibration"), state.get("perceptual_start")
        if type(step) is not int or not 0 <= step <= self.recipe.total_steps:
            raise ValueError("Invalid saved step")
        warmup = self.recipe.reconstruction_warmup_steps
        if (step < warmup and calibration is not None) or (step > warmup and calibration is None):
            raise ValueError("Saved calibration disagrees with the declared schedule")
        if calibration is None:
            if start is not None or any(bool(state["model"][name + ".statistics_frozen"])
                                        for name in ("stem_norm", "affine")):
                raise ValueError("Warmup statistics or stage were changed")
        elif (calibration.get("completed_step") != warmup or start != warmup
                or calibration.get("report", {}).get("provenance", {}).get("split") != "train"
                or not all(bool(state["model"][name + ".statistics_frozen"])
                           for name in ("stem_norm", "affine"))):
            raise ValueError("Missing fixed-weight training-only calibration evidence")
        if calibration:
            report = calibration["report"]
            if (report.get("method") != "sequential_fp64_sample_weighted_population_moments"
                    or report.get("parameters_unchanged") is not True
                    or report.get("statistics_frozen") is not True
                    or calibration.get("report_sha256") != _report_hash(report)
                    or calibration.get("fixed_buffers_sha256") != _norm_fingerprint(state["model"])):
                raise ValueError("Calibration report or fixed normalization buffers changed")
        if state.get("discriminator_updates") != max(0, step - warmup):
            raise ValueError("Discriminator updates do not match the step-based recipe")
        fraction = min(1., max(0., (step - warmup) / self.recipe.perceptual_ramp_steps))
        expected_weights = {"teacher_waveform": .5 - .2 * fraction,
            "teacher_mel": .5 - .1 * fraction, "feature_matching": .2 * fraction,
            "adversarial": .1 * fraction}
        if state.get("balancer", {}).get("weights") != expected_weights:
            raise ValueError("Saved loss shares differ from the declared step schedule")
        groups = state.get("discriminator_optimizer", {}).get("param_groups", [])
        if len(groups) != 1 or any(tuple(group.get("betas", ())) != (.8, .9)
                or group.get("lr") != self.recipe.discriminator_learning_rate
                or group.get("weight_decay") != 0 for group in groups):
            raise ValueError("Saved discriminator optimizer differs from the declared recipe")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.discriminators.load_state_dict(state["discriminators"], strict=True)
        self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        self.balancer.load_state_dict(state["balancer"])
        self.crop_generator.set_state(state["crop_rng"].cpu())
        self.step, self.calibration, self.perceptual_start = step, deepcopy(calibration), start
        self.discriminator_updates = state["discriminator_updates"]
