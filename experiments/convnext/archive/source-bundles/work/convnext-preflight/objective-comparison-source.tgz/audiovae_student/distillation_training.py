"""Explicit two-stage teacher distillation with masked waveform supervision.

This engine accepts already verified continuous teacher caches. It neither
selects data nor starts a job. Perceptual training requires an explicit passed
gate, and fixed normalization, before any discriminator update is allowed.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .batching import _validate_crop
from .cache import DECODER_HOP, TrainingCrop
from .discriminators import AudioDiscriminators, discriminator_loss, generator_losses
from .gradient_balancer import GradientBalancer, GradientBalancerConfig
from .losses_distillation import DistillationLossConfig, DistillationReconstructionLoss
from .model import StudentDecoder
from .optimizers import build_optimizer_bundle
from .teacher import CHECKPOINT_SHA256


@dataclass(frozen=True)
class DistillationTrainingConfig:
    learning_rate: float = 2e-4
    final_learning_rate: float = 2e-5
    discriminator_learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    total_steps: int = 500
    freeze_normalization_step: int = 100
    perceptual_ramp_steps: int = 500
    adversarial_samples: int = 9120  # 0.19 seconds at 48 kHz.
    gradient_clip: float = 1.0
    optimizer: str = "muon_adamw"
    crop_seed: int = 7
    reconstruction_waveform_share: float = 0.5
    learning_rate_schedule: str = "cosine"
    warmup_start_learning_rate: float | None = None
    parameter_update_metrics_interval: int = 0

    def __post_init__(self):
        for name in ("learning_rate", "final_learning_rate", "discriminator_learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.weight_decay or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final_learning_rate cannot exceed learning_rate")
        for name in ("total_steps", "perceptual_ramp_steps", "adversarial_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "freeze_normalization_step"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.total_steps or self.freeze_normalization_step >= self.total_steps:
            raise ValueError("Warmup and normalization freezing must precede completion")
        if type(self.crop_seed) is not int or self.crop_seed < 0:
            raise ValueError("crop_seed must be a nonnegative integer")
        if (not math.isfinite(self.reconstruction_waveform_share)
                or not 0 < self.reconstruction_waveform_share <= 1):
            raise ValueError("reconstruction_waveform_share must be finite in (0, 1]")
        if self.learning_rate_schedule not in ("cosine", "constant_after_warmup"):
            raise ValueError("Unknown learning_rate_schedule")
        if self.warmup_start_learning_rate is not None:
            if (not math.isfinite(self.warmup_start_learning_rate)
                    or not 0 < self.warmup_start_learning_rate <= self.learning_rate
                    or self.warmup_steps < 2):
                raise ValueError("Explicit warmup start requires at least two steps and a positive rate no larger than peak")
        if type(self.parameter_update_metrics_interval) is not int or self.parameter_update_metrics_interval < 0:
            raise ValueError("parameter_update_metrics_interval must be a nonnegative integer")


@dataclass
class ScoredBatch:
    prediction: Tensor
    score_mask: Tensor
    crops: tuple[TrainingCrop, ...]
    predictions: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]
    losses: dict[str, Tensor]


def scored_batch(model: StudentDecoder, crops: Sequence[TrainingCrop], criterion,
                 device: torch.device | str) -> ScoredBatch:
    crops = tuple(crops)
    if not crops:
        raise ValueError("At least one scored crop is required")
    for crop in crops:
        _validate_crop(crop)
        if crop.valid_scored_samples < max(criterion.config.fft_sizes):
            raise ValueError("Every scored region must accommodate all reconstruction FFTs")
    longest = max(c.latents.shape[-1] for c in crops)
    inputs = torch.cat([F.pad(c.latents.detach(), (0, longest - c.latents.shape[-1])) for c in crops]).to(device)
    latent_mask = torch.zeros((len(crops), longest), dtype=torch.bool, device=device)
    for i, crop in enumerate(crops):
        # A partly padded final latent is predicted/scored as appropriate, but
        # never contributes artificial samples to batch-statistic estimation.
        complete = crop.valid_scored_samples // DECODER_HOP
        latent_mask[i, crop.context_frames:crop.context_frames + complete] = True
    prediction = model(inputs, scored_latent_mask=latent_mask)
    expected = (len(crops), 1, longest * DECODER_HOP)
    if tuple(prediction.shape) != expected:
        raise ValueError("Decoder changed latent/sample accounting")
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    predicted, targets = [], []
    buckets = defaultdict(list)
    for i, crop in enumerate(crops):
        mask[i, :, crop.scored_slice] = True
        predicted.append(prediction[i:i + 1, :, crop.scored_slice])
        targets.append(crop.teacher_audio[..., crop.scored_slice].detach().to(device))
        buckets[crop.valid_scored_samples].append(i)
    values = defaultdict(list)
    for indices in buckets.values():
        group = criterion(torch.cat([predicted[i] for i in indices]),
                          torch.cat([targets[i] for i in indices]))
        for name, value in group.items():
            values[name].append(value * (len(indices) / len(crops)))
    losses = {name: torch.stack(items).sum() for name, items in values.items()}
    return ScoredBatch(prediction, mask, crops, tuple(predicted), tuple(targets), losses)


def _finite_parameters(parameters):
    # One device synchronization, including the normalization buffers that
    # define deployment behavior, rather than one synchronization per tensor.
    tensors = tuple(parameters)
    return bool(torch.stack([torch.isfinite(p).all() for p in tensors]).all().item()) if tensors else True


def model_state_fingerprint(model):
    """Bind acceptance evidence to all parameters and persistent buffers."""
    digest = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DistillationEngine:
    def __init__(self, model: StudentDecoder, *, config=DistillationTrainingConfig(),
                 loss_config=DistillationLossConfig(), balancer_config=None,
                 discriminators=None):
        if not isinstance(model, StudentDecoder):
            raise TypeError("Only a standalone StudentDecoder can be optimized")
        if model.config.normalization_mode != "masked_batch_norm":
            raise ValueError("The corrected recipe requires masked training normalization")
        self.model, self.config = model, config
        self.device = next(model.parameters()).device
        self.criterion = DistillationReconstructionLoss(loss_config).to(self.device)
        self.discriminators = (AudioDiscriminators() if discriminators is None else discriminators).to(self.device)
        self.optimizer = build_optimizer_bundle(model, optimizer=config.optimizer,
                                                lr=config.learning_rate, weight_decay=config.weight_decay)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminators.parameters(),
            lr=config.discriminator_learning_rate, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        self.balancer = GradientBalancer(balancer_config or GradientBalancerConfig(),
                                         weights=self.reconstruction_weights())
        self.step = 0
        self.perceptual_start = None
        self.gate = None
        self.crop_generator = torch.Generator(device="cpu").manual_seed(config.crop_seed)

    def reconstruction_weights(self):
        share = self.config.reconstruction_waveform_share
        return {"teacher_waveform": share, "teacher_mel": 1.0 - share,
                "feature_matching": 0.0, "adversarial": 0.0}

    def fork_reconstruction(self, config: DistillationTrainingConfig) -> dict:
        """Start an explicitly new experiment without resetting learned state.

        Load the parent checkpoint exactly before calling this method. The
        optimizer's internal counters/moments, discriminator, crop RNG and
        balancer EMA remain intact. Only the experiment clock and documented
        objective/schedule controls change; this is not an exact resume.
        """
        if not isinstance(config, DistillationTrainingConfig):
            raise TypeError("A validated DistillationTrainingConfig is required")
        if self.perceptual_start is not None or self.gate is not None:
            raise ValueError("Only reconstruction-stage checkpoints can fork this comparison")
        if not all(bool(getattr(self.model, name).statistics_frozen)
                   for name in ("stem_norm", "affine")):
            raise ValueError("A reconstruction fork requires already frozen normalization")
        allowed = {"learning_rate", "final_learning_rate", "warmup_steps", "total_steps",
                   "freeze_normalization_step", "reconstruction_waveform_share", "learning_rate_schedule",
                   "warmup_start_learning_rate", "parameter_update_metrics_interval"}
        before, after = asdict(self.config), asdict(config)
        if any(before[key] != after[key] for key in before if key not in allowed):
            raise ValueError("A controlled fork cannot change optimizer, clipping, seed or discriminator configuration")
        if config.freeze_normalization_step != 0:
            raise ValueError("A controlled fork must retain frozen normalization from update zero")
        provenance = {"parent_step": self.step, "parent_model_state_sha256": model_state_fingerprint(self.model),
                      "parent_config": before, "experiment_config": after}
        self.config = config
        self.step = 0
        self.balancer.set_weights(self.reconstruction_weights())
        return provenance

    def enable_perceptual(self, gate: dict):
        if (not isinstance(gate, dict) or gate.get("passed") is not True
                or gate.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
                or gate.get("model_config") != self.model.config.to_dict()
                or gate.get("evaluated_step") != self.step
                or gate.get("fixed_statistics") is not True
                or gate.get("model_state_sha256") != model_state_fingerprint(self.model)):
            raise ValueError("A passed teacher-identified gate at this exact model step is required")
        if self.perceptual_start is not None:
            raise ValueError("Perceptual training has already started")
        self.model.freeze_normalization_statistics()
        self.gate = dict(gate)
        self.perceptual_start = self.step

    def learning_rate(self):
        if self.config.warmup_steps and self.step < self.config.warmup_steps:
            if self.config.warmup_start_learning_rate is not None:
                fraction = self.step / (self.config.warmup_steps - 1)
                return self.config.warmup_start_learning_rate + fraction * (
                    self.config.learning_rate - self.config.warmup_start_learning_rate)
            return self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        if self.config.learning_rate_schedule == "constant_after_warmup":
            return self.config.learning_rate
        span = max(1, self.config.total_steps - self.config.warmup_steps - 1)
        fraction = min(1.0, max(0.0, (self.step - self.config.warmup_steps) / span))
        return self.config.final_learning_rate + (self.config.learning_rate - self.config.final_learning_rate) * (
            1 + math.cos(math.pi * fraction)) / 2

    def _perceptual_audio(self, batch):
        predictions, targets = [], []
        needed = self.config.adversarial_samples
        for prediction, target in zip(batch.predictions, batch.targets):
            if prediction.shape[-1] < needed:
                continue
            start = int(torch.randint(prediction.shape[-1] - needed + 1, (), generator=self.crop_generator).item())
            predictions.append(prediction[..., start:start + needed])
            targets.append(target[..., start:start + needed])
        if not predictions:
            raise ValueError("Perceptual batches require at least one valid complete adversarial crop")
        return torch.cat(predictions), torch.cat(targets), len(predictions)

    def train_step(self, crops: Sequence[TrainingCrop]) -> dict:
        if self.step >= self.config.total_steps:
            raise ValueError("The declared update budget is exhausted")
        self.model.train()
        if self.step >= self.config.freeze_normalization_step:
            self.model.freeze_normalization_statistics()
        lr = self.learning_rate()
        for optimizer in self.optimizer.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = lr
        self.optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        batch = scored_batch(self.model, crops, self.criterion, self.device)
        losses = {name: batch.losses[name] for name in ("teacher_waveform", "teacher_mel")}
        disc_loss = None
        adversarial_examples = 0
        if self.perceptual_start is not None:
            fraction = min(1.0, (self.step - self.perceptual_start + 1) / self.config.perceptual_ramp_steps)
            share = self.config.reconstruction_waveform_share
            # Preserve the historical floating-point expression for exact
            # resumes of the default balanced recipe.
            wave_weight = 0.5 - 0.2 * fraction if share == 0.5 else share * (1 - fraction) + 0.3 * fraction
            mel_weight = 0.5 - 0.1 * fraction if share == 0.5 else (1 - share) * (1 - fraction) + 0.4 * fraction
            self.balancer.set_weights({"teacher_waveform": wave_weight,
                "teacher_mel": mel_weight,
                "feature_matching": 0.2 * fraction, "adversarial": 0.1 * fraction})
            predicted, target, adversarial_examples = self._perceptual_audio(batch)
            self.discriminators.train()
            disc_loss = discriminator_loss(self.discriminators, predicted, target)
            if not torch.isfinite(disc_loss).item():
                raise FloatingPointError("Nonfinite discriminator loss")
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminators.parameters(), self.config.gradient_clip,
                                          error_if_nonfinite=True)
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            losses.update(generator_losses(self.discriminators, predicted, target))
        balanced = self.balancer.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        batch.prediction.backward(balanced.gradient)
        parameters = tuple(self.model.parameters())
        if any(p.grad is None for p in parameters):
            raise FloatingPointError("An intended student parameter has no gradient")
        update_metrics = {}
        interval = self.config.parameter_update_metrics_interval
        sample_update = interval > 0 and (self.step + 1) % interval == 0
        probes = {}
        if sample_update:
            for name in self.gradient_probe_names():
                parameter = self.model.get_parameter(name)
                probes[name] = parameter.detach().clone()
                prefix = f"parameter_update/{name}"
                update_metrics[f"{prefix}/gradient_norm_before_clip"] = float(parameter.grad.detach().float().norm())
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip, error_if_nonfinite=True)
        for name in probes:
            update_metrics[f"parameter_update/{name}/gradient_norm_after_clip"] = float(
                self.model.get_parameter(name).grad.detach().float().norm())
        self.optimizer.step()
        for name, before in probes.items():
            after = self.model.get_parameter(name).detach()
            delta, weight = (after.float() - before.float()).norm(), before.float().norm()
            prefix = f"parameter_update/{name}"
            update_metrics.update({f"{prefix}/delta_norm": float(delta), f"{prefix}/weight_norm": float(weight),
                                   f"{prefix}/relative_delta_norm": float(delta / weight.clamp_min(1e-12))})
        if (not _finite_parameters((*parameters, *self.model.buffers()))
                or not _finite_parameters((*self.discriminators.parameters(), *self.discriminators.buffers()))):
            raise FloatingPointError("Nonfinite parameters or normalization state after optimizer update")
        if any(p.grad is not None for p in self.discriminators.parameters()):
            raise RuntimeError("Generator backward accumulated discriminator parameter gradients")
        self.step += 1
        metrics = {name: float(value.detach().cpu()) for name, value in {**batch.losses, **losses}.items()}
        metrics.update({"step": self.step, "learning_rate": lr, "gradient_norm": float(norm),
                        "examples": len(crops), "adversarial_examples": adversarial_examples,
                        "scored_samples": sum(c.valid_scored_samples for c in crops)})
        metrics.update(balanced.metrics)
        metrics.update(update_metrics)
        if disc_loss is not None:
            metrics["discriminator"] = float(disc_loss.detach())
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("Nonfinite training metrics")
        return metrics

    def gradient_probe_names(self):
        return ("adapter.weight", f"blocks.{len(self.model.blocks) // 2}.project.weight", "output.weight")

    def state_dict(self):
        return {"format_version": 1, "config": asdict(self.config),
                "model_config": self.model.config.to_dict(), "loss_config": asdict(self.criterion.config),
                "discriminator_config": asdict(self.discriminators.config),
                "step": self.step, "perceptual_start": self.perceptual_start, "gate": self.gate,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "discriminators": self.discriminators.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "crop_rng": self.crop_generator.get_state(),
                "balancer": self.balancer.state_dict()}

    def load_state_dict(self, state):
        try:
            # New experiment controls default to the historical behavior when
            # loading earlier format-v1 checkpoints that predate these fields.
            saved_config = DistillationTrainingConfig(**state["config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid saved training configuration") from exc
        expected = (state.get("format_version") == 1 and saved_config == self.config
                    and state.get("model_config") == self.model.config.to_dict()
                    and state.get("loss_config") == asdict(self.criterion.config)
                    and state.get("discriminator_config") == asdict(self.discriminators.config))
        if not expected:
            raise ValueError("Changed decoder, objective or schedule cannot be an exact resume")
        step = state.get("step")
        start = state.get("perceptual_start")
        if start is None and state.get("balancer", {}).get("weights") != self.reconstruction_weights():
            raise ValueError("Saved reconstruction weights disagree with the training objective")
        if type(step) is not int or not 0 <= step <= self.config.total_steps:
            raise ValueError("Invalid saved update count")
        if start is not None and (type(start) is not int or not 0 <= start <= step or not state.get("gate", {}).get("passed")):
            raise ValueError("Invalid saved perceptual gate")
        if start is not None:
            gate = state["gate"]
            if (gate.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
                    or gate.get("model_config") != self.model.config.to_dict()
                    or gate.get("evaluated_step") != start or gate.get("fixed_statistics") is not True
                    or not isinstance(gate.get("model_state_sha256"), str)
                    or len(gate["model_state_sha256"]) != 64
                    or not all(bool(state["model"].get(f"{name}.statistics_frozen", False))
                               for name in ("stem_norm", "affine"))):
                raise ValueError("Perceptual resume requires a verified gate and frozen normalization")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.discriminators.load_state_dict(state["discriminators"], strict=True)
        self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        self.balancer.load_state_dict(state["balancer"])
        self.crop_generator.set_state(state["crop_rng"].cpu())
        self.step, self.perceptual_start, self.gate = step, start, state["gate"]


@torch.no_grad()
def evaluate_crops(engine: DistillationEngine, crops: Sequence[TrainingCrop]) -> list[dict]:
    previous = engine.model.training
    fingerprint = model_state_fingerprint(engine.model)
    rows = []
    try:
        engine.model.eval()
        # Evaluate independently so unrelated batch members cannot hide a
        # batch-statistics dependency or dominate the per-clip report.
        for crop in crops:
            batch = scored_batch(engine.model, [crop], engine.criterion, engine.device)
            predicted, target = batch.predictions[0], batch.targets[0]
            rms = target.square().mean().sqrt()
            prediction_rms = predicted.square().mean().sqrt()
            error = (predicted - target).abs().mean()
            baseline = target.abs().mean()
            cosine = F.cosine_similarity(predicted.flatten(), target.flatten(), dim=0, eps=1e-12)
            rows.append({"source_id": crop.source_id, "start_frame": crop.start_frame,
                "evaluated_step": engine.step, "model_state_sha256": fingerprint, "fixed_statistics": True,
                "samples": crop.valid_scored_samples,
                **{name: float(value) for name, value in batch.losses.items()},
                "teacher_rms": float(rms), "student_rms": float(prediction_rms),
                "waveform_to_silence_error_ratio": float(error / baseline.clamp_min(1e-8)),
                "rms_db_error": float(20 * torch.log10(prediction_rms.clamp_min(1e-8) / rms.clamp_min(1e-8))),
                "waveform_cosine": float(cosine)})
        return rows
    finally:
        engine.model.train(previous)


def waveform_gate(engine: DistillationEngine, rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("A waveform gate requires real per-clip evidence")
    failures, milestone_failures = [], []
    fingerprint = model_state_fingerprint(engine.model)
    for row in rows:
        if (row.get("evaluated_step") != engine.step or row.get("model_state_sha256") != fingerprint
                or row.get("fixed_statistics") is not True):
            raise ValueError("Waveform gate evidence is stale or lacks fixed-statistics identity")
        if not all(math.isfinite(float(row[key])) for key in ("teacher_rms", "student_rms",
                "waveform_to_silence_error_ratio", "rms_db_error", "waveform_cosine")):
            raise ValueError("Waveform gate evidence must contain finite measurements")
        if row["teacher_rms"] < 1e-3:
            # Quiet clips stay visible and must not acquire amplified noise.
            passed = row["student_rms"] <= max(1e-3, row["teacher_rms"] * 10 ** (1 / 20))
            milestone_passed = passed
        else:
            amplitude_and_error_passed = (row["waveform_to_silence_error_ratio"] <= 0.5
                                          and abs(row["rms_db_error"]) <= 1.0)
            passed = amplitude_and_error_passed and row["waveform_cosine"] >= 0.99
            milestone_passed = amplitude_and_error_passed and row["waveform_cosine"] >= 0.95
        if not passed:
            failures.append({"source_id": row["source_id"], "start_frame": row["start_frame"]})
        if not milestone_passed:
            milestone_failures.append({"source_id": row["source_id"], "start_frame": row["start_frame"]})
    return {"passed": not failures, "evaluated_step": engine.step,
            "model_state_sha256": fingerprint, "fixed_statistics": True,
            "teacher_checkpoint_sha256": CHECKPOINT_SHA256, "model_config": engine.model.config.to_dict(),
            "criteria": {"waveform_to_silence_error_ratio_max": 0.5, "absolute_rms_db_max": 1.0,
                         "waveform_cosine_min": 0.99, "quiet_rms_threshold": 1e-3},
            "milestone_095_passed": not milestone_failures,
            "milestone_095_failed_clips": milestone_failures,
            "failed_clips": failures, "clips": list(rows)}


def reconstruction_gradient_report(engine: DistillationEngine, crops: Sequence[TrainingCrop]) -> dict:
    """Read actual parameter gradients without updating weights or EMA state."""
    previous = engine.model.training
    try:
        engine.model.eval()
        batch = scored_batch(engine.model, crops, engine.criterion, engine.device)
        losses = {name: batch.losses[name] for name in ("teacher_waveform", "teacher_mel")}
        balancer = deepcopy(engine.balancer)
        balancer.set_weights(engine.reconstruction_weights())
        balanced = balancer.combine(losses, batch.prediction, valid_mask=batch.score_mask)
        for key in losses:
            if balancer.weights[key] == 0:
                balanced.metrics.update({f"{key}/scale": 0.0, f"{key}/scaled_norm": 0.0,
                                         f"{key}/target_share": 0.0})
        names = engine.gradient_probe_names()
        parameters = tuple(engine.model.get_parameter(name) for name in names)
        gradients = {key: torch.autograd.grad(value, parameters, retain_graph=True)
                     for key, value in losses.items()}
        rows = []
        for index, name in enumerate(names):
            wave, mel = (gradients[key][index].detach().float() for key in losses)
            wave_scaled = wave * balanced.metrics["teacher_waveform/scale"]
            mel_scaled = mel * balanced.metrics["teacher_mel/scale"]
            rows.append({"parameter": name, "waveform_raw_norm": float(wave.norm()),
                "mel_raw_norm": float(mel.norm()), "waveform_scaled_norm": float(wave_scaled.norm()),
                "mel_scaled_norm": float(mel_scaled.norm()),
                "cosine": float(F.cosine_similarity(wave.flatten(), mel.flatten(), dim=0, eps=1e-12))})
        return {"evaluated_step": engine.step, "probe_examples": len(crops),
                "probe_scored_samples": sum(c.valid_scored_samples for c in crops),
                "normalization_mode": "fixed_statistics",
                "reconstruction_weights": engine.reconstruction_weights(),
                "balancer_state_source": "copy_of_training_state_updated_on_probe_batch",
                "output_gradients": balanced.metrics, "parameters": rows}
    finally:
        engine.model.train(previous)
