"""Opt-in, bounded quiet-phase training for a separately declared comparison.

The active/base trainer is unchanged. A phase branch must first load its parent
exactly, fork its reconstruction clock and explicitly record activation evidence.
The caller still verifies frozen teacher weights and target-cache identities:
this engine owns only student weights and detached waveform targets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
import re

import torch

from .distillation_training import (DistillationEngine, DistillationTrainingConfig,
                                   _aligned_quiet_target, model_state_fingerprint)
from .gradient_balancer import MelGradientCapConfig
from .objective_comparison import state_fingerprint
from .quiet_phase import QuietPhaseConfig, _phase_values
from .teacher import CHECKPOINT_SHA256


@dataclass(frozen=True)
class QuietPhaseEngineConfig:
    gradient_share: float = 0.0
    loss: QuietPhaseConfig = QuietPhaseConfig()

    def __post_init__(self):
        if (isinstance(self.gradient_share, bool) or not math.isfinite(self.gradient_share)
                or not 0 <= self.gradient_share <= .05):
            raise ValueError("Phase gradient share must be finite in [0, 0.05]")
        if isinstance(self.loss, dict):
            object.__setattr__(self, "loss", QuietPhaseConfig(**self.loss))
        if not isinstance(self.loss, QuietPhaseConfig):
            raise TypeError("Phase loss requires a validated QuietPhaseConfig")
        if self.loss.period_samples != 480 or self.loss.window_samples != 960:
            raise ValueError("This engine candidate requires the decoder's 480-sample head period")


def _policy(value):
    if not isinstance(value, dict):
        raise ValueError("Explicit teacher-identified phase activation evidence is required")
    value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if (value.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or value.get("teacher_state_unchanged") is not True):
        raise ValueError("Phase evidence must identify the unchanged original teacher")
    for key in ("parent_checkpoint_sha256", "evidence_sha256"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]):
            raise ValueError(f"Phase evidence requires {key}")
    return value


def _statistics(model_state):
    names = ("stem_norm", "affine")
    keys = [f"{name}.{key}" for name in names for key in
            ("running_mean", "running_var", "num_batches_tracked", "statistics_frozen")]
    if not all(key in model_state for key in keys):
        raise ValueError("Phase comparison requires complete normalization statistics")
    return {key: model_state[key] for key in keys}


class QuietPhaseDistillationEngine(DistillationEngine):
    """Base-exact when disabled; current-batch phase norm share when enabled."""

    def __init__(self, model, *, phase_config=QuietPhaseEngineConfig(), mel_cap=None, **kwargs):
        if not isinstance(phase_config, QuietPhaseEngineConfig):
            raise TypeError("A validated QuietPhaseEngineConfig is required")
        super().__init__(model, **kwargs)
        if mel_cap is not None:
            if not isinstance(mel_cap, MelGradientCapConfig):
                raise TypeError("mel_cap requires a validated MelGradientCapConfig")
            self.balancer = self.balancer.fork_with_mel_cap(mel_cap)
        self.phase_config = phase_config
        self.phase_activation = None
        self._loaded_parent = None
        self._phase_fork = None

    def fork_reconstruction(self, config: DistillationTrainingConfig) -> dict:
        if self.phase_activation is not None or self.phase_config.gradient_share:
            raise ValueError("An active phase candidate cannot be silently forked as a new objective")
        if self._loaded_parent is None:
            raise ValueError("Load the exact parent checkpoint before the phase comparison fork")
        if state_fingerprint(super().state_dict()) != self._loaded_parent["state_sha256"]:
            raise ValueError("The loaded parent state changed before the comparison fork")
        result = super().fork_reconstruction(config)
        self._phase_fork = {"provenance": deepcopy(result),
                            "state_sha256": state_fingerprint(super().state_dict())}
        return result

    def enable_phase(self, *, gradient_share: float, policy: dict) -> dict:
        """Activate only immediately after an exact parent load and frozen fork.

        ``policy`` binds the caller-verified parent file, original teacher and
        diagnostic evidence file. Its teacher claim supplements, and does not
        replace, the caller's actual weight and target-cache verification.
        """
        config = replace(self.phase_config, gradient_share=gradient_share)
        if config.gradient_share == 0:
            raise ValueError("Phase activation requires a positive explicit share")
        policy = _policy(policy)
        if self.phase_activation is not None or self._loaded_parent is None or self._phase_fork is None:
            raise ValueError("Phase activation requires a fresh exact parent load and comparison fork")
        if self.step != 0 or state_fingerprint(super().state_dict()) != self._phase_fork["state_sha256"]:
            raise ValueError("The comparison fork changed before phase activation")
        self._require_reconstruction()
        activation = {"format_version": 1, "policy": policy,
                      "parent": deepcopy(self._loaded_parent), "fork": deepcopy(self._phase_fork),
                      "enabled_step": 0, "model_state_sha256": model_state_fingerprint(self.model),
                      "fixed_statistics_sha256": state_fingerprint(_statistics(self.model.state_dict())),
                      "training_config": asdict(self.config), "phase_config": asdict(config)}
        self.phase_config, self.phase_activation = config, activation
        return deepcopy(activation)

    def _require_reconstruction(self):
        if self.perceptual_start is not None or self.gate is not None:
            raise ValueError("Phase comparison requires reconstruction only, without GANs")
        if self.config.quiet_gradient_share != 0:
            raise ValueError("Phase comparison cannot combine the rejected general quiet objective")
        if self.config.freeze_normalization_step != 0 or not all(
                bool(getattr(self.model, name).statistics_frozen) for name in ("stem_norm", "affine")):
            raise ValueError("Phase comparison requires frozen normalization from update zero")

    def _require_activation(self):
        self._require_reconstruction()
        if (self.phase_activation is None
                or asdict(self.phase_config) != self.phase_activation["phase_config"]
                or asdict(self.config) != self.phase_activation["training_config"]):
            raise ValueError("Phase configuration changed or activation evidence is missing")

    def set_reconstruction_objective(self, *, waveform_share: float, quiet_share: float):
        if self.phase_activation is not None and (waveform_share != self.config.reconstruction_waveform_share
                                                 or quiet_share != 0):
            raise ValueError("A phase comparison cannot change its reconstruction objective")
        return super().set_reconstruction_objective(waveform_share=waveform_share, quiet_share=quiet_share)

    def enable_perceptual(self, gate):
        if self.phase_config.gradient_share or self.phase_activation is not None:
            raise ValueError("GAN activation cannot be combined with this isolated phase comparison")
        return super().enable_perceptual(gate)

    def train_step(self, crops):
        if self.phase_config.gradient_share or self.phase_activation is not None:
            self._require_activation()
        return super().train_step(crops)

    def _combine_quiet(self, batch, base_gradient):
        if self.phase_activation is not None:
            self._require_activation()
        if self.phase_config.gradient_share == 0:
            return super()._combine_quiet(batch, base_gradient)
        self._require_activation()
        share = self.phase_config.gradient_share
        loss, complete, quiet, eligible, active, _, template_rms, _ = _phase_values(
            batch.prediction, _aligned_quiet_target(batch), batch.score_mask, self.phase_config.loss)
        gradient, = torch.autograd.grad(loss, batch.prediction, retain_graph=True)
        gradient = gradient.detach().float().masked_fill(~batch.score_mask, 0)
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(base_gradient).all()):
            raise FloatingPointError("Nonfinite base or quiet-phase output gradient")
        norm = float(gradient.flatten(1).norm(dim=1).mean())
        base_norm = float(base_gradient.detach().float().flatten(1).norm(dim=1).mean())
        epsilon = self.balancer.config.epsilon
        scale, base_scale = 0.0, 1.0
        if norm > epsilon and base_norm > epsilon:
            scale = min(self.balancer.config.max_scale, share * base_norm / norm)
            base_scale = 1.0 - share
            combined = base_gradient * base_scale + gradient.to(base_gradient.dtype) * scale
        else:
            combined = base_gradient
        if not bool(torch.isfinite(combined).all()):
            raise FloatingPointError("Nonfinite combined quiet-phase gradient")
        scaled_norm, base_scaled_norm = norm * scale, base_norm * base_scale
        achieved = scaled_norm / max(epsilon, scaled_norm + base_scaled_norm)
        if achieved > share + 1e-12:
            raise FloatingPointError("Quiet-phase pre-sum gradient share exceeded its configured bound")
        values = {"teacher_phase": float(loss.detach()), "teacher_phase/raw_norm": norm,
                  "teacher_phase/scale": scale, "teacher_phase/scaled_norm": scaled_norm,
                  "teacher_phase/target_share": share, "teacher_phase/achieved_share": achieved,
                  "teacher_phase/base_scale": base_scale, "teacher_phase/base_scaled_norm": base_scaled_norm,
                  "teacher_phase/combined_norm": float(combined.detach().float().flatten(1).norm(dim=1).mean()),
                  "teacher_phase/zero_norm": float(norm <= epsilon),
                  "teacher_phase/zero_base_norm": float(base_norm <= epsilon),
                  "teacher_phase/scale_bound": float(scale >= self.balancer.config.max_scale),
                  "teacher_phase/complete_windows": float(complete.sum()),
                  "teacher_phase/teacher_quiet_windows": float(quiet.sum()),
                  "teacher_phase/eligible_windows": float(eligible.sum()),
                  "teacher_phase/active_examples": float(active.sum()),
                  "teacher_phase/min_windows_per_example": float(eligible.min()),
                  "teacher_phase/max_windows_per_example": float(eligible.max()),
                  "teacher_phase/template_rms_mean": float(template_rms.detach().sum() / active.sum().clamp_min(1))}
        if any(not math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite quiet-phase metrics")
        return combined, values

    def state_dict(self):
        result = super().state_dict()
        result["quiet_phase"] = {"format_version": 1, "config": asdict(self.phase_config),
                                 "activation": deepcopy(self.phase_activation)}
        return result

    def load_state_dict(self, state):
        saved = deepcopy(state)  # Optimizer tensors must not alias the caller's parent payload.
        extension = saved.pop("quiet_phase", None)
        activation = None
        if extension is None:
            if self.phase_config.gradient_share != 0 or self.phase_activation is not None:
                raise ValueError("A plain parent checkpoint can only load with phase disabled")
        else:
            if (not isinstance(extension, dict) or set(extension) != {"format_version", "config", "activation"}
                    or extension["format_version"] != 1
                    or QuietPhaseEngineConfig(**extension["config"]) != self.phase_config):
                raise ValueError("Changed phase configuration cannot be an exact resume")
            activation = extension["activation"]
            if self.phase_config.gradient_share:
                required = {"format_version", "policy", "parent", "fork", "enabled_step", "model_state_sha256",
                            "fixed_statistics_sha256", "training_config", "phase_config"}
                if (not isinstance(activation, dict) or set(activation) != required
                        or activation["format_version"] != 1
                        or type(activation["enabled_step"]) is not int or activation["enabled_step"] != 0
                        or activation["phase_config"] != asdict(self.phase_config)
                        or activation["training_config"] != asdict(self.config)
                        or saved.get("perceptual_start") is not None or saved.get("gate") is not None
                        or self.config.quiet_gradient_share != 0 or self.config.freeze_normalization_step != 0
                        or state_fingerprint(_statistics(saved["model"])) != activation["fixed_statistics_sha256"]):
                    raise ValueError("Invalid phase activation or changed fixed statistics on resume")
                _policy(activation["policy"])
                parent, fork = activation["parent"], activation["fork"]
                if (not isinstance(parent, dict) or set(parent) != {"state_sha256", "step", "model_state_sha256"}
                        or type(parent["step"]) is not int or parent["step"] < 0
                        or not isinstance(fork, dict) or set(fork) != {"provenance", "state_sha256"}
                        or not isinstance(fork["provenance"], dict)
                        or fork["provenance"].get("parent_step") != parent["step"]
                        or fork["provenance"].get("parent_model_state_sha256") != parent["model_state_sha256"]
                        or fork["provenance"].get("experiment_config") != asdict(self.config)
                        or activation["model_state_sha256"] != parent["model_state_sha256"]):
                    raise ValueError("Phase parent and fork provenance are inconsistent")
                for scope, key in ((activation, "model_state_sha256"), (activation["parent"], "state_sha256"),
                                   (activation["fork"], "state_sha256")):
                    if not isinstance(scope.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", scope[key]):
                        raise ValueError("Phase activation is missing its exact parent/fork fingerprint")
                if saved.get("step") == 0 and state_fingerprint(saved) != fork["state_sha256"]:
                    raise ValueError("Phase starting state differs from its exact fork")
            elif activation is not None:
                raise ValueError("Disabled phase checkpoints cannot carry active phase evidence")
        super().load_state_dict(saved)
        self.phase_activation = deepcopy(activation)
        self._loaded_parent = {"state_sha256": state_fingerprint(saved), "step": self.step,
                               "model_state_sha256": model_state_fingerprint(self.model)} if activation is None else None
        self._phase_fork = None
        if activation is not None:
            self._require_activation()
            if self.step == 0 and model_state_fingerprint(self.model) != activation["model_state_sha256"]:
                raise ValueError("Phase starting model differs from its activation evidence")
