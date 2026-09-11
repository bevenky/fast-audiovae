"""A bounded perceptual experiment with readiness and final quality kept separate.

The base update loop is reused, but its final-quality activation/loader is not
bypassed using a fabricated ``passed`` flag. This extension validates the actual
readiness artifacts, records their report as such, and owns the trial checkpoint
contract. The old engine therefore rejects an active trial checkpoint.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import re

import torch

from .distillation_training import DistillationEngine, DistillationTrainingConfig, model_state_fingerprint, waveform_gate
from .objective_comparison import state_fingerprint
from .perceptual_readiness import (PerceptualReadinessPolicy, _seal, _summary, _unseal,
    assess_perceptual_readiness, evaluation_evidence, readiness_binding)
from .restart_data import file_sha
from .teacher import CHECKPOINT_SHA256


@dataclass(frozen=True)
class PerceptualTrialConfig:
    max_updates: int = 1000
    review_update: int = 250
    review_error_ratio_max: float = 1.10
    review_level_error_increase_db_max: float = 1.0

    def __post_init__(self):
        if (type(self.max_updates) is not int or not 1 < self.max_updates <= 1000
                or type(self.review_update) is not int or not 1 <= self.review_update < self.max_updates
                or self.review_update > 250):
            raise ValueError("Trial must stop for review by 250 and finish by 1000 updates")
        if (not math.isfinite(self.review_error_ratio_max) or not 1 <= self.review_error_ratio_max <= 1.10
                or not math.isfinite(self.review_level_error_increase_db_max)
                or not 0 <= self.review_level_error_increase_db_max <= 1.0):
            raise ValueError("Trial rollback tolerances cannot exceed the declared limits")


D_SETTINGS = {"optimizer": "AdamW", "learning_rate": 2e-4, "betas": [.8, .9], "weight_decay": 0.0}


def _policy(value, readiness):
    if not isinstance(value, dict):
        raise ValueError("Trial requires an explicit immutable run policy")
    for key in ("parent_checkpoint_sha256", "data_plan_sha256"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]):
            raise ValueError(f"Trial policy requires verified {key}")
    if (value.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or value.get("readiness_sha256") != readiness["sha256"]):
        raise ValueError("Trial policy changes the teacher or readiness identity")
    return _unseal(_seal(deepcopy(value)))


def _active_phase(state):
    phase = state.get("quiet_phase")
    return phase is not None


def _fixed_statistics(state):
    return state_fingerprint({key: state[key] for name in ("stem_norm", "affine")
        for field in ("running_mean", "running_var", "num_batches_tracked", "statistics_frozen")
        for key in (name + "." + field,)})


class PerceptualTrialEngine(DistillationEngine):
    def __init__(self, model, *, trial_config=PerceptualTrialConfig(), **kwargs):
        if not isinstance(trial_config, PerceptualTrialConfig):
            raise TypeError("A validated PerceptualTrialConfig is required")
        super().__init__(model, **kwargs)
        self.trial_config = trial_config
        self.trial_activation = None
        self.trial_review = None
        self._loaded_parent = None
        self._activation_sha256 = None
        self._readiness_sha256 = None
        self._review_sha256 = None

    def enable_trial(self, readiness_report, policy, *, crops, current, previous, evidence):
        """Verify the exact parent, then atomically declare its experiment fork.

        The caller supplies all artifacts used to derive readiness, so an edited
        boolean or a correctly hashed but invented report cannot activate a run.
        No optimizer update occurs here. Control must use the returned fork
        configuration and the same parent/data before the paired trial runs.
        """
        if self.trial_activation is not None or self._loaded_parent is None:
            raise ValueError("Load an exact plain reconstruction parent before trial activation")
        if state_fingerprint(super().state_dict()) != self._loaded_parent["state_sha256"]:
            raise ValueError("The loaded parent changed before trial activation")
        body = _unseal(readiness_report)
        readiness_policy = PerceptualReadinessPolicy(**body["policy"])
        actual = assess_perceptual_readiness(self, crops, current=current, previous=previous,
                                            evidence=evidence, policy=readiness_policy)
        if actual != readiness_report or actual.get("ready_for_bounded_trial") is not True:
            raise ValueError("Actual current readiness does not authorize a bounded trial")
        policy = _policy(policy, actual)
        if (self.perceptual_start is not None or self.gate is not None
                or self.config.quiet_gradient_share != 0
                or self.balancer.mel_cap is not None
                or self.discriminator_optimizer.state):
            raise ValueError("This isolated trial requires plain reconstruction and fresh unused discriminators")
        if self.config.reconstruction_waveform_share != .75 or self.config.perceptual_ramp_steps != 500:
            raise ValueError("Trial requires the declared 75/25 parent and 500-update perceptual ramp")
        if any(p.grad is not None for p in self.discriminators.parameters()):
            raise ValueError("Fresh discriminators cannot carry gradients from earlier training")
        if (self.trial_config.max_updates > readiness_policy.max_trial_updates
                or self.trial_config.review_update > readiness_policy.safety_review_update):
            raise ValueError("Trial budget exceeds its readiness policy")
        parent_binding = readiness_binding(self, crops)
        parent_sha = self._loaded_parent["state_sha256"]
        fork_config = replace(self.config, total_steps=self.trial_config.max_updates, warmup_steps=0,
            freeze_normalization_step=0, learning_rate_schedule="constant_after_warmup",
            warmup_start_learning_rate=None, final_learning_rate=self.config.learning_rate)
        provenance = super().fork_reconstruction(fork_config)
        # Only the discriminator optimizer changes. Generator moments, Muon
        # state, crop RNG, model parameters and frozen statistics are retained.
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminators.parameters(),
            lr=D_SETTINGS["learning_rate"], betas=tuple(D_SETTINGS["betas"]), weight_decay=D_SETTINGS["weight_decay"])
        activation = _seal({"format_version": 1, "kind": "bounded_perceptual_activation",
            "parent": deepcopy(self._loaded_parent), "parent_binding": parent_binding,
            "parent_state_sha256": parent_sha, "fork": provenance,
            "readiness": deepcopy(actual), "initial_evaluation": deepcopy(current),
            "trial_config": asdict(self.trial_config), "training_config": asdict(self.config),
            "discriminator_optimizer": deepcopy(D_SETTINGS), "policy": policy,
            "fixed_statistics_sha256": _fixed_statistics(self.model.state_dict()),
            "implementation_sha256": file_sha(Path(__file__)), "enabled_step": 0})
        self.trial_activation = activation
        self._activation_sha256 = activation["sha256"]
        self._readiness_sha256 = actual["sha256"]
        self.perceptual_start = 0
        self.gate = deepcopy(actual)  # Actual readiness, without a final-quality passed flag.
        self.trial_review = None
        self._review_sha256 = None
        return deepcopy(activation)

    def _validate_live_trial(self):
        if self.trial_activation is None:
            raise ValueError("No perceptual trial is active")
        # The full panel evidence is checked on activation/load/checkpoint.
        # Do not rehash thousands of quiet-window rows on every GPU update.
        activation = self.trial_activation
        if (activation.get("sha256") != self._activation_sha256 or self._activation_sha256 is None
                or activation["trial_config"] != asdict(self.trial_config)
                or DistillationTrainingConfig(**activation["training_config"]) != self.config
                or activation["implementation_sha256"] != file_sha(Path(__file__))
                or self.perceptual_start != 0 or not isinstance(self.gate, dict)
                or self.gate.get("sha256") != self._readiness_sha256 or "passed" in self.gate
                or self.config.quiet_gradient_share != 0 or self.balancer.mel_cap is not None
                or _fixed_statistics(self.model.state_dict()) != activation["fixed_statistics_sha256"]
                or not all(bool(getattr(self.model, n).statistics_frozen) for n in ("stem_norm", "affine"))):
            raise ValueError("Trial policy, objective, implementation or normalization changed")
        if (self.trial_review is not None and (not isinstance(self.trial_review, dict)
                or self._review_sha256 is None or self.trial_review.get("sha256") != self._review_sha256)):
            raise ValueError("Unverified review cannot unlock the trial budget")
        for group in self.discriminator_optimizer.param_groups:
            if (group["lr"] != D_SETTINGS["learning_rate"] or tuple(group["betas"]) != tuple(D_SETTINGS["betas"])
                    or group["weight_decay"] != D_SETTINGS["weight_decay"]):
                raise ValueError("Discriminator optimizer settings changed")

    def train_step(self, crops):
        self._validate_live_trial()
        limit = self.trial_config.review_update if self.trial_review is None else self.trial_config.max_updates
        if self.step >= limit:
            raise ValueError("Perceptual trial is stopped at its explicit review or completion bound")
        return super().train_step(crops)

    def enable_perceptual(self, gate):
        raise ValueError("This experiment activates only from verified readiness via enable_trial")

    def fork_reconstruction(self, config):
        raise ValueError("Use enable_trial to verify readiness and declare the controlled fork together")

    def set_reconstruction_objective(self, *, waveform_share, quiet_share):
        raise ValueError("The bounded trial cannot change its declared reconstruction objective")

    def review_trial(self, current_evaluation, *, crops):
        """Explicitly review actual current-panel measurements before continuing.

        Passing this limited rollback screen allows the remainder of this
        experiment only. The caller must still inspect language/event results;
        final acceptance and listening qualification are separate.
        """
        self._validate_live_trial()
        if self.step != self.trial_config.review_update or self.trial_review is not None:
            raise ValueError("Review must occur once at the exact declared safety boundary")
        now = _unseal(current_evaluation)
        rebuilt = evaluation_evidence(crops, now["rows"], run_identity=now["run_identity"],
                                      model_config=self.model.config.to_dict())
        if (rebuilt != current_evaluation or now["model_state_sha256"] != model_state_fingerprint(self.model)
                or now["step"] != self.step
                or now["panel_sha256"] != self.trial_activation["parent_binding"]["panel_sha256"]):
            raise ValueError("Review needs the actual current model and unchanged development panel")
        prior = _unseal(self.trial_activation["initial_evaluation"])
        policy = PerceptualReadinessPolicy(**self.trial_activation["readiness"]["policy"])
        before, after = _summary(prior["rows"], policy), _summary(now["rows"], policy)
        checks = {name + "_within_rollback_limit": after[name] <= self.trial_config.review_error_ratio_max * before[name]
                  for name in ("waveform_error", "mel_error", "quiet_residual_mean")}
        checks.update({
            "level_error_within_rollback_limit": abs(after["active_median_level_db"]) <=
                abs(before["active_median_level_db"]) + self.trial_config.review_level_error_increase_db_max,
            "no_added_clipping": after["clipped_fraction"] <= before["clipped_fraction"],
            "no_added_active_collapse": after["active_nearzero_count"] <= before["active_nearzero_count"],
        })
        report = _seal({"format_version": 1, "kind": "bounded_perceptual_review", "step": self.step,
            "activation_sha256": self.trial_activation["sha256"],
            "model_state_sha256": model_state_fingerprint(self.model),
            "optimizer_sha256": state_fingerprint(self.optimizer.state_dict()),
            "evaluation": deepcopy(current_evaluation), "before": before, "after": after, "checks": checks,
            "continuation_authorized": all(checks.values()), "final_acceptance": waveform_gate(self, now["rows"])})
        if report["continuation_authorized"]:
            self.trial_review = report
            self._review_sha256 = report["sha256"]
        return deepcopy(report)

    def state_dict(self):
        if self.trial_activation is not None:
            _unseal(self.trial_activation)
            _unseal(self.gate)
            if self.gate != self.trial_activation["readiness"]:
                raise ValueError("Readiness evidence changed before checkpointing")
            if self.trial_review is not None:
                _unseal(self.trial_review)
        result = super().state_dict()
        result["perceptual_trial"] = {"format_version": 1, "config": asdict(self.trial_config),
            "activation": deepcopy(self.trial_activation), "review": deepcopy(self.trial_review),
            "base_state_sha256": state_fingerprint(result)}
        return result

    def load_state_dict(self, state):
        saved = deepcopy(state)
        if _active_phase(saved):
            raise ValueError("Quiet-phase parents require a separately declared transition")
        extension = saved.pop("perceptual_trial", None)
        if extension is not None and (not isinstance(extension, dict)
                or extension.get("base_state_sha256") != state_fingerprint(saved)):
            raise ValueError("Trial core-state checksum changed")
        if extension is None or extension.get("activation") is None:
            if self.trial_activation is not None or saved.get("perceptual_start") is not None or saved.get("gate") is not None:
                raise ValueError("Only a plain reconstruction parent may enter trial preparation")
            if extension is not None and (extension != {"format_version": 1, "config": asdict(self.trial_config),
                    "activation": None, "review": None, "base_state_sha256": state_fingerprint(saved)}):
                raise ValueError("Inactive trial checkpoint configuration changed")
            super().load_state_dict(saved)
            self._loaded_parent = {"state_sha256": state_fingerprint(super().state_dict()), "step": self.step,
                                   "model_state_sha256": model_state_fingerprint(self.model)}
            return
        if (set(extension) != {"format_version", "config", "activation", "review", "base_state_sha256"}
                or extension["format_version"] != 1 or PerceptualTrialConfig(**extension["config"]) != self.trial_config):
            raise ValueError("Trial checkpoint version or configuration changed")
        activation = _unseal(extension["activation"])
        readiness = _unseal(activation["readiness"])
        initial = _unseal(activation["initial_evaluation"])
        if (activation.get("format_version") != 1 or activation.get("kind") != "bounded_perceptual_activation"
                or activation.get("enabled_step") != 0 or activation.get("trial_config") != asdict(self.trial_config)
                or activation.get("discriminator_optimizer") != D_SETTINGS
                or activation.get("implementation_sha256") != file_sha(Path(__file__))
                or readiness.get("format_version") != 1 or readiness.get("kind") != "perceptual_trial_readiness"
                or readiness.get("ready_for_bounded_trial") is not True or not readiness.get("checks")
                or any(value is not True for value in readiness["checks"].values())
                or "passed" in readiness or readiness.get("automatic_activation") is not False
                or readiness.get("binding") != activation.get("parent_binding")):
            raise ValueError("Invalid actual-readiness activation evidence")
        # Keep final-quality evidence as recorded, never convert it to a trial gate.
        if (activation["readiness"]["evaluation_sha256"]["current"] != activation["initial_evaluation"]["sha256"]
                or initial["model_state_sha256"] != activation["parent_binding"]["model_state_sha256"]
                or initial["step"] != activation["parent_binding"]["step"]
                or initial["panel_sha256"] != activation["parent_binding"]["panel_sha256"]
                or activation["parent_state_sha256"] != activation["parent"]["state_sha256"]
                or activation["parent"]["model_state_sha256"] != activation["parent_binding"]["model_state_sha256"]
                or activation["parent"]["step"] != activation["parent_binding"]["step"]
                or activation["fork"]["parent_step"] != activation["parent"]["step"]
                or activation["fork"]["parent_config"] != activation["parent_binding"]["training_config"]
                or activation["fork"]["parent_model_state_sha256"] != activation["parent"]["model_state_sha256"]):
            raise ValueError("Trial parent, fork or evaluation provenance changed")
        _policy(activation["policy"], activation["readiness"])
        expected = (saved.get("format_version") == 1
            and DistillationTrainingConfig(**saved["config"]) == self.config
            and DistillationTrainingConfig(**activation["training_config"]) == self.config
            and DistillationTrainingConfig(**activation["fork"]["experiment_config"]) == self.config
            and saved.get("model_config") == self.model.config.to_dict()
            and saved.get("loss_config") == asdict(self.criterion.config)
            and saved.get("discriminator_config") == asdict(self.discriminators.config)
            and saved.get("perceptual_start") == 0 and saved.get("gate") == activation["readiness"])
        if not expected:
            raise ValueError("Changed trial model, objective, schedule or readiness cannot be an exact resume")
        step, review = saved.get("step"), extension["review"]
        if type(step) is not int or not 0 <= step <= self.trial_config.max_updates:
            raise ValueError("Invalid trial update count")
        if review is None and step > self.trial_config.review_update:
            raise ValueError("Checkpoint crossed its safety boundary without a review")
        if review is not None:
            checked_review = _unseal(review)
            reviewed_evaluation = _unseal(checked_review["evaluation"])
            if (checked_review.get("kind") != "bounded_perceptual_review"
                    or checked_review.get("step") != self.trial_config.review_update or step < checked_review["step"]
                    or checked_review.get("activation_sha256") != extension["activation"]["sha256"]
                    or checked_review.get("continuation_authorized") is not True
                    or not checked_review.get("checks") or not all(checked_review["checks"].values())
                    or reviewed_evaluation["model_state_sha256"] != checked_review["model_state_sha256"]
                    or reviewed_evaluation["step"] != checked_review["step"]
                    or reviewed_evaluation["panel_sha256"] != activation["parent_binding"]["panel_sha256"]):
                raise ValueError("Missing or changed safety review")
        # Validate the last actual ramp weights before loading any components.
        fraction = min(1.0, step / self.config.perceptual_ramp_steps)
        share = self.config.reconstruction_waveform_share
        weights = self.reconstruction_weights() if step == 0 else {
            "teacher_waveform": share * (1 - fraction) + .3 * fraction,
            "teacher_mel": (1 - share) * (1 - fraction) + .4 * fraction,
            "feature_matching": .2 * fraction, "adversarial": .1 * fraction}
        if saved["balancer"].get("weights") != weights:
            raise ValueError("Saved loss shares disagree with the actual perceptual ramp")
        checked_balancer = deepcopy(self.balancer)
        checked_balancer.load_state_dict(saved["balancer"])
        self.model.load_state_dict(saved["model"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.discriminators.load_state_dict(saved["discriminators"], strict=True)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminators.parameters(),
            lr=D_SETTINGS["learning_rate"], betas=tuple(D_SETTINGS["betas"]), weight_decay=D_SETTINGS["weight_decay"])
        self.discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        self.balancer = checked_balancer
        self.crop_generator.set_state(saved["crop_rng"].cpu())
        self.step, self.perceptual_start, self.gate = step, 0, deepcopy(saved["gate"])
        self.trial_activation, self.trial_review = deepcopy(extension["activation"]), deepcopy(review)
        self._activation_sha256 = self.trial_activation["sha256"]
        self._readiness_sha256 = self.gate["sha256"]
        self._review_sha256 = None if review is None else review["sha256"]
        self._loaded_parent = None
        self._validate_live_trial()
        if step == 0 and (model_state_fingerprint(self.model) != activation["parent_binding"]["model_state_sha256"]
                or state_fingerprint(self.optimizer.state_dict()) != activation["parent_binding"]["optimizer_sha256"]
                or state_fingerprint(self.balancer.state_dict()) != activation["parent_binding"]["balancer_sha256"]
                or self.discriminator_optimizer.state):
            raise ValueError("Trial initial weights, optimizer or EMA differ from readiness")
