"""Stateful, bounded output-gradient balancing for distillation.

This implements the EnCodec-style principle with explicit checkpoint state and
coefficient bounds. Shares apply before gradients are summed at audio, not to
parameter-update norms. Context/tail gradients are masked before norm analysis.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class GradientBalancerConfig:
    ema_decay: float = 0.999
    epsilon: float = 1e-12
    total_norm: float = 1.0
    min_scale: float = 1e-4
    max_scale: float = 1e4

    def __post_init__(self) -> None:
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must be finite in [0, 1)")
        for name in ("epsilon", "total_norm", "min_scale", "max_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_scale > self.max_scale:
            raise ValueError("Scale bounds are reversed")


@dataclass(frozen=True)
class BalancedGradients:
    gradient: Tensor
    metrics: dict[str, float]


def _weights(values: Mapping[str, float]) -> dict[str, float]:
    if not values or any(not isinstance(k, str) or not k for k in values):
        raise ValueError("Loss weights require nonempty string keys")
    result = {k: float(v) for k, v in values.items()}
    if any(not math.isfinite(v) or v < 0 for v in result.values()) or sum(result.values()) <= 0:
        raise ValueError("Weights must be finite, nonnegative and not all zero")
    return result


class GradientBalancer:
    def __init__(self, config: GradientBalancerConfig = GradientBalancerConfig(),
                 weights: Mapping[str, float] | None = None) -> None:
        self.config = config
        self.weights = _weights(weights if weights is not None else {
            "teacher_waveform": 0.5, "teacher_mel": 0.5,
            "feature_matching": 0.0, "adversarial": 0.0})
        self._ema = {name: {"total": 0.0, "weight": 0.0} for name in self.weights}
        self.updates = 0

    def set_weights(self, weights: Mapping[str, float]) -> None:
        updated = _weights(weights)
        if updated.keys() != self.weights.keys():
            raise ValueError("Stage changes must preserve the declared loss-name set")
        self.weights = updated

    @staticmethod
    def _mask(prediction: Tensor, valid_mask: Tensor | None,
              valid_lengths: Tensor | None) -> Tensor:
        if prediction.ndim != 3 or prediction.shape[0] < 1 or prediction.shape[1] != 1:
            raise ValueError("Prediction must have shape [batch, 1, samples]")
        if valid_mask is not None and valid_lengths is not None:
            raise ValueError("Specify a validity mask or prefix lengths, not both")
        if valid_lengths is not None:
            if (valid_lengths.ndim != 1 or len(valid_lengths) != prediction.shape[0]
                    or valid_lengths.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Valid lengths must be one integer per example")
            lengths = valid_lengths.to(prediction.device)
            if bool(((lengths <= 0) | (lengths > prediction.shape[-1])).any()):
                raise ValueError("Valid lengths must lie inside each prediction")
            valid_mask = torch.arange(prediction.shape[-1], device=prediction.device)[None, None] < lengths[:, None, None]
        if valid_mask is None:
            valid_mask = torch.ones_like(prediction, dtype=torch.bool)
        if valid_mask.shape != prediction.shape or valid_mask.dtype != torch.bool:
            raise ValueError("Validity mask must be bool with exactly the prediction shape")
        valid_mask = valid_mask.to(prediction.device)
        if bool((valid_mask.sum(dim=(1, 2)) == 0).any()):
            raise ValueError("Every example must contain scored samples")
        return valid_mask

    def combine(self, losses: Mapping[str, Tensor], prediction: Tensor,
                valid_mask: Tensor | None = None,
                valid_lengths: Tensor | None = None) -> BalancedGradients:
        if not prediction.requires_grad or not prediction.is_floating_point():
            raise ValueError("Prediction must be differentiable floating-point audio")
        mask = self._mask(prediction, valid_mask, valid_lengths)
        active = {name for name, weight in self.weights.items() if weight > 0}
        if not active.issubset(losses) or not set(losses).issubset(self.weights):
            raise ValueError("Provide every active loss and no undeclared loss names")
        total_weight = sum(self.weights.values())
        gradients, norms = {}, {}
        for name in self.weights:
            if name not in active:
                continue
            loss = losses[name]
            if loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise ValueError(f"{name} must be a finite differentiable scalar")
            grad, = torch.autograd.grad(loss, prediction, retain_graph=True, create_graph=False)
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError(f"Nonfinite output gradient for {name}")
            grad = grad.detach().float().masked_fill(~mask, 0)
            gradients[name] = grad
            # Per-example L2 norms ignore context and padded samples. They are
            # then averaged, preserving EnCodec-style norm semantics.
            norms[name] = float(grad.flatten(1).norm(dim=1).mean())

        # Validate every gradient before changing EMA state.
        metrics: dict[str, float] = {}
        combined = torch.zeros_like(prediction, dtype=torch.float32)
        norm_total = sum(norms.values())
        for name, grad in gradients.items():
            norm = norms[name]
            share = self.weights[name] / total_weight
            metrics[f"{name}/raw_norm"] = norm
            metrics[f"{name}/target_share"] = share
            metrics[f"{name}/raw_share"] = norm / norm_total if norm_total > 0 else 0.0
            if norm <= self.config.epsilon:
                # A solved/constant term must not get a huge epsilon-derived
                # coefficient or contaminate an EMA initialized by real signal.
                metrics.update({f"{name}/scale": 0.0, f"{name}/scaled_norm": 0.0,
                                f"{name}/saturated": 0.0, f"{name}/zero_norm": 1.0})
                continue
            ema = self._ema[name]
            ema["total"] = self.config.ema_decay * ema["total"] + norm
            ema["weight"] = self.config.ema_decay * ema["weight"] + 1.0
            average = ema["total"] / ema["weight"]
            desired = share * self.config.total_norm / average
            scale = min(self.config.max_scale, max(self.config.min_scale, desired))
            combined.add_(grad, alpha=scale)
            metrics.update({f"{name}/ema_norm": average, f"{name}/scale": scale,
                            f"{name}/scaled_norm": norm * scale,
                            f"{name}/saturated": float(scale != desired), f"{name}/zero_norm": 0.0})
        if not bool(torch.isfinite(combined).all()):
            raise FloatingPointError("Combined output gradient is not finite")
        self.updates += 1
        metrics["combined_norm"] = float(combined.flatten(1).norm(dim=1).mean())
        metrics["valid_samples_mean"] = float(mask.sum(dim=(1, 2)).float().mean())
        return BalancedGradients(combined.to(prediction.dtype), metrics)

    def backward(self, losses: Mapping[str, Tensor], prediction: Tensor,
                 valid_mask: Tensor | None = None,
                 valid_lengths: Tensor | None = None) -> dict[str, float]:
        result = self.combine(losses, prediction, valid_mask, valid_lengths)
        prediction.backward(result.gradient)
        return result.metrics

    def state_dict(self) -> dict:
        return {"format_version": 1, "config": asdict(self.config), "weights": dict(self.weights),
                "updates": self.updates, "ema": {k: dict(v) for k, v in self._ema.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("format_version") != 1 or state.get("config") != asdict(self.config):
            raise ValueError("Gradient balancer checkpoint format/config does not match")
        weights = _weights(state.get("weights", {}))
        if weights.keys() != self.weights.keys() or state.get("ema", {}).keys() != self.weights.keys():
            raise ValueError("Gradient balancer loss identities do not match")
        updates = state.get("updates")
        if type(updates) is not int or updates < 0:
            raise ValueError("Invalid gradient balancer update count")
        ema = {}
        for name, values in state["ema"].items():
            if set(values) != {"total", "weight"}:
                raise ValueError("Invalid EMA fields")
            total, weight = float(values["total"]), float(values["weight"])
            if (not math.isfinite(total) or not math.isfinite(weight) or total < 0 or weight < 0
                    or (weight == 0 and total != 0) or (weight > 0 and total <= 0)):
                raise ValueError("Invalid EMA values")
            ema[name] = {"total": total, "weight": weight}
        self.weights, self._ema, self.updates = weights, ema, updates
