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
class MelGradientCapConfig:
    """Opt-in cap on the current batch's mel pre-sum norm contribution.

    Removed mel norm is assigned to teacher_waveform. Other losses and the
    total pre-sum norm are held fixed. If a zero waveform gradient or a scale
    bound prevents that transfer, the original combination is retained and the
    exception is reported. This caps neither parameter norms nor individual
    examples' shares: norms are averaged per example before taking the share.
    """

    max_share: float = 0.25
    format_version: int = 1

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("Unsupported mel gradient cap config version")
        if not math.isfinite(self.max_share) or not 0 < self.max_share < 1:
            raise ValueError("Mel gradient cap share must be finite in (0, 1)")


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
                 weights: Mapping[str, float] | None = None, *,
                 mel_cap: MelGradientCapConfig | None = None) -> None:
        self.config = config
        self.weights = _weights(weights if weights is not None else {
            "teacher_waveform": 0.5, "teacher_mel": 0.5,
            "feature_matching": 0.0, "adversarial": 0.0})
        if mel_cap is not None and not isinstance(mel_cap, MelGradientCapConfig):
            raise TypeError("mel_cap must be a MelGradientCapConfig or None")
        if mel_cap is not None and not {"teacher_waveform", "teacher_mel"} <= self.weights.keys():
            raise ValueError("Mel gradient cap requires teacher_waveform and teacher_mel losses")
        self.mel_cap = mel_cap
        self._ema = {name: {"total": 0.0, "weight": 0.0} for name in self.weights}
        self.updates = 0

    def fork_with_mel_cap(self, mel_cap: MelGradientCapConfig | None) -> "GradientBalancer":
        """Explicit experimental fork preserving the old EMA and update count.

        Ordinary load_state_dict remains strict about the policy. Callers must
        separately record the source checkpoint and fork policy in their run
        identity; this method does not authorize reusing a continuation ledger.
        """
        fork = GradientBalancer(self.config, self.weights, mel_cap=mel_cap)
        fork._ema = {name: dict(values) for name, values in self._ema.items()}
        fork.updates = self.updates
        return fork

    def _cap_mel(self, norms: Mapping[str, float], scales: dict[str, float],
                 metrics: dict[str, float]) -> None:
        assert self.mel_cap is not None
        waveform, mel = "teacher_waveform", "teacher_mel"
        before = {name: norm * scales.get(name, 0.0) for name, norm in norms.items()}
        total = sum(before.values())
        wave_before, mel_before = before.get(waveform, 0.0), before.get(mel, 0.0)
        share_before = mel_before / total if total > 0 else 0.0
        requested = share_before > self.mel_cap.max_share
        wave_zero = norms.get(waveform, 0.0) <= self.config.epsilon
        mel_zero = norms.get(mel, 0.0) <= self.config.epsilon
        wave_bound = mel_bound = False
        feasible = True
        if requested:
            mel_after = self.mel_cap.max_share * total
            wave_after = wave_before + (mel_before - mel_after)
            if not wave_zero:
                wave_scale = wave_after / norms[waveform]
                wave_bound = not self.config.min_scale <= wave_scale <= self.config.max_scale
            if not mel_zero:
                mel_scale = mel_after / norms[mel]
                mel_bound = not self.config.min_scale <= mel_scale <= self.config.max_scale
            feasible = not (wave_zero or mel_zero or wave_bound or mel_bound)
            if feasible:
                scales[waveform], scales[mel] = wave_scale, mel_scale

        # Existing scale/scaled_norm metrics describe the gradient actually
        # used, while these policy metrics also preserve its uncapped values.
        after = {name: norm * scales.get(name, 0.0) for name, norm in norms.items()}
        after_total = sum(after.values())
        after_share = after.get(mel, 0.0) / after_total if after_total > 0 else 0.0
        tolerance = 1e-12  # Scalar arithmetic only, not a relaxation of the policy.
        metrics.update({
            "mel_cap/enabled": 1.0,
            "mel_cap/max_share": self.mel_cap.max_share,
            "mel_cap/requested": float(requested),
            "mel_cap/applied": float(requested and feasible),
            "mel_cap/feasible": float(feasible),
            "mel_cap/satisfied": float(after_share <= self.mel_cap.max_share + tolerance),
            "mel_cap/share_defined": float(after_total > 0),
            "mel_cap/share_before": share_before,
            "mel_cap/share_after": after_share,
            "mel_cap/total_norm_before": total,
            "mel_cap/total_norm_after": after_total,
            "mel_cap/total_norm_delta": after_total - total,
            "mel_cap/total_norm_preserved": float(math.isclose(total, after_total, rel_tol=tolerance, abs_tol=tolerance)),
            "mel_cap/waveform_norm_before": wave_before,
            "mel_cap/waveform_norm_after": after.get(waveform, 0.0),
            "mel_cap/mel_norm_before": mel_before,
            "mel_cap/mel_norm_after": after.get(mel, 0.0),
            "mel_cap/zero_waveform": float(wave_zero),
            "mel_cap/zero_mel": float(mel_zero),
            "mel_cap/zero_total": float(total == 0),
            "mel_cap/waveform_bound_exception": float(wave_bound),
            "mel_cap/mel_bound_exception": float(mel_bound),
        })
        for name in norms:
            metrics[f"{name}/scale_before_cap"] = metrics[f"{name}/scale"]
            metrics[f"{name}/scaled_norm_before_cap"] = before[name]
            metrics[f"{name}/scale"] = scales.get(name, 0.0)
            metrics[f"{name}/scaled_norm"] = after[name]
            metrics[f"{name}/achieved_share"] = after[name] / after_total if after_total > 0 else 0.0

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
        scales: dict[str, float] = {}
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
            scales[name] = scale
            combined.add_(grad, alpha=scale)
            metrics.update({f"{name}/ema_norm": average, f"{name}/scale": scale,
                            f"{name}/scaled_norm": norm * scale,
                            f"{name}/saturated": float(scale != desired), f"{name}/zero_norm": 0.0})
        if self.mel_cap is not None:
            self._cap_mel(norms, scales, metrics)
            combined.zero_()
            for name, grad in gradients.items():
                if name in scales:
                    combined.add_(grad, alpha=scales[name])
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
        state = {"format_version": 1, "config": asdict(self.config), "weights": dict(self.weights),
                 "updates": self.updates, "ema": {k: dict(v) for k, v in self._ema.items()}}
        if self.mel_cap is not None:
            state.update(format_version=2, mel_cap=asdict(self.mel_cap))
        return state

    def load_state_dict(self, state: Mapping) -> None:
        expected_version = 1 if self.mel_cap is None else 2
        expected_cap = None if self.mel_cap is None else asdict(self.mel_cap)
        if (state.get("format_version") != expected_version
                or state.get("config") != asdict(self.config)
                or state.get("mel_cap") != expected_cap):
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
