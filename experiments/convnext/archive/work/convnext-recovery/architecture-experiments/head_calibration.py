"""Pure CPU linear algebra for disposable frozen-feature readout experiments.

The four fitted phases are calibration constraints, not held-out quality proof.
No module, optimizer, checkpoint, streaming history or input tensor is mutated.
All solves use CPU FP64. The result is one ordinary readout weight; there is no
new inference operation. The constraint remains valid only for frozen features.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor


SHRINKAGE_GRID = (0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class HeadCalibrationResult:
    weight: Tensor
    weight_fp64: Tensor
    delta_fp64: Tensor
    report: dict[str, Any]


def _matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _inputs(weight: Tensor, features: Tensor, target: Tensor,
            max_condition: float) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    if weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("Source weight must be FP32 or FP64")
    if not math.isfinite(max_condition) or max_condition < 1:
        raise ValueError("max_condition must be finite and at least one")
    w, h, t = (_matrix(weight, "weight"), _matrix(features, "features"),
               _matrix(target, "target"))
    if h.shape[1] != 4 or h.shape[0] < 4:
        raise ValueError("features must have shape [features >= 4, four phases]")
    if w.shape[0] == 0 or w.shape[1] != h.shape[0] or t.shape != (w.shape[0], 4):
        raise ValueError("Incompatible weight, features and target shapes")
    u, singular, vh = torch.linalg.svd(h, full_matrices=False)
    threshold = torch.finfo(torch.float64).eps * max(h.shape) * float(singular[0])
    rank = int((singular > threshold).sum())
    if rank != 4:
        raise ValueError(f"Stationary features must have rank four; observed {rank}")
    condition = float(singular[0] / singular[-1])
    if condition > max_condition:
        raise ValueError(f"Stationary features are ill-conditioned: {condition}")
    inverse = (vh.T * singular.reciprocal()[None, :]) @ u.T
    return w, h, t, inverse, {
        "feature_shape": list(h.shape), "feature_rank": rank,
        "feature_singular_values": singular.tolist(),
        "feature_rank_threshold": threshold, "feature_condition": condition,
        "maximum_allowed_feature_condition": max_condition,
    }


def _desired(w: Tensor, h: Tensor, t: Tensor,
             mode: Literal["shared", "full"]) -> tuple[Tensor, Tensor]:
    before = w @ h
    error = t - before
    if mode == "full":
        desired = t
    elif mode == "shared":
        error = error.mean(dim=1, keepdim=True).expand_as(error)
        desired = before + error
    else:
        raise ValueError("mode must be 'shared' or 'full'")
    return desired, error


def _finish(w: Tensor, h: Tensor, t: Tensor, desired: Tensor, delta: Tensor,
            source_dtype: torch.dtype, report: dict[str, Any]) -> HeadCalibrationResult:
    fitted = w + delta
    installed = fitted.to(dtype=source_dtype)
    if not bool(torch.isfinite(fitted).all() and torch.isfinite(installed).all()):
        raise ValueError("The fitted or installed weight is non-finite")
    residual64 = fitted @ h - desired
    # This isolates coefficient rounding. Ordinary FP32 execution can have
    # additional input rounding and summation error; do not call it runtime
    # output parity or a measured GPU/CPU numerical tolerance.
    installed_residual = installed.double() @ h - desired
    teacher_residual = fitted @ h - t
    eps = torch.finfo(source_dtype).eps
    dot_terms = h.shape[0]
    gamma = ((2 * dot_terms + 1) * eps /
             (1 - (2 * dot_terms + 1) * eps))
    summation_bound = gamma * (installed.double().abs() @ h.abs())
    report.update({
        "source_dtype": str(source_dtype), "solve_dtype": "torch.float64",
        "solve_device": "cpu", "returned_weight_device": "cpu",
        "weight_shape": list(w.shape),
        "weight_norm_before": float(w.norm()), "delta_weight_norm": float(delta.norm()),
        "delta_weight_norm_fraction": float(delta.norm() / w.norm()) if float(w.norm()) else None,
        "constraint_residual_fp64_max_abs": float(residual64.abs().max()),
        "constraint_residual_fp64_rms": float(residual64.square().mean().sqrt()),
        "installed_coefficient_rounding_constraint_max_abs": float(installed_residual.abs().max()),
        "installed_coefficient_rounding_constraint_rms": float(installed_residual.square().mean().sqrt()),
        "calibration_teacher_residual_fp64_rms": float(teacher_residual.square().mean().sqrt()),
        "conservative_installed_dot_rounding_bound_max": float(summation_bound.max()),
        "rounding_scope": "Coefficient rounding is evaluated in FP64. Dot-product bound excludes feature rounding and is not an observed execution error.",
        "inference_added_parameters": 0, "inference_added_macs": 0,
        "scope": "Four fitted stationary phases with frozen upstream features. Calibration constraints do not establish held-out quality or CPU RTF.",
    })
    # Reject a numerically failed solve; report the much smaller actual result.
    scale = max(float(desired.abs().max()), float((fitted.abs() @ h.abs()).max()),
                torch.finfo(torch.float64).tiny)
    tolerance = 4096 * torch.finfo(torch.float64).eps * max(h.shape) * scale
    report["constraint_residual_fp64_guard_tolerance"] = tolerance
    if float(residual64.abs().max()) > tolerance:
        raise ValueError("Fitted FP64 weight does not satisfy its declared constraint")
    return HeadCalibrationResult(installed, fitted, delta, report)


def minimum_norm_anchor(weight: Tensor, features: Tensor, target: Tensor, *,
                        mode: Literal["shared", "full"] = "full",
                        max_condition: float = 1e8) -> HeadCalibrationResult:
    """Minimum Frobenius-norm readout edit for one four-phase constraint.

    In shared mode each phase receives the same waveform correction, preserving
    the current differences between its four phase outputs. Full mode matches
    all four teacher blocks. Neither is an unconditional waveform subtraction.
    """
    w, h, t, inverse, report = _inputs(weight, features, target, max_condition)
    desired, error = _desired(w, h, t, mode)
    delta = error @ inverse
    report.update({"method": "minimum_frobenius_anchor", "constraint_mode": mode})
    return _finish(w, h, t, desired, delta, weight.dtype, report)


def covariance_ridge_fit(weight: Tensor, features: Tensor, target: Tensor,
                         covariance: Tensor, residual_cross_covariance: Tensor, *,
                         shrinkage: float,
                         mode: Literal["shared", "full"] = "full",
                         anchor_only: bool = False,
                         max_condition: float = 1e8) -> HeadCalibrationResult:
    """Constrained ridge residual fit with training-only sufficient statistics.

    A = mean(h h^T), B = mean(h (teacher - W h)^T), C = A + lambda I,
    lambda = shrinkage * trace(A) / D. The fitted delta minimizes
    0.5 tr(delta C delta^T) - tr(delta B), subject to the selected anchor.
    anchor_only=True sets the linear term to zero and minimizes calibration
    output displacement plus ridge rather than fitting teacher residuals.

    Callers must form A and B on the same complete-frame training set and fixed
    W. This helper cannot establish that provenance or held-out separation.
    """
    if type(anchor_only) is not bool:
        raise ValueError("anchor_only must be boolean")
    if not math.isfinite(shrinkage) or shrinkage not in SHRINKAGE_GRID:
        raise ValueError(f"shrinkage must be in the predeclared grid {SHRINKAGE_GRID}")
    w, h, t, _, report = _inputs(weight, features, target, max_condition)
    desired, _ = _desired(w, h, t, mode)
    a = _matrix(covariance, "covariance")
    b = _matrix(residual_cross_covariance, "residual_cross_covariance")
    d, m = w.shape[1], w.shape[0]
    if a.shape != (d, d) or b.shape != (d, m):
        raise ValueError("Sufficient statistics have incompatible shapes")
    asymmetry = float((a - a.T).abs().max())
    symmetry_tolerance = 1e-10 * max(float(a.abs().max()), torch.finfo(a.dtype).tiny)
    if asymmetry > symmetry_tolerance:
        raise ValueError("Covariance is not symmetric")
    a = (a + a.T) * 0.5
    scale = float(a.trace() / d)
    if scale <= 0 or bool((a.diag() < -symmetry_tolerance).any()):
        raise ValueError("Covariance must have positive mean diagonal and nonnegative variances")
    ridge = shrinkage * scale
    c = a + ridge * torch.eye(d, dtype=torch.float64)
    chol, info = torch.linalg.cholesky_ex(c)
    if int(info) != 0:
        raise ValueError("Regularized covariance is not positive definite")
    effective_b = torch.zeros_like(b) if anchor_only else b
    delta0 = torch.cholesky_solve(effective_b, chol).T
    inverse_c_h = torch.cholesky_solve(h, chol)
    gram = h.T @ inverse_c_h
    gram = (gram + gram.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) <= 0:
        raise ValueError("Constraint covariance Gram matrix is not positive definite")
    gram_condition = float(eigenvalues[-1] / eigenvalues[0])
    if not math.isfinite(gram_condition) or gram_condition > 1e12:
        raise ValueError("Constraint covariance Gram matrix is ill-conditioned")
    remaining = desired - (w + delta0) @ h
    multiplier = torch.linalg.solve(gram, remaining.T).T
    delta = delta0 + multiplier @ inverse_c_h.T
    gradient = delta @ c - effective_b.T
    # For feasible directions V with V H = 0, first-order stationarity requires
    # gradient V^T = 0. Project using the already verified SVD rather than a
    # dense D-by-D nullspace projector.
    q = torch.linalg.qr(h, mode="reduced")[0]
    feasible_gradient = gradient - (gradient @ q) @ q.T
    report.update({
        "method": "covariance_anchor" if anchor_only else "constrained_ridge_residual_fit",
        "constraint_mode": mode, "anchor_only": anchor_only,
        "shrinkage": shrinkage, "ridge_lambda": ridge,
        "covariance_mean_diagonal": scale,
        "covariance_asymmetry_max_abs": asymmetry,
        "regularized_covariance_cholesky_passed": True,
        "constraint_gram_eigenvalues": eigenvalues.tolist(),
        "constraint_gram_condition": gram_condition,
        "feasible_gradient_norm": float(feasible_gradient.norm()),
        "gradient_norm": float(gradient.norm()),
        "feasible_gradient_norm_fraction": float(feasible_gradient.norm() / gradient.norm()) if float(gradient.norm()) else 0.0,
        "ridge_objective_at_delta": float(0.5 * (delta @ c * delta).sum() - (delta * effective_b.T).sum()),
        "unconstrained_delta_norm": float(delta0.norm()),
        "statistics_contract": "Caller supplies training-only complete-frame A=mean(hhT) and B=mean(h residualT), at the unchanged W.",
    })
    return _finish(w, h, t, desired, delta, weight.dtype, report)
