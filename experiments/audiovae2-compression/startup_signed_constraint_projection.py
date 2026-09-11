"""Signed linear bounds for the isolated startup normal-correction probe.

Reuse the frozen complete-row dual solver and its numerical certificate. Unlike
the original update projection, zero need not be feasible here: a violated
nonlinear constraint at a trial point gives a negative bound for its correction.
No numerical certificate substitutes for the caller's exact nonlinear checks.
"""
from __future__ import annotations

import math

import torch

import startup_constraint_projection as complete


VERSION = 'audiovae2_signed_startup_constraint_projection_v1'
MAX_CONSTRAINTS = complete.MAX_CONSTRAINTS


class InfeasibleProjectionError(RuntimeError):
    """A zero linear row has a strictly negative bound."""


def project_displacement(rows, displacement, bounds):
    """Minimize .5*||d-displacement||^2 with all row[i].d <= bounds[i].

    Bounds may be signed. A normal correction uses zero ``displacement`` and
    bounds equal to minus the actual trial-point constraint excesses. Return
    detached FP64 parameter-shaped tensors and the original Q-compatible report.
    Inputs are never changed; failed or unresolved solves raise, never fall back
    to an unverified zero correction. Every original row is checked again after
    constructing the complete parameter vector.
    """
    if len(rows) != len(bounds):
        raise ValueError('Constraint rows and signed bounds differ in length')
    try:
        finite_bounds = all(math.isfinite(bound) for bound in bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Signed bounds must be finite real scalars') from exc
    if not finite_bounds:
        raise ValueError('Signed bounds must be finite real scalars')
    # Reuse the frozen vector/shape/device validation without imposing its
    # nonnegative-bound policy. The actual signed bounds are preserved below.
    complete._validate_vectors(rows, displacement, [0.] * len(bounds))
    norms = [math.sqrt(complete._dot(row, row)) for row in rows]
    for row, norm, bound in zip(rows, norms, bounds):
        if norm == 0:
            if any(bool(torch.count_nonzero(t)) for t in row):
                raise ValueError('A nonzero constraint row norm underflowed in FP64')
            if bound < 0:
                raise InfeasibleProjectionError('Zero-gradient row with negative bound is linearly infeasible')
    nonzero = [i for i, norm in enumerate(norms) if norm > 0]
    gram = torch.tensor([
        [complete._dot(rows[i], rows[j]) / (norms[i] * norms[j]) for j in nonzero]
        for i in nonzero
    ], dtype=torch.float64).reshape(len(nonzero), len(nonzero))
    gram = (gram + gram.T) / 2
    violation = torch.tensor([
        (complete._dot(rows[i], displacement) - bounds[i]) / norms[i] for i in nonzero
    ], dtype=torch.float64)
    # This existing dual already accepts signed violations. Its full KKT and
    # rank checks remain unchanged, including failure on no verified candidate.
    multiplier, report = complete.solve_small_qp(gram, violation)
    corrected = bool((multiplier != 0).any())
    projected = [d.detach().double().clone() for d in displacement]
    if corrected:
        for coefficient, index in zip(multiplier.tolist(), nonzero):
            if coefficient:
                for d, gradient in zip(projected, rows[index]):
                    d.add_(gradient.detach().double(), alpha=-coefficient / norms[index])
    if any(not torch.isfinite(d).all() for d in projected):
        raise RuntimeError('Signed projected displacement is nonfinite')

    physical_positive = normalized_positive = 0.
    for row, bound, norm in zip(rows, bounds, norms):
        residual = complete._dot(row, projected) - bound
        physical_positive = max(physical_positive, residual)
        if norm:
            normalized = residual / norm
            normalized_positive = max(normalized_positive, normalized)
            if normalized > report['tolerance'] * 4:
                raise RuntimeError('Reconstructed signed displacement fails a full normalized primal check')
        elif residual > 0:
            raise InfeasibleProjectionError('A zero-gradient signed constraint is infeasible')

    correction = [p - d.detach().double() for p, d in zip(projected, displacement)]
    pn = math.sqrt(complete._dot(projected, projected))
    dn = math.sqrt(complete._dot(displacement, displacement))
    report.update({
        'version': VERSION, 'solver_version': complete.VERSION,
        'rows': len(rows), 'nonzero_rows': len(nonzero), 'zero_rows': len(rows) - len(nonzero),
        'linear_conflicts': int((violation > 0).sum()), 'corrected': corrected,
        'correction_norm': math.sqrt(complete._dot(correction, correction)), 'proposal_norm': dn,
        'proposal_projected_cosine': complete._dot(projected, displacement) / (pn * dn) if pn * dn else None,
        'active_constraint_indices': [nonzero[i] for i in report['active_set']],
        'row_norm_min': min(norms) if norms else None,
        'row_norm_max': max(norms) if norms else None,
        'negative_bounds': sum(bound < 0 for bound in bounds),
        'zero_displacement_feasible': all(bound >= 0 for bound in bounds),
        'full_primal_verified': True,
        'reconstructed_normalized_primal_violation_max': normalized_positive,
        'reconstructed_physical_primal_violation_max': physical_positive,
        'maximum_constraints': MAX_CONSTRAINTS,
        'rank_policy': 'Frozen FP64 eigensystem at256*eps*max(1,spectral_radius); original equations and every inequality verified; no jitter',
        'nonlinear_feasibility_checked': False,
    })
    return projected, report
