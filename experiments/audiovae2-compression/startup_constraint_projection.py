"""Project one realized Adam displacement against up to twelve scalar rows.

The small dual problem is solved in CPU FP64 by bounded active-set enumeration.
Dependent active rows use a rank-revealing eigensystem, without diagonal jitter
or removal of inequalities. Every accepted solution is checked against the full
system. Numerical linear certificates do not replace true waveform feasibility.
"""
from __future__ import annotations

import itertools
import math

import torch


VERSION = 'audiovae2_complete_startup_constraint_projection_v1'
MAX_CONSTRAINTS = 12
QP_EPS_FACTOR = 256


def _spectrum(matrix):
    values = torch.linalg.eigvalsh(matrix)
    largest = float(values.abs().max()) if values.numel() else 0.
    cutoff = QP_EPS_FACTOR * torch.finfo(torch.float64).eps * max(1., largest)
    if values.numel() and float(values.min()) < -cutoff:
        raise ValueError('Constraint Gram is not numerically positive semidefinite')
    positive = values[values > cutoff]
    return {
        'rank': int(positive.numel()),
        'rank_tolerance': cutoff,
        'eigen_min': float(values.min()) if values.numel() else None,
        'eigen_max': float(values.max()) if values.numel() else None,
        'condition_on_resolved_subspace': float(positive.max()/positive.min()) if positive.numel() else None,
    }


def solve_small_qp(gram, violation):
    """Solve min .5*lambda.T*G*lambda - lambda.T*v, lambda >= 0.

    G is the Gram of normalized inequality gradients and v = A*d_Adam - b.
    The dual right-hand side can have either sign. At most 2**12 active sets
    are examined. Rank-deficient subsets must satisfy their original equations;
    a pseudoinverse is never accepted merely because its residual is small in a
    truncated subspace. All omitted inequalities also participate in KKT checks.
    """
    if (not isinstance(gram, torch.Tensor) or not isinstance(violation, torch.Tensor)
            or gram.device.type != 'cpu' or violation.device.type != 'cpu'
            or gram.dtype != torch.float64 or violation.dtype != torch.float64
            or violation.ndim != 1 or gram.ndim != 2
            or gram.shape != (violation.numel(),)*2
            or violation.numel() > MAX_CONSTRAINTS
            or not torch.isfinite(gram).all() or not torch.isfinite(violation).all()):
        raise ValueError('QP requires finite CPU FP64 Gram/vector with at most twelve rows')
    n = violation.numel()
    if n == 0:
        return violation.detach().clone(), {
            'active_set': [], 'rank': 0, 'sets_checked': 1,
            'verified_candidates': 1, 'kkt_passed': True, 'tolerance': 0.,
            'primal_violation_max': 0., 'complementarity_max': 0., 'dual_min': 0.,
            'active_equation_residual_max': 0., 'dual_objective': 0.,
            'gram_rank': 0, 'gram_rank_tolerance': 0.,
            'gram_eigen_min': None, 'gram_eigen_max': None,
            'gram_condition_on_resolved_subspace': None,
            'active_condition_on_resolved_subspace': None,
        }

    gram, violation = gram.detach(), violation.detach()
    eps = torch.finfo(torch.float64).eps
    scale = max(1., float(gram.abs().max()), float(violation.abs().max()))
    tolerance = QP_EPS_FACTOR * eps * scale
    if float((gram-gram.T).abs().max()) > tolerance:
        raise ValueError('Constraint Gram is not symmetric')
    g = (gram+gram.T)/2
    spectrum = _spectrum(g)
    best = None
    checked = verified = 0

    for count in range(n+1):
        for active in itertools.combinations(range(n), count):
            checked += 1
            multiplier = torch.zeros(n, dtype=torch.float64)
            rank = 0
            active_condition = None
            equation_residual = 0.
            if active:
                indices = list(active)
                block = g[indices][:, indices]
                rhs = violation[indices]
                values, vectors = torch.linalg.eigh(block)
                cutoff = QP_EPS_FACTOR * eps * max(1., float(values.abs().max()))
                if float(values.min()) < -cutoff:
                    raise RuntimeError('Active Gram is not numerically positive semidefinite')
                positive = values > cutoff
                rank = int(positive.sum())
                resolved = values[positive]
                if rank:
                    active_condition = float(resolved.max()/resolved.min())
                solution = vectors[:, positive] @ ((vectors[:, positive].T @ rhs)/resolved)
                if not torch.isfinite(solution).all():
                    continue
                equation_residual = float((block @ solution-rhs).abs().max())
                if equation_residual > tolerance or float(solution.min()) < -tolerance:
                    continue
                multiplier[indices] = solution.clamp_min(0)
                # Clipping numerical negative multipliers must not invalidate the
                # original active equations or any full-system inequality.
                equation_residual = float((block @ multiplier[indices]-rhs).abs().max())
                if equation_residual > tolerance:
                    continue

            slack = violation-g @ multiplier
            complement = multiplier*slack
            if not torch.isfinite(slack).all() or not torch.isfinite(complement).all():
                continue
            if (float(slack.max()) > tolerance
                    or float(complement.abs().max()) > tolerance*max(1., float(multiplier.abs().max()))):
                continue
            objective = float(.5*multiplier @ (g @ multiplier)-multiplier @ violation)
            if not math.isfinite(objective):
                continue
            verified += 1
            if best is None or objective < best[0]:
                best = (objective, multiplier, active, rank, slack, complement,
                        equation_residual, active_condition)

    if best is None:
        raise RuntimeError('No fully verified twelve-row QP solution; no jitter or unconstrained fallback')
    objective, multiplier, active, rank, slack, complement, equation_residual, active_condition = best
    return multiplier, {
        'active_set': list(active), 'rank': rank, 'sets_checked': checked,
        'verified_candidates': verified, 'kkt_passed': True, 'tolerance': tolerance,
        'primal_violation_max': max(0., float(slack.max())),
        'complementarity_max': float(complement.abs().max()),
        'dual_min': float(multiplier.min()),
        'active_equation_residual_max': equation_residual,
        'dual_objective': objective,
        'gram_rank': spectrum['rank'], 'gram_rank_tolerance': spectrum['rank_tolerance'],
        'gram_eigen_min': spectrum['eigen_min'], 'gram_eigen_max': spectrum['eigen_max'],
        'gram_condition_on_resolved_subspace': spectrum['condition_on_resolved_subspace'],
        'active_condition_on_resolved_subspace': active_condition,
    }


def _dot(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('Parameter vector topology differs or is empty')
    result = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for a, b in zip(left, right):
        if a.shape != b.shape or a.device != b.device or a.device != left[0].device:
            raise ValueError('Parameter vector shapes/devices differ')
        result += (a.detach().double()*b.detach().double()).sum()
    value = float(result)
    if not math.isfinite(value):
        raise ValueError('Nonfinite parameter-vector dot product')
    return value


def _validate_vectors(rows, displacement, bounds):
    if (not displacement or len(rows) != len(bounds) or len(rows) > MAX_CONSTRAINTS
            or any(not math.isfinite(b) or b < 0 for b in bounds)):
        raise ValueError('Require at most twelve rows with finite nonnegative bounds')
    if any(not isinstance(t, torch.Tensor) or not t.is_floating_point()
           or not torch.isfinite(t).all() for t in displacement):
        raise ValueError('Displacement must contain finite real floating tensors')
    device = displacement[0].device
    if any(t.device != device for t in displacement):
        raise ValueError('Displacement spans multiple devices')
    for row in rows:
        if len(row) != len(displacement):
            raise ValueError('Constraint row has a different parameter topology')
        if any(not isinstance(a, torch.Tensor) or not a.is_floating_point()
               or a.shape != d.shape or a.device != device or not torch.isfinite(a).all()
               for a, d in zip(row, displacement)):
            raise ValueError('Constraint row must have matching finite real shapes/devices')


def project_displacement(rows, displacement, bounds):
    """Minimize ||d-d_Adam||^2/2 subject to every row[i].d <= bounds[i].

    Input parameter tensors are read only. Bounds must be nonnegative so that
    the original zero displacement is feasible. Return detached FP64 tensors,
    preserving the input displacement exactly when no correction is needed.
    A caller must still check the actual FP32-written nonlinear waveform.
    """
    _validate_vectors(rows, displacement, bounds)
    norms = [math.sqrt(_dot(row, row)) for row in rows]
    nonzero = [i for i, norm in enumerate(norms) if norm > 0]
    gram = torch.tensor([
        [_dot(rows[i], rows[j])/(norms[i]*norms[j]) for j in nonzero]
        for i in nonzero
    ], dtype=torch.float64).reshape(len(nonzero), len(nonzero))
    gram = (gram+gram.T)/2
    violation = torch.tensor([
        (_dot(rows[i], displacement)-bounds[i])/norms[i] for i in nonzero
    ], dtype=torch.float64)
    multiplier, report = solve_small_qp(gram, violation)
    corrected = bool((multiplier != 0).any())
    projected = [d.detach().double().clone() for d in displacement]
    if corrected:
        for coefficient, index in zip(multiplier.tolist(), nonzero):
            if coefficient:
                for d, gradient in zip(projected, rows[index]):
                    d.add_(gradient.detach().double(), alpha=-coefficient/norms[index])
    if any(not torch.isfinite(d).all() for d in projected):
        raise RuntimeError('Projected displacement is nonfinite')

    # Independently re-evaluate every original row after constructing the large
    # vector. A Gram-space certificate alone can hide reconstruction roundoff.
    physical_positive = normalized_positive = 0.
    for row, bound, norm in zip(rows, bounds, norms):
        residual = _dot(row, projected)-bound
        physical_positive = max(physical_positive, residual)
        if norm:
            normalized = residual/norm
            normalized_positive = max(normalized_positive, normalized)
            if normalized > report['tolerance']*4:
                raise RuntimeError('Reconstructed displacement fails a complete normalized primal check')
        elif residual > 0:
            raise RuntimeError('A zero-gradient constraint is infeasible')

    correction = [p-d.detach().double() for p, d in zip(projected, displacement)]
    pn = math.sqrt(_dot(projected, projected))
    dn = math.sqrt(_dot(displacement, displacement))
    report.update({
        'rows': len(rows), 'nonzero_rows': len(nonzero), 'zero_rows': len(rows)-len(nonzero),
        'linear_conflicts': int((violation > 0).sum()), 'corrected': corrected,
        'correction_norm': math.sqrt(_dot(correction, correction)), 'proposal_norm': dn,
        'proposal_projected_cosine': _dot(projected, displacement)/(pn*dn) if pn*dn else None,
        'active_constraint_indices': [nonzero[i] for i in report['active_set']],
        'row_norm_min': min(norms) if norms else None,
        'row_norm_max': max(norms) if norms else None,
        'full_primal_verified': True,
        'reconstructed_normalized_primal_violation_max': normalized_positive,
        'reconstructed_physical_primal_violation_max': physical_positive,
        'maximum_constraints': MAX_CONSTRAINTS,
        'rank_policy': 'FP64 eigensystem at256*eps*max(1,spectral_radius); original equations and every inequality verified; no jitter',
        'nonlinear_feasibility_checked': False,
    })
    return projected, report
