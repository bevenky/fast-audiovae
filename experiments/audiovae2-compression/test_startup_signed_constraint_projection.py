"""Signed correction bounds require inward movement and full-system feasibility."""
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_signed_constraint_projection as signed


def vectors(matrix):
    return [[torch.tensor(row, dtype=torch.float64)] for row in matrix]


def test_negative_bound_produces_inward_correction_and_preserves_inputs():
    rows, proposal = vectors([[1., 0.], [0., 1.]]), [torch.tensor([0., .25], dtype=torch.float64)]
    saved = proposal[0].clone()
    correction, report = signed.project_displacement(rows, proposal, [-2., .5])
    torch.testing.assert_close(correction[0], torch.tensor([-2., .25], dtype=torch.float64), rtol=0, atol=1e-14)
    assert torch.equal(proposal[0], saved)
    assert report['kkt_passed'] and report['full_primal_verified']
    assert report['rows'] == 2
    # Scaling a physical inequality must also scale its negative right-hand side.
    rescaled, _ = signed.project_displacement(vectors([[1e-5, 0.], [0., 9.]]), proposal, [-2e-5, 4.5])
    torch.testing.assert_close(rescaled[0], correction[0], rtol=0, atol=1e-13)


def test_inward_correction_must_not_harm_another_currently_satisfied_constraint():
    proposal = [torch.zeros(2, dtype=torch.float64)]
    rows = vectors([[1., 0.], [-1., 1.]])
    first_only, _ = signed.project_displacement(rows[:1], proposal, [-1.])
    assert torch.dot(rows[1][0], first_only[0]) > 0
    full, report = signed.project_displacement(rows, proposal, [-1., 0.])
    torch.testing.assert_close(full[0], torch.tensor([-1., -1.], dtype=torch.float64), rtol=0, atol=1e-13)
    assert torch.dot(rows[0][0], full[0]) <= -1 + 1e-13
    assert torch.dot(rows[1][0], full[0]) <= 1e-13
    assert report['full_primal_verified'] and report['kkt_passed']


@pytest.mark.parametrize('matrix,bounds', [([[0., 0.]], [-1e-12]), ([[1., 0.], [-1., 0.]], [-1., -1.])])
def test_infeasible_signed_system_fails_without_returning_an_unchecked_fallback(matrix, bounds):
    proposal = [torch.zeros(2, dtype=torch.float64)]
    with pytest.raises(RuntimeError):
        signed.project_displacement(vectors(matrix), proposal, bounds)
    assert torch.equal(proposal[0], torch.zeros_like(proposal[0]))
