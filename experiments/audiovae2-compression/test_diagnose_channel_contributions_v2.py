"""The same contribution contracts plus the approved input-alignment correction."""
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "convnext"))

import diagnose_channel_contributions_v2 as v2
import test_diagnose_channel_contributions as contracts


@pytest.fixture(autouse=True)
def bind_frozen_contracts_to_v2(monkeypatch):
    # Scope the substitution to this test item so joint collection with v1
    # continues to validate each module independently.
    monkeypatch.setattr(contracts, "audit", v2)


deterministic_cpu = contracts.deterministic_cpu
for name, function in list(vars(contracts).items()):
    if name.startswith("test_") and callable(function):
        globals()[name] = function


def test_alignment_accepts_measured_fp32_roundoff_at_large_feature_amplitude():
    teacher = torch.tensor([[[100.], [2.753337]]], dtype=torch.float32)
    student = torch.tensor([[[2.7533245]]], dtype=torch.float32)
    saved = teacher.clone()
    result = v2.first_input_alignment(teacher, student, [1])
    assert result["passed"] is True
    assert float((teacher[:, 1:] - student).abs().max()) > 1e-5
    assert torch.equal(teacher, saved)


@pytest.mark.parametrize("teacher_value,student_value", [(0., 2e-5), (2.753337, 2.75), (0., float("nan"))])
def test_alignment_still_rejects_nearzero_or_substantive_coordinate_mismatch(teacher_value, student_value):
    teacher = torch.tensor([[[teacher_value]]], dtype=torch.float32)
    student = torch.tensor([[[student_value]]], dtype=torch.float32)
    assert v2.first_input_alignment(teacher, student, [0])["passed"] is False
