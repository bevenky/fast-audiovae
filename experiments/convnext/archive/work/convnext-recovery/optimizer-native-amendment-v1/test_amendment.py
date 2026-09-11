from copy import deepcopy
import json
from pathlib import Path

import pytest

from run_amendment import PRIOR_REPORT_SHA, validate_prior


@pytest.fixture
def original_report():
    path = Path(__file__).resolve().parents[3] / "outputs/convnext-recovery/optimizer-silence-causal/optimizer-comparison.json"
    return json.loads(path.read_text())


def test_amendment_uses_only_largest_prequalified_native_radius(original_report):
    result = validate_prior(original_report, PRIOR_REPORT_SHA)
    assert result["fraction"] == 1 / 16
    assert result["radius"] == .0026359612391104728
    assert [b["id"] for b in result["comparison_batches"]] == ["comparison_" + str(i) for i in range(4)]
    assert sum(len(b["indices"]) for b in result["comparison_batches"]) == 128
    assert result["calibration_batches_repeated"] == result["calibration_vjps_repeated"] == 0


@pytest.mark.parametrize("field,value", [("comparisons", [{}]), ("resolved", True), ("state_restored", False), ("retained_updates", 1)])
def test_rejects_post_comparison_or_changed_retained_state(original_report, field, value):
    bad = deepcopy(original_report); bad[field] = value
    with pytest.raises(ValueError, match="unresolved"):
        validate_prior(bad, PRIOR_REPORT_SHA)


def test_rejects_changed_report_bytes_and_hidden_radius_change(original_report):
    with pytest.raises(ValueError, match="bytes"):
        validate_prior(original_report, "wrong")
    bad = deepcopy(original_report)
    bad["calibration"]["initial_radius"] *= 2
    with pytest.raises(ValueError, match="1/16"):
        validate_prior(bad, PRIOR_REPORT_SHA)


def test_rejects_incomplete_three_batch_evidence(original_report):
    bad = deepcopy(original_report)
    bad["calibration"]["attempts"][0]["cases"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_prior(bad, PRIOR_REPORT_SHA)
