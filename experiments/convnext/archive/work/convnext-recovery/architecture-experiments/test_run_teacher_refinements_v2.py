"""Run the complete refinement suite against the isolated v2 entry policy.

The original test module and v1 runner remain unchanged. Its existing tests
are collected here with a function-scoped module binding to the v2 runner.
Additional checks distinguish disposable entry tolerance from strict quality
acceptance and retained overshoot-count/source protections.
"""
from copy import deepcopy
import inspect

import pytest

import test_run_teacher_refinements as original_suite
import run_teacher_refinements as v1
import run_teacher_refinements_v2 as v2

# Reuse the exact original tests and fixtures, rather than copying a divergent
# second suite. The original module binding is restored after every test.
deterministic_cpu = original_suite.deterministic_cpu
objectives = original_suite.objectives
for _name, _value in vars(original_suite).items():
    if _name.startswith("test_") and callable(_value):
        globals()[_name] = _value


@pytest.fixture(autouse=True)
def bind_v2_runner(monkeypatch):
    monkeypatch.setattr(original_suite, "refine", v2)


def entry_report():
    report = original_suite.quality_report()
    report["rows"] = [{"source_id": "a", "start_frame": 0, "peak": 1.2, "overshoot_samples": 4},
                      {"source_id": "b", "start_frame": 0, "peak": .9, "overshoot_samples": 0}]
    return report


@pytest.mark.parametrize("relative_peak", [1., 1.005, 1.01, 1.011])
def test_v2_entry_peak_tolerance_is_one_percent_but_final_gate_stays_strict(relative_peak):
    baseline = entry_report()
    candidate = deepcopy(baseline)
    candidate["aggregate"]["peak"] = 1.2*relative_peak
    candidate["rows"][0]["peak"] = 1.2*relative_peak
    entry = v2.finite_update_checks(candidate, baseline)
    assert entry["passed"] == (relative_peak <= 1.01)
    peak_check = next(c for c in entry["checks"] if c["name"] == "peak/aggregate")
    assert peak_check["baseline"] == 1.2
    assert peak_check["candidate"] == candidate["aggregate"]["peak"]
    assert peak_check["limit"] == pytest.approx(1.2*1.01+2e-6)
    candidate["aggregate"]["quiet_mse"] *= .8**2
    final = v2.teacher_selection_checks(candidate, baseline)
    assert final["qualified"] == (relative_peak <= 1.)
    old_peak = original_suite.peak_report([1.2, .9], [4, 0])
    new_peak = original_suite.peak_report([1.2*relative_peak, .9], [4, 0])
    assert v2.refinement_peak_screen(old_peak, new_peak)["passed"] == (relative_peak <= 1.)


@pytest.mark.parametrize("problem", ["count", "new_source"])
def test_v2_entry_does_not_relax_overshoot_counts_or_affected_sources(problem):
    baseline = entry_report()
    candidate = deepcopy(baseline)
    candidate["aggregate"]["peak"] = 1.19
    candidate["rows"][0]["peak"] = 1.19
    if problem == "count":
        candidate["aggregate"]["overshoot_samples"] = 5
        candidate["rows"][0]["overshoot_samples"] = 5
    else:
        candidate["rows"][0]["overshoot_samples"] = 3
        candidate["rows"][1]["overshoot_samples"] = 1
        candidate["rows"][1]["peak"] = 1.001
    assert not v2.finite_update_checks(candidate, baseline)["passed"]


def test_v2_final_quality_functions_are_identical_to_v1():
    for name in ("teacher_selection_checks", "refinement_peak_screen"):
        assert inspect.getsource(getattr(v2, name)) == inspect.getsource(getattr(v1, name))
