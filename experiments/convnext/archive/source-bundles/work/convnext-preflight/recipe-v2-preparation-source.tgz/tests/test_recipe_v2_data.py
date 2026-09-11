from dataclasses import replace
import json

import pytest

from audiovae_student.comparison_data import load_comparison_plan
from audiovae_student.recipe_v2_data import plan_recipe_v2_data, write_recipe_v2_plan
from audiovae_student.sampling import SamplerExhausted
from test_restart_data import row, counts, WINDOW


def pool():
    rows = [row(f"en-{i}", language="en", samples=WINDOW * 4 + 3040) for i in range(30)]
    rows += [row(f"{language}-{i}", language=language, samples=WINDOW * 3)
             for language in ("hi", "ar") for i in range(4)]
    rows += [row("cry", language="und", samples=WINDOW * 7),
             row("giggle", language="und", samples=WINDOW * 4)]
    return rows


def plan(rows=None, **kwargs):
    rows = pool() if rows is None else rows
    return plan_recipe_v2_data(rows, counts(rows), optimization_windows=160, calibration_windows=8,
        event_labels={"cry": ["Crying_and_sobbing"], "giggle": ["Giggle"]},
        required_indic=("hi",), required_other=("ar",), **kwargs)


def test_fresh_finite_plan_separates_calibration_intervals_without_retiring_sources():
    result = plan()
    optimization, calibration = result["optimization"], result["calibration"]
    assert len(optimization["windows"]) == 160
    assert len(calibration["windows"]) == 8
    all_windows = optimization["windows"] + calibration["windows"]
    assert len({(w.source_id, w.start_frame) for w in all_windows}) == 168
    assert result["report"]["shared_source_files"] > 0
    assert not optimization["ledger"].entries
    assert optimization["metadata"]["gradient_passes"] == 1
    assert calibration["metadata"]["gradient_passes"] == 0
    assert calibration["metadata"]["statistics_only_passes"] == 2
    assert all(w.valid_output_samples48k >= 9120 for w in all_windows)
    assert any(w.valid_output_samples48k == 9120 for w in all_windows)


def test_capacity_capping_is_fixed_and_scarce_events_span_the_run():
    result = plan()
    metadata = result["optimization"]["metadata"]
    assert metadata["by_subgroup_windows"]["indic:hi"] < .30 * 160
    assert metadata["by_subgroup_windows"]["indic:hi"] == result["report"]["capacity"]["indic:hi"]["available_windows"] - 1
    windows = result["optimization"]["windows"]
    for condition in ("Crying_and_sobbing", "Giggle"):
        indices = [i for i, w in enumerate(windows) if w.condition == condition]
        assert len(indices) >= 3
        assert indices[0] > 0
        assert indices[-1] > len(windows) * .75
    assert {w.condition for w in result["calibration"]["windows"]} >= {"Crying_and_sobbing", "Giggle"}


def test_known_heldout_people_and_fitted_source_exclusions_remain_reserved():
    rows = pool()
    dev = replace(rows[0], source_id="dev-copy", split="dev")
    excluded = rows[1]
    result = plan(rows, reserved_rows=[dev], excluded_sources=[excluded])
    ids = {w.source_id for key in ("optimization", "calibration") for w in result[key]["windows"]}
    assert rows[0].source_id not in ids and rows[1].source_id not in ids
    assert result["report"]["heldout_reservations"] == result["report"]["source_exclusions"] == 1


def test_identity_roundtrip_is_deterministic_and_tampering_rejected(tmp_path):
    result = plan()
    repeated = plan(list(reversed(pool())))
    for key in ("optimization", "calibration"):
        assert result[key]["windows"] == repeated[key]["windows"]
    root = tmp_path / "fresh"
    ready = write_recipe_v2_plan(result, root, provenance={"fresh_student": True})
    for key in ("optimization", "calibration"):
        loaded = load_comparison_plan(root / key)
        assert loaded["windows"] == result[key]["windows"]
        assert loaded["identity"]["identity_sha256"] == ready[f"{key}_plan_identity"]
    path = root / "calibration/input-sample-counts.json"
    counts_now = json.loads(path.read_text())
    counts_now[next(iter(counts_now))] += 1
    path.write_text(json.dumps(counts_now))
    with pytest.raises(ValueError, match="metadata changed"):
        load_comparison_plan(root / "calibration")


def test_capacity_failure_does_not_repeat_or_silently_drop_required_languages():
    rows = pool()
    with pytest.raises(SamplerExhausted):
        plan_recipe_v2_data(rows, counts(rows), optimization_windows=1000, calibration_windows=8,
                            required_indic=(), required_other=())
    with pytest.raises(ValueError, match="Missing required"):
        plan_recipe_v2_data(rows, counts(rows), optimization_windows=100, calibration_windows=8,
                            required_indic=("ta",), required_other=())


def test_partial_tail_below_gan_minimum_is_disclosed_not_scored():
    rows = pool() + [row("short-tail", language="hi", samples=WINDOW + 3039)]
    result = plan(rows)
    assert result["report"]["capacity"]["indic:hi"]["discarded_short_tail_input_samples"] == 3039
    assert not any(w.source_id == "short-tail" and w.start_frame == 64
                   for key in ("optimization", "calibration") for w in result[key]["windows"])
