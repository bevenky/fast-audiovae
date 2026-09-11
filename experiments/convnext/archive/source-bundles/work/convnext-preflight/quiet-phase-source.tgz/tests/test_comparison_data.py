"""Metadata-only checks for the matched data plan and per-student exclusions."""
from dataclasses import replace
import json

import pytest

from audiovae_student.comparison_data import (
    assert_comparison_disjoint, parent_student_ledger, plan_comparison,
    load_comparison_plan, write_comparison_plan, verify_comparison_windows)
from audiovae_student.restart_data import FixedWindowSampler
from audiovae_student.sampling import SamplerExhausted
from test_restart_data import row, counts, WINDOW


def ledger(rows=()):
    return parent_student_ledger(rows, counts(rows), checkpoint_sha256="a" * 64)


def pool():
    return [row(f"{language}-{i}", samples=WINDOW * 3 + 1366,
                language=language) for language in ("en", "hi", "ar", "und") for i in range(8)]


def plan(rows, **kwargs):
    return plan_comparison(rows, counts(rows), ledger(), windows_count=64,
                           required_indic=("hi",), required_other=("ar",), **kwargs)


def test_pilot_lineage_excluded_but_previous_independent_audio_allowed():
    rows = pool()
    parent = rows[0]
    result = plan_comparison(rows, counts(rows), ledger([parent]), windows_count=64,
                             required_indic=("hi",), required_other=("ar",))
    assert parent.source_id not in {r.source_id for r in result["rows"]}
    assert any(r.source_id == rows[1].source_id for r in result["rows"])
    assert len(result["windows"]) == 64
    verify_comparison_windows(result["windows"], {r.source_id: r for r in result["rows"]},
                              result["counts"], result["ledger"])


def test_unknown_fleurs_placeholder_is_disclosed_but_real_speaker_is_blocked():
    source = replace(row("es-train", language="es"), speaker_id=None,
                     session_id="fleurs:unknown-session-group:es_419")
    dev = replace(row("es-dev", language="es"), split="dev", speaker_id=None,
                  session_id=source.session_id)
    assert_comparison_disjoint([source], [dev])
    with pytest.raises(ValueError, match="reserved identity"):
        assert_comparison_disjoint([replace(source, speaker_id="real-person")],
                                   [replace(dev, speaker_id="real-person")])
    with pytest.raises(ValueError, match="reserved identity"):
        assert_comparison_disjoint([source], [replace(dev, audio_sha256=source.audio_sha256)])
    with pytest.raises(ValueError, match="reserved identity"):
        assert_comparison_disjoint([replace(source, session_id="real-session")],
                                   [replace(dev, session_id="real-session")])


def test_finite_partial_windows_and_rare_events_are_not_replayed():
    rows = pool()
    events = {"und-0": ["Laughter", "Giggle"], "und-1": ["Crying_and_sobbing"]}
    result = plan(rows, event_labels=events)
    windows = result["windows"]
    assert len({(w.source_id, w.start_frame) for w in windows}) == len(windows)
    assert any(w.valid_output_samples48k == 4098 for w in windows)
    assert any(w.condition == "Crying_and_sobbing" for w in windows)
    assert any(w.condition == "Giggle" for w in windows)
    assert all(w.condition != "Laughter" for w in windows if w.source_id == "und-0")
    assert all(w.valid_output_samples48k >= 4096 for w in windows)
    sampler = FixedWindowSampler(windows)
    assert len(sampler.take_batch(64)) == 64
    with pytest.raises(SamplerExhausted):
        sampler.take_batch(1)


def test_missing_language_fails_instead_of_silent_renormalization():
    rows = pool()
    with pytest.raises(ValueError, match="Missing required Indic"):
        plan_comparison(rows, counts(rows), ledger(), required_indic=("sat",), required_other=())
    result = plan(rows)
    assert any(item["bucket"] == "events" and item["reallocated_to"] == "english"
               for item in result["metadata"]["drained_capacity"])


def test_mixed_japanese_recordings_are_not_relabeled_speech():
    rows = pool() + [replace(row("mixed", language="ja", samples=WINDOW * 3,
                                dataset="jvnv"), license="CC-BY-SA-4.0")]
    result = plan(rows)
    selected = [w for w in result["windows"] if w.source_id == "mixed"]
    assert selected
    assert all(w.condition == "jvnv_verbal_and_nonverbal" for w in selected)


def test_roundtrip_and_tampering_fail_closed(tmp_path):
    result = plan(pool())
    root = tmp_path / "comparison"
    ready = write_comparison_plan(result, root)
    loaded = load_comparison_plan(root)
    assert [w.to_dict() for w in loaded["windows"]] == [w.to_dict() for w in result["windows"]]
    assert loaded["identity"] == ready
    values = json.loads((root / "input-sample-counts.json").read_text())
    values[next(iter(values))] += 1
    (root / "input-sample-counts.json").write_text(json.dumps(values))
    with pytest.raises(ValueError, match="metadata changed"):
        load_comparison_plan(root)
