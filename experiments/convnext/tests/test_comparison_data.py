"""Metadata-only checks for the matched data plan and per-student exclusions."""
from dataclasses import replace
import json

import pytest

from audiovae_student.comparison_data import (
    assert_comparison_disjoint, parent_student_ledger, plan_comparison,
    load_comparison_plan, write_comparison_plan, verify_comparison_windows)
from audiovae_student.restart_data import FixedWindowSampler, canonical, digest
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


def test_explicit_default_minimum_preserves_windows_and_metadata():
    default = plan(pool())
    explicit = plan(pool(), minimum_input_samples=1366)
    assert canonical(default['metadata']) == canonical(explicit['metadata'])
    assert canonical([w.to_dict() for w in default['windows']]) == canonical(
        [w.to_dict() for w in explicit['windows']])
    # Captured from the frozen r6 planner, source SHA eeb3627755bfc0475fd063f96ea854e829cbe78ac9fdf3cf016da658ddac3611.
    assert digest(default['metadata']) == 'c12ae50b9640798d661b9a3e1f019bedcf01cfd08ec3bf922bd835e7ad18b7a4'
    assert digest([w.to_dict() for w in default['windows']]) == '41362de784143e08ade6c39e4e019cb6ffe2ac56f4aebe7a81ccfca38ce5cae6'


def test_gan_minimum_retains_all_qualifying_partial_tails(tmp_path):
    rows = [row('boundary', language='en', samples=WINDOW + 3040),
            row('larger-tail', language='en', samples=WINDOW + 3500),
            row('too-short-tail', language='en', samples=WINDOW + 3039)]
    result = plan_comparison(rows, counts(rows), ledger(), windows_count=5,
        required_indic=(), required_other=(), minimum_input_samples=3040)
    assert len(result['windows']) == 5
    tails = {w.source_id: w.valid_input_samples16k for w in result['windows'] if w.start_frame == 64}
    assert tails == {'boundary': 3040, 'larger-tail': 3500}
    assert result['metadata']['minimum_output_samples'] == 9120
    assert result['metadata']['capacity']['english']['discarded_tail_samples'] == 3039
    write_comparison_plan(result, tmp_path / 'gan-minimum')
    loaded = load_comparison_plan(tmp_path / 'gan-minimum')
    assert all(w.valid_output_samples48k >= 9120 for w in loaded['windows'])
    assert any(w.valid_output_samples48k == 9120 for w in loaded['windows'])


def test_loader_rejects_validly_checksummed_windows_below_declared_minimum(tmp_path):
    result = plan(pool())
    assert any(w.valid_output_samples48k < 9120 for w in result['windows'])
    result['metadata']['minimum_output_samples'] = 9120
    # A complete file hash set does not excuse an inconsistent length contract.
    write_comparison_plan(result, tmp_path / 'inconsistent-minimum')
    with pytest.raises(ValueError, match='declared minimum'):
        load_comparison_plan(tmp_path / 'inconsistent-minimum')
    ordinary = plan(pool())
    ordinary['metadata']['minimum_input_samples'] = 3040
    write_comparison_plan(ordinary, tmp_path / 'conflicting-minimum')
    with pytest.raises(ValueError, match='input/output minimum'):
        load_comparison_plan(tmp_path / 'conflicting-minimum')


@pytest.mark.parametrize('minimum', [1365, 3040.0, True, WINDOW + 1])
def test_invalid_minimum_is_rejected(minimum):
    with pytest.raises(ValueError, match='minimum_input_samples'):
        plan(pool(), minimum_input_samples=minimum)
