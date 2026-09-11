"""No-repeat ledger, quota and reserve checks using synthetic metadata only."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from audiovae_student.data import ManifestRow
from audiovae_student.restart_data import (
    ConsumedLedger, FixedWindowSampler, HOUR, PilotWindow, default_language_quotas,
    default_quotas, digest, merge_intervals, plan_pilot, rebuild_ledger,
    reserve_unseen, verify_windows, windows_for, write_plan,
)
from audiovae_student.sampling import NoRepeatSegmentSampler, SamplerExhausted

WINDOW = 64 * 640
CONFIG = dict(scored_frames=64, seed=7, include_short_tail=True, min_input_samples=1366)


def row(identifier, *, samples=WINDOW * 3, dataset="fleurs", language="hi", speaker=None):
    return ManifestRow(
        dataset=dataset, source_revision="synthetic-v1", source_id=identifier,
        source_url="https://example.test/synthetic", audio_path=f"{identifier}.wav",
        audio_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
        parent_recording_id=f"parent:{identifier}", parent_start_seconds=0,
        speaker_id=speaker or f"speaker:{identifier}", session_id=f"session:{identifier}",
        language=language, sample_rate_hz=16000, original_sample_rate_hz=16000,
        bandwidth_hz=8000, bandwidth_class="speech_band", bandwidth_evidence="synthetic sample bounds",
        native_recording=True, enhanced=False, duration_seconds=samples / 16000,
        split="train", source_split="train", license="CC-BY-4.0",
        license_url="https://example.test/license", attribution="synthetic",
        access_record="synthetic fixture", gain_policy="unchanged", resampler_policy="none",
        teacher_cache_key=None)


def counts(rows):
    return {r.source_id: round(r.duration_seconds * 16000) for r in rows}


def ledger_for(rows, consumed=0, bootstrap=()):
    sampler = NoRepeatSegmentSampler(rows, counts(rows), **CONFIG)
    if consumed:
        sampler.take_batch(consumed)
    return rebuild_ledger(rows, counts(rows), sampler.state_dict(), CONFIG, bootstrap)


def test_replay_exact_intervals_retire_bootstrap_and_reject_mutation():
    rows = [row("a"), row("b", language="en"), row("c")]
    sampler = NoRepeatSegmentSampler(rows, counts(rows), **CONFIG)
    emitted = sampler.take_batch(5)
    bootstrap = [row("bootstrap", samples=12345)]
    ledger = rebuild_ledger(rows, counts(rows), sampler.state_dict(), CONFIG, bootstrap)
    for window in emitted:
        source = next(r for r in rows if r.source_id == window.row_id)
        assert ledger.overlaps(source, window.start_input_sample, window.stop_input_sample)
    assert ledger.entries["bootstrap"]["intervals"] == [[0, 12345]]
    assert ledger.entries["bootstrap"]["retired_whole"]
    changed = deepcopy(ledger.document)
    changed["sources"][0]["input_samples"] += 1
    with pytest.raises(ValueError, match="checksum"):
        ConsumedLedger(changed)
    state = sampler.state_dict()
    state["next_language"] = (state["next_language"] + 1) % 2
    with pytest.raises(ValueError, match="reproducible"):
        rebuild_ledger(rows, counts(rows), state, CONFIG, [])
    wrong_counts = counts(rows)
    wrong_counts["a"] += 1
    with pytest.raises(ValueError):
        rebuild_ledger(rows, wrong_counts, sampler.state_dict(), CONFIG, [])


def test_overlap_detects_parent_alias_hash_alias_and_touching_boundary():
    source = row("a", samples=WINDOW * 2)
    ledger = ledger_for([source], 1)
    alias = replace(row("alias"), parent_recording_id=source.parent_recording_id)
    assert ledger.overlaps(alias, 0, 640)
    assert not ledger.overlaps(alias, WINDOW, WINDOW + 640)
    moved = replace(alias, parent_start_seconds=WINDOW / 16000)
    assert not ledger.overlaps(moved, 0, 640)
    hashed = replace(row("hashed"), audio_sha256=source.audio_sha256)
    assert ledger.overlaps(hashed, 0, 640)
    assert not ledger.whole_untouched(alias)
    changed_id = replace(source, audio_sha256="a" * 64)
    with pytest.raises(ValueError, match="provenance"):
        ledger.overlaps(changed_id, WINDOW, WINDOW + 1)


def test_manifest_relocation_does_not_change_ledger_replay_identity():
    rows = [row("a"), row("b")]
    sampler = NoRepeatSegmentSampler(rows, counts(rows), **CONFIG)
    sampler.take_batch(2)
    moved = [replace(r, audio_path="/new/" + r.audio_path, access_record="relocated") for r in rows]
    ledger = rebuild_ledger(moved, counts(rows), sampler.state_dict(), CONFIG, [])
    assert ledger.document["saved_sampler_state"] == sampler.state_dict()


def test_merge_intervals_and_corrupt_canonical_ledger():
    assert merge_intervals([[10, 20], [0, 5], [4, 10], [30, 40]]) == [[0, 20], [30, 40]]
    with pytest.raises(ValueError):
        merge_intervals([[5, 5]])
    value = deepcopy(ledger_for([row("a")], 1).document)
    value["sources"][0]["intervals"] = [[0, 20], [10, 30]]
    value.pop("identity_sha256")
    value["identity_sha256"] = digest(value)
    with pytest.raises(ValueError, match="canonical"):
        ConsumedLedger(value)


def test_unseen_speaker_reserve_excludes_all_linked_exposed_sessions():
    old = row("old", dataset="librispeech", language="en", speaker="seen")
    ledger = ledger_for([old], 1)
    future = [row("same-speaker", dataset="librispeech", language="en", speaker="seen"),
              replace(row("same-session", dataset="librispeech", language="en", speaker="linked"),
                      session_id=old.session_id),
              row("linked-other-session", dataset="librispeech", language="en", speaker="linked"),
              row("safe-a", dataset="librispeech", language="en", speaker="safe-a"),
              row("safe-b", dataset="librispeech", language="en", speaker="safe-b")]
    dev, test = reserve_unseen(future, ledger)
    assert {r.source_id for r in dev + test} == {"safe-a", "safe-b"}
    assert len(dev) == len(test) == 1
    assert not {r.speaker_id for r in dev} & {r.speaker_id for r in test}


def small_plan(rows, ledger, **kwargs):
    return plan_pilot(rows, counts(rows), ledger, quotas={"fleurs": WINDOW * 4},
                      language_quotas={"fleurs": {"hi": WINDOW * 2, "en": WINDOW * 2}},
                      language_minimums={}, diagnostic_size=0, sentinel_size=0,
                      reserve_datasets=(), **kwargs)


def test_pilot_deterministic_quotas_balancing_no_repeated_windows():
    rows = [row("hi-a"), row("en-a", language="en"), row("hi-b"), row("en-b", language="en")]
    ledger = ledger_for(rows)
    plan = small_plan(rows, ledger)
    reversed_plan = small_plan(list(reversed(rows)), ledger)
    assert plan["windows"] == reversed_plan["windows"]
    assert [w.language for w in plan["windows"]] == ["en", "hi", "en", "hi"]
    assert plan["summary"]["actual_input_samples"] == {"fleurs": WINDOW * 4}
    assert len({(w.source_id, w.start_frame) for w in plan["windows"]}) == 4


def test_pilot_does_not_fall_back_to_consumed_tail_or_renormalize_missing_language():
    rows = [row("hi"), row("en", language="en")]
    ledger = ledger_for(rows, 1)  # English first, but two English windows are locally unused.
    with pytest.raises(SamplerExhausted, match="fleurs/en"):
        small_plan(rows, ledger)


def test_diagnostic_and_sentinel_whole_sources_are_reserved_before_pilot():
    old = row("old", dataset="librispeech", language="en", speaker="seen")
    rows = [old, *[row(f"same-{i}", dataset="librispeech", language="en", speaker="seen") for i in range(5)],
            *[row(f"new-{i}", dataset="librispeech", language="en") for i in range(6)]]
    plan = plan_pilot(rows, counts(rows), ledger_for([old], 1), quotas={"librispeech": WINDOW * 4},
                      language_quotas={}, language_minimums={}, diagnostic_size=2, sentinel_size=2)
    diag_ids = {r.source_id for r in plan["diagnostic"]}
    reserve_ids = {r.source_id for r in plan["dev_reserve"] + plan["test_reserve"]}
    train_ids = {w.source_id for w in plan["windows"]}
    assert not diag_ids & train_ids and not reserve_ids & (diag_ids | train_ids)
    assert {r.speaker_id for r in plan["sentinel"]}.isdisjoint({"seen"})
    assert len(plan["sentinel"]) == 2


def test_tail_counts_and_output_units_and_global_overlap_verifier():
    source = row("a", samples=WINDOW + 1401)
    windows = list(windows_for(source, WINDOW + 1401))
    tail = windows[-1]
    assert tail.scored_frames == 3
    assert tail.to_dict()["valid_output_samples48k"] == 4203
    assert tail.as_segment().valid_input_samples == 1401
    ledger = ledger_for([source])
    verify_windows(windows, {source.source_id: source}, counts([source]), ledger)
    with pytest.raises(ValueError, match="Repeated"):
        verify_windows(windows + windows[-1:], {source.source_id: source}, counts([source]), ledger)
    consumed = ledger_for([source], 1)
    with pytest.raises(ValueError, match="globally consumed"):
        verify_windows(windows, {source.source_id: source}, counts([source]), consumed)
    with pytest.raises(ValueError, match="Reserved"):
        verify_windows(windows, {source.source_id: source}, counts([source]), ledger,
                       reserved_rows=[replace(row("dev"), speaker_id=source.speaker_id)])


def test_fixed_plan_resume_and_batch_exhaustion_are_atomic():
    source = row("a")
    windows = list(windows_for(source, WINDOW * 3))
    sampler = FixedWindowSampler(windows)
    sampler.take_batch(2)
    state = sampler.state_dict()
    with pytest.raises(SamplerExhausted, match="never wraps"):
        sampler.take_batch(2)
    assert sampler.state_dict() == state
    resumed = FixedWindowSampler(windows)
    resumed.load_state_dict(state)
    assert resumed.take_batch(1) == sampler.take_batch(1)
    with pytest.raises(ValueError, match="identity"):
        FixedWindowSampler(windows[::-1]).load_state_dict(state)


def test_plan_checksums_and_refuses_to_overwrite(tmp_path):
    rows = [row("hi"), row("en", language="en")]
    ledger = ledger_for(rows)
    plan = small_plan(rows, ledger)
    report = write_plan(tmp_path / "plan", plan, ledger, counts(rows))
    assert report["files_sha256"]["windows.jsonl"] == hashlib.sha256((tmp_path / "plan/windows.jsonl").read_bytes()).hexdigest()
    ready = json.loads((tmp_path / "plan/ready.json").read_text())
    assert ready == report
    with pytest.raises(FileExistsError):
        write_plan(tmp_path / "plan", plan, ledger, counts(rows))


def test_default_twenty_hour_source_and_fleurs_language_quotas():
    quotas = default_quotas()
    assert sum(quotas.values()) == 20 * HOUR
    assert sum(default_language_quotas()["fleurs"].values()) == quotas["fleurs"]
    assert "gigaspeech" not in quotas
