"""Synthetic metadata tests for unique-window accounting, not speech quality."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import random

import pytest

from audiovae_student.data import ManifestRow, ManifestValidationError
from audiovae_student.sampling import NoRepeatSegmentSampler, SamplerExhausted, normalize_language


WINDOW = 640 * 64


def make_row(identifier, *, samples=WINDOW * 2, language="hi", dataset="fleurs", split="train"):
    return ManifestRow(
        dataset=dataset, source_revision="synthetic-unit-test-v1", source_id=identifier,
        source_url="https://example.test/synthetic", audio_path=f"{identifier}.wav",
        audio_sha256=hashlib.sha256(identifier.encode()).hexdigest(), parent_recording_id=f"parent:{identifier}",
        parent_start_seconds=0, speaker_id=f"speaker:{identifier}", session_id=f"session:{identifier}",
        language=language, sample_rate_hz=16000, original_sample_rate_hz=16000, bandwidth_hz=7600,
        bandwidth_class="speech_band", bandwidth_evidence="synthetic fixture band contract",
        native_recording=True, enhanced=False, duration_seconds=samples / 16000,
        split=split, source_split="train" if split == "train" else "validation",
        license="CC-BY-4.0", license_url="https://example.test/license", attribution="Synthetic fixture",
        access_record="synthetic fixture; no downloaded audio", gain_policy="unchanged",
        resampler_policy="none", teacher_cache_key=None)


def make_corpus():
    counts = {"english-a": WINDOW * 3 + 123, "english-b": WINDOW, "english-c": WINDOW * 2,
              "hindi-a": WINDOW, "hindi-b": WINDOW * 3, "short-tamil": WINDOW - 1}
    rows = [make_row("english-a", samples=counts["english-a"], language="en_US", dataset="librispeech"),
            make_row("english-b", samples=counts["english-b"], language="en", dataset="librispeech"),
            make_row("english-c", samples=counts["english-c"], language="eng"),
            make_row("hindi-a", samples=counts["hindi-a"], language="hi_IN"),
            make_row("hindi-b", samples=counts["hindi-b"], language="Hindi"),
            make_row("short-tamil", samples=counts["short-tamil"], language="ta_IN")]
    return rows, counts


@pytest.mark.parametrize("language,expected", [
    ("en", "en"), ("en_US", "en"), ("English", "en"), ("eng-GB", "en"),
    ("Assamese", "as"), ("ben_IN", "bn"), ("Bodo", "brx"), ("Dogri", "doi"),
    ("Gujarati", "gu"), ("hin_Deva", "hi"), ("Kannada", "kn"), ("Kashmiri", "ks"),
    ("Konkani", "kok"), ("Maithili", "mai"), ("Malayalam", "ml"), ("mni_Mtei", "mni"),
    ("Marathi", "mr"), ("Nepali", "ne"), ("Odia", "or"), ("Punjabi", "pa"),
    ("Sanskrit", "sa"), ("Santali", "sat"), ("Sindhi", "sd"), ("Tamil", "ta"),
    ("Telugu", "te"), ("Urdu", "ur"), ("bho_IN", "bho"), ("es_ES", "es"),
    ("bod_CN", "bod"),  # Tibetan's code is not Bodo.
])
def test_language_normalization(language, expected):
    assert normalize_language(language) == expected


def test_full_windows_never_repeat_or_overlap_and_tails_are_counted():
    rows, counts = make_corpus()
    sampler = NoRepeatSegmentSampler(rows, counts)
    segments = sampler.take_batch(sampler.total_segments)
    assert len(segments) == len({(segment.row_id, segment.start_frame) for segment in segments}) == 10
    by_row = defaultdict(list)
    for segment in segments:
        assert segment.input_samples == WINDOW
        assert segment.stop_input_sample <= counts[segment.row_id]
        by_row[segment.row_id].append(segment)
    for row_id, row_segments in by_row.items():
        assert [segment.start_frame for segment in row_segments] == list(range(0, 64 * (counts[row_id] // WINDOW), 64))
        for first, second in zip(row_segments, row_segments[1:]):
            assert first.stop_input_sample == second.start_input_sample
    assert "short-tamil" not in by_row
    report = sampler.inventory()
    assert report["segments"] == report["emitted_segments"] == 10
    assert report["remaining_segments"] == 0
    assert report["utterances"] == 6 and report["usable_utterances"] == 5
    assert report["unique_scored_input_samples"] == 10 * WINDOW
    assert report["excluded_tail_input_samples"] == 123 + WINDOW - 1
    assert report["unique_scored_audio_hours"] == pytest.approx(10 * 2.56 / 3600)
    assert report["unique_scored_input_samples"] + report["excluded_tail_input_samples"] == sum(counts.values())
    assert set(report["by_language"]) == {"en", "hi", "ta"}
    assert report["by_language"]["en"]["segments"] == 6
    assert report["by_language"]["ta"]["segments"] == 0
    with pytest.raises(SamplerExhausted, match="never starts a new epoch"):
        sampler.next_segment()


def test_languages_cycle_without_oversampling_and_utterances_stay_sequential():
    rows, counts = make_corpus()
    sampler = NoRepeatSegmentSampler(rows, counts)
    segments = sampler.take_batch(10)
    assert [segment.language for segment in segments[:8]] == ["en", "hi"] * 4
    assert [segment.language for segment in segments[8:]] == ["en", "en"]
    for language in ("en", "hi"):
        selected = [segment for segment in segments if segment.language == language]
        completed = set()
        previous = None
        for segment in selected:
            if segment.row_id != previous:
                assert segment.row_id not in completed
                completed.add(segment.row_id)
                previous = segment.row_id
    english_utterances = []
    for segment in (value for value in segments if value.language == "en"):
        if not english_utterances or english_utterances[-1][0] != segment.row_id:
            english_utterances.append((segment.row_id, segment.dataset))
    assert [dataset for _, dataset in english_utterances] == ["fleurs", "librispeech", "librispeech"]


@pytest.mark.parametrize("boundary", [0, 1, 3, 8, 9, 10])
def test_checkpoint_resume_replays_exactly_even_when_language_is_exhausted(boundary):
    rows, counts = make_corpus()
    sampler = NoRepeatSegmentSampler(rows, counts, seed=17)
    if boundary:
        sampler.take_batch(boundary)
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.take_batch(sampler.remaining_segments) if sampler.remaining_segments else []
    resumed = NoRepeatSegmentSampler(reversed(rows), counts, seed=17)
    resumed.load_state_dict(state)
    actual = resumed.take_batch(resumed.remaining_segments) if resumed.remaining_segments else []
    assert actual == expected
    assert resumed.state_dict() == sampler.state_dict()


def test_full_batch_exhaustion_is_atomic_and_sampler_does_not_change_global_rng():
    rows, counts = make_corpus()
    random.seed(991)
    global_rng = random.getstate()
    sampler = NoRepeatSegmentSampler(rows, counts)
    sampler.take_batch(8)
    before = sampler.state_dict()
    with pytest.raises(SamplerExhausted, match="only 2 unique windows remain"):
        sampler.take_batch(3)
    assert sampler.state_dict() == before
    assert random.getstate() == global_rng
    sampler.take_batch(2)
    with pytest.raises(SamplerExhausted):
        sampler.take_batch(1)


@pytest.mark.parametrize("change", ["seed", "samples", "language", "source", "hash", "frames", "tails", "tail_minimum"])
def test_checkpoint_rejects_changed_selection_identity(change):
    rows, counts = make_corpus()
    sampler = NoRepeatSegmentSampler(rows, counts)
    sampler.take_batch(3)
    state = sampler.state_dict()
    kwargs = {}
    if change == "seed":
        kwargs["seed"] = 8
    elif change == "samples":
        counts["english-a"] += 1
    elif change == "language":
        rows[3] = replace(rows[3], language="bn_IN")
    elif change == "source":
        rows[0] = replace(rows[0], source_revision="another-immutable-revision")
    elif change == "hash":
        rows[0] = replace(rows[0], audio_sha256="a" * 64)
    elif change == "frames":
        kwargs["scored_frames"] = 32
    elif change == "tails":
        kwargs["include_short_tail"] = True
    elif change == "tail_minimum":
        kwargs["min_input_samples"] = 1500
    changed = NoRepeatSegmentSampler(rows, counts, **kwargs)
    before = changed.state_dict()
    with pytest.raises(ValueError, match="identity mismatch"):
        changed.load_state_dict(state)
    assert changed.state_dict() == before


@pytest.mark.parametrize("change", ["counter", "sample_counter", "utterance", "segment", "language_cursor", "missing_language"])
def test_checkpoint_rejects_corrupt_cursors_atomically(change):
    rows, counts = make_corpus()
    sampler = NoRepeatSegmentSampler(rows, counts)
    sampler.take_batch(3)
    before = sampler.state_dict()
    state = deepcopy(before)
    if change == "counter":
        state["emitted_segments"] += 1
    elif change == "sample_counter":
        state["emitted_input_samples"] += 1
    elif change == "utterance":
        state["cursors"]["en"][0] = 100
    elif change == "segment":
        state["cursors"]["hi"][1] = -1
    elif change == "language_cursor":
        state["next_language"] = 100
    else:
        del state["cursors"]["hi"]
    with pytest.raises(ValueError, match="checkpoint"):
        sampler.load_state_dict(state)
    assert sampler.state_dict() == before


def test_heldout_reserved_previous_audio_and_overlapping_parents_are_rejected():
    row = make_row("train")
    with pytest.raises(ManifestValidationError, match="training-only"):
        NoRepeatSegmentSampler([replace(row, split="dev", source_split="validation")], {row.source_id: WINDOW * 2})
    with pytest.raises(ManifestValidationError, match="leakage"):
        NoRepeatSegmentSampler([row], {row.source_id: WINDOW * 2}, reserved_rows=[row])
    other = replace(make_row("second"), parent_recording_id=row.parent_recording_id, parent_start_seconds=2.56)
    with pytest.raises(ManifestValidationError, match="overlapping training segments"):
        NoRepeatSegmentSampler([row, other], {row.source_id: WINDOW * 2, other.source_id: WINDOW * 2})
    # Incorrectly short duration metadata cannot hide overlap in actual counts.
    row = replace(row, duration_seconds=2.56)
    with pytest.raises(ValueError, match="Actual scored windows overlap"):
        NoRepeatSegmentSampler([row, other], {row.source_id: WINDOW * 2, other.source_id: WINDOW * 2})


def test_actual_sample_counts_are_required_and_zero_inventory_is_reported():
    row = make_row("short", samples=1000)
    with pytest.raises(ValueError, match="exact training row IDs"):
        NoRepeatSegmentSampler([row], {})
    with pytest.raises(ValueError, match="positive integer"):
        NoRepeatSegmentSampler([row], {row.source_id: 1000.0})
    sampler = NoRepeatSegmentSampler([row], {row.source_id: 1000})
    assert sampler.inventory()["segments"] == 0
    assert sampler.inventory()["excluded_tail_input_samples"] == 1000
    sampler.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
    with pytest.raises(SamplerExhausted):
        sampler.take_batch(1)


def test_relocating_files_does_not_change_sampler_identity():
    rows, counts = make_corpus()
    original = NoRepeatSegmentSampler(rows, counts)
    original.take_batch(3)
    relocated = [replace(row, audio_path=f"/new-root/{row.audio_path}", access_record="/new-root/access.json")
                 for row in rows]
    sampler = NoRepeatSegmentSampler(relocated, counts)
    sampler.load_state_dict(original.state_dict())
    assert sampler.take_batch(7) == original.take_batch(7)


def test_short_expressive_tails_keep_actual_samples_without_scoring_padding():
    counts = {"too-short": 1365, "minimum": 1366, "one-second": 16000, "almost-full": WINDOW - 1,
              "short-tail": WINDOW + 1365, "valid-tail": WINDOW + 1366, "two-full": WINDOW * 2 + 137}
    rows = [make_row(row_id, samples=count) for row_id, count in counts.items()]
    sampler = NoRepeatSegmentSampler(rows, counts, include_short_tail=True)
    segments = sampler.take_batch(sampler.total_segments)
    by_row = defaultdict(list)
    for segment in segments:
        by_row[segment.row_id].append(segment)
        assert 1366 <= segment.valid_input_samples <= WINDOW
        assert segment.valid_output_samples >= 4096
        assert segment.input_samples == segment.valid_input_samples
        assert segment.scored_frames == (segment.valid_input_samples + 639) // 640
        assert segment.stop_input_sample <= counts[segment.row_id]
        assert segment.scored_frames <= 64
    assert "too-short" not in by_row
    assert [(segment.start_frame, segment.input_samples) for segment in by_row["valid-tail"]] == [(0, WINDOW), (64, 1366)]
    assert [(segment.start_frame, segment.input_samples) for segment in by_row["one-second"]] == [(0, 16000)]
    for row_segments in by_row.values():
        for first, second in zip(row_segments, row_segments[1:]):
            assert first.stop_input_sample == second.start_input_sample
    report = sampler.inventory()
    assert report["segments"] == 8
    assert report["full_segments"] == report["tail_segments"] == 4
    assert report["excluded_tail_input_samples"] == 1365 * 2 + 137
    assert report["unique_scored_input_samples"] == sum(segment.input_samples for segment in segments)
    assert report["emitted_input_samples"] == report["unique_scored_input_samples"]
    assert report["remaining_scored_input_samples"] == 0
    assert report["full_window_equivalents"] == report["unique_scored_input_samples"] / WINDOW


@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_tail_sampler_checkpoint_replays_exact_samples(boundary):
    counts = {"english": WINDOW + 1500, "hindi": WINDOW + 16000}
    rows = [make_row(row_id, samples=count, language="en" if row_id == "english" else "hi")
            for row_id, count in counts.items()]
    original = NoRepeatSegmentSampler(rows, counts, include_short_tail=True)
    original.take_batch(boundary)
    state = json.loads(json.dumps(original.state_dict()))
    resumed = NoRepeatSegmentSampler(rows, counts, include_short_tail=True)
    resumed.load_state_dict(state)
    while original.remaining_segments:
        assert original.next_segment() == resumed.next_segment()
        assert original.state_dict() == resumed.state_dict()
    assert resumed.emitted_input_samples == sum(counts.values())


def test_tail_threshold_and_language_aliases_apply_to_source_policy():
    row = make_row("english", language="English", dataset="librispeech", samples=16000)
    sampler = NoRepeatSegmentSampler([row], {row.source_id: 16000}, include_short_tail=True, min_input_samples=16001)
    assert sampler.total_segments == 0
    sampler = NoRepeatSegmentSampler([row], {row.source_id: 16000}, include_short_tail=True, min_input_samples=16000)
    assert sampler.next_segment().language == "en"
    with pytest.raises(ManifestValidationError, match="requires non-English"):
        NoRepeatSegmentSampler([replace(row, dataset="mls_non_english")], {row.source_id: 16000})
    with pytest.raises(ValueError, match="exceeds the configured scored window"):
        NoRepeatSegmentSampler([row], {row.source_id: 16000}, include_short_tail=True, min_input_samples=WINDOW + 1)
