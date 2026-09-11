"""Build an immutable restart plan from verified previous exposure, without audio IO.

The ledger describes scored 16 kHz intervals, not the overlapping causal context.
It also retires the whole bootstrap corpus because its old sampler repeated data.
The conservative pilot uses wholly untouched utterances. Deduplication trusts
canonical parent/session identities and exact file hashes, not audio fingerprints.
This module never downloads audio, loads a teacher or starts training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data import ManifestRow, load_manifest, validate_manifest
from .sampling import NoRepeatSegmentSampler, ScoredSegment, SamplerExhausted, normalize_language

RATE = 16000
HOP = 640
MINIMUM = 1366
HOUR = RATE * 3600
INDIC = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "pa", "sd", "ta", "te", "ur")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(row: ManifestRow) -> dict:
    value = row.to_dict()
    for key in ("audio_path", "access_record", "teacher_cache_key"):
        value.pop(key)
    return value


def identities(row: ManifestRow, *, people: bool = True) -> set[tuple[str, str]]:
    result = {("source", row.source_id), ("parent", row.parent_recording_id),
              ("sha256", row.audio_sha256)}
    if people:
        result.update((name, value) for name, value in
                      (("speaker", row.speaker_id), ("session", row.session_id)) if value is not None)
    return result


def _counts(rows: Iterable[ManifestRow], counts: Mapping[str, int]) -> None:
    rows = list(rows)
    if len({r.source_id for r in rows}) != len(rows) or set(counts) != {r.source_id for r in rows}:
        raise ValueError("Counts must match unique source IDs exactly")
    for row in rows:
        count = counts[row.source_id]
        if (type(count) is not int or count < 1 or row.sample_rate_hz != RATE or
                abs(row.duration_seconds * RATE - count) > 1e-5):
            raise ValueError(f"Counts must describe exact prepared 16 kHz lengths: {row.source_id}")


def merge_intervals(intervals: Iterable[Iterable[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for pair in sorted([list(v) for v in intervals]):
        if len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1]:
            raise ValueError("Invalid scored interval")
        if merged and pair[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], pair[1])
        else:
            merged.append(pair)
    return merged


class ConsumedLedger:
    """Verified immutable source intervals, with canonical cross-file overlap checks."""

    def __init__(self, document: Mapping[str, Any]):
        body = dict(document)
        expected = body.pop("identity_sha256", None)
        if expected != digest(body) or body.get("format_version") != 1 or body.get("input_sample_rate") != RATE:
            raise ValueError("Ledger checksum or version mismatch")
        self.document = json.loads(canonical(document))
        self.identity_sha256 = expected
        self.entries = {}
        self.touched = set()
        self.whole_keys = set()
        self._parents = defaultdict(list)
        self._hashes = defaultdict(list)
        for entry in body["sources"]:
            row = ManifestRow.from_dict(entry["row"])
            count = entry["input_samples"]
            _counts([row], {row.source_id: count})
            if row.source_id in self.entries:
                raise ValueError("Duplicate source in ledger")
            intervals = entry["intervals"]
            if merge_intervals(intervals) != intervals or any(stop > count for _, stop in intervals):
                raise ValueError("Ledger intervals are not canonical or exceed source length")
            if type(entry["retired_whole"]) is not bool or not intervals:
                raise ValueError("Ledger must contain actual exposure or a whole-source retirement")
            if entry["retired_whole"] and intervals != [[0, count]]:
                raise ValueError("Retired bootstrap source must cover the entire file")
            self.entries[row.source_id] = entry
            self.touched.update(identities(row))
            if entry["retired_whole"]:
                self.whole_keys.update(identities(row, people=False))
            origin = row.parent_start_seconds * RATE
            for start, stop in intervals:
                # Outward rounding remains conservative for nonintegral parent timestamps.
                self._parents[row.parent_recording_id].append((math.floor(origin + start), math.ceil(origin + stop)))
                self._hashes[row.audio_sha256].append((start, stop))
        for index in (self._parents, self._hashes):
            for key in index:
                index[key] = merge_intervals(index[key])

    def overlaps(self, row: ManifestRow, start_input_sample: int, stop_input_sample: int) -> bool:
        if (type(start_input_sample) is not int or type(stop_input_sample) is not int or
                not 0 <= start_input_sample < stop_input_sample):
            raise ValueError("Invalid candidate interval")
        old = self.entries.get(row.source_id)
        if old is not None and identity(row) != identity(ManifestRow.from_dict(old["row"])):
            raise ValueError("Previously seen source ID changed immutable provenance")
        if identities(row, people=False) & self.whole_keys:
            return True
        candidates = list(self._hashes.get(row.audio_sha256, ()))
        if any(start_input_sample < stop and start < stop_input_sample for start, stop in candidates):
            return True
        origin = row.parent_start_seconds * RATE
        lo, hi = math.floor(origin + start_input_sample), math.ceil(origin + stop_input_sample)
        return any(lo < stop and start < hi for start, stop in self._parents.get(row.parent_recording_id, ()))

    def whole_untouched(self, row: ManifestRow) -> bool:
        """Reject any previously touched file or canonical parent, including aliases."""
        if identities(row, people=False) & self.touched:
            return False
        # Indic chunk offsets inside the full recording were not published.
        if row.dataset == "indicvoices" and row.session_id and ("session", row.session_id) in self.touched:
            return False
        return True


def rebuild_ledger(rows: Iterable[ManifestRow], counts: Mapping[str, int], saved_state: Mapping[str, Any],
                   sampler_config: Mapping[str, Any], bootstrap_rows: Iterable[ManifestRow], *,
                   provenance: Mapping[str, Any] | None = None) -> ConsumedLedger:
    """Replay the complete committed prefix and require exact cursor/counter equality.

    Counts must come from the previously verified prepared corpus. The old sampler
    identity pins counts, source metadata, seed, ordering and scored/tail settings.
    No private sampler queue fields are used.
    """
    rows = sorted(rows, key=lambda r: r.source_id)
    _counts(rows, counts)
    required = {"scored_frames", "seed", "include_short_tail", "min_input_samples"}
    if set(sampler_config) != required:
        raise ValueError(f"Sampler config must contain exactly {sorted(required)}")
    checked = NoRepeatSegmentSampler(rows, counts, **sampler_config)
    checked.load_state_dict(saved_state)
    replay = NoRepeatSegmentSampler(rows, counts, **sampler_config)
    intervals = defaultdict(list)
    for _ in range(saved_state["emitted_segments"]):
        segment = replay.next_segment()
        intervals[segment.row_id].append([segment.start_input_sample, segment.stop_input_sample])
    if replay.state_dict() != dict(saved_state):
        raise ValueError("Saved sampler state is not the exact reproducible committed prefix")
    by_id = {row.source_id: row for row in rows}
    retired = set()
    combined_counts = dict(counts)
    bootstrap_rows = list(bootstrap_rows)
    if bootstrap_rows:
        validate_manifest(bootstrap_rows, training_only=True)
    for row in bootstrap_rows:
        if row.split != "train":
            raise ValueError("Only prior bootstrap training rows may be retired")
        if row.source_id in by_id and identity(row) != identity(by_id[row.source_id]):
            raise ValueError("Bootstrap source identity conflicts with expanded corpus")
        count = round(row.duration_seconds * RATE)
        _counts([row], {row.source_id: count})
        by_id[row.source_id], combined_counts[row.source_id] = row, count
        retired.add(row.source_id)
        intervals[row.source_id] = [[0, count]]
    body = {"format_version": 1, "input_sample_rate": RATE,
            "deduplication_scope": "canonical source/parent/session identities and exact prepared-file hashes; no acoustic fingerprints",
            "sampler_config": dict(sampler_config), "saved_sampler_state": dict(saved_state),
            "provenance": dict(provenance or {}),
            "bootstrap_policy": "whole training utterances retired, including unobserved bootstrap crops",
            "sources": [{"row": by_id[key].to_dict(), "input_samples": combined_counts[key],
                         "retired_whole": key in retired, "intervals": merge_intervals(intervals[key])}
                        for key in sorted(intervals)]}
    return ConsumedLedger({**body, "identity_sha256": digest(body)})


@dataclass(frozen=True)
class PilotWindow:
    source_id: str
    start_frame: int
    scored_frames: int
    valid_input_samples16k: int
    dataset: str
    language: str
    condition: str | None = None

    @property
    def valid_output_samples48k(self):
        return self.valid_input_samples16k * 3

    def to_dict(self):
        return {**asdict(self), "row_id": self.source_id,
                "valid_output_samples48k": self.valid_output_samples48k}

    def as_segment(self):
        return ScoredSegment(self.source_id, self.start_frame, self.scored_frames,
                             self.language, self.dataset, self.valid_input_samples16k)


def windows_for(row: ManifestRow, count: int, *, scored_frames: int = 64, minimum: int = MINIMUM):
    for start in range(0, count, scored_frames * HOP):
        valid = min(count - start, scored_frames * HOP)
        if valid >= minimum:
            yield PilotWindow(row.source_id, start // HOP, math.ceil(valid / HOP), valid,
                              row.dataset, normalize_language(row.language))


def _order(rows: Iterable[ManifestRow], seed: int) -> list[ManifestRow]:
    """Deterministic speaker cycles; missing speakers use honest session groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.speaker_id or row.session_id].append(row)
    queues = []
    for key in sorted(groups, key=lambda value: digest([seed, value])):
        queues.append(deque(sorted(groups[key], key=lambda r: digest([seed, r.source_id]))))
    result = []
    while queues:
        for queue in queues:
            result.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    return result


def reserve_unseen(rows: Iterable[ManifestRow], ledger: ConsumedLedger, *, seed: int = 19,
                   datasets=("librispeech", "indicvoices")) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Reserve complete real-speaker/session components before selecting pilot audio."""
    rows = list(rows)
    # A speaker with any linked exposed session is not eligible, even if another
    # one of that speaker's utterances has no direct link to the exposed session.
    linked = defaultdict(set)
    for row in rows:
        if row.speaker_id:
            linked[row.speaker_id].update(identities(row))
    eligible_speakers = {speaker for speaker, keys in linked.items() if not keys & ledger.touched}
    candidates = [r for r in rows if r.dataset in datasets and r.speaker_id in eligible_speakers
                  and ledger.whole_untouched(r)]
    # Union speakers that share any session/parent/hash, so dev/test cannot split copies.
    groups = {r.speaker_id: r.speaker_id for r in candidates}
    def find(key):
        while groups[key] != key:
            groups[key] = groups[groups[key]]
            key = groups[key]
        return key
    owner = {}
    for row in candidates:
        for key in identities(row):
            if key in owner:
                a, b = find(row.speaker_id), find(owner[key])
                groups[max(a, b)] = min(a, b)
            owner[key] = row.speaker_id
    components = defaultdict(list)
    for row in candidates:
        components[find(row.speaker_id)].append(row)
    buckets = defaultdict(list)
    for group, values in components.items():
        buckets[(values[0].dataset, normalize_language(values[0].language))].append(group)
    dev, test = [], []
    for bucket in sorted(buckets):
        for index, group in enumerate(sorted(buckets[bucket], key=lambda k: digest([seed, k]))):
            split = "dev" if index % 2 == 0 else "test"
            target = dev if split == "dev" else test
            target.extend(replace(row, split=split) for row in components[group])
    return sorted(dev, key=lambda r: r.source_id), sorted(test, key=lambda r: r.source_id)


def default_quotas() -> dict[str, int]:
    return {"librispeech": 10 * HOUR, "fleurs": 5 * HOUR, "indicvoices": HOUR,
            "emogator": round(2.4 * HOUR), "crema_d": round(.8 * HOUR),
            "jvnv": round(.4 * HOUR), "thorsten_emotional": round(.4 * HOUR)}


def default_language_quotas() -> dict[str, dict[str, int]]:
    return {"fleurs": {**{lang: HOUR // 4 for lang in INDIC},
                       "en": round(.6 * HOUR), "de": round(.35 * HOUR), "ja": round(.8 * HOUR)}}


def _take_quota(rows, counts, samples, *, seed, frames, minimum):
    selected, total = [], 0
    for row in _order(rows, seed):
        for window in windows_for(row, counts[row.source_id], scored_frames=frames, minimum=minimum):
            selected.append(window)
            total += window.valid_input_samples16k
            if total >= samples:
                return selected
    raise SamplerExhausted(f"Quota requires {samples} valid input samples but only {total} remain")


def _interleave(queues: Mapping[str, list[PilotWindow]], weights: Mapping[str, int]) -> list[PilotWindow]:
    """Valid-sample weighted finite queues. Exhaustion never changes an unmet quota."""
    pending = {key: deque(value) for key, value in queues.items()}
    emitted = dict.fromkeys(pending, 0)
    result = []
    while pending:
        key = min(pending, key=lambda k: (emitted[k] / weights[k], k))
        value = pending[key].popleft()
        result.append(value)
        emitted[key] += value.valid_input_samples16k
        if not pending[key]:
            del pending[key]
    return result


def select_diagnostic(rows, counts, *, size=16, seed=19, maximum_seconds=12, minimum_samples=20480):
    """Whole short fresh utterances, balanced among available approved pilot buckets."""
    groups = defaultdict(deque)
    for row in _order(rows, seed):
        if minimum_samples <= counts[row.source_id] <= maximum_seconds * RATE:
            groups[row.dataset].append(row)
    result = []
    while groups and len(result) < size:
        for key in sorted(list(groups)):
            if len(result) == size:
                break
            result.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
    if len(result) != size:
        raise ValueError("Not enough wholly untouched short utterances for the diagnostic")
    return result


def plan_pilot(rows: Iterable[ManifestRow], counts: Mapping[str, int], ledger: ConsumedLedger, *,
               quotas: Mapping[str, int] | None = None,
               language_quotas: Mapping[str, Mapping[str, int]] | None = None,
               language_minimums: Mapping[str, Mapping[str, int]] | None = None,
               reserved_rows: Iterable[ManifestRow] = (), seed: int = 19,
               scored_frames: int = 64, minimum: int = MINIMUM, diagnostic_size: int = 16,
               sentinel_size: int = 16, reserve_datasets=("librispeech", "indicvoices")) -> dict:
    rows = sorted(rows, key=lambda r: r.source_id)
    _counts(rows, counts)
    validate_manifest(rows)
    quotas = dict(default_quotas() if quotas is None else quotas)
    language_quotas = dict(default_language_quotas() if language_quotas is None else language_quotas)
    language_minimums = dict({"indicvoices": {lang: 3 * 60 * RATE for lang in INDIC}}
                            if language_minimums is None else language_minimums)
    if not quotas or any(type(v) is not int or v <= 0 for v in quotas.values()):
        raise ValueError("Every source quota must be an explicit positive input-sample count")
    if set(language_quotas) - set(quotas) or set(language_minimums) - set(quotas):
        raise ValueError("Language allocation names a source without a quota")
    for name, allocations in [*language_quotas.items(), *language_minimums.items()]:
        if (not allocations or any(type(v) is not int or v <= 0 for v in allocations.values()) or
                any(normalize_language(k) != k for k in allocations)):
            raise ValueError("Language quotas must use normalized labels and positive sample counts")
        if sum(allocations.values()) > quotas[name]:
            raise ValueError("Language allocations exceed the source quota")
    for name, allocations in language_quotas.items():
        if sum(allocations.values()) != quotas[name] or name in language_minimums:
            raise ValueError("Exact language quotas must sum to the source quota and cannot also have minimums")
    if type(scored_frames) is not int or scored_frames < 1 or not 1 <= minimum <= scored_frames * HOP:
        raise ValueError("Invalid scored window or minimum")
    train = [r for r in rows if r.split == "train"]
    heldout = [r for r in rows if r.split != "train"] + list(reserved_rows)
    dev, test = reserve_unseen(train, ledger, seed=seed, datasets=reserve_datasets)
    heldout.extend(dev + test)
    heldout_keys = set().union(*(identities(row) for row in heldout)) if heldout else set()
    eligible = [r for r in train if ledger.whole_untouched(r) and not identities(r) & heldout_keys
                and r.dataset in quotas]
    diagnostic = select_diagnostic(eligible, counts, size=diagnostic_size, seed=seed) if diagnostic_size else []
    diagnostic_keys = set().union(*(identities(r, people=False) for r in diagnostic)) if diagnostic else set()
    eligible = [r for r in eligible if not identities(r, people=False) & diagnostic_keys]
    # Sentinel speakers come exclusively from the new unexposed reserves; one
    # utterance per real speaker, so all are distinct from diagnostic speakers.
    sentinel = []
    speakers = set()
    sentinel_groups = defaultdict(lambda: defaultdict(deque))
    for row in _order(dev, seed):
        if counts[row.source_id] >= max(minimum, 20480):
            sentinel_groups[row.dataset][normalize_language(row.language)].append(row)
    language_cursor = defaultdict(int)
    # Balance source buckets first, then languages, instead of allowing hundreds
    # of Indic speaker groups to crowd English out of the small sentinel panel.
    while sentinel_groups and len(sentinel) < sentinel_size:
        for dataset in sorted(list(sentinel_groups)):
            if len(sentinel) == sentinel_size:
                break
            languages = sentinel_groups[dataset]
            added = False
            while languages and not added:
                keys = sorted(languages)
                language = keys[language_cursor[dataset] % len(keys)]
                language_cursor[dataset] += 1
                queue = languages[language]
                while queue:
                    row = queue.popleft()
                    if row.speaker_id not in speakers:
                        sentinel.append(row)
                        speakers.add(row.speaker_id)
                        added = True
                        break
                if not queue:
                    del languages[language]
            if not languages:
                del sentinel_groups[dataset]
    if len(sentinel) != sentinel_size:
        raise ValueError("Not enough real unseen-speaker utterances for the sentinel")
    by_source = defaultdict(list)
    for row in eligible:
        by_source[row.dataset].append(row)
    selections = {}
    for dataset, quota in sorted(quotas.items()):
        pool = by_source[dataset]
        specified = language_quotas.get(dataset, language_minimums.get(dataset, {}))
        parts = {}
        for language, samples in sorted(specified.items()):
            try:
                parts[language] = _take_quota([r for r in pool if normalize_language(r.language) == language],
                                              counts, samples, seed=seed, frames=scored_frames, minimum=minimum)
            except SamplerExhausted as error:
                raise SamplerExhausted(f"{dataset}/{language}: {error}") from error
        selected_keys = {(w.source_id, w.start_frame) for values in parts.values() for w in values}
        selected_total = sum(w.valid_input_samples16k for values in parts.values() for w in values)
        remaining = quota - selected_total
        if remaining > 0:
            extra = []
            for row in _order(pool, seed):
                for window in windows_for(row, counts[row.source_id], scored_frames=scored_frames, minimum=minimum):
                    if (window.source_id, window.start_frame) not in selected_keys:
                        extra.append(window)
                        remaining -= window.valid_input_samples16k
                        if remaining <= 0:
                            break
                if remaining <= 0:
                    break
            if remaining > 0:
                raise SamplerExhausted(f"{dataset}: quota short by {remaining} valid input samples; no renormalization")
            parts["__remainder__"] = extra
        weights = {key: sum(w.valid_input_samples16k for w in values) for key, values in parts.items()}
        selections[dataset] = _interleave(parts, weights)
    windows = _interleave(selections, quotas)
    by_id = {row.source_id: row for row in rows}
    verify_windows(windows, by_id, counts, ledger, reserved_rows=heldout, excluded_sources=diagnostic)
    selected_ids = {w.source_id for w in windows}
    pilot_rows = [row for row in rows if row.source_id in selected_ids]
    actual = {name: sum(w.valid_input_samples16k for w in values) for name, values in selections.items()}
    return {"rows": pilot_rows, "windows": windows, "diagnostic": diagnostic,
            "dev_reserve": dev, "test_reserve": test, "sentinel": sentinel,
            "summary": {"format_version": 1, "seed": seed, "ledger_sha256": ledger.identity_sha256,
                        "scored_frames": scored_frames, "minimum_input_samples": minimum,
                        "requested_input_samples": quotas, "actual_input_samples": actual,
                        "language_quotas": language_quotas, "language_minimums": language_minimums,
                        "quota_policy": "minimum valid scored samples; whole final window may overshoot by less than one window per allocation",
                        "selection_policy": "whole untouched utterances; real-speaker reserves first; no replacement or wrapping",
                        "condition_policy": "null means unverified; emotion is not substituted for a vocal event",
                        "diagnostic_policy": "separate whole utterances; repetition is an explicit diagnostic exception; discard fitted weights",
                        "windows": len(windows), "unique_scored_hours": sum(actual.values()) / HOUR,
                        "diagnostic_rows": len(diagnostic), "sentinel_rows": len(sentinel),
                        "dev_reserve_rows": len(dev), "test_reserve_rows": len(test)}}


def verify_windows(windows: Iterable[PilotWindow], rows: Mapping[str, ManifestRow], counts: Mapping[str, int],
                   ledger: ConsumedLedger, *, reserved_rows: Iterable[ManifestRow] = (),
                   excluded_sources: Iterable[ManifestRow] = ()) -> None:
    """Reject overlap with old scored intervals, reservations or another new window."""
    reserved = set().union(*(identities(r) for r in reserved_rows))
    excluded = set().union(*(identities(r, people=False) for r in excluded_sources))
    parents = defaultdict(list)
    hashes = defaultdict(list)
    for window in windows:
        row = rows[window.source_id]
        # Diagnostic and pilot can share a speaker; their complete source files
        # are excluded by the planner, while actual dev/test blocks people too.
        if identities(row) & reserved or identities(row, people=False) & excluded:
            raise ValueError(f"Reserved source/speaker/session in pilot: {row.source_id}")
        start = window.start_frame * HOP
        stop = start + window.valid_input_samples16k
        if (type(window.start_frame) is not int or window.start_frame < 0 or
                type(window.valid_input_samples16k) is not int or window.valid_input_samples16k < MINIMUM or
                stop > counts[row.source_id] or window.scored_frames != math.ceil(window.valid_input_samples16k / HOP) or
                window.dataset != row.dataset or window.language != normalize_language(row.language)):
            raise ValueError("Window shape, length or identity differs from source metadata")
        if ledger.overlaps(row, start, stop):
            raise ValueError(f"Pilot overlaps globally consumed audio: {row.source_id}")
        origin = row.parent_start_seconds * RATE
        parents[row.parent_recording_id].append((math.floor(origin + start), math.ceil(origin + stop)))
        hashes[row.audio_sha256].append((start, stop))
    for index in (parents, hashes):
        for key, intervals in index.items():
            previous_stop = -1
            for start, stop in sorted(intervals):
                if start < previous_stop:
                    raise ValueError(f"Repeated or overlapping planned window: {key}")
                previous_stop = stop


class FixedWindowSampler:
    """Finite ordered plan cursor; callers save state after successful updates only."""
    def __init__(self, windows: Iterable[PilotWindow]):
        self.windows = tuple(windows)
        self.identity_sha256 = digest([w.to_dict() for w in self.windows])
        self.cursor = 0

    @property
    def remaining_segments(self):
        return len(self.windows) - self.cursor

    def take_batch(self, size: int):
        if type(size) is not int or size < 1:
            raise ValueError("Batch size must be positive")
        if size > self.remaining_segments:
            raise SamplerExhausted("Fixed restart plan exhausted; never wraps or renormalizes")
        result = self.windows[self.cursor:self.cursor + size]
        self.cursor += size
        return [w.as_segment() for w in result]

    def state_dict(self):
        return {"format_version": 1, "identity_sha256": self.identity_sha256, "cursor": self.cursor}

    def load_state_dict(self, state):
        if (set(state) != {"format_version", "identity_sha256", "cursor"} or state["format_version"] != 1 or
                state["identity_sha256"] != self.identity_sha256 or type(state["cursor"]) is not int or
                not 0 <= state["cursor"] <= len(self.windows)):
            raise ValueError("Restart plan cursor or identity mismatch")
        self.cursor = state["cursor"]


def write_plan(output: Path, plan: dict, ledger: ConsumedLedger, counts: Mapping[str, int], *, provenance=None):
    """Publish into a new directory; ready.json appears only after all checksums exist."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "ledger.json").write_bytes(canonical(ledger.document))
    for key in ("rows", "diagnostic", "dev_reserve", "test_reserve", "sentinel"):
        name = "train-manifest.jsonl" if key == "rows" else key.replace("_", "-") + ".jsonl"
        (output / name).write_bytes(b"".join(canonical(r.to_dict()) for r in plan[key]))
    (output / "windows.jsonl").write_bytes(b"".join(canonical(w.to_dict()) for w in plan["windows"]))
    needed = {r.source_id for key in ("rows", "diagnostic", "dev_reserve", "test_reserve", "sentinel") for r in plan[key]}
    (output / "input-sample-counts.json").write_bytes(canonical({key: counts[key] for key in sorted(needed)}))
    report = {**plan["summary"], "state": "ready", "provenance": dict(provenance or {}),
              "fixed_sampler_identity": FixedWindowSampler(plan["windows"]).identity_sha256,
              "files_sha256": {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    report["files"] = {name: {"sha256": value} for name, value in report["files_sha256"].items()}
    report["identity_sha256"] = digest(report)
    (output / "ready.json").write_bytes(canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True,
                        help="Existing verified corpus report that pins the exact source manifest")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Existing completed source-training checkpoint, loaded on CPU with weights_only")
    parser.add_argument("--bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True, help="New isolated directory; refuses overwrite")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; previous plans are immutable")
    import torch
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    readiness = json.loads(args.readiness.read_text())
    if (readiness.get("ready") is not True or readiness.get("state") != "ready" or readiness.get("issues") or
            readiness.get("source_manifest_sha256") != file_sha(args.manifest)):
        raise ValueError("Existing readiness does not pin this exact source manifest")
    if checkpoint.get("identity", {}).get("kind") != "source_corpus_one_pass":
        raise ValueError("Only an identified previous source-corpus checkpoint can supply the ledger")
    preparation = checkpoint["identity"].get("preparation", {})
    if (preparation.get("audit_sha256") != file_sha(args.readiness) or
            preparation.get("manifest_sha256") != file_sha(args.manifest)):
        raise ValueError("Checkpoint preparation does not pin the supplied readiness and manifest")
    prior_hashes = readiness.get("prior_training_manifest_sha256", {})
    if file_sha(args.bootstrap_manifest) not in prior_hashes.values():
        raise ValueError("Bootstrap exclusion manifest does not match the previously audited exclusions")
    rows = load_manifest(args.manifest)
    counts = {r.source_id: round(r.duration_seconds * RATE) for r in rows}
    config = checkpoint["identity"]["training_config"]
    loss = checkpoint["identity"]["loss_config"]
    minimum = max(math.ceil(max(loss["teacher_fft_sizes"]) / 3), max(loss["reference_fft_sizes_16k"]))
    sampler_config = {"scored_frames": config["scored_frames"], "seed": config["seed"],
                      "include_short_tail": True, "min_input_samples": minimum}
    origin_step = (checkpoint.get("lineage") or {}).get("checkpoint_step", 0)
    if checkpoint["sampler"]["emitted_segments"] != (checkpoint["step"] - origin_step) * config["batch_size"]:
        raise ValueError("Checkpoint updates and committed sampler exposure disagree")
    provenance = {name: {"path": str(path.resolve()), "sha256": file_sha(path)} for name, path in
                  (("manifest", args.manifest), ("readiness", args.readiness),
                   ("checkpoint", args.checkpoint), ("bootstrap_manifest", args.bootstrap_manifest))}
    provenance["reserved_manifests"] = [{"path": str(p.resolve()), "sha256": file_sha(p)} for p in args.reserved_manifest]
    provenance["planner_code_sha256"] = file_sha(__file__)
    train = [r for r in rows if r.split == "train"]
    ledger = rebuild_ledger(train, {r.source_id: counts[r.source_id] for r in train}, checkpoint["sampler"],
                            sampler_config, load_manifest(args.bootstrap_manifest), provenance=provenance)
    reserved = [r for p in args.reserved_manifest for r in load_manifest(p)]
    plan = plan_pilot(rows, counts, ledger, reserved_rows=reserved)
    print(json.dumps(write_plan(args.output, plan, ledger, counts, provenance=provenance), indent=2))


if __name__ == "__main__":
    main()
