"""Deterministic speech windows without replacement, repeated epochs, or padding.

Counts refer to actual audio samples after the teacher's 16 kHz input conversion.
Each utterance contributes floor(samples / (640 * scored_frames)) disjoint scored
windows. By default, the remainder is excluded. An explicit tail option retains
remainders long enough for the requested loss. A caller may prepend causal
context to each window, but that context is not additional unique scored audio.

Languages cycle evenly while they have data. Within each language, datasets
cycle at utterance boundaries, and every window of the current utterance is
used in order before selecting another utterance. This favors a rolling teacher
cache. Dataset balance is by utterance, not an assertion of equal audio hours.
Deduplication relies on the supplied canonical identities and hashes; unknown
re-encodings cannot be detected by this sampler.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

from .data import ManifestRow, validate_manifest


INPUT_SAMPLE_RATE = 16000
INPUT_SAMPLES_PER_FRAME = 640
POLICY = "language_cycle_dataset_utterance_cycle_no_replacement_v1"

_LANGUAGE_ALIASES = {
    "en": ("eng", "english"),
    "as": ("asm", "assamese"),
    "bn": ("ben", "bengali", "bangla"),
    "brx": ("bodo", "boro"),
    "doi": ("dogri",),
    "gu": ("guj", "gujarati"),
    "hi": ("hin", "hindi"),
    "kn": ("kan", "kannada"),
    "ks": ("kas", "kashmiri"),
    "kok": ("konkani",),
    "mai": ("maithili",),
    "ml": ("mal", "malayalam"),
    "mni": ("manipuri", "meitei", "meetei"),
    "mr": ("mar", "marathi"),
    "ne": ("nep", "npi", "nepali"),
    "or": ("ori", "ory", "odia", "oriya"),
    "pa": ("pan", "punjabi", "panjabi"),
    "sa": ("san", "sanskrit"),
    "sat": ("santali", "santhali"),
    "sd": ("snd", "sindhi"),
    "ta": ("tam", "tamil"),
    "te": ("tel", "telugu"),
    "ur": ("urd", "urdu"),
}
_CANONICAL_LANGUAGE = {alias: canonical for canonical, aliases in _LANGUAGE_ALIASES.items()
                       for alias in (canonical, *aliases)}


def normalize_language(language: str) -> str:
    """Normalize known language aliases and remove BCP-47 region/script suffixes.

    Unknown primary codes remain distinct, including languages outside the 22
    scheduled Indian languages. No dialect-to-macrolanguage guessing is applied.
    """
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a nonempty string")
    primary = language.strip().casefold().replace("_", "-").split("-", 1)[0]
    return _CANONICAL_LANGUAGE.get(primary, primary)


class SamplerExhausted(RuntimeError):
    """The requested batch exceeds the remaining unique scored windows."""


@dataclass(frozen=True)
class ScoredSegment:
    row_id: str
    start_frame: int
    scored_frames: int
    language: str
    dataset: str
    valid_input_samples: int

    @property
    def input_samples(self) -> int:
        """Number of scored 16 kHz samples, excluding any causal context."""
        return self.valid_input_samples

    @property
    def start_input_sample(self) -> int:
        return self.start_frame * INPUT_SAMPLES_PER_FRAME

    @property
    def stop_input_sample(self) -> int:
        return self.start_input_sample + self.input_samples

    @property
    def valid_output_samples(self) -> int:
        return self.valid_input_samples * 3


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class NoRepeatSegmentSampler:
    """Cycle through a finite corpus once, with an atomic full-batch request.

    source_id must be globally unique within rows. input_sample_counts maps
    those IDs to actual post-conversion 16 kHz sample counts, not an estimate
    from manifest duration. Reserved rows are checked before building queues.
    Pass the already filtered training manifest; this does not acquire data or
    silently remove duplicates, held-out examples, or previous training audio.
    include_short_tail retains short expressive clips and utterance remainders.
    Its default minimum of 1366 input samples gives 4098 valid output samples,
    sufficient for a 4096-sample teacher FFT. Callers using a larger loss window
    must raise the minimum explicitly. Partial latent-frame padding is unscored.
    """

    def __init__(self, rows: Iterable[ManifestRow], input_sample_counts: Mapping[str, int], *,
                 scored_frames: int = 64, seed: int = 7,
                 reserved_rows: Iterable[ManifestRow] = (), include_short_tail: bool = False,
                 min_input_samples: int = 1366):
        _positive_int(scored_frames, "scored_frames")
        _positive_int(min_input_samples, "min_input_samples")
        if type(include_short_tail) is not bool:
            raise ValueError("include_short_tail must be an explicit boolean")
        if include_short_tail and min_input_samples > scored_frames * INPUT_SAMPLES_PER_FRAME:
            raise ValueError("min_input_samples exceeds the configured scored window")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        rows = list(rows)
        # Normalize before source-language policy checks as well as balancing.
        # For example, English aliases must not bypass MLS's non-English rule.
        validate_manifest((replace(row, language=normalize_language(row.language)) for row in rows),
                          reserved_rows=reserved_rows, training_only=True)
        self._rows = {row.source_id: row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("Sampler source_id must be globally unique")
        if set(input_sample_counts) != set(self._rows):
            raise ValueError("input_sample_counts must match the exact training row IDs")
        for row_id, count in input_sample_counts.items():
            _positive_int(count, f"input_sample_counts[{row_id!r}]")
        self.scored_frames, self.seed = scored_frames, seed
        self.include_short_tail, self.min_input_samples = include_short_tail, min_input_samples
        self._samples = dict(input_sample_counts)
        self._window_samples = scored_frames * INPUT_SAMPLES_PER_FRAME
        self._counts = {}
        self._scored_samples = {}
        for row_id, count in self._samples.items():
            full, tail = divmod(count, self._window_samples)
            accepted_tail = tail if include_short_tail and tail >= min_input_samples else 0
            self._counts[row_id] = full + int(accepted_tail > 0)
            self._scored_samples[row_id] = full * self._window_samples + accepted_tail
        self._languages = {row_id: normalize_language(row.language) for row_id, row in self._rows.items()}
        self.total_segments = sum(self._counts.values())
        self.total_scored_input_samples = sum(self._scored_samples.values())
        self._validate_scored_parent_intervals()

        # Shuffle utterances, not windows. No random draws occur after queue
        # construction, so checkpoint cursors are sufficient to replay exactly.
        groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for row_id in sorted(self._rows):
            if self._counts[row_id]:
                groups[self._languages[row_id]][self._rows[row_id].dataset].append(row_id)
        rng = random.Random(seed)
        self._queues: dict[str, list[str]] = {}
        for language in sorted(groups):
            sources = [groups[language][dataset] for dataset in sorted(groups[language])]
            for source in sources:
                rng.shuffle(source)
            queue: list[str] = []
            # Cycle datasets until each finite utterance queue is exhausted.
            for position in range(max(map(len, sources))):
                queue.extend(source[position] for source in sources if position < len(source))
            self._queues[language] = queue
        self._language_order = tuple(self._queues)
        self._prefix_counts: dict[str, list[int]] = {}
        self._prefix_samples: dict[str, list[int]] = {}
        for language, queue in self._queues.items():
            prefixes, samples = [0], [0]
            for row_id in queue:
                prefixes.append(prefixes[-1] + self._counts[row_id])
                samples.append(samples[-1] + self._scored_samples[row_id])
            self._prefix_counts[language] = prefixes
            self._prefix_samples[language] = samples
        # Metadata changes that affect eligibility or provenance are guarded;
        # relocating source files and adding cache paths do not change windows.
        digest = hashlib.sha256()
        digest.update(_canonical_bytes({"format_version": 1, "policy": POLICY, "seed": seed,
                                         "scored_frames": scored_frames, "input_sample_rate": INPUT_SAMPLE_RATE,
                                         "include_short_tail": include_short_tail, "min_input_samples": min_input_samples,
                                         "input_samples_per_frame": INPUT_SAMPLES_PER_FRAME, "queues": self._queues}))
        for row_id in sorted(self._rows):
            row_identity = self._rows[row_id].to_dict()
            for field in ("audio_path", "access_record", "teacher_cache_key"):
                row_identity.pop(field)
            digest.update(_canonical_bytes({"row": row_identity, "input_samples": self._samples[row_id]}))
        self.identity_sha256 = digest.hexdigest()
        self._cursors = {language: [0, 0] for language in self._language_order}
        self._next_language = 0
        self.emitted_segments = 0
        self.emitted_input_samples = 0

    def _validate_scored_parent_intervals(self) -> None:
        # Manifest duration is provenance, while these intervals use supplied
        # actual sample counts. Catch overlap even if duration metadata is short.
        parents: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for row_id, row in self._rows.items():
            if self._counts[row_id]:
                seconds = self._scored_samples[row_id] / INPUT_SAMPLE_RATE
                parents[row.parent_recording_id].append((row.parent_start_seconds,
                                                         row.parent_start_seconds + seconds, row_id))
        for parent, segments in parents.items():
            segments.sort()
            for previous, current in zip(segments, segments[1:]):
                if current[0] < previous[1] - 1e-9:
                    raise ValueError(f"Actual scored windows overlap in parent {parent}: "
                                     f"{previous[2]}, {current[2]}")

    @property
    def remaining_segments(self) -> int:
        return self.total_segments - self.emitted_segments

    def next_segment(self) -> ScoredSegment:
        if self.remaining_segments == 0:
            raise SamplerExhausted("No unique scored windows remain; this sampler never starts a new epoch")
        for _ in self._language_order:
            language = self._language_order[self._next_language]
            self._next_language = (self._next_language + 1) % len(self._language_order)
            utterance, segment = self._cursors[language]
            if utterance == len(self._queues[language]):
                continue
            row_id = self._queues[language][utterance]
            valid = min(self._window_samples, self._samples[row_id] - segment * self._window_samples)
            frames = (valid + INPUT_SAMPLES_PER_FRAME - 1) // INPUT_SAMPLES_PER_FRAME
            result = ScoredSegment(row_id, segment * self.scored_frames, frames,
                                    language, self._rows[row_id].dataset, valid)
            segment += 1
            if segment == self._counts[row_id]:
                utterance, segment = utterance + 1, 0
            self._cursors[language] = [utterance, segment]
            self.emitted_segments += 1
            self.emitted_input_samples += valid
            return result
        raise RuntimeError("Sampler cursors disagree with the remaining segment count")

    def take_batch(self, size: int) -> list[ScoredSegment]:
        """Return exactly size unique windows or leave state unchanged and fail.

        A training checkpoint should persist this state only after that batch's
        optimizer update completes. Prefetched batches must not move the saved
        consumption boundary ahead of committed updates.
        """
        _positive_int(size, "batch size")
        if size > self.remaining_segments:
            raise SamplerExhausted(f"Requested {size} windows but only {self.remaining_segments} unique windows remain")
        return [self.next_segment() for _ in range(size)]

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, "identity_sha256": self.identity_sha256,
                "emitted_segments": self.emitted_segments, "next_language": self._next_language,
                "emitted_input_samples": self.emitted_input_samples,
                "cursors": {language: list(cursor) for language, cursor in self._cursors.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate the whole state before mutating any sampler cursor."""
        expected = {"format_version", "identity_sha256", "emitted_segments", "emitted_input_samples", "next_language", "cursors"}
        if set(state) != expected or state.get("format_version") != 1 or state.get("identity_sha256") != self.identity_sha256:
            raise ValueError("Sampler checkpoint corpus/order/config identity mismatch")
        next_language, emitted, cursors = state["next_language"], state["emitted_segments"], state["cursors"]
        emitted_samples = state["emitted_input_samples"]
        if (type(next_language) is not int or not 0 <= next_language < max(1, len(self._language_order)) or
                type(emitted) is not int or not 0 <= emitted <= self.total_segments or
                type(emitted_samples) is not int or not 0 <= emitted_samples <= self.total_scored_input_samples or
                not isinstance(cursors, dict) or set(cursors) != set(self._queues)):
            raise ValueError("Invalid sampler checkpoint counters")
        checked: dict[str, list[int]] = {}
        consumed, consumed_samples = 0, 0
        for language, cursor in cursors.items():
            if not isinstance(cursor, (list, tuple)) or len(cursor) != 2 or any(type(value) is not int for value in cursor):
                raise ValueError("Invalid sampler checkpoint cursor")
            utterance, segment = cursor
            queue = self._queues[language]
            if not 0 <= utterance <= len(queue):
                raise ValueError("Invalid sampler checkpoint utterance cursor")
            limit = self._counts[queue[utterance]] if utterance < len(queue) else 1
            if not 0 <= segment < limit or (utterance == len(queue) and segment != 0):
                raise ValueError("Invalid sampler checkpoint segment cursor")
            consumed += self._prefix_counts[language][utterance] + segment
            consumed_samples += self._prefix_samples[language][utterance] + segment * self._window_samples
            checked[language] = [utterance, segment]
        if consumed != emitted or consumed_samples != emitted_samples:
            raise ValueError("Sampler checkpoint exposure does not match its cursors")
        self._next_language, self.emitted_segments, self._cursors = next_language, emitted, checked
        self.emitted_input_samples = emitted_samples

    def inventory(self) -> dict[str, Any]:
        """Report actual available scored windows and excluded tails, not quotas."""
        def totals(row_ids):
            row_ids = list(row_ids)
            segments = sum(self._counts[row_id] for row_id in row_ids)
            samples = sum(self._samples[row_id] for row_id in row_ids)
            scored = sum(self._scored_samples[row_id] for row_id in row_ids)
            full_segments = sum(self._samples[row_id] // self._window_samples for row_id in row_ids)
            return {"utterances": len(row_ids), "usable_utterances": sum(self._counts[row_id] > 0 for row_id in row_ids),
                    "segments": segments, "input_samples": samples, "unique_scored_input_samples": scored,
                    "full_segments": full_segments, "tail_segments": segments - full_segments,
                    "full_window_equivalents": scored / self._window_samples,
                    "excluded_tail_input_samples": samples - scored,
                    "source_audio_hours": samples / INPUT_SAMPLE_RATE / 3600,
                    "unique_scored_audio_hours": scored / INPUT_SAMPLE_RATE / 3600,
                    "excluded_tail_audio_hours": (samples - scored) / INPUT_SAMPLE_RATE / 3600}

        language_rows: dict[str, list[str]] = defaultdict(list)
        dataset_rows: dict[str, list[str]] = defaultdict(list)
        for row_id, row in self._rows.items():
            language_rows[self._languages[row_id]].append(row_id)
            dataset_rows[row.dataset].append(row_id)
        return {"policy": POLICY, "identity_sha256": self.identity_sha256,
                "input_sample_rate": INPUT_SAMPLE_RATE, "input_samples_per_frame": INPUT_SAMPLES_PER_FRAME,
                "scored_frames": self.scored_frames, "segment_seconds": self._window_samples / INPUT_SAMPLE_RATE,
                "formula": "sum(floor(N / W) + int(include_short_tail and N % W >= min_input_samples)); W = 640 * scored_frames",
                "include_short_tail": self.include_short_tail, "min_input_samples": self.min_input_samples,
                "tail_policy": ("retain qualifying tails using actual valid samples; padding is never scored" if self.include_short_tail
                                else "exclude all incomplete scored windows; never pad them into training targets"),
                "context_policy": "causal context may overlap; it is excluded from unique scored audio totals",
                **totals(self._rows), "emitted_segments": self.emitted_segments,
                "emitted_input_samples": self.emitted_input_samples,
                "emitted_scored_audio_hours": self.emitted_input_samples / INPUT_SAMPLE_RATE / 3600,
                "remaining_scored_input_samples": self.total_scored_input_samples - self.emitted_input_samples,
                "remaining_segments": self.remaining_segments,
                "by_language": {key: totals(value) for key, value in sorted(language_rows.items())},
                "by_dataset": {key: totals(value) for key, value in sorted(dataset_rows.items())}}
