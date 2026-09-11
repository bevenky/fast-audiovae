"""Offline provenance, split and quota checks for decoder training manifests.

No data is downloaded, access terms accepted, or audio decoded by this module.
One JSONL row describes one source segment on its canonical parent timeline.
Parent, speaker and session IDs must be globally scoped by the data preparer,
including across related datasets. Checks cover supplied identities, overlapping
segments and exact file hashes, not acoustic fingerprints or unknown duplicates.
Bandwidth fields record externally established provenance, not measurements
performed by this validator. Never infer native bandwidth from a WAV header.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


DATASET_LICENSES = {
    "librispeech": "CC-BY-4.0",
    "common_voice_scripted_26_en": "CC0-1.0",
    "voxpopuli_en": "CC0-1.0",
    "fleurs": "CC-BY-4.0",
    "indicvoices": "CC-BY-4.0",
    "indicvoices_r": "CC-BY-4.0",
    "hifi_tts": "CC-BY-4.0",
    "vctk_0_92": "CC-BY-4.0",
    "mls_non_english": "CC-BY-4.0",
    "aishell_3": "Apache-2.0",
}
_OFFICIAL_TRAIN_SOURCES = {
    "librispeech", "common_voice_scripted_26_en", "voxpopuli_en", "fleurs",
    "hifi_tts", "mls_non_english", "aishell_3",
}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class ManifestValidationError(ValueError):
    """An invalid row, an excluded source, or a corpus identity collision."""


def _text(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{name} must be a nonempty string")


def _number(value: Any, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ManifestValidationError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class ManifestRow:
    """Required JSONL schema; teacher_cache_key is null before cache preparation.

    audio_sha256 hashes the source file bytes, not normalized PCM. parent_start_seconds
    and duration_seconds identify this segment on the original parent timeline.
    source_revision pins an immutable release, not a floating main/latest alias.
    access_record references a locally recorded source/access-terms review; its
    presence is provenance, not an automated legal approval or acceptance.
    """

    dataset: str
    source_revision: str
    source_id: str
    source_url: str
    audio_path: str
    audio_sha256: str
    parent_recording_id: str
    parent_start_seconds: float
    speaker_id: str | None
    session_id: str | None
    language: str
    sample_rate_hz: int
    original_sample_rate_hz: int
    bandwidth_hz: float
    bandwidth_class: str
    bandwidth_evidence: str
    native_recording: bool
    enhanced: bool
    duration_seconds: float
    split: str
    source_split: str
    license: str
    license_url: str
    attribution: str
    access_record: str
    gain_policy: str
    resampler_policy: str
    teacher_cache_key: str | None

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {
                "dataset", "source_revision", "source_id", "source_url", "audio_path",
                "parent_recording_id", "language", "bandwidth_evidence", "source_split",
                "license", "license_url", "attribution", "access_record", "gain_policy",
                "resampler_policy",
            }:
                _text(getattr(self, field.name), field.name)
        if self.source_revision.casefold() in {"main", "master", "latest", "head"}:
            raise ManifestValidationError("source_revision must pin a release or immutable revision")
        for name in ("speaker_id", "session_id"):
            _text(getattr(self, name), name, nullable=True)
        if self.speaker_id is None and self.session_id is None:
            raise ManifestValidationError("at least one canonical speaker_id or session_id is required")
        for name in ("audio_sha256", "teacher_cache_key"):
            value = getattr(self, name)
            if name == "teacher_cache_key" and value is None:
                continue
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ManifestValidationError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("sample_rate_hz", "original_sample_rate_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ManifestValidationError(f"{name} must be a positive integer")
        _number(self.duration_seconds, "duration_seconds", positive=True)
        _number(self.parent_start_seconds, "parent_start_seconds")
        _number(self.bandwidth_hz, "bandwidth_hz", positive=True)
        if type(self.native_recording) is not bool or type(self.enhanced) is not bool:
            raise ManifestValidationError("native_recording and enhanced must be explicit booleans")
        if self.bandwidth_hz > self.sample_rate_hz / 2:
            raise ManifestValidationError("bandwidth_hz exceeds the current Nyquist limit")
        if not self.enhanced and self.bandwidth_hz > self.original_sample_rate_hz / 2:
            raise ManifestValidationError("bandwidth_hz exceeds the original recording Nyquist limit")
        if self.native_recording and self.enhanced:
            raise ManifestValidationError("enhanced audio cannot be marked as a native recording")
        if self.bandwidth_class not in {"speech_band", "native_fullband", "enhanced"}:
            raise ManifestValidationError("bandwidth_class must be speech_band, native_fullband or enhanced")
        if self.bandwidth_class == "native_fullband":
            if not self.native_recording or self.enhanced or self.original_sample_rate_hz < 44100:
                raise ManifestValidationError("native_fullband requires original 44.1/48 kHz provenance")
            if self.bandwidth_hz <= 8000:
                raise ManifestValidationError("native_fullband requires observed bandwidth above 8 kHz")
        if self.bandwidth_class == "speech_band" and self.bandwidth_hz > 8000:
            raise ManifestValidationError("speech_band supervision must be capped at 8 kHz")
        if (self.bandwidth_class == "enhanced") != self.enhanced:
            raise ManifestValidationError("enhancement status must match the enhanced bandwidth class")
        if self.bandwidth_evidence.casefold().strip() in {"header", "sample_rate", "sample rate", "unknown"}:
            raise ManifestValidationError("bandwidth_evidence must establish more than a sample-rate header")
        if self.split not in {"train", "dev", "test", "regression"}:
            raise ManifestValidationError("split must be train, dev, test or regression")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestRow":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("each manifest row must be an object")
        expected = {field.name for field in fields(cls)}
        missing, extra = expected - value.keys(), value.keys() - expected
        if missing or extra:
            raise ManifestValidationError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Read and validate row schemas. Corpus/split checks require validate_manifest."""
    rows: list[ManifestRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(ManifestRow.from_dict(json.loads(line)))
            except (ManifestValidationError, json.JSONDecodeError) as error:
                raise ManifestValidationError(f"{path}:{line_number}: {error}") from error
    return rows


def _training_policy(row: ManifestRow) -> None:
    provenance = f"{row.dataset} {row.source_url}".casefold()
    if "gigaspeech" in re.sub(r"[^a-z]", "", provenance):
        raise ManifestValidationError("GigaSpeech is excluded from the commercial training corpus")
    if row.dataset not in DATASET_LICENSES:
        raise ManifestValidationError(f"unreviewed training dataset: {row.dataset}")
    if row.license != DATASET_LICENSES[row.dataset]:
        raise ManifestValidationError(f"unexpected data license for {row.dataset}: {row.license}")
    source_split = row.source_split.casefold()
    if any(word in source_split for word in ("test", "dev", "validation")):
        raise ManifestValidationError(f"held-out source partition cannot enter training: {row.source_split}")
    if row.dataset in _OFFICIAL_TRAIN_SOURCES and not source_split.startswith("train"):
        raise ManifestValidationError(f"{row.dataset} training requires an official train source partition")
    if row.dataset == "fleurs" and source_split != "train":
        raise ManifestValidationError("FLEURS must use its train partition exactly")
    language = re.split(r"[-_]", row.language.casefold())[0]
    if row.dataset in {"librispeech", "common_voice_scripted_26_en", "voxpopuli_en", "hifi_tts", "vctk_0_92"} and language != "en":
        raise ManifestValidationError(f"the {row.dataset} allocation requires English recordings")
    if row.dataset == "mls_non_english" and language == "en":
        raise ManifestValidationError("the MLS allocation requires non-English recordings")
    if row.dataset == "aishell_3" and language not in {"zh", "cmn"}:
        raise ManifestValidationError("the AISHELL-3 allocation requires Mandarin recordings")
    if row.dataset == "indicvoices" and row.bandwidth_class != "native_fullband":
        raise ManifestValidationError("the IndicVoices allocation requires verified original fullband recordings")
    if row.dataset == "indicvoices_r" and not row.enhanced:
        raise ManifestValidationError("IndicVoices-R must retain enhanced provenance")
    if row.dataset in {"hifi_tts", "vctk_0_92", "aishell_3"} and row.bandwidth_class != "native_fullband":
        raise ManifestValidationError(f"{row.dataset} allocation requires native fullband references")


def _identities(row: ManifestRow) -> Iterable[tuple[str, str]]:
    yield "parent", row.parent_recording_id
    yield "audio_sha256", row.audio_sha256
    if row.speaker_id is not None:
        yield "speaker", row.speaker_id
    if row.session_id is not None:
        yield "session", row.session_id


def validate_manifest(
    rows: Iterable[ManifestRow], *, reserved_rows: Iterable[ManifestRow] = (),
    training_only: bool = False,
) -> None:
    """Reject split leakage and duplicate/overlapping training source segments.

    reserved_rows is the existing regression/test exclusion manifest. It is always
    treated as reserved, irrespective of any stale split label inside it. This
    function trusts supplied canonical identities; it cannot infer missing source
    relationships, verify licenses, or detect unknown re-encodings by listening.
    """
    rows, reserved_rows = list(rows), list(reserved_rows)
    if not rows:
        raise ManifestValidationError("manifest must contain at least one row")
    owner: dict[tuple[str, str], tuple[str, str]] = {}
    for row in reserved_rows:
        for identity in _identities(row):
            owner[identity] = ("reserved", row.source_id)
    source_ids: set[tuple[str, str, str]] = set()
    train_hashes: dict[str, str] = {}
    train_segments: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        if training_only and row.split != "train":
            raise ManifestValidationError(f"training-only manifest contains {row.split}: {row.source_id}")
        if row.split == "train":
            _training_policy(row)
        source_key = (row.dataset, row.source_revision, row.source_id)
        if source_key in source_ids:
            raise ManifestValidationError(f"duplicate source row: {source_key}")
        source_ids.add(source_key)
        for identity in _identities(row):
            prior = owner.get(identity)
            if prior is not None and prior[0] != row.split:
                raise ManifestValidationError(
                    f"{identity[0]} leakage: {row.source_id} ({row.split}) overlaps "
                    f"{prior[1]} ({prior[0]}) via {identity[1]}"
                )
            owner[identity] = (row.split, row.source_id)
        if row.split == "train":
            prior_hash = train_hashes.get(row.audio_sha256)
            if prior_hash is not None:
                raise ManifestValidationError(f"duplicate training audio file: {prior_hash}, {row.source_id}")
            train_hashes[row.audio_sha256] = row.source_id
            start = row.parent_start_seconds
            train_segments[row.parent_recording_id].append((start, start + row.duration_seconds, row.source_id))
    for parent, segments in train_segments.items():
        segments.sort()
        for previous, current in zip(segments, segments[1:]):
            if current[0] < previous[1] - 1e-9:
                raise ManifestValidationError(
                    f"overlapping training segments in {parent}: {previous[2]}, {current[2]}; "
                    "original/restored and microphone copies cannot count as unique hours"
                )


def build_training_manifest(
    rows: Iterable[ManifestRow], *, reserved_rows: Iterable[ManifestRow] = (),
) -> list[ManifestRow]:
    """Select training rows only after checking the complete input split manifest."""
    rows, reserved_rows = list(rows), list(reserved_rows)
    validate_manifest(rows, reserved_rows=reserved_rows)
    result = [row for row in rows if row.split == "train"]
    validate_manifest(result, reserved_rows=reserved_rows, training_only=True)
    return result


def summarize_manifest(rows: Iterable[ManifestRow]) -> dict[str, Any]:
    """Return audited hours and coverage. No quota or acquisition is implied."""
    rows = list(rows)
    validate_manifest(rows)
    totals: dict[str, dict[str, float]] = {
        key: defaultdict(float) for key in ("dataset", "split", "language", "bandwidth_class")
    }
    for row in rows:
        for key, mapping in totals.items():
            mapping[getattr(row, key)] += row.duration_seconds / 3600
    return {
        "rows": len(rows),
        "hours": sum(row.duration_seconds for row in rows) / 3600,
        "hours_by": {key: dict(sorted(value.items())) for key, value in totals.items()},
        "speakers": len({row.speaker_id for row in rows if row.speaker_id is not None}),
        "sessions": len({row.session_id for row in rows if row.session_id is not None}),
        "cached_teacher_rows": sum(row.teacher_cache_key is not None for row in rows),
        "deduplication_scope": "supplied canonical identities, parent intervals and exact file SHA-256; no acoustic fingerprints",
    }


def validate_mixture(config: Mapping[str, Any]) -> None:
    """Validate the approved source quotas, not actual data availability/rights."""
    if config.get("schema_version") != 1 or config.get("commercial_checkpoint") is not True:
        raise ManifestValidationError("mixture requires schema_version=1 and commercial_checkpoint=true")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ManifestValidationError("mixture sources must be a nonempty list")
    names = [source["dataset"] for source in sources]
    if len(set(names)) != len(names) or set(names) != set(DATASET_LICENSES):
        raise ManifestValidationError("mixture must contain each approved source exactly once")
    for source in sources:
        if source["license"] != DATASET_LICENSES[source["dataset"]]:
            raise ManifestValidationError(f"unexpected mixture license: {source['dataset']}")
        for key in ("pilot_hours", "main_hours"):
            _number(source[key], key, positive=True)
    for stage, expected in (("pilot", 100), ("main", 1000)):
        total = sum(source[f"{stage}_hours"] for source in sources)
        if not math.isclose(total, expected, abs_tol=1e-9):
            raise ManifestValidationError(f"{stage} quotas sum to {total}, expected {expected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "summarize"):
        subparser = commands.add_parser(command)
        subparser.add_argument("manifest", type=Path)
        subparser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
        subparser.add_argument("--training-only", action="store_true")
    mixture = commands.add_parser("validate-mixture")
    mixture.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-mixture":
            with args.config.open(encoding="utf-8") as handle:
                validate_mixture(json.load(handle))
            print(json.dumps({"valid": True, "pilot_hours": 100, "main_hours": 1000}))
            return 0
        rows = load_manifest(args.manifest)
        reserved = [row for path in args.reserved_manifest for row in load_manifest(path)]
        validate_manifest(rows, reserved_rows=reserved, training_only=args.training_only)
        report = summarize_manifest(rows) if args.command == "summarize" else {"valid": True, "rows": len(rows)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ManifestValidationError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
