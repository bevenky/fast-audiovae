"""Verify completed acquisition sources and freeze a train/dev source corpus.

No acquisition, resampling, model inference or training is performed. Hours are
computed from SHA-256 verified mono16k audio frames. A ready manifest and audit
are immutable; incomplete reports cannot authorize training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import soundfile as sf

from .acquire import FLEURS_REVISION, _atomic_bytes, _fleurs_catalog, load_evaluation_exclusions
from .acquire_indic import LANGUAGES
from .data import ManifestRow, validate_manifest


RATE = 16000
_KINDS = {"acquire", "indic", "expressive"}
_ALLOWED = {"acquire": {"librispeech", "fleurs"}, "indic": {"indicvoices"},
            "expressive": {"thorsten_emotional", "jnv", "jvnv", "crema_d", "emogator",
                           "fsd50k_vocal_cc0", "fsd50k_vocal_cc_by_3", "freesound_human_whistle_cc0",
                           "freesound_human_whistle_cc_by_3", "freesound_human_whistle_cc_by_4"}}


class FrozenCorpusError(ValueError):
    """Existing ready artifacts or their pinned inputs changed."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _resolve(value, base):
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _language(value):
    return re.split(r"[-_]", value.casefold())[0]


def _configuration(value):
    return value.casefold().replace("-", "_")


def _source_key(row):
    return row.dataset, row.source_id


def _without_paths(row):
    value = row.to_dict()
    value.pop("audio_path")
    value.pop("access_record")
    return value


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.payloads = {}
        self.objects = {}

    def read(self, path):
        path = Path(path).resolve()
        if str(path) in self.payloads:
            return self.payloads[str(path)]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError("Input metadata changed during assembly")
        self.hashes[str(path)] = digest
        self.payloads[str(path)] = payload
        return payload

    def json(self, path):
        key = str(Path(path).resolve())
        if key not in self.objects:
            self.objects[key] = json.loads(self.read(path))
        return self.objects[key]

    def manifest(self, path):
        path = Path(path).resolve()
        rows = []
        for index, line in enumerate(self.read(path).decode().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = ManifestRow.from_dict(json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid manifest row {path.name}:{index}: {error}") from error
            rows.append(replace(row, audio_path=str(_resolve(row.audio_path, path.parent)),
                                access_record=str(_resolve(row.access_record, path.parent))))
        return rows

    def unchanged(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Input metadata changed during assembly")


def _file_frames(row):
    """Hash and inspect the same open file; never trust a planned duration."""
    with Path(row.audio_path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row.audio_sha256:
            raise ValueError(f"Audio SHA-256 mismatch: {row.source_id}")
        stream.seek(0)
        with sf.SoundFile(stream) as audio:
            frames, channels, rate = int(audio.frames), audio.channels, audio.samplerate
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Audio changed while being verified")
    if rate != RATE or channels != 1 or row.sample_rate_hz != RATE or frames < 1:
        raise ValueError(f"Source must be nonempty prepared mono16k: {row.source_id}")
    if abs(row.duration_seconds * RATE - frames) > 1.00001:
        raise ValueError(f"Declared duration disagrees with actual file frames: {row.source_id}")
    return frames


def _readiness(kind, marker, preparation_version=2):
    if kind == "acquire":
        return marker.get("state") == "ready"
    if kind == "indic":
        return marker.get("status") == "complete"
    return marker.get("state") == "complete" and marker.get("preparation_version") == preparation_version


def _summary_matches(kind, marker, rows, frames):
    summary = marker.get("summary", {}) if kind == "acquire" else marker
    actual_hours = sum(frames[row.source_id] for row in rows) / RATE / 3600
    if summary.get("rows") != len(rows) or not isinstance(summary.get("hours"), (float, int)):
        raise ValueError("Completed source summary row/hour fields do not match its final manifests")
    if abs(float(summary["hours"]) - actual_hours) > 1 / RATE / 3600 * max(1, len(rows)):
        raise ValueError("Completed source summary hours disagree with verified audio")
    split_hours = summary.get("hours_by", {}).get("split")
    if split_hours is not None:
        for split in {row.split for row in rows}:
            actual = sum(frames[row.source_id] for row in rows if row.split == split) / RATE / 3600
            if abs(float(split_hours.get(split, -1)) - actual) > 1 / RATE / 3600 * max(1, len(rows)):
                raise ValueError("Completed source summary split hours disagree with verified audio")


def _rows_bytes(rows):
    return b"".join(_canonical(row.to_dict()) + b"\n" for row in sorted(rows, key=lambda row: row.source_id))


def _frozen_existing(output, config_digest):
    audit_path = output / "readiness.json"
    if not audit_path.exists():
        return None
    existing = json.loads(audit_path.read_bytes())
    if existing.get("state") != "ready":
        return None
    if existing.get("configuration_sha256") != config_digest:
        raise FrozenCorpusError("Ready corpus uses a different config; reuse its final config or choose a new output directory")
    for path, expected in existing["input_metadata_sha256"].items():
        if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise FrozenCorpusError("A ready corpus input manifest or provenance record changed")
    manifest = output / "source-manifest.jsonl"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != existing["source_manifest_sha256"]:
        raise FrozenCorpusError("Ready source manifest changed")
    # Return exactly the stored audit. In particular, never rewrite its timestamp
    # or bytes: training checkpoints bind the complete readiness.json SHA-256.
    return existing


def assemble_corpus(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    if config.get("format_version") != 1:
        raise ValueError("Assembly config requires format_version1")
    config_digest = _hash(config)
    output.mkdir(parents=True, exist_ok=True)
    previous = _frozen_existing(output, config_digest)
    if previous is not None:
        return previous
    if (output / "source-manifest.jsonl").exists():
        raise FrozenCorpusError("An existing source manifest has no matching ready audit; do not overwrite it")
    minimum_hours = config.get("minimum_train_hours", 500)
    indic_hours = config.get("minimum_indic_hours_per_language", 2)
    maximum_seconds = config.get("maximum_utterance_seconds", 180)
    min_fleurs = config.get("minimum_fleurs_configurations", 102)
    for value in (minimum_hours, indic_hours, maximum_seconds):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Hour requirements and maximum utterance length must be positive and finite")
    if type(min_fleurs) is not int or min_fleurs < 1:
        raise ValueError("minimum_fleurs_configurations must be positive")
    specs = config.get("sources", [])
    names = [spec["name"] for spec in specs]
    if not names or len(set(names)) != len(names) or any(spec["kind"] not in _KINDS for spec in specs):
        raise ValueError("Sources require unique names and supported readiness kinds")
    for spec in specs:
        if "preparation_version" in spec and (spec["kind"] != "expressive" or
                type(spec["preparation_version"]) is not int or spec["preparation_version"] < 1):
            raise ValueError("preparation_version must be a positive integer for an expressive source")
    initial = config.get("initial_source_names", ["core", "indic", "expressive"])
    if not initial or set(initial) - set(names):
        raise ValueError("Every initial source name must be listed")
    base, inputs = config_path.parent, Inputs()
    inputs.hashes[str(config_path)] = hashlib.sha256(raw_config).hexdigest()
    report = {"format_version": 1, "state": "pending_sources", "ready": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "configuration_sha256": config_digest,
              "minimum_train_hours": minimum_hours, "maximum_utterance_seconds": maximum_seconds,
              "source_states": {}, "issues": []}
    rows, frames, original_hashes, access_hashes, observed_lengths = [], {}, {}, {}, []
    pending, invalid = False, False
    def issue(source, error):
        report["issues"].append({"source": source, "reason": str(error)})
    def verify_rows(selected):
        if not selected:
            raise ValueError("Completed source has no audio rows")
        validate_manifest(selected)
        for row in selected:
            if row.split not in {"train", "dev"}:
                raise ValueError("Only train/dev rows may enter the assembled source manifest")
            tokens = set(re.findall(r"[a-z]+", row.source_split.casefold()))
            if row.split == "train" and tokens & {"valid", "val", "validation", "dev", "test", "eval", "evaluation"}:
                raise ValueError("Held-out source split cannot be relabeled training")
            if row.source_id in frames:
                raise ValueError("Source IDs must be globally unique across all manifests and revisions")
            count = _file_frames(row)
            observed_lengths.append(count / RATE)
            if count > maximum_seconds * RATE:
                raise ValueError(f"Utterance exceeds maximum_utterance_seconds: {row.source_id}")
            access_path = Path(row.access_record)
            access = inputs.json(access_path)
            if "prepared_audio_sha256" in access and access["prepared_audio_sha256"] != row.audio_sha256:
                raise ValueError("Prepared audio provenance SHA-256 disagrees with the row")
            if "prepared_frames" in access and access["prepared_frames"] != count:
                raise ValueError("Prepared audio provenance frame count disagrees with the file")
            original = access.get("original_audio_sha256", row.audio_sha256)
            if not isinstance(original, str) or not re.fullmatch(r"[a-f0-9]{64}", original):
                raise ValueError("Invalid original audio provenance hash")
            frames[row.source_id] = count
            original_hashes[row.source_id] = original
            access_hashes[row.source_id] = inputs.hashes[str(access_path.resolve())]
        return [replace(row, duration_seconds=frames[row.source_id] / RATE) for row in selected]

    for spec in specs:
        state = {"kind": spec["kind"], "state": "pending"}
        report["source_states"][spec["name"]] = state
        try:
            marker_path = _resolve(spec["readiness"], base)
            if not marker_path.is_file():
                pending = True
                state["reason"] = "completion marker missing"
                continue
            marker = inputs.json(marker_path)
            state["observed_state"] = marker.get("status", marker.get("state"))
            if not _readiness(spec["kind"], marker, spec.get("preparation_version", 2)):
                pending = True
                state["reason"] = "source has not completed its acquisition and validation"
                continue
            selected = [row for path in spec["manifests"] for row in inputs.manifest(_resolve(path, base))]
            if any(row.dataset not in _ALLOWED[spec["kind"]] for row in selected):
                raise ValueError("Manifest dataset does not match source readiness kind")
            selected = verify_rows(selected)
            _summary_matches(spec["kind"], marker, selected, frames)
            rows.extend(selected)
            state.update(state="verified", rows=len(selected),
                         train_hours=sum(frames[row.source_id] for row in selected if row.split == "train") / RATE / 3600)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            invalid = True
            state.update(state="invalid", reason=str(error))
            issue(spec["name"], error)
    try:
        for path in config.get("dev_manifests", []):
            selected = inputs.manifest(_resolve(path, base))
            if any(row.split != "dev" for row in selected):
                raise ValueError("Fixed held-out dev manifests must contain only dev rows")
            rows.extend(verify_rows(selected))
        reserved = [row for path in config.get("reserved_manifests", []) for row in inputs.manifest(_resolve(path, base))]
        prior = [row for path in config.get("prior_training_manifests", []) for row in inputs.manifest(_resolve(path, base))]
        if any(row.split != "train" for row in prior):
            raise ValueError("Prior training exclusion manifests must contain only train rows")
        legacy_paths = [_resolve(path, base) for path in config.get("legacy_evaluation_manifests", [])]
        for path in legacy_paths:
            inputs.read(path)
        legacy = load_evaluation_exclusions(legacy_paths)
        prior_keys, prior_parents = {_source_key(row) for row in prior}, {row.parent_recording_id for row in prior}
        prior_hashes = {row.audio_sha256 for row in prior}
        reserved_keys = {_source_key(row) for row in reserved}
        if rows:
            validate_manifest(rows, reserved_rows=reserved)
        owner = {}
        for row in rows:
            if _source_key(row) in reserved_keys:
                raise ValueError("Reserved source identity cannot enter the assembled corpus")
            if row.split == "train" and (_source_key(row) in prior_keys or row.parent_recording_id in prior_parents
                                         or row.audio_sha256 in prior_hashes or original_hashes[row.source_id] in prior_hashes):
                raise ValueError("Previously used training source/file/parent appears in the fresh corpus")
            if row.audio_sha256 in legacy["sha256"] or original_hashes[row.source_id] in legacy["sha256"]:
                raise ValueError("Known legacy evaluation audio hash appears in the corpus")
            original = original_hashes[row.source_id]
            if original in owner:
                raise ValueError("Duplicate original file or alternate preparation cannot count as unique audio")
            owner[original] = row.source_id
        report["prior_training_exclusion_fingerprint"] = _hash(sorted(
            [{"dataset": row.dataset, "source_id": row.source_id, "parent": row.parent_recording_id,
              "sha256": row.audio_sha256} for row in prior], key=lambda value: (value["dataset"], value["source_id"])))
        report["prior_training_exclusion_rows"] = len(prior)
        report["prior_training_manifest_sha256"] = {str(_resolve(path, base)): inputs.hashes[str(_resolve(path, base))]
                                                   for path in config.get("prior_training_manifests", [])}
        report["legacy_evaluation_exclusion_fingerprint"] = _hash({key: sorted(value) for key, value in legacy.items()})
        report["legacy_evaluation_exclusion_counts"] = {key: len(value) for key, value in legacy.items()}
        report["reserved_manifest_rows"] = len(reserved)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        invalid = True
        issue("exclusions", error)
        prior, legacy = [], {"sha256": set(), "filenames": set(), "text_ids": set()}
    coverage_missing = []
    try:
        plan_path = _resolve(config["fleurs_plan"], base)
        plan = inputs.json(plan_path)
        if plan["configuration"]["fleurs_revision"] != FLEURS_REVISION:
            raise ValueError("FLEURS coverage plan must pin the approved immutable revision")
        expected = plan["configuration"]["languages"]
        if len(set(expected)) != len(expected) or len(expected) < min_fleurs:
            raise ValueError("FLEURS plan does not contain the requested broad configuration coverage")
        actual = Counter()
        catalogs = {}
        legacy_names = {(_configuration(language), name) for language, name in legacy["filenames"]}
        legacy_texts = {(_configuration(language), text_id) for language, text_id in legacy["text_ids"]}
        for row in rows:
            if row.dataset != "fleurs" or row.split != "train":
                continue
            language, filename = _configuration(row.language), Path(row.audio_path).name
            if language not in expected or row.source_revision != FLEURS_REVISION:
                raise ValueError("FLEURS row lies outside the pinned requested coverage plan")
            if language not in catalogs:
                metadata = inputs.read(plan_path.parent / language / "train.tsv")
                if hashlib.sha256(metadata).hexdigest() != plan["metadata_sha256"][language]:
                    raise ValueError("Pinned FLEURS metadata hash changed")
                catalogs[language] = _fleurs_catalog(metadata)
            entry = catalogs[language].get(filename)
            if entry is None or entry["num_samples"] != frames[row.source_id]:
                raise ValueError("FLEURS file frames do not match pinned official train metadata")
            if (language, filename) in legacy_names or (language, entry["dataset_id"]) in legacy_texts:
                raise ValueError("FLEURS legacy evaluation filename or text ID appears in training")
            actual[language] += frames[row.source_id]
        for language in expected:
            requested_seconds = plan.get("language_seconds", {}).get(language, 0)
            if actual[language] < max(1, math.ceil(float(requested_seconds) * RATE - 1e-6)):
                coverage_missing.append({"dataset": "fleurs", "language": language})
        indic = Counter()
        for row in rows:
            if row.dataset == "indicvoices" and row.split == "train":
                indic[_language(row.language)] += frames[row.source_id]
        for language in sorted(set(LANGUAGES.values())):
            if indic[language] < math.ceil(indic_hours * 3600 * RATE - 1e-6):
                coverage_missing.append({"dataset": "indicvoices", "language": language})
        report["coverage"] = {"required_fleurs_configurations": len(expected),
                              "verified_fleurs_configurations": len([value for value in actual.values() if value]),
                              "required_indic_languages": 22, "verified_indic_languages": len([value for value in indic.values() if value]),
                              "minimum_indic_hours_per_language": indic_hours, "missing": coverage_missing}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        invalid = True
        issue("coverage", error)
    train, dev = [row for row in rows if row.split == "train"], [row for row in rows if row.split == "dev"]
    train_frames = sum(frames[row.source_id] for row in train)
    train_hours = train_frames / RATE / 3600
    remaining = max(0, minimum_hours - train_hours)
    initial_complete = all(report["source_states"][name]["state"] == "verified" for name in initial)
    if not dev:
        invalid = True
        issue("dev", "A nonempty verified held-out dev split is required")
    try:
        inputs.unchanged()
    except (ValueError, OSError) as error:
        invalid = True
        issue("input_snapshot", error)
    state = ("invalid_sources" if invalid else "pending_sources" if pending else
             "insufficient_coverage" if coverage_missing else "insufficient_hours" if remaining > 1e-12 else "ready")
    dataset_frames, language_frames = Counter(), Counter()
    for row in train:
        dataset_frames[row.dataset] += frames[row.source_id]
        language_frames[_language(row.language)] += frames[row.source_id]
    metadata_fingerprint = _hash([{"source": _without_paths(row), "input_frames": frames[row.source_id],
                                   "access_record_sha256": access_hashes[row.source_id]}
                                  for row in sorted(rows, key=lambda row: row.source_id)])
    report.update(state=state, ready=state == "ready", verified_train_rows=len(train), verified_dev_rows=len(dev),
                  verified_train_hours=train_hours, verified_dev_hours=sum(frames[row.source_id] for row in dev) / RATE / 3600,
                  actual_maximum_utterance_seconds=max(observed_lengths, default=0),
                  remaining_train_hours=remaining, initial_sources_complete=initial_complete,
                  filler_permitted=initial_complete and not pending and not invalid and not coverage_missing and remaining > 0,
                  train_hours_by_dataset={key: value / RATE / 3600 for key, value in sorted(dataset_frames.items())},
                  train_hours_by_normalized_language={key: value / RATE / 3600 for key, value in sorted(language_frames.items())},
                  metadata_fingerprint=metadata_fingerprint, input_metadata_sha256=dict(sorted(inputs.hashes.items())),
                  verification="SHA-256 verified file bytes plus actual mono16k audio header frame counts; no resampling or duration truncation",
                  exclusion_scope="Canonical source/file/parent identities and train/dev speaker/session separation; known original/prepared hashes and FLEURS legacy filenames/text IDs; no acoustic/fuzzy deduplication claim")
    if not invalid:
        _atomic_bytes(output / "candidate-train.jsonl", _rows_bytes(train))
        _atomic_bytes(output / "candidate-dev.jsonl", _rows_bytes(dev))
        exclusions = {_source_key(row): row for row in prior}
        exclusions.update({_source_key(row): row for row in train})
        _atomic_bytes(output / "filler-exclusion.jsonl", _rows_bytes(list(exclusions.values())))
        report["filler_exclusion_manifest"] = str(output / "filler-exclusion.jsonl")
        report["filler_exclusion_manifest_sha256"] = hashlib.sha256((output / "filler-exclusion.jsonl").read_bytes()).hexdigest()
    report["filler_recommendation"] = {
        "status": "eligible" if report["filler_permitted"] else "wait_for_all_initial_sources_and_valid_counts",
        "source": "librispeech/train-other-500", "fresh_minutes_required": math.ceil(remaining * 60),
        "method": "Run acquire.py in a new filler root with FLEURS/train-clean/dev quotas0, train-other minutes at least this shortfall, exclude filler-exclusion.jsonl, reserve candidate-dev.jsonl plus external reserved manifests and legacy evaluation manifests. Add its final ready source as kind=acquire, then rerun this gate. Acquisition quotas never count as acquired hours.",
    }
    if state == "ready":
        payload = _rows_bytes(rows)
        _atomic_bytes(output / "source-manifest.jsonl", payload)
        report["source_manifest"] = str(output / "source-manifest.jsonl")
        report["source_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    _atomic_bytes(output / "readiness.json", _canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = assemble_corpus(args.config, args.output_dir)
    print(json.dumps({key: report[key] for key in ("state", "verified_train_hours", "remaining_train_hours",
                                                  "initial_sources_complete", "filler_permitted")}, sort_keys=True))
    return 0 if report["state"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
