"""Acquire a small real-speech bootstrap from official, separate source splits.

FLEURS uses only train.tar.gz at a pinned repository revision. LibriSpeech uses
official train-clean-100 and optional dev-clean, with real reader/chapter IDs.
Archives are streamed and stopped at the requested quota, not stored or fully
checksum-verified. Selected original files are SHA-256 checked and kept intact.
This is an archive-order bootstrap, not the final balanced 100-hour pilot.
No dataset loading scripts, access gates, resampling or model code are executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable
import urllib.request

import soundfile as sf

from .data import ManifestRow, load_manifest, summarize_manifest, validate_manifest


FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}"
LIBRI_BASE = "https://www.openslr.org/resources/12"
LIBRI_MD5 = {"train-clean-100": "2a93770f6d5c6c964bc36631d331a522", "dev-clean": "42e2234ba48799c1f50f24a7926300a1"}
DEFAULT_LANGUAGES = ("en_us", "hi_in", "es_419", "pt_br", "fr_fr", "ta_in", "te_in", "bn_in", "mr_in", "gu_in", "de_de", "ar_eg")
_LIBRI_NAME = re.compile(r"^(\d+)-(\d+)-(\d+)\.flac$")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".acquire-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _open_url(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fast-audiovae-data/1"}), timeout=60)


class DownloadBudget:
    def __init__(self, limit_bytes: int):
        if type(limit_bytes) is not int or limit_bytes < 1:
            raise ValueError("download limit must be a positive integer")
        self.limit_bytes = limit_bytes
        self.read_bytes = 0

    def add(self, count: int) -> None:
        self.read_bytes += count
        if self.read_bytes > self.limit_bytes:
            raise ValueError("download byte budget exceeded; completed source manifests remain available")


class _CountedReader:
    def __init__(self, stream: BinaryIO, budget: DownloadBudget):
        self.stream, self.budget = stream, budget
        self.sha256 = hashlib.sha256()
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self.budget.add(len(value))
        self.sha256.update(value)
        self.read_bytes += len(value)
        return value


def _fleurs_catalog(payload: bytes) -> dict[str, dict[str, Any]]:
    result = {}
    for line in csv.reader(io.StringIO(payload.decode("utf-8")), delimiter="\t"):
        if len(line) != 7 or not line[0].isdigit() or not line[5].isdigit():
            raise ValueError("unexpected FLEURS train TSV schema")
        filename = line[1]
        if PurePosixPath(filename).name != filename or not filename.endswith(".wav"):
            raise ValueError("unsafe FLEURS audio filename")
        if filename in result:
            raise ValueError("duplicate FLEURS filename in train metadata")
        result[filename] = {"dataset_id": int(line[0]), "num_samples": int(line[5]), "gender": line[6]}
    if not result:
        raise ValueError("empty FLEURS train catalog")
    return result


def load_evaluation_exclusions(paths: Iterable[str | Path]) -> dict[str, set[Any]]:
    """Read existing legacy evaluation metadata, never its audio files.

    FLEURS dataset IDs are text IDs, not speaker IDs. Excluding them in addition
    to original filenames is deliberately conservative across prompt variants.
    """
    result: dict[str, set[Any]] = {"filenames": set(), "text_ids": set(), "sha256": set()}
    for path in paths:
        value = json.loads(Path(path).read_text())
        samples = value.get("samples") if isinstance(value, dict) else None
        if not isinstance(samples, list):
            raise ValueError(f"unsupported evaluation manifest schema: {path}")
        for sample in samples:
            language = sample.get("language")
            filename = sample.get("original_audio_path")
            if not isinstance(language, str):
                raise ValueError("evaluation sample requires an explicit language")
            if filename:
                result["filenames"].add((language, PurePosixPath(filename).name))
            if sample.get("dataset_id") is not None:
                result["text_ids"].add((language, int(sample["dataset_id"])))
            if sample.get("sha256"):
                result["sha256"].add(sample["sha256"])
            if not filename and sample.get("dataset_id") is None and not sample.get("sha256"):
                raise ValueError("evaluation sample has no usable exclusion identity")
    return result


def _row(dataset: str, partition: str, language: str, filename: str, audio_path: Path,
         payload: bytes, frames: int, source_url: str, access_record: str) -> ManifestRow:
    if dataset == "fleurs":
        speaker = None
        # This is explicitly a conservative grouping, not an invented session ID.
        session = f"fleurs:unknown-session-group:{language}"
        parent = f"fleurs:{language}:audio:{filename}"
        source_id = f"{language}:train:{filename}"
        revision = FLEURS_REVISION
    else:
        match = _LIBRI_NAME.fullmatch(filename)
        if match is None:
            raise ValueError("unexpected LibriSpeech utterance name")
        reader, chapter, utterance = match.groups()
        speaker = f"librispeech:speaker:{reader}"
        session = f"librispeech:chapter:{reader}:{chapter}"
        parent = f"librispeech:utterance:{reader}-{chapter}-{utterance}"
        source_id = filename.removesuffix(".flac")
        revision = f"openslr12:{partition}:md5:{LIBRI_MD5[partition]}"
    return ManifestRow(
        dataset=dataset, source_revision=revision, source_id=source_id, source_url=source_url,
        audio_path=str(audio_path.resolve()), audio_sha256=hashlib.sha256(payload).hexdigest(),
        parent_recording_id=parent, parent_start_seconds=0, speaker_id=speaker, session_id=session,
        language=language, sample_rate_hz=16000, original_sample_rate_hz=16000,
        bandwidth_hz=7600, bandwidth_class="speech_band",
        bandwidth_evidence="Official original 16 kHz speech release; conservative 7.6 kHz loss limit, not a measured highband claim",
        native_recording=True, enhanced=False, duration_seconds=frames / 16000,
        split="dev" if partition == "dev-clean" else "train", source_split=partition,
        license="CC-BY-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="FLEURS: Google Research, Conneau et al." if dataset == "fleurs" else "LibriSpeech: Panayotov, Chen, Povey and Khudanpur; LibriVox recordings",
        access_record=access_record, gain_policy="none: preserve original amplitude",
        resampler_policy="none: original mono 16000 Hz", teacher_cache_key=None,
    )


def collect_archive(
    stream: BinaryIO, *, dataset: str, partition: str, language: str, root: Path,
    seconds: float, source_url: str, budget: DownloadBudget,
    catalog: dict[str, dict[str, Any]] | None = None,
    exclusions: dict[str, set[Any]] | None = None, reserved_rows: Iterable[ManifestRow] = (),
    min_seconds: float = 2, max_seconds: float = 30, speaker_seconds: float = 60,
) -> tuple[list[ManifestRow], dict[str, Any]]:
    """Select complete original utterances while streaming an approved archive."""
    if dataset not in {"fleurs", "librispeech"}:
        raise ValueError("only FLEURS and LibriSpeech acquisition are implemented")
    if (dataset == "fleurs" and partition != "train") or (dataset == "librispeech" and partition not in LIBRI_MD5):
        raise ValueError("unapproved acquisition partition")
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("requested seconds must be positive")
    if dataset == "fleurs" and catalog is None:
        raise ValueError("FLEURS requires pinned train metadata")
    exclusions = exclusions or {"filenames": set(), "text_ids": set(), "sha256": set()}
    reserved_rows = list(reserved_rows)
    reader = _CountedReader(stream, budget)
    rows: list[ManifestRow] = []
    total, by_speaker, skips, by_gender = 0.0, Counter(), Counter(), Counter()
    balance_gender = catalog is not None and {"MALE", "FEMALE"}.issubset({item["gender"] for item in catalog.values()})
    access = str((root / "provenance" / "sources.json").resolve())
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member path")
            if not member.isfile():
                continue
            filename = name.name
            if dataset == "fleurs":
                if filename not in catalog:
                    continue
                info = catalog[filename]
                if ((language, filename) in exclusions["filenames"] or
                        (language, info["dataset_id"]) in exclusions["text_ids"]):
                    skips["evaluation_identity"] += 1
                    continue
                if balance_gender and by_gender[info["gender"]] >= seconds * 0.65:
                    skips["gender_exposure_cap"] += 1
                    continue
            else:
                match = _LIBRI_NAME.fullmatch(filename)
                if match is None:
                    continue
                # Do not trust a filename when an archive member labels another split.
                if partition not in name.parts:
                    raise ValueError("LibriSpeech member partition does not match requested archive")
                if by_speaker[match.group(1)] >= speaker_seconds:
                    skips["speaker_exposure_cap"] += 1
                    continue
            if member.size > 8 * 1024 * 1024:
                skips["oversized_file"] += 1
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("could not read an archive file")
            payload = handle.read()
            if len(payload) != member.size:
                raise ValueError("truncated archive file")
            digest = hashlib.sha256(payload).hexdigest()
            if digest in exclusions["sha256"]:
                skips["evaluation_hash"] += 1
                continue
            audio = sf.info(io.BytesIO(payload))
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError("bootstrap requires original mono 16000 Hz audio")
            duration = audio.frames / 16000
            if not min_seconds <= duration <= max_seconds:
                skips["duration_filter"] += 1
                continue
            if dataset == "fleurs" and audio.frames != catalog[filename]["num_samples"]:
                raise ValueError("FLEURS audio frame count disagrees with pinned train TSV")
            path = root / "audio" / dataset / partition / language / filename
            row = _row(dataset, partition, language, filename, path, payload, audio.frames, source_url, access)
            validate_manifest([row], reserved_rows=reserved_rows)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"existing audio file has different bytes: {path}")
            else:
                _atomic_bytes(path, payload)
            # Verify bytes that subsequent target generation will consume.
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.audio_sha256:
                raise ValueError("written source audio failed SHA-256 verification")
            rows.append(row)
            total += duration
            if dataset == "librispeech":
                by_speaker[_LIBRI_NAME.fullmatch(filename).group(1)] += duration
            else:
                by_gender[catalog[filename]["gender"]] += duration
            if total >= seconds:
                break
    if not rows:
        raise ValueError(f"no eligible audio found for {dataset}/{language}")
    validate_manifest(rows, reserved_rows=reserved_rows)
    receipt = {"dataset": dataset, "partition": partition, "language": language,
               "url": source_url, "selected_seconds": total, "requested_seconds": seconds,
               "rows": len(rows), "downloaded_prefix_bytes": reader.read_bytes,
               "downloaded_prefix_sha256": reader.sha256.hexdigest(),
               "complete_archive_digest_verified": False, "skips": dict(skips),
               "selected_speakers": dict(by_speaker), "selected_gender_seconds": dict(by_gender),
               "selection": "archive order, whole utterances, duration filter; not final corpus balance"}
    return rows, receipt


def acquire_bootstrap(
    root: str | Path, *, fleurs_languages: Iterable[str] = DEFAULT_LANGUAGES,
    minutes_per_language: float = 3, librispeech_train_minutes: float = 8,
    librispeech_dev_minutes: float = 2, reserved_rows: Iterable[ManifestRow] = (),
    evaluation_manifest_paths: Iterable[str | Path] = (), max_download_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    root = Path(root).resolve()
    languages = list(fleurs_languages)
    if len(set(languages)) != len(languages) or any(not re.fullmatch(r"[a-z]{2,3}_[a-z0-9_]+", language) for language in languages):
        raise ValueError("FLEURS languages must be unique official configuration names")
    for value in (minutes_per_language, librispeech_train_minutes, librispeech_dev_minutes):
        if not math.isfinite(value) or value < 0:
            raise ValueError("minute quotas must be finite and nonnegative")
    budget = DownloadBudget(max_download_bytes)
    reserved_rows = list(reserved_rows)
    exclusions = load_evaluation_exclusions(evaluation_manifest_paths)
    rows: list[ManifestRow] = []
    receipts: list[dict[str, Any]] = []

    def publish() -> dict[str, Any]:
        validate_manifest(rows, reserved_rows=reserved_rows)
        for split in ("train", "dev"):
            selected = [row for row in rows if row.split == split]
            payload = "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in selected).encode()
            _atomic_bytes(root / f"{split}.jsonl", payload)
        report = {
            "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_receipts": receipts, "summary": summarize_manifest(rows),
            "downloaded_bytes": budget.read_bytes, "commercial_audio_licenses": ["CC-BY-4.0"],
            "license_evidence": {"fleurs": f"{FLEURS_BASE}/README.md", "librispeech": "https://www.openslr.org/12/"},
            "fleurs_speaker_limit": "No speaker/session IDs are published. speaker_id is null; each language is one conservative unknown-session group. No FLEURS speaker-disjoint development claim.",
            "librispeech_identity_limit": "Reader/chapter IDs are real dataset IDs. Mapping these to shared LibriVox identities is still required before mixing related MLS/Hi-Fi-TTS sources.",
            "bandwidth": "Original 16 kHz recordings only; no native fullband reference or quality claim.",
            "evaluation_exclusions": {key: len(value) for key, value in exclusions.items()},
            "rights": "Public CC-BY-4.0 sources; attribution retained. No gated access or new terms accepted.",
        }
        _atomic_bytes(root / "provenance" / "sources.json", _json_bytes(report))
        print(json.dumps({"completed_source": receipts[-1], "total_minutes": report["summary"]["hours"] * 60}), flush=True)
        return report

    for language in languages if minutes_per_language else ():
        metadata_url = f"{FLEURS_BASE}/data/{language}/train.tsv"
        with _open_url(metadata_url) as response:
            metadata = response.read(16 * 1024 * 1024 + 1)
        budget.add(len(metadata))
        if len(metadata) > 16 * 1024 * 1024:
            raise ValueError("unexpectedly large FLEURS metadata")
        catalog = _fleurs_catalog(metadata)
        _atomic_bytes(root / "provenance" / language / "train.tsv", metadata)
        url = f"{FLEURS_BASE}/data/{language}/audio/train.tar.gz"
        with _open_url(url) as response:
            selected, receipt = collect_archive(response, dataset="fleurs", partition="train", language=language,
                root=root, seconds=minutes_per_language * 60, source_url=url, budget=budget,
                catalog=catalog, exclusions=exclusions, reserved_rows=reserved_rows)
        receipt.update({"revision": FLEURS_REVISION, "metadata_url": metadata_url,
                        "metadata_sha256": hashlib.sha256(metadata).hexdigest()})
        rows.extend(selected)
        receipts.append(receipt)
        publish()
    for partition, minutes in (("train-clean-100", librispeech_train_minutes), ("dev-clean", librispeech_dev_minutes)):
        if not minutes:
            continue
        url = f"{LIBRI_BASE}/{partition}.tar.gz"
        with _open_url(url) as response:
            selected, receipt = collect_archive(response, dataset="librispeech", partition=partition,
                language="en", root=root, seconds=minutes * 60, source_url=url, budget=budget,
                exclusions=exclusions, reserved_rows=reserved_rows, speaker_seconds=60)
        receipt.update({"official_archive_md5": LIBRI_MD5[partition],
                        "archive_checksum_source": f"{LIBRI_BASE}/md5sum.txt"})
        rows.extend(selected)
        receipts.append(receipt)
        publish()
    if not rows:
        raise ValueError("at least one positive acquisition quota is required")
    return publish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fleurs-languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--minutes-per-language", type=float, default=3)
    parser.add_argument("--librispeech-train-minutes", type=float, default=8)
    parser.add_argument("--librispeech-dev-minutes", type=float, default=2)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-evaluation-manifest", type=Path, action="append", default=[])
    parser.add_argument("--max-download-gb", type=float, default=4)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_download_gb) or args.max_download_gb <= 0:
        parser.error("--max-download-gb must be positive")
    reserved = [row for path in args.reserved_manifest for row in load_manifest(path)]
    acquire_bootstrap(args.output_root, fleurs_languages=filter(None, args.fleurs_languages.split(",")),
        minutes_per_language=args.minutes_per_language, librispeech_train_minutes=args.librispeech_train_minutes,
        librispeech_dev_minutes=args.librispeech_dev_minutes, reserved_rows=reserved,
        evaluation_manifest_paths=args.reserved_evaluation_manifest, max_download_bytes=int(args.max_download_gb * 1024**3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
