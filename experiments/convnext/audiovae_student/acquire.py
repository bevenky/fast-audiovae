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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import threading
import time
from typing import Any, BinaryIO, Iterable
import urllib.request

import soundfile as sf

from .data import ManifestRow, load_manifest, summarize_manifest, validate_manifest


FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}"
LIBRI_BASE = "https://www.openslr.org/resources/12"
LIBRI_MD5 = {"train-clean-100": "2a93770f6d5c6c964bc36631d331a522", "train-other-500": "d1a0fd59409feb2c614ce4d30c387708", "dev-clean": "42e2234ba48799c1f50f24a7926300a1"}
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
    def __init__(self, limit_bytes: int, initial_bytes: int = 0):
        if type(limit_bytes) is not int or limit_bytes < 1:
            raise ValueError("download limit must be a positive integer")
        if type(initial_bytes) is not int or initial_bytes < 0 or initial_bytes > limit_bytes:
            raise ValueError("previous transfer exceeds the download byte budget")
        self.limit_bytes = limit_bytes
        self.read_bytes = initial_bytes
        self._reserved_bytes = 0
        self._lock = threading.Lock()

    def add(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("download byte count must be nonnegative")
        with self._lock:
            self.read_bytes += count
            if self.read_bytes > self.limit_bytes:
                raise ValueError("download byte budget exceeded; completed source manifests remain available")

    def read(self, stream: BinaryIO, size: int) -> bytes:
        """Reserve each read before network I/O so concurrent reads share a cap."""
        with self._lock:
            available = self.limit_bytes - self.read_bytes - self._reserved_bytes
            if available <= 0:
                raise ValueError("download byte budget exceeded; completed source manifests remain available")
            count = min(available, size if size >= 0 else 64 * 1024)
            self._reserved_bytes += count
        try:
            value = stream.read(count)
        except BaseException:
            with self._lock:
                self._reserved_bytes -= count
            raise
        with self._lock:
            self._reserved_bytes -= count
            self.read_bytes += len(value)
        return value


class AudioStorageBudget:
    """Bound only this task's audio directory, while preserving filesystem reserve."""

    def __init__(self, root: Path, *, max_bytes: int | None = None, min_free_bytes: int = 0):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max audio bytes must be positive")
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("filesystem reserve must be nonnegative")
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        self._lock = threading.RLock()
        self.used_bytes = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        self.check(0)

    def check(self, added_bytes: int) -> None:
        with self._lock:
            if self.max_bytes is not None and self.used_bytes + added_bytes > self.max_bytes:
                raise ValueError("audio storage cap reached; no existing files were deleted")
            if shutil.disk_usage(self.root).free - added_bytes < self.min_free_bytes:
                raise ValueError("audio filesystem reserve would be crossed; no existing files were deleted")

    def add(self, count: int) -> None:
        with self._lock:
            self.used_bytes += count

    def store(self, path: Path, payload: bytes, digest: str) -> None:
        """Serialize only local check/write/accounting, never archive downloads."""
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("audio storage path is outside the bounded root")
        with self._lock:
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"existing audio file has different bytes: {path}")
                return
            self.check(len(payload))
            _atomic_bytes(path, payload)
            self.used_bytes += len(payload)


class _AcquisitionCancelled(RuntimeError):
    pass


def _run_source_tasks(tasks, acquire_one, record_source, *, workers: int,
                      stop_event: threading.Event) -> None:
    """Bound in-flight archives; only the calling thread publishes results.

    A failed source stops new work. Completed siblings are still checkpointed,
    while partially read archives stop at the next read/member boundary.
    """
    pending = iter(enumerate(tasks))
    failure = None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio-source") as pool:
        futures = {}

        def submit_next():
            item = next(pending, None)
            if item is not None:
                index, task = item
                futures[pool.submit(acquire_one, task)] = index

        for _ in range(workers):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(finished, key=futures.__getitem__):
                futures.pop(future)
                try:
                    selected, receipt = future.result()
                    record_source(selected, receipt)
                except Exception as error:
                    if failure is None:
                        failure = error
                    stop_event.set()
                # Drain completed results even after a sibling fails. Do not
                # start another archive until all completions are inspected.
            if failure is None:
                for _ in range(workers - len(futures)):
                    submit_next()
    if failure is not None:
        raise failure


def discover_fleurs_languages() -> list[str]:
    """Discover every configuration from the pinned official data directory."""
    url = f"https://huggingface.co/api/datasets/google/fleurs/tree/{FLEURS_REVISION}/data?limit=1000"
    with _open_url(url) as response:
        value = json.load(response)
    languages = sorted(item["path"].removeprefix("data/") for item in value if item.get("type") == "directory")
    if len(languages) != 102 or len(set(languages)) != 102 or any(not re.fullmatch(r"[a-z]{2,3}_[a-z0-9_]+", language) for language in languages):
        raise ValueError("pinned FLEURS inventory must contain exactly 102 valid configurations")
    return languages


def _used_training_exclusions(rows: Iterable[ManifestRow]) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {"source_ids": set(), "parents": set(), "sha256": set(), "filenames": set()}
    for row in rows:
        if row.split != "train":
            raise ValueError("prior-training exclusion manifest must contain only train rows; use reserved manifests for held-out data")
        result["source_ids"].add((row.dataset, row.source_id))
        result["parents"].add(row.parent_recording_id)
        result["sha256"].add(row.audio_sha256)
        if row.dataset == "fleurs":
            result["filenames"].add((row.language, Path(row.audio_path).name))
    return result


def nonoverlapping_window_inventory(rows: Iterable[ManifestRow], scored_frames: int = 64) -> dict[str, Any]:
    """Count disjoint scored windows at starts 0,S,2S,... within each utterance.

    Causal context may precede a window but is never counted as new scored data.
    This is a capacity inventory. The training sampler must enforce one-use
    window IDs and checkpoint its cursor; counting alone does not enforce it.
    """
    if type(scored_frames) is not int or scored_frames < 1:
        raise ValueError("scored_frames must be positive")
    rows = [row for row in rows if row.split == "train"]
    if rows:
        validate_manifest(rows, training_only=True)
    samples_per_window = scored_frames * 640
    by_dataset, by_language, entries = Counter(), Counter(), []
    for row in rows:
        if row.sample_rate_hz != 16000:
            raise ValueError("unique-window accounting requires prepared original 16 kHz source files")
        samples = round(row.duration_seconds * 16000)
        windows = samples // samples_per_window
        by_dataset[row.dataset] += windows
        by_language[row.language] += windows
        entries.append({"dataset": row.dataset, "source_id": row.source_id,
                        "audio_sha256": row.audio_sha256, "windows": windows})
    total = sum(by_dataset.values())
    return {"scored_frames": scored_frames, "window_seconds": samples_per_window / 16000,
            "windows": total, "unique_scored_hours": total * samples_per_window / 16000 / 3600,
            "by_dataset": dict(by_dataset), "by_language": dict(by_language),
            "inventory_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
            "policy": "full windows at nonoverlapping frame starts 0,S,2S; context and leftover tails excluded; sampler must enforce no reuse"}


def allocate_language_seconds(capacities: dict[str, float], target_seconds: float,
                              minimum_seconds: float = 1800) -> dict[str, float]:
    """Water-fill actual eligible capacities while retaining all requested languages."""
    if not capacities or not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("positive target and language capacities are required")
    if not math.isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("minimum language seconds must be nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in capacities.values()):
        raise ValueError("every requested language must have eligible fresh audio")
    floors = {language: min(capacity, minimum_seconds) for language, capacity in capacities.items()}
    if target_seconds + 1e-6 < sum(floors.values()):
        raise ValueError("target is below the requested minimum language coverage")
    if target_seconds > sum(capacities.values()) + 1e-6:
        raise ValueError("insufficient eligible fresh FLEURS duration for the requested target")
    result = dict(floors)
    remaining = target_seconds - sum(result.values())
    while remaining > 1e-6:
        active = [language for language in capacities if capacities[language] - result[language] > 1e-6]
        if not active:
            raise ValueError("language allocation exhausted capacity")
        share = remaining / len(active)
        allocated = 0.0
        for language in active:
            value = min(share, capacities[language] - result[language])
            result[language] += value
            allocated += value
        remaining -= allocated
    return result


def _catalog_capacity(catalog: dict[str, dict[str, Any]], language: str,
                      exclusions: dict[str, set[Any]], used: dict[str, set[Any]]) -> float:
    by_gender = Counter()
    for filename, item in catalog.items():
        duration = item["num_samples"] / 16000
        if not 2 <= duration <= 30:
            continue
        if ((language, filename) in exclusions["filenames"] or
                (language, item["dataset_id"]) in exclusions["text_ids"] or
                (language, filename) in used["filenames"]):
            continue
        by_gender[item["gender"]] += duration
    total = sum(by_gender.values())
    if {"MALE", "FEMALE"}.issubset(by_gender):
        # A 65% exposure cap needs enough audio outside each gender category.
        total = min(total, *((sum(by_gender.values()) - amount) / 0.35 for amount in by_gender.values()))
    return total * 0.99  # Reserve for whole-utterance selection and exclusions by file hash.


class _CountedReader:
    def __init__(self, stream: BinaryIO, budget: DownloadBudget, stop_event: threading.Event | None = None):
        self.stream, self.budget = stream, budget
        self.stop_event = stop_event
        self.sha256 = hashlib.sha256()
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        if self.stop_event is not None and self.stop_event.is_set():
            raise _AcquisitionCancelled("source cancelled after another archive failed")
        value = self.budget.read(self.stream, size)
        self.sha256.update(value)
        self.read_bytes += len(value)
        return value


def _fleurs_catalog(payload: bytes) -> dict[str, dict[str, Any]]:
    result = {}
    # FLEURS is literal TSV, not RFC CSV: transcript quotation marks are data.
    # CSV quote handling can join physical rows or swallow field delimiters.
    reader = csv.reader(io.StringIO(payload.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE)
    for number, line in enumerate(reader, 1):
        if len(line) != 7 or not line[0].isdigit() or not line[5].isdigit():
            raise ValueError(f"unexpected FLEURS train TSV schema on physical line {number}")
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
        bandwidth_hz=8000, bandwidth_class="speech_band",
        bandwidth_evidence="Original mono 16 kHz source; 8 kHz Nyquist upper bound, not a measured physical bandwidth estimate",
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
    prior_training_rows: Iterable[ManifestRow] = (), audio_root: Path | None = None,
    storage_budget: AudioStorageBudget | None = None,
    stop_event: threading.Event | None = None,
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
    used = _used_training_exclusions(prior_training_rows)
    audio_root = (audio_root or root / "audio").resolve()
    last_progress = time.monotonic()
    reader = _CountedReader(stream, budget, stop_event)
    rows: list[ManifestRow] = []
    total, by_speaker, skips, by_gender = 0.0, Counter(), Counter(), Counter()
    quarantine_count = 0
    quarantine_attempt = str(time.time_ns())
    quarantine_path = root / "provenance" / "quarantine" / f"{dataset}-{partition}-{language}.jsonl"
    balance_gender = catalog is not None and {"MALE", "FEMALE"}.issubset({item["gender"] for item in catalog.values()})
    access = str((root / "provenance" / "sources.json").resolve())
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            if stop_event is not None and stop_event.is_set():
                raise _AcquisitionCancelled("source cancelled after another archive failed")
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member path")
            if not member.isfile():
                continue
            filename = name.name
            if time.monotonic() - last_progress >= 15:
                progress = {"state": "downloading", "dataset": dataset, "partition": partition,
                            "language": language, "selected_seconds": total, "requested_seconds": seconds,
                            "rows": len(rows), "downloaded_prefix_bytes": reader.read_bytes,
                            "total_downloaded_bytes": budget.read_bytes,
                            "quarantined_records": quarantine_count}
                _atomic_bytes(root / "provenance" / "progress" / f"{dataset}-{partition}-{language}.json", _json_bytes(progress))
                _atomic_bytes(root / "provenance" / "download-progress.json", _json_bytes(progress))
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
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
                if ((language, filename) in used["filenames"] or
                        (dataset, f"{language}:train:{filename}") in used["source_ids"]):
                    skips["prior_training_source"] += 1
                    continue
            else:
                match = _LIBRI_NAME.fullmatch(filename)
                if match is None:
                    continue
                # Do not trust a filename when an archive member labels another split.
                if partition not in name.parts:
                    raise ValueError("LibriSpeech member partition does not match requested archive")
                if (dataset, filename.removesuffix(".flac")) in used["source_ids"]:
                    skips["prior_training_source"] += 1
                    continue
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
            if digest in used["sha256"]:
                skips["prior_training_hash"] += 1
                continue
            audio = sf.info(io.BytesIO(payload))
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError("bootstrap requires original mono 16000 Hz audio")
            if dataset == "fleurs" and audio.frames != catalog[filename]["num_samples"]:
                # The pinned af_za release contains, for example, files whose
                # WAV frame count is twice the TSV count. Never truncate,
                # reinterpret float byte widths, or trust either duration for
                # training when the source identities disagree. Exclude the
                # whole record and seek another valid utterance for the quota.
                quarantine = {"attempt": quarantine_attempt, "reason": "metadata_audio_frame_mismatch",
                              "dataset": dataset, "partition": partition, "language": language,
                              "filename": filename, "archive_member": member.name,
                              "source_url": source_url, "source_revision": FLEURS_REVISION,
                              "dataset_id": catalog[filename]["dataset_id"],
                              "gender": catalog[filename]["gender"],
                              "expected_tsv_samples": catalog[filename]["num_samples"],
                              "decoded_audio_frames": audio.frames, "sample_rate_hz": audio.samplerate,
                              "channels": audio.channels, "subtype": audio.subtype,
                              "source_bytes": len(payload), "audio_sha256": digest,
                              "outcome": "excluded; audio not saved or added to training manifest"}
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                # Append records instead of rewriting a growing audit list.
                # On retries attempt IDs distinguish repeated observations.
                with quarantine_path.open("a", encoding="utf-8") as quarantine_file:
                    quarantine_file.write(json.dumps(quarantine, ensure_ascii=False) + "\n")
                quarantine_count += 1
                skips["metadata_audio_frame_mismatch"] += 1
                continue
            duration = audio.frames / 16000
            if not min_seconds <= duration <= max_seconds:
                skips["duration_filter"] += 1
                continue
            path = audio_root / dataset / partition / language / filename
            row = _row(dataset, partition, language, filename, path, payload, audio.frames, source_url, access)
            if row.parent_recording_id in used["parents"]:
                skips["prior_training_parent"] += 1
                continue
            validate_manifest([row], reserved_rows=reserved_rows)
            if storage_budget is not None:
                storage_budget.store(path, payload, digest)
            elif path.exists():
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
        raise ValueError(f"no eligible audio found for {dataset}/{language}; quarantined={quarantine_count}, log={quarantine_path}")
    validate_manifest(rows, reserved_rows=reserved_rows)
    receipt = {"dataset": dataset, "partition": partition, "language": language,
               "url": source_url, "selected_seconds": total, "requested_seconds": seconds,
               "quota_met": total >= seconds,
               "rows": len(rows), "downloaded_prefix_bytes": reader.read_bytes,
               "downloaded_prefix_sha256": reader.sha256.hexdigest(),
               "complete_archive_digest_verified": False, "skips": dict(skips),
               "selected_speakers": dict(by_speaker), "selected_gender_seconds": dict(by_gender),
               "quarantined_records": quarantine_count, "quarantine_attempt": quarantine_attempt,
               "quarantine_log": str(quarantine_path.relative_to(root)) if quarantine_count else None,
               "selection": "archive order, whole utterances, duration filter; not final corpus balance"}
    return rows, receipt


def acquire_bootstrap(
    root: str | Path, *, fleurs_languages: Iterable[str] | str = DEFAULT_LANGUAGES,
    minutes_per_language: float = 3, librispeech_train_minutes: float = 8,
    librispeech_dev_minutes: float = 2, reserved_rows: Iterable[ManifestRow] = (),
    evaluation_manifest_paths: Iterable[str | Path] = (), max_download_bytes: int = 4 * 1024**3,
    target_fleurs_hours: float | None = None, minimum_language_minutes: float = 30,
    librispeech_other_minutes: float = 0, librispeech_speaker_minutes: float = 1,
    prior_training_rows: Iterable[ManifestRow] = (), audio_root: str | Path | None = None,
    max_audio_bytes: int | None = None, min_free_audio_bytes: int = 0,
    resume: bool = False, minimum_fresh_hours: float = 0, minimum_unique_windows: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Acquire configurable speech quotas with durable source-boundary resumes.

    Use a fresh root for expanded data and pass the old training manifest via
    prior_training_rows. Unlike held-out exclusions, prior training exclusions
    reject used utterances, not their whole speakers or unknown session groups.
    Metadata/checkpoints can live on a durable volume while audio_root is on a
    separate scratch filesystem. No data is deleted to meet a storage budget.
    """
    root = Path(root).resolve()
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 to 16")
    audio_root = Path(audio_root).resolve() if audio_root is not None else root / "audio"
    languages = discover_fleurs_languages() if fleurs_languages == "all" else list(fleurs_languages)
    if len(set(languages)) != len(languages) or any(not re.fullmatch(r"[a-z]{2,3}_[a-z0-9_]+", language) for language in languages):
        raise ValueError("FLEURS languages must be unique official configuration names")
    for value in (minutes_per_language, librispeech_train_minutes, librispeech_dev_minutes,
                  librispeech_other_minutes, minimum_language_minutes, minimum_fresh_hours):
        if not math.isfinite(value) or value < 0:
            raise ValueError("duration quotas must be finite and nonnegative")
    if not math.isfinite(librispeech_speaker_minutes) or librispeech_speaker_minutes <= 0:
        raise ValueError("speaker minutes must be positive")
    if target_fleurs_hours is not None and (not math.isfinite(target_fleurs_hours) or target_fleurs_hours <= 0):
        raise ValueError("target FLEURS hours must be positive")
    if type(minimum_unique_windows) is not int or minimum_unique_windows < 0:
        raise ValueError("minimum unique windows must be a nonnegative integer")
    storage = AudioStorageBudget(audio_root, max_bytes=max_audio_bytes, min_free_bytes=min_free_audio_bytes)
    reserved_rows, prior_training_rows = list(reserved_rows), list(prior_training_rows)
    exclusions = load_evaluation_exclusions(evaluation_manifest_paths)
    used = _used_training_exclusions(prior_training_rows)
    exclusion_identity = hashlib.sha256(_json_bytes({
        "evaluation": {key: sorted(value) for key, value in exclusions.items()},
        "used": {key: sorted(value) for key, value in used.items()},
        "reserved": [row.to_dict() for row in reserved_rows],
    })).hexdigest()
    configuration = {"format_version": 2, "fleurs_revision": FLEURS_REVISION,
        "languages": languages, "minutes_per_language": minutes_per_language,
        "target_fleurs_hours": target_fleurs_hours, "minimum_language_minutes": minimum_language_minutes,
        "librispeech_train_minutes": librispeech_train_minutes, "librispeech_other_minutes": librispeech_other_minutes,
        "librispeech_dev_minutes": librispeech_dev_minutes, "librispeech_speaker_minutes": librispeech_speaker_minutes,
        "exclusion_identity": exclusion_identity, "audio_root": str(audio_root),
        "minimum_fresh_hours": minimum_fresh_hours, "minimum_unique_windows": minimum_unique_windows,
        "selection_version": "whole_utterances_2_to_30_seconds_gender65_v1"}
    plan_path, report_path = root / "provenance/acquisition-plan.json", root / "provenance/sources.json"
    existing_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if existing_plan is not None and not resume:
        raise ValueError("acquisition root already has a plan; use --resume or a new root")
    if resume and (existing_plan is None or existing_plan.get("configuration") != configuration):
        raise ValueError("resume requires the same saved acquisition configuration and exclusions")
    saved = json.loads(report_path.read_text()) if resume and report_path.exists() else None
    if saved is not None and saved.get("acquisition_configuration") != configuration:
        raise ValueError("saved source receipts do not match this acquisition plan")
    prior_downloaded_bytes = saved["downloaded_bytes"] if saved is not None else 0
    if resume:
        progress_paths = list((root / "provenance/progress").glob("*.json"))
        latest_path = root / "provenance/download-progress.json"
        if latest_path.exists():
            progress_paths.append(latest_path)
        completed_saved = {(item["dataset"], item["partition"], item["language"])
                           for item in saved["source_receipts"]} if saved else set()
        for path in progress_paths:
            progress = json.loads(path.read_text())
            if "total_downloaded_bytes" in progress:
                prior_downloaded_bytes = max(prior_downloaded_bytes, progress["total_downloaded_bytes"])
            elif (path == latest_path and (saved is None or "execution" not in saved)
                  and (progress["dataset"], progress["partition"], progress["language"]) not in completed_saved):
                # Older serial jobs recorded the in-progress prefix separately
                # from the last completed-source report. Retain that transfer.
                legacy_total = (saved["downloaded_bytes"] if saved else 0) + progress["downloaded_prefix_bytes"]
                prior_downloaded_bytes = max(prior_downloaded_bytes, legacy_total)
    budget = DownloadBudget(max_download_bytes, initial_bytes=prior_downloaded_bytes)
    catalogs, metadata_hashes = {}, {}
    active_languages = languages if target_fleurs_hours is not None or minutes_per_language else []
    for language in active_languages:
        metadata_url = f"{FLEURS_BASE}/data/{language}/train.tsv"
        metadata_path = root / "provenance" / language / "train.tsv"
        if resume:
            metadata = metadata_path.read_bytes()
        else:
            with _open_url(metadata_url) as response:
                metadata = response.read(16 * 1024 * 1024 + 1)
            budget.add(len(metadata))
            if len(metadata) > 16 * 1024 * 1024:
                raise ValueError("unexpectedly large FLEURS metadata")
            _atomic_bytes(metadata_path, metadata)
        metadata_hashes[language] = hashlib.sha256(metadata).hexdigest()
        if resume and metadata_hashes[language] != existing_plan["metadata_sha256"][language]:
            raise ValueError("saved pinned FLEURS metadata SHA-256 changed")
        catalogs[language] = _fleurs_catalog(metadata)
    if target_fleurs_hours is not None:
        capacities = {language: _catalog_capacity(catalogs[language], language, exclusions, used) for language in active_languages}
        language_seconds = allocate_language_seconds(capacities, target_fleurs_hours * 3600,
                                                    minimum_language_minutes * 60)
    else:
        capacities = None
        language_seconds = {language: minutes_per_language * 60 for language in active_languages}
    plan = {"configuration": configuration, "language_seconds": language_seconds,
            "eligible_capacity_seconds": capacities, "metadata_sha256": metadata_hashes}
    if resume and existing_plan != plan:
        raise ValueError("saved acquisition allocation differs; do not silently change resumed data")
    if not resume:
        _atomic_bytes(plan_path, _json_bytes(plan))
    rows: list[ManifestRow] = []
    receipts: list[dict[str, Any]] = []
    if saved is not None:
        receipts = saved["source_receipts"]
        for receipt in receipts:
            shard = root / receipt["manifest"]
            if hashlib.sha256(shard.read_bytes()).hexdigest() != receipt["manifest_sha256"]:
                raise ValueError("saved source manifest SHA-256 changed")
            rows.extend(load_manifest(shard))
        for row in rows:
            if not Path(row.audio_path).is_file():
                raise ValueError(f"source file missing during resume: {row.audio_path}; restore it before continuing")
        if rows:
            validate_manifest(rows, reserved_rows=reserved_rows)
    completed = {(receipt["dataset"], receipt["partition"], receipt["language"]) for receipt in receipts}
    tasks = [("fleurs", "train", language, language_seconds[language]) for language in active_languages]
    tasks.extend(("librispeech", partition, "en", minutes * 60)
                 for partition, minutes in (("train-clean-100", librispeech_train_minutes),
                                            ("train-other-500", librispeech_other_minutes),
                                            ("dev-clean", librispeech_dev_minutes)) if minutes)
    source_order = {task[:3]: index for index, task in enumerate(tasks)}

    def row_source_key(row):
        return row.dataset, row.source_split, row.language

    # Completion order may vary. Keep final manifests/window inventory in the
    # same planned source order as serial execution, with archive order inside.
    rows.sort(key=lambda row: source_order[row_source_key(row)])
    receipts.sort(key=lambda item: source_order[(item["dataset"], item["partition"], item["language"])])

    def publish(state: str = "acquiring") -> dict[str, Any]:
        if rows:
            validate_manifest(rows, reserved_rows=reserved_rows)
        if state == "checking_capacity" or state.startswith("insufficient"):
            for split in ("train", "dev"):
                selected = [row for row in rows if row.split == split]
                payload = "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in selected).encode()
                _atomic_bytes(root / f"{split}.jsonl", payload)
        windows = nonoverlapping_window_inventory(rows)
        training_hours = sum(row.duration_seconds for row in rows if row.split == "train") / 3600
        report = {
            "format_version": 2, "state": state, "created_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition_configuration": configuration,
            "source_receipts": receipts, "summary": summarize_manifest(rows) if rows else {"rows": 0, "hours": 0},
            "fresh_training_hours": training_hours, "nonoverlapping_windows": windows,
            "downloaded_bytes": budget.read_bytes,
            "execution": {"workers": workers, "prior_downloaded_bytes": prior_downloaded_bytes,
                          "transfer_accounting": "includes prior persisted transfer and reread archive prefixes; abrupt termination can lose transfer since last progress snapshot"},
            "audio_storage_bytes": storage.used_bytes, "audio_filesystem_free_bytes": shutil.disk_usage(audio_root).free,
            "commercial_audio_licenses": ["CC-BY-4.0"],
            "license_evidence": {"fleurs": f"{FLEURS_BASE}/README.md", "librispeech": "https://www.openslr.org/12/"},
            "fleurs_speaker_limit": "No speaker/session IDs are published. speaker_id is null; each language is one conservative unknown-session group. No FLEURS speaker-disjoint development claim.",
            "librispeech_identity_limit": "Reader/chapter IDs are real dataset IDs. Mapping these to shared LibriVox identities is still required before mixing related MLS/Hi-Fi-TTS sources.",
            "bandwidth": "Original 16 kHz recordings only; no native fullband reference or expressive-condition labels are claimed.",
            "evaluation_exclusions": {key: len(value) for key, value in exclusions.items()},
            "prior_training_exclusions": {key: len(value) for key, value in used.items()},
            "rights": "Public CC-BY-4.0 sources; attribution retained. No gated access or new terms accepted.",
        }
        _atomic_bytes(report_path, _json_bytes(report))
        print(json.dumps({"state": state, "completed_sources": len(receipts),
                          "last_source": receipts[-1] if receipts else None,
                          "fresh_training_hours": training_hours, "unique_windows": windows["windows"]}), flush=True)
        return report

    def record_source(selected, receipt):
        validate_manifest([*rows, *selected], reserved_rows=reserved_rows)
        shard = Path("manifests") / f"{receipt['dataset']}-{receipt['partition']}-{receipt['language']}.jsonl"
        payload = "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in selected).encode()
        _atomic_bytes(root / shard, payload)
        receipt["manifest"], receipt["manifest_sha256"] = str(shard), hashlib.sha256(payload).hexdigest()
        rows.extend(selected)
        receipts.append(receipt)
        rows.sort(key=lambda row: source_order[row_source_key(row)])
        receipts.sort(key=lambda item: source_order[(item["dataset"], item["partition"], item["language"])])
        publish()
        if not receipt["quota_met"]:
            publish("insufficient_source_capacity")
            raise ValueError(f"source quota was not met: {receipt['dataset']}/{receipt['partition']}/{receipt['language']}; do not train against planned hours")

    stop_event = threading.Event()

    def acquire_one(task):
        dataset, partition, language, seconds = task
        if stop_event.is_set():
            raise _AcquisitionCancelled("source cancelled before opening another archive")
        url = (f"{FLEURS_BASE}/data/{language}/audio/train.tar.gz" if dataset == "fleurs"
               else f"{LIBRI_BASE}/{partition}.tar.gz")
        with _open_url(url) as response:
            selected, receipt = collect_archive(response, dataset=dataset, partition=partition,
                language=language, root=root, audio_root=audio_root, storage_budget=storage,
                seconds=seconds, source_url=url, budget=budget, exclusions=exclusions,
                catalog=catalogs[language] if dataset == "fleurs" else None,
                reserved_rows=reserved_rows, prior_training_rows=prior_training_rows,
                speaker_seconds=librispeech_speaker_minutes * 60, stop_event=stop_event)
        if dataset == "fleurs":
            receipt.update({"revision": FLEURS_REVISION, "metadata_url": f"{FLEURS_BASE}/data/{language}/train.tsv",
                            "metadata_sha256": metadata_hashes[language]})
        else:
            receipt.update({"official_archive_md5": LIBRI_MD5[partition],
                            "archive_checksum_source": f"{LIBRI_BASE}/md5sum.txt"})
        return selected, receipt

    try:
        unfinished = [task for task in tasks if task[:3] not in completed]
        if workers > 1:
            # Let the larger LibriSpeech archives make progress immediately,
            # alongside FLEURS, without changing canonical manifest ordering.
            fleurs = iter(task for task in unfinished if task[0] == "fleurs")
            libri = iter(task for task in unfinished if task[0] == "librispeech")
            unfinished = []
            while True:
                pair = [item for item in (next(fleurs, None), next(libri, None)) if item is not None]
                if not pair:
                    break
                unfinished.extend(pair)
        _run_source_tasks(unfinished, acquire_one,
                          record_source, workers=workers, stop_event=stop_event)
    except Exception:
        publish("failed")
        raise
    if not rows:
        raise ValueError("at least one positive acquisition quota is required")
    report = publish("checking_capacity")
    actual_languages = {row.language for row in rows if row.dataset == "fleurs" and row.split == "train"}
    if set(active_languages) - actual_languages:
        publish("insufficient_language_coverage")
        raise ValueError("one or more requested FLEURS languages have no fresh training audio")
    if any(not receipt["quota_met"] for receipt in receipts):
        publish("insufficient_source_capacity")
        raise ValueError("one or more source quotas remain unmet")
    if report["fresh_training_hours"] + 1e-9 < minimum_fresh_hours or report["nonoverlapping_windows"]["windows"] < minimum_unique_windows:
        publish("insufficient_unique_data")
        raise ValueError("actual fresh training hours or disjoint scored-window count is below the required minimum")
    return publish("ready")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fleurs-languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--minutes-per-language", type=float, default=3)
    parser.add_argument("--target-fleurs-hours", type=float, help="Allocate this total using actual language capacities instead of equal minute quotas")
    parser.add_argument("--minimum-language-minutes", type=float, default=30)
    parser.add_argument("--librispeech-train-minutes", type=float, default=8)
    parser.add_argument("--librispeech-other-minutes", type=float, default=0)
    parser.add_argument("--librispeech-dev-minutes", type=float, default=2)
    parser.add_argument("--librispeech-speaker-minutes", type=float, default=1)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-evaluation-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--max-download-gb", type=float, default=4)
    parser.add_argument("--audio-root", type=Path, help="Separate scratch audio storage; metadata remains under output-root")
    parser.add_argument("--max-audio-gb", type=float)
    parser.add_argument("--min-free-audio-gb", type=float, default=0)
    parser.add_argument("--minimum-fresh-hours", type=float, default=0)
    parser.add_argument("--minimum-unique-windows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent source archives, 1 to 16; does not change data selection")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_download_gb) or args.max_download_gb <= 0:
        parser.error("--max-download-gb must be positive")
    if args.max_audio_gb is not None and (not math.isfinite(args.max_audio_gb) or args.max_audio_gb <= 0):
        parser.error("--max-audio-gb must be positive")
    if not math.isfinite(args.min_free_audio_gb) or args.min_free_audio_gb < 0:
        parser.error("--min-free-audio-gb must be nonnegative")
    reserved = [row for path in args.reserved_manifest for row in load_manifest(path)]
    used = [row for path in args.exclude_training_manifest for row in load_manifest(path)]
    languages = "all" if args.fleurs_languages == "all" else filter(None, args.fleurs_languages.split(","))
    acquire_bootstrap(args.output_root, fleurs_languages=languages,
        minutes_per_language=args.minutes_per_language, librispeech_train_minutes=args.librispeech_train_minutes,
        librispeech_dev_minutes=args.librispeech_dev_minutes, reserved_rows=reserved,
        evaluation_manifest_paths=args.reserved_evaluation_manifest, max_download_bytes=int(args.max_download_gb * 1024**3),
        target_fleurs_hours=args.target_fleurs_hours, minimum_language_minutes=args.minimum_language_minutes,
        librispeech_other_minutes=args.librispeech_other_minutes, librispeech_speaker_minutes=args.librispeech_speaker_minutes,
        prior_training_rows=used, audio_root=args.audio_root,
        max_audio_bytes=None if args.max_audio_gb is None else int(args.max_audio_gb * 1024**3),
        min_free_audio_bytes=int(args.min_free_audio_gb * 1024**3), resume=args.resume,
        minimum_fresh_hours=args.minimum_fresh_hours, minimum_unique_windows=args.minimum_unique_windows,
        workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
