"""Bounded original IndicVoices acquisition with immutable source provenance.

Only official train Parquet row groups are streamed. Selected embedded FLAC
bytes are kept unchanged; no complete corpus or shard cache is created. The
existing Hugging Face login is used by the Hub API, never read or printed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import signal
import threading
from typing import Any

import numpy as np
import soundfile as sf

from .acquire import AudioStorageBudget, _atomic_bytes, _json_bytes, load_evaluation_exclusions
from .data import ManifestRow, load_manifest, validate_manifest


REPO = "ai4bharat/IndicVoices"
REVISION = "c96f9088f138cf89d419da7e8e643e1f05c00a87"
LANGUAGES = {
    "assamese": "as", "bengali": "bn", "bodo": "brx", "dogri": "doi", "gujarati": "gu",
    "hindi": "hi", "kannada": "kn", "kashmiri": "ks", "konkani": "kok", "maithili": "mai",
    "malayalam": "ml", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
    "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
    "telugu": "te", "urdu": "ur",
}
_CHUNK = re.compile(r"^(\d+)_chunk_(\d+)\.(?:flac|wav)$")
_CONVERSATION_CHUNK = re.compile(
    r"^([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})_\d+_chunk_(\d+)\.(?:flac|wav)$")
_COLUMNS = ["audio_filepath.path", "duration", "samples", "speaker_id", "lang", "scenario", "task_name"]


def source_identifiers(metadata: dict[str, Any], language: str) -> dict[str, str]:
    """Keep actual chunk, full-recording and speaker identifiers separate."""
    path = metadata["audio_filepath"]["path"]
    if not isinstance(path, str) or PurePosixPath(path).name != path:
        raise ValueError("Unexpected original IndicVoices chunk path")
    # Conversation filenames publish one UUID with separate numeric speaker
    # tracks. Share its session across tracks so alternate channels cannot count
    # twice or leak across splits; keep the exact delivered chunk as parent.
    match = _CHUNK.fullmatch(path) or _CONVERSATION_CHUNK.fullmatch(path)
    if match is None:
        raise ValueError("Original chunk path no longer exposes a verified recording identifier")
    speaker = metadata.get("speaker_id")
    if not isinstance(speaker, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", speaker):
        raise ValueError("A real source speaker_id is required")
    if metadata.get("lang") != LANGUAGES[language]:
        raise ValueError("Source language does not match its pinned configuration")
    return {"source_id": f"{language}:{path}", "filename": path,
            "speaker_id": f"indicvoices:speaker:{speaker}",
            "session_id": f"indicvoices:recording:{match.group(1)}",
            "parent_recording_id": f"indicvoices:utterance:{PurePosixPath(path).stem}"}


def make_row(metadata: dict[str, Any], payload: bytes, *, language: str, audio_path: Path,
             parquet_path: str, row_group: int, row_number: int, access_record: Path) -> tuple[ManifestRow, dict]:
    ids = source_identifiers(metadata, language)
    with sf.SoundFile(io.BytesIO(payload)) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("Original IndicVoices file is not mono16k; stop before any conversion")
        frames, subtype, format_name = audio.frames, audio.subtype, audio.format
        decoded = audio.read(dtype="float32", always_2d=True)
    if decoded.shape != (frames, 1) or not np.isfinite(decoded).all():
        raise ValueError("Original audio has invalid decoded samples")
    if metadata.get("samples") != frames:
        raise ValueError("Source sample count differs from decoded original audio")
    duration = frames / 16000
    if not math.isfinite(float(metadata["duration"])) or abs(float(metadata["duration"]) - duration) > 0.001:
        raise ValueError("Source duration differs from decoded original audio")
    digest = hashlib.sha256(payload).hexdigest()
    source_url = f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/{parquet_path}"
    row = ManifestRow(
        dataset="indicvoices", source_revision=REVISION, source_id=ids["source_id"],
        source_url=source_url, audio_path=str(audio_path.resolve()), audio_sha256=digest,
        parent_recording_id=ids["parent_recording_id"], parent_start_seconds=0,
        speaker_id=ids["speaker_id"], session_id=ids["session_id"], language=LANGUAGES[language],
        sample_rate_hz=16000, original_sample_rate_hz=16000, bandwidth_hz=8000,
        bandwidth_class="speech_band", native_recording=False, enhanced=False,
        bandwidth_evidence="Official original unenhanced IndicVoices release delivered mono16k; 8k Nyquist supervision limit, not a claim about microphone capture bandwidth",
        duration_seconds=duration, split="train", source_split="train", license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="AI4Bharat, IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian Languages",
        access_record=str(access_record.resolve()), gain_policy="none: preserve original amplitude",
        resampler_policy="none: original mono 16000 Hz", teacher_cache_key=None,
    )
    receipt = {"manifest": row.to_dict(), "source": {
        "configuration": language, "parquet_path": parquet_path, "row_group": row_group,
        "row_number_in_group": row_number, "original_audio_path": ids["filename"],
        "original_bytes": len(payload), "decoded_frames": frames, "format": format_name, "subtype": subtype,
        "scenario": metadata.get("scenario"), "task_name": metadata.get("task_name"),
        "original_speaker_id": metadata["speaker_id"],
        "parent_timeline_policy": "Canonical parent is this delivered chunk at offset0. Full-recording chunk offset is unpublished and not fabricated; its real full-recording ID is retained as session_id.",
    }}
    return row, receipt


def _manifest_bytes(rows):
    return ("".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode()


class Collector:
    def __init__(self, args):
        self.args, self.root = args, args.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.rows, self.receipts = [], []
        self.hours = Counter()
        self.speaker_seconds, self.seen_ids, self.seen_hashes, self.seen_sessions = Counter(), set(), set(), set()
        self.skipped, self.completed_groups, self.source_files = Counter(), {}, {}
        self.reserved = [row for path in args.reserved_manifest for row in load_manifest(path)]
        prior = [row for path in args.prior_training_manifest for row in load_manifest(path)]
        if any(row.split != "train" for row in prior):
            raise ValueError("Prior training exclusions must contain train rows only")
        self.excluded_hashes = load_evaluation_exclusions(args.reserved_evaluation_manifest)["sha256"]
        self.excluded_ids, self.excluded_speakers, self.excluded_sessions, self.excluded_parents = set(), set(), set(), set()
        for row in self.reserved + prior:
            self.excluded_hashes.add(row.audio_sha256)
            self.excluded_ids.add((row.dataset, row.source_id))
            self.excluded_parents.add(row.parent_recording_id)
            if row.speaker_id:
                self.excluded_speakers.add(row.speaker_id)
            if row.session_id:
                self.excluded_sessions.add(row.session_id)
        self.audio_budget = AudioStorageBudget(self.root / "audio", max_bytes=int(args.max_audio_gib * 2**30),
                                               min_free_bytes=int(args.min_free_gib * 2**30))
        self.policy = {"format_version": 1, "repo": REPO, "revision": REVISION,
                       "source_partition": "train", "languages": args.languages,
                       "hours_per_language": args.hours_per_language, "speaker_seconds_cap": args.speaker_seconds,
                       "minimum_clip_seconds": args.minimum_clip_seconds,
                       "one_chunk_per_full_recording": True,
                       "exclusion_sha256": hashlib.sha256(_json_bytes({
                           "hashes": sorted(self.excluded_hashes), "sources": sorted(self.excluded_ids),
                           "speakers": sorted(self.excluded_speakers), "sessions": sorted(self.excluded_sessions),
                           "parents": sorted(self.excluded_parents)})).hexdigest()}
        policy_path = self.root / "acquisition-policy.json"
        if policy_path.exists() and json.loads(policy_path.read_text()) != self.policy:
            raise ValueError("Resume acquisition policy differs; use a separate output directory")
        _atomic_bytes(policy_path, _json_bytes(self.policy))
        state_path = self.root / "progress.json"
        if state_path.exists():
            old = json.loads(state_path.read_text())
            self.completed_groups = old.get("completed_row_groups", {})
            self.source_files = old.get("source_files", {})
        # Individual atomic receipts are authoritative if a process stopped
        # between an audio write and a periodic full manifest snapshot.
        for path in sorted((self.root / "receipts").glob("*/*.json")):
            receipt = json.loads(path.read_text())
            row = ManifestRow.from_dict(receipt["manifest"])
            if row.source_revision != REVISION or row.dataset != "indicvoices":
                raise ValueError("Resume receipt provenance mismatch")
            if hashlib.sha256(Path(row.audio_path).read_bytes()).hexdigest() != row.audio_sha256:
                raise ValueError("Resume original audio hash mismatch")
            self._remember(row, receipt)
        if self.rows:
            validate_manifest(self.rows, reserved_rows=self.reserved, training_only=True)
        self.staging = None
        if getattr(args, "download_method", "ranged") == "native":
            from .indic_staging import NativeShardStager
            self.staging = NativeShardStager(self.root / "native-staging", repo=REPO, revision=REVISION,
                                            lock=self.lock, stop=self.stop,
                                            max_bytes=int(getattr(args, "max_staging_gib", 6) * 2**30),
                                            reserve_bytes=int(args.min_free_gib * 2**30))
            self.staging.discard_completed(lambda path: self.hours[path.split("/", 1)[0]] >= args.hours_per_language
                or self._shard_complete(path))
        self.snapshot("starting")

    def _shard_complete(self, path):
        count = self.source_files.get(path, {}).get("row_groups")
        return isinstance(count, int) and set(self.completed_groups.get(path, [])) == set(range(count))

    @contextmanager
    def _source_stream(self, fs, item):
        if self.staging is not None:
            with self.staging.shard(item) as path:
                with path.open("rb") as stream:
                    yield stream
        else:
            with fs.open(f"datasets/{REPO}@{REVISION}/{item.path}", "rb",
                         block_size=1024**2, cache_type="none") as stream:
                yield stream

    def _remember(self, row, receipt):
        if row.source_id in self.seen_ids or row.audio_sha256 in self.seen_hashes or row.session_id in self.seen_sessions:
            raise ValueError("Duplicate original chunk or full-recording session in acquisition receipts")
        self.rows.append(row)
        self.receipts.append(receipt)
        self.seen_ids.add(row.source_id)
        self.seen_hashes.add(row.audio_sha256)
        self.seen_sessions.add(row.session_id)
        self.speaker_seconds[row.speaker_id] += row.duration_seconds
        self.hours[receipt["source"]["configuration"]] += row.duration_seconds / 3600

    def snapshot(self, status, error_type=None):
        with self.lock:
            _atomic_bytes(self.root / "train.jsonl", _manifest_bytes(self.rows))
            speakers = {language: len({row.speaker_id for row, receipt in zip(self.rows, self.receipts)
                                      if receipt["source"]["configuration"] == language}) for language in self.args.languages}
            value = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
                     "download_method": getattr(self.args, "download_method", "ranged"),
                     "workers": getattr(self.args, "workers", 2),
                     "rows": len(self.rows), "hours": sum(self.hours.values()),
                     "hours_by_language": dict(self.hours), "speakers_by_language": speakers,
                     "maximum_speaker_seconds": max(self.speaker_seconds.values(), default=0),
                     "full_2_56s_window_hours": sum(int(round(row.duration_seconds * 16000)) // 40960 * 2.56 for row in self.rows) / 3600,
                     "audio_bytes": self.audio_budget.used_bytes,
                     "skipped": dict(self.skipped), "completed_row_groups": self.completed_groups,
                     "source_files": self.source_files,
                     "coverage_limit": "Exact source, speaker, session, parent and file-hash exclusions; no acoustic or fuzzy cross-publisher deduplication claim"}
            if error_type:
                value["error_type"] = error_type
            _atomic_bytes(self.root / "progress.json", _json_bytes(value))

    def candidate(self, metadata, language):
        ids = source_identifiers(metadata, language)
        duration = float(metadata["duration"])
        if not math.isfinite(duration) or duration < self.args.minimum_clip_seconds:
            return False
        with self.lock:
            if self.hours[language] >= self.args.hours_per_language:
                return False
            if ids["source_id"] in self.seen_ids or ids["session_id"] in self.seen_sessions:
                return False
            if (("indicvoices", ids["source_id"]) in self.excluded_ids
                    or ids["speaker_id"] in self.excluded_speakers
                    or ids["session_id"] in self.excluded_sessions
                    or ids["parent_recording_id"] in self.excluded_parents):
                return False
            return self.speaker_seconds[ids["speaker_id"]] + duration <= self.args.speaker_seconds

    def collect_language(self, language):
        # Executor tasks that were queued when SIGTERM arrived must not start
        # fresh Hub listings while the existing readers drain and checkpoint.
        if self.stop.is_set() or self.hours[language] >= self.args.hours_per_language:
            return
        from huggingface_hub import HfApi, HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem() if self.staging is None else None
        files = [item for item in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION,
                                                       path_in_repo=language)
                 if item.path.startswith(language + "/train-") and item.path.endswith(".parquet")]
        if not files:
            raise ValueError("Pinned language has no official training Parquet files")
        # Spread the starting source shard across a reproducible revision-scoped
        # ordering; no source-order quality or scenario preference is assumed.
        files.sort(key=lambda item: hashlib.sha256((REVISION + item.path).encode()).hexdigest())
        scanned_audio_bytes = 0
        for item in files:
            if self.stop.is_set() or self.hours[language] >= self.args.hours_per_language:
                break
            if self._shard_complete(item.path):
                continue
            with self._source_stream(fs, item) as stream:
                parquet = pq.ParquetFile(stream)
                with self.lock:
                    self.source_files[item.path] = {"bytes": item.size, "row_groups": parquet.num_row_groups,
                                                   "rows": parquet.metadata.num_rows}
                completed = set(self.completed_groups.get(item.path, []))
                for group in range(parquet.num_row_groups):
                    if self.stop.is_set() or self.hours[language] >= self.args.hours_per_language:
                        break
                    if group in completed:
                        continue
                    metadata_rows = parquet.read_row_group(group, columns=_COLUMNS).to_pylist()
                    candidates = [index for index, metadata in enumerate(metadata_rows)
                                  if self.candidate(metadata, language)]
                    if candidates:
                        audio_columns = [parquet.metadata.row_group(group).column(index)
                                         for index in range(parquet.metadata.num_columns)
                                         if parquet.metadata.row_group(group).column(index).path_in_schema == "audio_filepath.bytes"]
                        scanned_audio_bytes += sum(column.total_compressed_size for column in audio_columns)
                        if scanned_audio_bytes > self.args.max_scanned_audio_gib_per_language * 2**30:
                            raise ValueError("Bounded audio-column scan exceeded its per-language quota")
                        blobs = parquet.read_row_group(group, columns=["audio_filepath.bytes"])["audio_filepath"].to_pylist()
                        for index in candidates:
                            if self.stop.is_set() or self.hours[language] >= self.args.hours_per_language:
                                break
                            metadata = metadata_rows[index]
                            if not self.candidate(metadata, language):
                                continue
                            payload = blobs[index]["bytes"]
                            if not isinstance(payload, bytes):
                                raise ValueError("Training audio is not embedded original file bytes")
                            filename = metadata["audio_filepath"]["path"]
                            audio_path = self.root / "audio" / language / filename
                            row, receipt = make_row(metadata, payload, language=language, audio_path=audio_path,
                                                    parquet_path=item.path, row_group=group, row_number=index,
                                                    access_record=self.root / "source-access.json")
                            with self.lock:
                                if row.audio_sha256 in self.excluded_hashes or row.audio_sha256 in self.seen_hashes:
                                    self.skipped["known_exact_audio_hash"] += 1
                                    continue
                                if not self.candidate(metadata, language):
                                    continue
                                validate_manifest([row], reserved_rows=self.reserved, training_only=True)
                                self.audio_budget.check(len(payload))
                                if self.staging is not None:
                                    self.staging.check_audio_write(len(payload))
                                if audio_path.exists():
                                    if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row.audio_sha256:
                                        raise ValueError("Existing original audio path has different bytes")
                                else:
                                    _atomic_bytes(audio_path, payload)
                                    self.audio_budget.add(len(payload))
                                receipt_path = self.root / "receipts" / language / (filename + ".json")
                                _atomic_bytes(receipt_path, _json_bytes(receipt))
                                self._remember(row, receipt)
                                if len(self.rows) % 32 == 0:
                                    self.snapshot("downloading")
                        del blobs
                    with self.lock:
                        if not self.stop.is_set():
                            self.completed_groups.setdefault(item.path, []).append(group)
                    self.snapshot("downloading")
                    print(json.dumps({"language": language, "hours": self.hours[language],
                                      "total_hours": sum(self.hours.values()), "row_group": group,
                                      "audio_column_scan_gib": scanned_audio_bytes / 2**30}), flush=True)
        if not self.stop.is_set() and self.hours[language] < self.args.hours_per_language:
            raise ValueError("Language source exhausted before its varied-speaker quota")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=sorted(LANGUAGES), default=list(LANGUAGES))
    parser.add_argument("--hours-per-language", type=float, default=2)
    parser.add_argument("--speaker-seconds", type=float, default=180)
    parser.add_argument("--minimum-clip-seconds", type=float, default=2.56)
    parser.add_argument("--max-audio-gib", type=float, default=12)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--max-scanned-audio-gib-per-language", type=float, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download-method", choices=("ranged", "native"), default="ranged")
    parser.add_argument("--max-staging-gib", type=float, default=6)
    parser.add_argument("--reserved-manifest", type=Path, action="append", default=[])
    parser.add_argument("--prior-training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reserved-evaluation-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if len(set(args.languages)) != len(args.languages) or args.workers not in range(1, 9):
        parser.error("Languages must be unique and workers between 1 and 8")
    for key in ("hours_per_language", "speaker_seconds", "minimum_clip_seconds", "max_audio_gib",
                "min_free_gib", "max_scanned_audio_gib_per_language", "max_staging_gib"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive and finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.download_method == "native":
        from .indic_staging import configure_native_transfer
        configure_native_transfer(args.root / "native-staging")
    collector = Collector(args)
    from huggingface_hub import hf_hub_download
    readme_path = hf_hub_download(REPO, "README.md", repo_type="dataset", revision=REVISION)
    readme = Path(readme_path).read_bytes()
    _atomic_bytes(collector.root / "source-readme.md", readme)
    _atomic_bytes(collector.root / "source-access.json", _json_bytes({
        "repo": REPO, "revision": REVISION, "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        "authentication": "Existing local Hugging Face login; no credential copied into artifacts",
        "selection": "Official train only; configured per-language duration and per-speaker caps; at most one chosen chunk per full source recording; source order not perceptual quality selection",
        "bandwidth": "Original distributed16k file bytes preserved; speech-band upper bound only; microphone capture bandwidth unverified",
        "identity": "Delivered chunk is canonical parent at0; full-recording chunk offsets unpublished, real full-recording ID retained as session; no original-full-timeline offset invented",
        "conditions": "Source scenario and task_name copied only to receipts; no condition or emotion invented",
    }))
    def stop(signum, frame):
        collector.stop.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(collector.collect_language, language) for language in args.languages]
            for future in as_completed(futures):
                try:
                    future.result()
                except InterruptedError:
                    if not collector.stop.is_set():
                        raise
                except BaseException:
                    collector.stop.set()
                    raise
        if collector.rows:
            validate_manifest(collector.rows, reserved_rows=collector.reserved, training_only=True)
        collector.snapshot("paused" if collector.stop.is_set() else "complete")
    except BaseException as error:
        collector.snapshot("failed", type(error).__name__)
        # Hub exceptions can contain signed download URLs. Only the class is
        # printed; fixed locally generated ValueErrors are safe to identify.
        detail = str(error) if type(error) is ValueError else type(error).__name__
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "detail": detail}), flush=True)
        return 1
    finally:
        if collector.staging is not None:
            collector.staging.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
