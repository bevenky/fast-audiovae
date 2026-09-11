"""On-demand continuous teacher targets in a bounded, owned disk cache.

Only already prepared mono 16 kHz source files are supported. No resampling,
normalization, cropping, data download, or device selection occurs here. A
custom reader receives verified file bytes and must preserve the supplied row's
prepared-audio contract. The caller loads and qualifies the pinned FP32 teacher,
including any required JIT warmup, before constructing this corpus.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import suppress
from dataclasses import replace
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor

from .cache import FORMAT_VERSION, UtteranceCache, prepare_utterance_cache
from .data import ManifestRow, load_manifest, validate_manifest
from .teacher import CHECKPOINT_SHA256, SOURCE_SHA256, FrozenAudioVAE2


GiB = 1024**3
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREPARED_POLICIES = {"prepared-soxr-vhq-to-16000-v1", "prepared-native-16000-float-v1"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".identity-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_native_16k(payload: bytes, row: ManifestRow) -> Tensor:
    """Decode verified bytes without changing sample rate, channels or gain."""
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Install soundfile to read prepared 16 kHz source audio") from error
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if rate != 16000 or audio.shape[1] != 1 or row.sample_rate_hz != 16000:
        raise ValueError("SourceCorpus requires prepared mono 16 kHz audio; no implicit conversion is allowed")
    return torch.from_numpy(audio.T.copy()).unsqueeze(0)


def _default_reader_identity() -> dict[str, Any]:
    import importlib.metadata

    try:
        version = importlib.metadata.version("soundfile")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {"name": "soundfile_prepared_mono16k", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config": {"dtype": "float32", "sample_rate": 16000, "channels": 1, "gain": "unchanged",
                       "resample": False, "soundfile_version": version}}


class SourceCorpus:
    """Read a pinned source corpus lazily, caching complete teacher utterances.

    input_sample_counts contains exact prepared 16 kHz lengths, including dev
    rows if supplied. Manifest-relative audio paths resolve against the manifest
    directory, or source_root when rows are passed directly. Source files are
    SHA-256 checked before a reader sees their bytes. A custom reader requires
    an explicit identity containing name, code_sha256, and serialized config.
    That hash is the caller's declared implementation pin; it is not inferred
    from a callable's name or used as a substitute for source-file verification.
    allow_prepared_source accepts the two pinned upstream preparation policies
    for already converted mono16k files. The reader still performs no conversion.

    The target-byte limit excludes the small SQLite metadata index. Free-space
    reserve is checked before and after writes, but other processes can change
    available space. Only digest-named targets in this cache's owned targets
    directory can be evicted. There is one writer per cache directory.
    """

    def __init__(self, rows_or_manifest: Iterable[ManifestRow] | str | Path, teacher: FrozenAudioVAE2, *,
                 cache_dir: str | Path, input_sample_counts: Mapping[str, int],
                 reserved_rows: Iterable[ManifestRow] = (), source_root: str | Path | None = None,
                 max_disk_bytes: int = 12 * GiB, min_free_bytes: int = 4 * GiB,
                 max_memory_utterances: int = 4,
                 audio_reader: Callable[[bytes, ManifestRow], Tensor] | None = None,
                 reader_identity: Mapping[str, Any] | None = None, allow_prepared_source: bool = False):
        self._closed = False
        self._guard = threading.RLock()
        self._database = None
        self._lock_file = None
        if type(allow_prepared_source) is not bool:
            raise ValueError("allow_prepared_source must be an explicit boolean")
        if not isinstance(teacher, FrozenAudioVAE2):
            raise TypeError("teacher must be the explicitly loaded FrozenAudioVAE2")
        self._teacher = teacher
        self.teacher_identity = json.loads(_json(teacher.provenance))
        if (self.teacher_identity.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                self.teacher_identity.get("source_sha256") != SOURCE_SHA256):
            raise ValueError("SourceCorpus requires the pinned original AudioVAE2 source and checkpoint")
        if self.teacher_identity.get("dtype") != "float32" or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("SourceCorpus teacher must remain frozen FP32")
        for value, name, minimum in ((max_disk_bytes, "max_disk_bytes", 1), (min_free_bytes, "min_free_bytes", 0),
                                      (max_memory_utterances, "max_memory_utterances", 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_disk_bytes, self.min_free_bytes = max_disk_bytes, min_free_bytes
        self.max_memory_utterances = max_memory_utterances
        if isinstance(rows_or_manifest, (str, Path)):
            manifest_path = Path(rows_or_manifest).resolve(strict=True)
            rows = load_manifest(manifest_path)
            root = manifest_path.parent if source_root is None else Path(source_root)
        else:
            rows = list(rows_or_manifest)
            root = Path.cwd() if source_root is None else Path(source_root)
        self._source_root = root.resolve()
        self._reserved = tuple(reserved_rows)
        validate_manifest(rows, reserved_rows=self._reserved)
        self.rows = tuple(rows)
        self.rows_by_id = MappingProxyType({row.source_id: row for row in rows})
        if len(self.rows_by_id) != len(rows):
            raise ValueError("SourceCorpus source_id must be globally unique")
        if set(input_sample_counts) != set(self.rows_by_id):
            raise ValueError("input_sample_counts must match every source row exactly")
        self.input_sample_counts = MappingProxyType(dict(input_sample_counts))
        digest = hashlib.sha256()
        for row in sorted(rows, key=lambda value: value.source_id):
            count = input_sample_counts[row.source_id]
            if type(count) is not int or count < 1:
                raise ValueError("Prepared input sample counts must be positive integers")
            if row.sample_rate_hz != 16000 or abs(row.duration_seconds - count / 16000) > 1 / 16000 + 1e-10:
                raise ValueError("Rows must describe the exact prepared 16 kHz audio and its duration")
            policy_prefixes = [value.casefold().split(":", 1)[0].strip()
                               for value in (row.resampler_policy, row.gain_policy)]
            native_unchanged = (all(value in {"none", "unchanged"} for value in policy_prefixes) and
                                row.original_sample_rate_hz == 16000)
            explicit_prepared = (allow_prepared_source and row.resampler_policy in _PREPARED_POLICIES and
                                 row.gain_policy == "no-additional-gain-normalization")
            if explicit_prepared:
                is_native_policy = row.resampler_policy == "prepared-native-16000-float-v1"
                if is_native_policy != (row.original_sample_rate_hz == 16000):
                    raise ValueError("Prepared-source policy contradicts the original sample rate")
            if not native_unchanged and not explicit_prepared:
                raise ValueError("Source preparation requires native none/unchanged policy or explicit "
                                 "allow_prepared_source with a supported prepared policy")
            digest.update((_json({"source": row.to_dict(), "input_samples": count}) + "\n").encode())
        self.data_fingerprint = digest.hexdigest()
        if audio_reader is None:
            if reader_identity is not None:
                raise ValueError("reader_identity without a custom audio_reader is ambiguous")
            self._reader, self.reader_identity = read_native_16k, _default_reader_identity()
        else:
            if not callable(audio_reader) or not isinstance(reader_identity, Mapping):
                raise ValueError("A custom audio_reader requires an explicit reader_identity")
            self._reader, self.reader_identity = audio_reader, json.loads(_json(dict(reader_identity)))
            if (not isinstance(self.reader_identity.get("name"), str) or not self.reader_identity["name"] or
                    not isinstance(self.reader_identity.get("config"), dict) or
                    not isinstance(self.reader_identity.get("code_sha256"), str) or
                    not _SHA256.fullmatch(self.reader_identity["code_sha256"])):
                raise ValueError("reader_identity must pin name, code_sha256 and config")
        self.reader_identity["prepared_source_policy"] = {
            "allow_prepared_source": allow_prepared_source,
            "accepted_policies": sorted(_PREPARED_POLICIES) if allow_prepared_source else [],
            "reader_resampling": False,
        }
        self.identity = {"format_version": 1, "data_fingerprint": self.data_fingerprint,
                         "teacher": self.teacher_identity, "reader": self.reader_identity,
                         "target_policy": "continuous_whole_utterance_raw_mu_fp32_with_reference16k",
                         "config": {"max_disk_bytes": max_disk_bytes, "min_free_bytes": min_free_bytes,
                                    "max_memory_utterances": max_memory_utterances,
                                    "allow_prepared_source": allow_prepared_source}}
        self.identity_sha256 = _digest(self.identity)
        self._directory = Path(cache_dir).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        identity_path = self._directory / "identity.json"
        if not identity_path.exists() and any(path.name != ".writer.lock" for path in self._directory.iterdir()):
            raise ValueError("Use an empty cache directory or one with the matching SourceCorpus identity")
        try:
            self._lock_file = (self._directory / ".writer.lock").open("a+b")
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("SourceCorpus cache already has an active writer") from error
            if identity_path.exists():
                if identity_path.is_symlink() or json.loads(identity_path.read_text()) != self.identity:
                    raise ValueError("SourceCorpus cache manifest/teacher/reader/config identity mismatch")
            else:
                _atomic_json(identity_path, self.identity)
            self._targets = self._directory / "targets"
            if self._targets.is_symlink():
                raise ValueError("Cache targets directory cannot be a symlink")
            self._targets.mkdir(exist_ok=True)
            # A previous process may have stopped before publishing a staged
            # file. The exclusive lock proves no live writer owns these temps.
            for pending_path in self._targets.glob(".pending-*"):
                self._check_file(pending_path)
                pending_path.unlink()
            index_path = self._directory / "index.sqlite3"
            if index_path.is_symlink():
                raise ValueError("Cache index cannot be a symlink")
            # The RLock serializes access when a caller prepares targets from a
            # worker thread. SQLite itself still has a single cache writer.
            self._database = sqlite3.connect(index_path, check_same_thread=False)
            self._database.execute("PRAGMA journal_mode=WAL")
            self._database.execute("CREATE TABLE IF NOT EXISTS entries (lookup_key TEXT PRIMARY KEY, info TEXT NOT NULL)")
            self._database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
            pending = []
            for key, encoded in self._database.execute("SELECT lookup_key,info FROM entries"):
                info = json.loads(encoded)
                path = self._cache_path(key)
                if path.exists():
                    self._check_file(path)
                    if path.stat().st_size != info["bytes"]:
                        raise ValueError("Cached target size differs from its index")
                    pending.append((key, info))
                else:
                    self._database.execute("DELETE FROM entries WHERE lookup_key=?", (key,))
            self._entries.update(sorted(pending, key=lambda item: item[1]["last_access"]))
            # Recover published files left between atomic publication and index
            # commit without loading any tensors at initialization.
            for path in self._targets.glob("*.pt"):
                key = path.stem
                if _SHA256.fullmatch(key) and key not in self._entries:
                    self._check_file(path)
                    self._entries[key] = {"bytes": path.stat().st_size, "last_access": 0,
                                          "row_id": None, "cache_key": None, "file_sha256": None}
            self._entries = OrderedDict(sorted(self._entries.items(), key=lambda item: item[1]["last_access"]))
            self._clock = max((info["last_access"] for info in self._entries.values()), default=0)
            self._memory: OrderedDict[str, UtteranceCache] = OrderedDict()
            self._disk_bytes = sum(info["bytes"] for info in self._entries.values())
            stored = self._database.execute("SELECT value FROM metadata WHERE key='stats'").fetchone()
            self._stats = json.loads(stored[0]) if stored else {}
            for name in ("memory_hits", "disk_hits", "cache_misses", "prepared_utterances", "prepared_input_samples",
                         "source_bytes_read", "cache_bytes_read", "cache_bytes_written", "evictions", "evicted_bytes",
                         "source_hash_seconds", "audio_read_seconds", "teacher_seconds", "cache_load_seconds", "cache_write_seconds"):
                self._stats.setdefault(name, 0)
            self._dirty = set(self._entries)
            self._make_space(0)
            self.flush()
        except BaseException:
            self.close(flush=False)
            raise

    def _cache_path(self, key: str) -> Path:
        if not isinstance(key, str) or not _SHA256.fullmatch(key):
            raise ValueError("Invalid owned cache filename")
        if self._targets.is_symlink() or self._targets.resolve() != self._directory / "targets":
            raise ValueError("Cache targets directory moved outside its owned location")
        return self._targets / f"{key}.pt"

    @staticmethod
    def _check_file(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Owned cache target must be a regular file, not a symlink")

    def _lookup_key(self, row: ManifestRow) -> str:
        return _digest({"source": row.to_dict(), "input_samples": self.input_sample_counts[row.source_id],
                        "teacher": self.teacher_identity, "reader": self.reader_identity,
                        "target_policy": self.identity["target_policy"]})

    def _touch(self, key: str) -> None:
        if key in self._entries:
            self._clock += 1
            self._entries[key]["last_access"] = self._clock
            self._entries.move_to_end(key)
            self._dirty.add(key)

    def _make_space(self, new_bytes: int) -> None:
        if new_bytes > self.max_disk_bytes:
            raise OSError("One whole-utterance target exceeds the disk-cache byte budget")
        while self._disk_bytes + new_bytes > self.max_disk_bytes or shutil.disk_usage(self._directory).free - new_bytes < self.min_free_bytes:
            if not self._entries:
                raise OSError("Insufficient cache space while preserving the configured free-space reserve")
            key, info = next(iter(self._entries.items()))
            path = self._cache_path(key)
            if path.exists() or path.is_symlink():
                self._check_file(path)
                path.unlink()
            self._entries.pop(key)
            self._dirty.discard(key)
            self._memory.pop(info.get("row_id"), None)
            self._disk_bytes -= info["bytes"]
            self._stats["evictions"] += 1
            self._stats["evicted_bytes"] += info["bytes"]
            self._database.execute("DELETE FROM entries WHERE lookup_key=?", (key,))
            self._database.commit()

    def _verify_record(self, record: UtteranceCache, row: ManifestRow, key: str) -> None:
        identity = record.metadata["identity"]
        expected_source = row.to_dict()
        expected_source["teacher_cache_key"] = None
        if (identity["source"] != expected_source or identity["teacher"] != self.teacher_identity or
                record.input_samples != self.input_sample_counts[row.source_id] or record.reference16k is None or
                record.metadata.get("source_preparation") != {"reader": self.reader_identity, "lookup_key": key,
                                                                "source_file_sha256": row.audio_sha256}):
            raise ValueError("Cached target source/teacher/preparation provenance mismatch")
        if row.teacher_cache_key is not None and row.teacher_cache_key != record.cache_key:
            raise ValueError("Cached target differs from the manifest's teacher_cache_key")

    def _load_record(self, path: Path, info: dict[str, Any], row: ManifestRow, key: str) -> UtteranceCache:
        self._check_file(path)
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        if info.get("file_sha256") is not None and file_hash != info["file_sha256"]:
            raise ValueError("Cached target file SHA-256 mismatch")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        fields = {"format_version", "cache_key", "metadata", "latents", "teacher_audio", "reference16k"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported owned target-cache format")
        if info.get("cache_key") is not None and payload["cache_key"] != info["cache_key"]:
            raise ValueError("Cached target key differs from its index")
        record = UtteranceCache(payload["latents"], payload["teacher_audio"], payload["reference16k"],
                                payload["metadata"], payload["cache_key"])
        record.validate()
        self._verify_record(record, row, key)
        info.update(row_id=row.source_id, cache_key=record.cache_key, file_sha256=file_hash)
        self._stats["cache_bytes_read"] += len(raw)
        return record

    def _prepare_record(self, row: ManifestRow, key: str) -> UtteranceCache:
        path = Path(row.audio_path)
        if not path.is_absolute():
            path = self._source_root / path
        started = time.perf_counter()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.audio_sha256:
            raise ValueError(f"Source file SHA-256 mismatch: {row.source_id}")
        self._stats["source_hash_seconds"] += time.perf_counter() - started
        self._stats["source_bytes_read"] += len(raw)
        started = time.perf_counter()
        audio = self._reader(raw, row)
        if (not isinstance(audio, Tensor) or audio.dtype != torch.float32 or audio.device.type != "cpu" or
                tuple(audio.shape) != (1, 1, self.input_sample_counts[row.source_id]) or
                audio.requires_grad or not torch.isfinite(audio).all()):
            raise ValueError("Audio reader violated finite detached CPU FP32 [1,1,N] prepared-input contract")
        self._stats["audio_read_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        record = prepare_utterance_cache(audio, row, self._teacher, reserved_rows=self._reserved, include_reference=True)
        self._stats["teacher_seconds"] += time.perf_counter() - started
        self._stats["prepared_utterances"] += 1
        self._stats["prepared_input_samples"] += audio.shape[-1]
        metadata = json.loads(_json(record.metadata))
        metadata["source_preparation"] = {"reader": self.reader_identity, "lookup_key": key,
                                           "source_file_sha256": row.audio_sha256}
        record = replace(record, metadata=metadata)
        self._verify_record(record, row, key)
        return record

    def _store_record(self, record: UtteranceCache, row: ManifestRow, key: str) -> None:
        # Serialize one utterance in memory to know the exact write size before
        # eviction or reserve checks. This avoids temporary on-disk overshoot.
        payload = {"format_version": FORMAT_VERSION, "cache_key": record.cache_key, "metadata": record.metadata,
                   "latents": record.latents, "teacher_audio": record.teacher_audio, "reference16k": record.reference16k}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getbuffer()
        size = len(raw)
        self._make_space(size)
        destination, temporary = self._cache_path(key), None
        try:
            with tempfile.NamedTemporaryFile(dir=self._targets, prefix=".pending-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            if shutil.disk_usage(self._directory).free < self.min_free_bytes:
                raise OSError("Cache write would violate the configured free-space reserve")
            os.link(temporary, destination)
            self._entries[key] = {"row_id": row.source_id, "cache_key": record.cache_key,
                                  "bytes": size, "file_sha256": hashlib.sha256(raw).hexdigest(), "last_access": 0}
            self._disk_bytes += size
            self._stats["cache_bytes_written"] += size
            self._touch(key)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get(self, row_id: str) -> UtteranceCache:
        """Return immutable continuous targets; no reset is introduced by crops."""
        with self._guard:
            if self._closed:
                raise RuntimeError("SourceCorpus is closed")
            if self._teacher.provenance != self.teacher_identity:
                raise ValueError("Teacher identity changed after SourceCorpus construction")
            row = self.rows_by_id[row_id]
            key = self._lookup_key(row)
            if row_id in self._memory:
                self._stats["memory_hits"] += 1
                self._memory.move_to_end(row_id)
                self._touch(key)
                return self._memory[row_id]
            path = self._cache_path(key)
            if key in self._entries:
                started = time.perf_counter()
                record = self._load_record(path, self._entries[key], row, key)
                self._stats["cache_load_seconds"] += time.perf_counter() - started
                self._stats["disk_hits"] += 1
                self._touch(key)
            else:
                self._stats["cache_misses"] += 1
                record = self._prepare_record(row, key)
                started = time.perf_counter()
                self._store_record(record, row, key)
                self._stats["cache_write_seconds"] += time.perf_counter() - started
            if self.max_memory_utterances:
                self._memory[row_id] = record
                while len(self._memory) > self.max_memory_utterances:
                    self._memory.popitem(last=False)
            self.flush()
            return record

    def metrics(self) -> dict[str, Any]:
        with self._guard:
            return {**self._stats, "disk_bytes": self._disk_bytes, "disk_records": len(self._entries),
                    "memory_records": len(self._memory), "max_disk_bytes": self.max_disk_bytes,
                    "min_free_bytes": self.min_free_bytes, "identity_sha256": self.identity_sha256}

    def flush(self) -> None:
        """Persist metadata/exposure at a trainer checkpoint or clean shutdown."""
        with self._guard:
            if self._closed or self._database is None:
                return
            with self._database:
                for key in self._dirty:
                    self._database.execute("INSERT OR REPLACE INTO entries VALUES (?,?)", (key, _json(self._entries[key])))
                self._database.execute("INSERT OR REPLACE INTO metadata VALUES ('stats',?)", (_json(self._stats),))
            self._dirty.clear()

    def close(self, *, flush: bool = True) -> None:
        with self._guard:
            if self._closed:
                return
            try:
                if flush:
                    self.flush()
            finally:
                if self._database is not None:
                    self._database.close()
                if self._lock_file is not None:
                    with suppress(OSError):
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    self._lock_file.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
