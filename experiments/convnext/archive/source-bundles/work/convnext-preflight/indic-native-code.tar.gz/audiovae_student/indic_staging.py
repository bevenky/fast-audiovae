"""Bounded, task-owned native Hub shard staging for Indic acquisition."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading

from .acquire import _atomic_bytes, _json_bytes


OVERHEAD_BYTES = 1024**2
XET_OVERHEAD_BYTES = 64 * 1024**2


class NativeShardStager:
    """Reserve full shard space before network I/O; retain interrupted caches.

    The shared collector lock protects reservations and original-audio writes.
    Network transfer, hashing, and cache deletion happen outside that lock.
    """

    def __init__(self, root, *, repo, revision, lock, stop, max_bytes=6 * 1024**3,
                 reserve_bytes=8 * 1024**3, download=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo, self.revision, self.stop = repo, revision, stop
        self.max_bytes, self.reserve_bytes, self.download = max_bytes, reserve_bytes, download
        self.condition = threading.Condition(lock)
        self.entries, self.active = {}, set()
        if type(max_bytes) is not int or max_bytes <= OVERHEAD_BYTES or reserve_bytes < 0:
            raise ValueError("Invalid native staging limits")
        owner = {"version": 1, "repo": repo, "revision": revision, "purpose": "temporary original IndicVoices Parquet shards"}
        marker = self.root / "owner.json"
        if marker.exists():
            if json.loads(marker.read_bytes()) != owner:
                raise ValueError("Native staging ownership/provenance mismatch")
        elif any(self.root.iterdir()):
            raise ValueError("Refuse an existing staging directory without an ownership marker")
        else:
            _atomic_bytes(marker, _json_bytes(owner))
        # A second process must never manipulate the same temporary caches.
        import fcntl
        self._lock_file = (self.root / ".lock").open("a+b")
        fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in self.root.iterdir():
            if folder.name in {"owner.json", ".lock", "xet"}:
                continue
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("Unexpected native staging entry")
            spec = json.loads((folder / "spec.json").read_bytes())
            self._validate_spec(spec)
            if folder.name != self._key(spec["path"]):
                raise ValueError("Native staging path/spec mismatch")
            self.entries[folder.name] = spec
        if self._reserved() > self.max_bytes:
            raise ValueError("Existing resumable native caches exceed the staging cap")
        self.pending_bytes()  # Validate stored files against their reservations.

    def close(self):
        self._lock_file.close()

    def _key(self, path):
        return hashlib.sha256(path.encode()).hexdigest()

    def _validate_spec(self, spec):
        path = PurePosixPath(spec["path"])
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or not path.name.endswith(".parquet"):
            raise ValueError("Unsafe native shard source path")
        if type(spec["bytes"]) is not int or spec["bytes"] < 1 or spec["bytes"] + OVERHEAD_BYTES + XET_OVERHEAD_BYTES > self.max_bytes:
            raise ValueError("Native shard cannot fit within its staging cap")
        if not isinstance(spec["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
            raise ValueError("Native shard requires the pinned upstream SHA-256")

    def _reserved(self):
        return XET_OVERHEAD_BYTES + sum(spec["bytes"] + OVERHEAD_BYTES for spec in self.entries.values())

    def _stored(self, key):
        folder = self.root / key
        total = 0
        if folder.is_symlink():
            raise ValueError("Symlinks are not allowed in native staging")
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Symlinks are not allowed in native staging")
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic download rename: conservatively reserve these bytes again.
        return total

    def pending_bytes(self):
        """Unwritten reserved bytes; disk free already accounts for stored data."""
        xet_bytes = self._stored("xet")
        if xet_bytes > XET_OVERHEAD_BYTES:
            raise ValueError("Disabled Xet cache/logs exceeded their metadata allowance")
        remaining = XET_OVERHEAD_BYTES - xet_bytes
        for key, spec in self.entries.items():
            allocated, stored = spec["bytes"] + OVERHEAD_BYTES, self._stored(key)
            if stored > allocated:
                raise ValueError("Native cache grew beyond its reserved byte cap")
            remaining += allocated - stored
        return remaining

    def check_audio_write(self, added):
        if shutil.disk_usage(self.root).free - added - self.pending_bytes() < self.reserve_bytes:
            raise ValueError("Original audio write would consume reserved native staging space")

    def discard_completed(self, predicate):
        for key, spec in list(self.entries.items()):
            if predicate(spec["path"]):
                self._remove(key)

    def _remove(self, key):
        # Only descendants created under our ownership marker may be removed.
        folder = self.root / key
        if folder.exists():
            if folder.is_symlink() or folder.parent != self.root:
                raise ValueError("Refuse unsafe native cache cleanup")
            shutil.rmtree(folder)
        with self.condition:
            self.entries.pop(key, None)
            self.active.discard(key)
            self.condition.notify_all()

    @contextmanager
    def shard(self, item):
        lfs = getattr(item, "lfs", None)
        digest = getattr(lfs, "sha256", None) if not isinstance(lfs, dict) else lfs.get("sha256")
        spec = {"path": item.path, "bytes": item.size, "sha256": digest}
        self._validate_spec(spec)
        key = self._key(item.path)
        with self.condition:
            while True:
                if self.stop.is_set():
                    raise InterruptedError("Native staging stopped before a new transfer")
                if key in self.active:
                    raise ValueError("A native shard is already being consumed")
                if key in self.entries:
                    if self.entries[key] != spec:
                        raise ValueError("Resumable native shard differs from its pinned source")
                    self.active.add(key)
                    break
                allocated = item.size + OVERHEAD_BYTES
                if self._reserved() + allocated <= self.max_bytes:
                    if shutil.disk_usage(self.root).free - self.pending_bytes() - allocated < self.reserve_bytes:
                        raise ValueError("Native staging would cross the filesystem reserve")
                    self.entries[key] = spec
                    self.active.add(key)
                    break
                self.condition.wait(timeout=0.5)
        folder = self.root / key
        success = False
        try:
            folder.mkdir(exist_ok=True)
            _atomic_bytes(folder / "spec.json", _json_bytes(spec))
            downloader = self.download
            if downloader is None:
                from huggingface_hub import hf_hub_download
                downloader = hf_hub_download
            result = Path(downloader(self.repo, item.path, repo_type="dataset", revision=self.revision,
                                     local_dir=folder)).resolve()
            expected = (folder / item.path).resolve()
            if result != expected or not result.is_relative_to(folder) or result.is_symlink():
                raise ValueError("Hub returned an unexpected native staging path")
            if result.stat().st_size != item.size:
                raise ValueError("Native shard size differs from the pinned source")
            with result.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError("Native shard SHA-256 differs from the pinned source")
            with self.condition:
                self.pending_bytes()
            yield result
            success = not self.stop.is_set()
        finally:
            if success:
                self._remove(key)
            else:
                with self.condition:
                    self.active.discard(key)
                    self.condition.notify_all()


def configure_native_transfer(root):
    """Run before importing Hub/Xet; do not relocate the existing login cache."""
    os.environ["HF_XET_CACHE"] = str(Path(root).resolve() / "xet")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ["HF_XET_SHARD_CACHE_SIZE_LIMIT"] = "0"
    os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "16"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
