"""Read the parent's exact teacher crops without loading or rerunning a teacher."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .cache import load_cache
from .preflight_distillation import PreflightConfig, _crop_identity, fixed_crops
from .teacher import CHECKPOINT_SHA256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parent_crops(parent: dict, cache_dir: str | Path):
    """Verify immutable parent identities and return the same ordered crop tensors.

    The existing writer lock is held shared; the SQLite index is opened as an
    immutable read-only snapshot. No cache identity, access time, row, target or
    source file is rewritten. Missing or changed targets fail closed.
    """
    directory = Path(cache_dir).resolve(strict=True)
    identity = parent["identity"]
    data = identity["data"]
    expected = data["source_corpus"]
    if (data.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256 or
            expected["teacher"].get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Parent must identify the pinned frozen AudioVAE2 teacher")
    for name in ("identity.json", "index.sqlite3", ".writer.lock", "initial-prefill.json", "targets"):
        path = directory / name
        if path.is_symlink() or not path.exists():
            raise ValueError("Parent cache requires original nonsymlinked files")
    with (directory / ".writer.lock").open("rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The teacher cache has an active writer") from error
        if json.loads((directory / "identity.json").read_text()) != expected:
            raise ValueError("Teacher cache identity differs from the parent")
        if _sha(directory / "initial-prefill.json") != data["initial_teacher_prefill_sha256"]:
            raise ValueError("Teacher batching qualification differs from the parent")
        wal = directory / "index.sqlite3-wal"
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Parent cache has an uncheckpointed SQLite WAL")
        db = sqlite3.connect((directory / "index.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            entries = {}
            for key, encoded in db.execute("SELECT lookup_key, info FROM entries"):
                if not re.fullmatch(r"[0-9a-f]{64}", key):
                    raise ValueError("Invalid target lookup key")
                info = json.loads(encoded)
                cache_key = info.get("cache_key")
                if cache_key in entries:
                    raise ValueError("Ambiguous duplicate cache target")
                entries[cache_key] = (key, info)
        finally:
            db.close()
        config = PreflightConfig(**identity["config"])
        groups, files = {}, {}
        for role in ("diagnostic", "sentinel"):
            wanted = identity[role]
            records = {}
            for spec in wanted:
                cache_key = spec["cache_key"]
                if cache_key in records:
                    continue
                if cache_key not in entries:
                    raise ValueError("A parent teacher target is missing")
                key, info = entries[cache_key]
                path = directory / "targets" / f"{key}.pt"
                if path.is_symlink() or not path.is_file():
                    raise ValueError("A parent teacher target is missing or symlinked")
                if path.stat().st_size != info["bytes"] or _sha(path) != info["file_sha256"]:
                    raise ValueError("Parent teacher target file checksum changed")
                record = load_cache(path, expected_cache_key=cache_key)
                source = record.metadata["identity"]["source"]
                if (record.metadata["identity"]["teacher"] != expected["teacher"] or
                        source["source_id"] != spec["source_id"] or info["row_id"] != source["source_id"] or
                        record.metadata.get("source_preparation") != {
                            "reader": expected["reader"], "lookup_key": key,
                            "source_file_sha256": source["audio_sha256"]}):
                    raise ValueError("Parent target source or teacher provenance changed")
                records[cache_key] = fixed_crops(record, role=role, config=config)
                files[key] = {"file_sha256": info["file_sha256"], "cache_key": cache_key,
                              "source_id": source["source_id"]}
            selected = []
            for spec in wanted:
                matches = [c for c in records[spec["cache_key"]] if c.start_frame == spec["start_frame"]]
                if len(matches) != 1:
                    raise ValueError("Parent crop position changed")
                selected.append(matches[0])
            if _crop_identity(selected) != wanted:
                raise ValueError("Latent, target or crop accounting differs from the parent")
            groups[role] = tuple(selected)
        if {c.source_id for c in groups["diagnostic"]} & {c.source_id for c in groups["sentinel"]}:
            raise ValueError("Diagnostic and held-out identities overlap")
        evidence = deepcopy(data)
        evidence["read_only_parent_cache"] = {
            "identity_sha256": _sha(directory / "identity.json"),
            "index_sha256": _sha(directory / "index.sqlite3"),
            "targets": files, "teacher_inference_calls": 0,
            "all_parent_crop_hashes_match": True,
            "diagnostic_crops": len(groups["diagnostic"]), "sentinel_crops": len(groups["sentinel"])}
        return groups["diagnostic"], groups["sentinel"], evidence
