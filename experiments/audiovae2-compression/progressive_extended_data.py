"""Append source-disjoint targets without changing the original pruning plan."""
from __future__ import annotations

import json
from pathlib import Path

from fresh_training_data import (
    CHECKPOINT_SHA256, SOURCE_SHA256, FreshTrainingData, digest, sha, stat_identity,
)

VERSION = "audiovae2_progressive_extension_5000_v1"
PARENT_CHECKPOINT_SHA256 = "a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f"
ORIGINAL_SOURCES, APPENDED_SOURCES = 30000, 30000
START, STOP = 24000, 60000
KEYS = ("source_id", "audio_sha256", "parent_recording_id")


def validate_extension(original, extension):
    """Authenticate the complete old prefix and every appended source identity."""
    fresh = original.fresh
    old_ids = tuple(original.source_ids)
    parent_rows = fresh.plan["rows"]
    if (len(old_ids) != ORIGINAL_SOURCES or len(set(old_ids)) != ORIGINAL_SOURCES
            or len(original.fit) != 3000 or len(parent_rows) != 27000
            or old_ids != tuple(row["source_id"] for row in original.fit) + tuple(fresh.source_ids)
            or tuple(fresh.source_ids) != tuple(row["source_id"] for row in parent_rows)):
        raise ValueError("Expected the unchanged original30000-source stream")
    identity = extension.get("identity_sha256")
    if (extension.get("version") != VERSION
            or digest({k:v for k,v in extension.items() if k != "identity_sha256"}) != identity
            or extension.get("parent_plan_sha256") != sha(fresh.plan_path)
            or extension.get("parent_plan_identity_sha256") != fresh.identity
            or extension.get("original_manifest_sha256") != sha(fresh._original_manifest)
            or extension.get("original_manifest_sha256") != fresh.plan["original_manifest_sha256"]
            or extension.get("starting_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
            or extension.get("starting_optimizer_step") != 2000
            or extension.get("target_optimizer_step") != 5000
            or extension.get("appended_start_index") != ORIGINAL_SOURCES
            or extension.get("authorized_source_interval") != [START, STOP]
            or extension.get("teacher_source_sha256") != SOURCE_SHA256
            or extension.get("teacher_checkpoint_sha256") != CHECKPOINT_SHA256
            or fresh.plan["teacher_source_sha256"] != SOURCE_SHA256
            or fresh.plan["teacher_checkpoint_sha256"] != CHECKPOINT_SHA256):
        raise ValueError("Progressive extension provenance or recovery contract differs")
    rows = extension.get("appended_rows")
    if not isinstance(rows, list) or len(rows) != APPENDED_SOURCES:
        raise ValueError("The extension requires exactly30000 appended sources")
    blocked = {key:set(fresh.plan["blocked_identities"][key]) for key in KEYS}
    for row in parent_rows:
        for key in KEYS:
            blocked[key].add(row["manifest_row"][key])
    # The original reader authenticates the fitting and held-out exclusion ledger.
    # Check the entire stream again here, including its fitting-source prefix.
    blocked["source_id"].update(old_ids)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("manifest_row"), dict):
            raise ValueError("Invalid appended source row")
        source = row["manifest_row"]
        if row.get("source_id") != source.get("source_id") or source.get("split") != "train":
            raise ValueError("Appended identity or training split differs")
        FreshTrainingData._geometry(row)
        for key in KEYS:
            value = source.get(key)
            if not isinstance(value, str) or not value or value in blocked[key]:
                raise ValueError("Repeated, previously planned or held-out identity: " + key)
            blocked[key].add(value)
    ids = old_ids + tuple(row["source_id"] for row in rows)
    if extension.get("source_ids_sha256") != digest(list(ids)):
        raise ValueError("Full60000-source order differs from the sealed extension")
    return rows, ids


class _ExtensionShards(FreshTrainingData):
    """Reuse the sealed-target reader with local extension indices0:30000."""
    def __init__(self, plan_path, shards_dir, extension, rows):
        self.plan_path, self.shards_dir = Path(plan_path), Path(shards_dir)
        self.identity = extension["identity_sha256"]
        self.plan = {"rows":rows}
        self.source_ids = tuple(row["source_id"] for row in rows)
        self._hashes = {str(self.plan_path):sha(self.plan_path)}
        self._stats = {path:stat_identity(path) for path in self._hashes}
        self._cached_index, self._cached_crops = None, None


class ExtendedProgressiveData:
    """Read the old stream unchanged, followed by separately sealed new targets."""
    def __init__(self, original_stream, extension_plan_path, extension_shards_dir):
        self.original, self.fresh = original_stream, original_stream.fresh
        self.plan_path = Path(extension_plan_path)
        original_stream.assert_unchanged()
        extension = json.loads(self.plan_path.read_text())
        rows, self.source_ids = validate_extension(original_stream, extension)
        self.identity, self.extension_plan = extension["identity_sha256"], extension
        self.extension = _ExtensionShards(self.plan_path, extension_shards_dir, extension, rows)

    def take(self, global_cursor, count):
        if (type(global_cursor) is not int or type(count) is not int
                or global_cursor < 0 or count < 1 or global_cursor+count > STOP):
            raise ValueError("Invalid global progressive-source interval")
        stop = global_cursor+count
        result = []
        if global_cursor < ORIGINAL_SOURCES:
            result.extend(self.original.take(global_cursor, min(stop, ORIGINAL_SOURCES)-global_cursor))
        if stop > ORIGINAL_SOURCES:
            start = max(global_cursor, ORIGINAL_SOURCES)-ORIGINAL_SOURCES
            result.extend(self.extension.take(start, stop-max(global_cursor, ORIGINAL_SOURCES)))
        if [row["source_id"] for row in result] != list(self.source_ids[global_cursor:stop]):
            raise RuntimeError("Global progressive-source order changed")
        return result

    def assert_unchanged(self):
        self.original.assert_unchanged()
        self.extension.assert_unchanged()
