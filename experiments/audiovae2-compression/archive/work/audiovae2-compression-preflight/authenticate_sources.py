"""Authenticate selected existing source file bytes; no audio/model dependency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(item):
    row = item["manifest_row"]
    result = {"source_id": item["source_id"], "path": row["audio_path"],
              "expected_audio_sha256": item["audio_sha256"], "passed": False}
    try:
        if row["source_id"] != item["source_id"] or row["audio_sha256"] != item["audio_sha256"]:
            raise ValueError("Source row and selection identity disagree")
        path = Path(row["audio_path"])
        if not path.is_absolute():
            raise ValueError("Source path must be absolute")
        if not path.is_file():
            raise ValueError("Selected source is not an existing file")
        before = path.stat()
        if before.st_size < 1:
            raise ValueError("Selected source is empty")
        result["observed_audio_sha256"] = file_hash(path)
        after = path.stat()
        stable = all(getattr(before, key) == getattr(after, key)
                     for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns"))
        result.update(bytes=after.st_size, stat_unchanged_during_read=stable,
                      mtime_ns=after.st_mtime_ns)
        result["passed"] = stable and result["observed_audio_sha256"] == result["expected_audio_sha256"]
        if not result["passed"]:
            result["error"] = "Source changed during read" if not stable else "Source byte hash differs"
    except (OSError, ValueError, KeyError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Worker count must be between one and eight")
    if args.output.exists():
        raise FileExistsError("Refusing to replace a source-authentication receipt")
    payload = args.selection.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != args.expected_selection_sha256:
        raise ValueError("Selection file byte hash differs")
    selection = json.loads(payload)
    if selection["identity_sha256"] != digest({k: v for k, v in selection.items() if k != "identity_sha256"}):
        raise ValueError("Selection identity differs")
    expected_counts = {"calibration": 72, "development": 96, "fit": 3000}
    if set(selection["splits"]) != set(expected_counts):
        raise ValueError("Unexpected split set")
    entries = []
    for name, count in expected_counts.items():
        split = selection["splits"][name]
        if len(split["rows"]) != count or len(split["indices"]) != count:
            raise ValueError("Unexpected split size")
        if [row["pool_index"] for row in split["rows"]] != split["indices"]:
            raise ValueError("Source and pool index ordering differ")
        entries.extend(split["rows"])
    for key in ("source_id", "audio_sha256", "parent_recording_id", "pool_index"):
        if len({entry[key] for entry in entries}) != 3168:
            raise ValueError("Expected exactly 3,168 distinct " + key)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(verify, entries))
    checks = {row["source_id"]: row for row in rows}
    split_checks = {name: {"sources": len(split["rows"]),
        "passed": sum(checks[row["source_id"]]["passed"] for row in split["rows"]),
        "bytes_read": sum(checks[row["source_id"]].get("bytes", 0) for row in split["rows"])}
        for name, split in selection["splits"].items()}
    result = {"version": "audiovae2-compression-source-byte-authentication-v1",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - clock, "workers": args.workers,
        "selection_path": str(args.selection), "selection_file_sha256": observed,
        "selection_identity_sha256": selection["identity_sha256"],
        "script_sha256": file_hash(__file__), "source_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
        "total_bytes_read": sum(row.get("bytes", 0) for row in rows),
        "split_checks": split_checks, "rows": rows,
        "audio_decode_operations": 0, "model_operations": 0,
        "limitations": ["This verifies encoded file bytes against the pinned acquisition manifest; it does not decode or assess audio.",
            "No teacher-pair cache bytes, tensor values or model numerical parity were checked here.",
            "The receipt establishes source integrity at inspection, not protection against later file changes."]}
    result["identity_sha256"] = digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second run from overwriting this evidence.
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({key: result[key] for key in ("passed", "source_count", "passed_count",
        "failed_count", "total_bytes_read", "elapsed_seconds", "identity_sha256")}))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
