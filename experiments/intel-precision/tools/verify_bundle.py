#!/usr/bin/env python3
"""Verify the source bundle without loading native code or numerical packages."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-sha256", help="Optional independently supplied manifest digest")
    a = p.parse_args()
    path = ROOT / "manifest.json"
    digest = sha(path)
    if a.manifest_sha256 and digest != a.manifest_sha256:
        raise ValueError("Bundle manifest hash differs")
    manifest = json.loads(path.read_text())
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        file = ROOT / relative
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(ROOT):
            raise ValueError("Expected a regular bundled file: " + name)
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError("Bundle file differs: " + name)
    print(json.dumps({"status": "verified", "files": len(manifest["files"]),
                      "manifest_sha256": digest, "native_execution": False,
                      "model_execution": False, "timing_collected": False}))


if __name__ == "__main__":
    main()
