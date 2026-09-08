"""Publish complete generated bundles without reusing interrupted outputs."""
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from ..assets import sha256


def _files(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != ".recipe-ready.json"}


def completed_bundle(destination, create):
    destination = Path(destination).resolve()
    marker = destination / ".recipe-ready.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("version") != 1 or record.get("files") != _files(destination):
            raise RuntimeError("A generated recipe bundle changed after preparation")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve interrupted work for diagnosis. It is never adopted as complete.
    if destination.exists():
        destination.rename(destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex[:8]))
    with tempfile.TemporaryDirectory(prefix=".recipe-", dir=destination.parent) as directory:
        staged = Path(directory) / "bundle"
        create(staged)
        manifest = json.loads((staged / "bundle.json").read_text())
        if not (staged / manifest["fallback"]).is_file():
            raise RuntimeError("Prepared fallback graph is missing")
        record = {"version": 1, "files": _files(staged)}
        (staged / ".recipe-ready.json").write_text(json.dumps(record, indent=2) + "\n")
        staged.rename(destination)
    return destination
