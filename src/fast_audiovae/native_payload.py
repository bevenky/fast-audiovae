"""Verified native libraries supplied by a platform wheel."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import os
import subprocess
import sys
from functools import lru_cache

from .assets import sha256

PAYLOAD_ROOT = Path(__file__).resolve().parent / "_native"


def _path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise RuntimeError("Invalid native payload path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Native payload path escapes its directory")
    result = root / name
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Native payload contains an unsafe path")
    return result


def inspect_payload():
    root = PAYLOAD_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    from .build_resources import validate_native_payload
    manifest = validate_native_payload(root)
    return {"root": root, "manifest": manifest, "identity": sha256(manifest_path)}


def materialize_payload(payload, destination, recipe):
    """Copy wheel artifacts, then adapt relative build records for the cache."""
    manifest = payload["manifest"]
    entry = manifest.get("recipes", {}).get(recipe)
    if entry is None:
        return None
    destination = Path(destination).resolve()
    for name, digest in manifest["files"].items():
        target = _path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError("Cached native wheel artifact changed: " + name)
        else:
            descriptor, temporary_name = tempfile.mkstemp(prefix=".native-", dir=target.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(_path(payload["root"], name), temporary)
                if sha256(temporary) != digest:
                    raise RuntimeError("Native wheel changed while copying: " + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    result = {**entry, "root": str(destination), "identity": payload["identity"]}
    if "payload_dir" in entry:
        result["root"] = str(_path(destination, entry["payload_dir"]))
    if isinstance(entry.get("manifest"), str):
        relative = str(PurePosixPath(entry.get("payload_dir", "")) / entry["manifest"])
        if relative not in manifest["files"]:
            raise RuntimeError("CPU recipe manifest is outside the wheel inventory")
        result["manifest"] = json.loads(_path(destination, relative).read_text())
    if "base_build" in entry:
        if entry["base_build"] not in manifest["files"]:
            raise RuntimeError("Native build record is outside the wheel inventory")
        path = _path(destination, entry["base_build"])
        record = json.loads(path.read_text())
        if record.get("library") not in manifest["files"]:
            raise RuntimeError("Native library is outside the wheel inventory")
        library = _path(destination, record["library"])
        if sha256(library) != record["library_sha256"]:
            raise RuntimeError("Native build record library does not match the wheel")
        record["library"] = str(library)
        derived = destination.parent / (recipe + "-native-build.json")
        derived.write_text(json.dumps(record, indent=2) + "\n")
        result["base_build"] = str(derived)
    return result


@lru_cache(maxsize=8)
def _load_probe(paths, identities):
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
                       HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
    try:
        result = subprocess.run([sys.executable, "-c",
            "import ctypes,sys; handles=[ctypes.CDLL(p) for p in sys.argv[1:]]", *paths],
            env=environment, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    return result.returncode == 0, result.stderr.strip()


def probe_payload(prebuilt):
    """Check dynamic dependency compatibility in a child before parent loading."""
    root = Path(prebuilt["root"])
    if "manifest" in prebuilt:
        records = prebuilt["manifest"]["libraries"]
        # Resolve the shared core first, then its dependent operator bridges.
        names = sorted(records, key=lambda name: (name != "native", name != "core", name))
        paths = tuple(str(_path(root, records[name]["path"])) for name in names)
        identities = tuple(records[name]["sha256"] for name in names)
    else:
        record = json.loads(Path(prebuilt["base_build"]).read_text())
        paths, identities = (record["library"],), (record["library_sha256"],)
    return _load_probe(paths, identities)
