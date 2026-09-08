"""Locate and materialize the small source tree shipped with the Python wheel.

Build resources are copied from the maintained repository sources at wheel-build
time. Native compilation happens only in the caller's writable cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


# Explicit paths prevent local checkpoints, build products, credentials and
# editor conflict copies from entering an installation or its build cache.
RESOURCE_FILES = tuple("""
LICENSE
native/apple/native_kernels.c
native/apple/native_kernels.h
native/apple/custom_ops.cpp
native/x86/native_kernels.c
native/x86/native_kernels.h
native/x86/custom_ops.cpp
native/amd/packed_a.c
native/amd/packed_a.h
native/amd/custom_op.cpp
tools/build.py
tools/build_apple.py
tools/build_x86.py
tools/build_amd.py
tools/build_aocl.py
tools/build_sleef.py
tools/fetch_headers.py
experiments/cpu-stage/dependency-pins.json
experiments/cpu-stage/compose_candidates.py
experiments/cpu-stage/matrix/build.py
experiments/cpu-stage/matrix/custom_op.cpp
experiments/cpu-stage/matrix/fused_pointwise.c
experiments/cpu-stage/matrix/fused_pointwise.h
experiments/cpu-stage/matrix/rewrite.py
experiments/cpu-stage/matrix/LICENSE-LIBXSMM.md
experiments/cpu-stage/stage/build.py
experiments/cpu-stage/stage/custom_op.cpp
experiments/cpu-stage/stage/rewrite.py
experiments/cpu-stage/stage/stage_dw.h
experiments/cpu-stage/mkl/dependency-pins.json
experiments/cpu-stage/mkl/matrix_common.py
experiments/cpu-stage/mkl/prepare_decoder.py
experiments/cpu-stage/upsample/build.py
experiments/cpu-stage/upsample/custom_op.cpp
experiments/cpu-stage/upsample/projection.c
experiments/cpu-stage/upsample/projection.h
experiments/cpu-stage/upsample/rewrite.py
experiments/intel-precision/LICENSE
experiments/intel-precision/THIRD_PARTY_NOTICES.md
experiments/intel-precision/native/precision.cpp
experiments/intel-precision/native/precision.h
experiments/intel-precision/native/custom_op.cpp
experiments/intel-precision/fused/stage_precision.cpp
experiments/intel-precision/fused/upsample_precision.cpp
experiments/intel-precision/support/matrix/fused_pointwise.c
experiments/intel-precision/support/matrix/fused_pointwise.h
experiments/intel-precision/support/native/native_kernels.h
experiments/intel-precision/support/stage/stage_dw.h
experiments/intel-precision/support/upsample/projection.c
experiments/intel-precision/support/upsample/projection.h
experiments/intel-precision/tools/rewrite.py
experiments/intel-precision/pins/dependencies.json
experiments/intel-precision/pins/mkl.json
experiments/intel-precision/pins/ort.json
experiments/intel-precision/licenses/libxsmm-LICENSE.md
experiments/intel-precision/licenses/onnxruntime-LICENSE
experiments/intel-precision/licenses/sleef-LICENSE.txt
experiments/intel-precision/licenses/onemkl/LICENSE.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-IntelMPI.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-benchmarks.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-level-zero.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-oneMath.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-oneTBB.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-openmp.txt
experiments/intel-precision/licenses/onemkl/third-party-programs-safestring.txt
experiments/intel-precision/licenses/onemkl/third-party-programs.txt
experiments/intel-precision/licenses/onemkl/top_level.txt
experiments/amd-precision/aocl/source/custom_op.cpp
experiments/amd-precision/aocl/source/precision.cpp
experiments/amd-precision/aocl/source/precision.h
experiments/amd-precision/common/fused/stage_precision.cpp
experiments/amd-precision/common/fused/upsample_precision.cpp
experiments/amd-precision/common/native/precision.h
experiments/amd-precision/common/support/matrix/fused_pointwise.c
experiments/amd-precision/common/support/matrix/fused_pointwise.h
experiments/amd-precision/common/support/native/native_kernels.h
experiments/amd-precision/common/support/stage/stage_dw.h
experiments/amd-precision/common/support/upsample/projection.c
experiments/amd-precision/common/support/upsample/projection.h
experiments/amd-precision/pins/dependencies.json
experiments/amd-precision/pins/rebuild.json
experiments/amd-precision/licenses/AOCL-LICENSE
experiments/amd-precision/licenses/AOCL-NOTICES
experiments/amd-precision/licenses/LICENSE
experiments/amd-precision/licenses/libxsmm-LICENSE.md
experiments/amd-precision/licenses/onnxruntime-LICENSE
experiments/amd-precision/licenses/sleef-LICENSE.txt
experiments/apple-precision/licenses/LICENSE
experiments/apple-precision/licenses/ONNXRUNTIME-LICENSE
experiments/apple-precision/pins/ort.json
experiments/streaming-matrix/apple/prepare.py
experiments/streaming-matrix/intel/build_candidate.py
experiments/streaming-matrix/intel/paired_projection.cpp
experiments/streaming-matrix/intel/precision.h
""".split())


def resource_files(root: Path) -> tuple[Path, ...]:
    """Return the checked source allowlist, without discovering extra files."""
    root = Path(root).resolve()
    result = []
    for relative in RESOURCE_FILES:
        path = root / relative
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise RuntimeError("Missing or unsafe build resource: " + relative)
        result.append(path)
    return tuple(result)


def resource_root() -> Path:
    """Return maintained checkout sources or the wheel's bundled source tree."""
    package = Path(__file__).resolve().parent
    installed = package / "_build_resources"
    root = installed if installed.is_dir() else package.parents[1]
    resource_files(root)
    return root


def _contents(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    data = {str(path.relative_to(root)): path.read_bytes() for path in resource_files(root)}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    return data, hashes


def resource_fingerprint() -> str:
    """Hash every shipped compiler input and recipe, independent of its path."""
    _, hashes = _contents(resource_root())
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_resources(destination: str | Path) -> str:
    """Refresh only bundled source files; preserve cached .deps and .build.

    The caller serializes setup for this destination. Returns the source-tree
    fingerprint used by the cache manifest. No compiler, network or model runs.
    """
    source = resource_root()
    destination = Path(destination).resolve()
    if destination == source:
        raise ValueError("Materialize build resources outside their maintained source tree")
    data, hashes = _contents(source)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in data.items():
        target = destination / relative
        if target.is_symlink() or not target.resolve().is_relative_to(destination):
            raise ValueError("Build destination contains an unsafe source path: " + relative)
        if not target.is_file() or target.read_bytes() != content:
            _write(target, content)
    record = destination / ".build-resources.json"
    if record.is_symlink() or not record.resolve().is_relative_to(destination):
        raise ValueError("Build destination contains an unsafe resource manifest")
    _write(record, (json.dumps({"version": 1, "fingerprint": fingerprint, "files": hashes},
                             sort_keys=True, indent=2) + "\n").encode())
    return fingerprint


def validate_native_payload(root: str | Path) -> dict:
    """Validate an explicit maintainer-built binary payload before packaging.

    A source distribution never embeds this external payload. Linux uses the
    generic platform tag until its dependency closure is separately audited.
    """
    import re

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Native payload requires a regular manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Native payload manifest version 1 is required")
    tag = manifest.get("wheel_platform", "")
    if not isinstance(tag, str) or (tag != "linux_x86_64" and not re.fullmatch(r"macosx_\d+_\d+_arm64", tag)):
        raise ValueError("Native wheel requires linux_x86_64 or a macosx versioned arm64 platform")
    if tag.startswith("macosx_"):
        _, major, minor, _ = tag.split("_")
        if int(major) >= 11 and int(minor) != 0:
            raise ValueError("Modern macOS wheel tags require a major_0 deployment tag")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Native payload requires an explicit file hash inventory")
    def check_metadata(value):
        if isinstance(value, dict):
            for key, item in value.items():
                check_metadata(key)
                check_metadata(item)
        elif isinstance(value, list):
            for item in value:
                check_metadata(item)
        elif isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
            raise ValueError("Native payload metadata must use relative paths")

    check_metadata(manifest)
    encoded = json.dumps(manifest, sort_keys=True)
    if any(prefix in encoded for prefix in ("/Users/", "/home/", "/root/", "/workspace/", "/var/tmp/", "/dev/shm/")):
        raise ValueError("Native payload manifest contains a private absolute path")
    for relative, expected in files.items():
        path = Path(relative)
        if (not relative or path.is_absolute() or "\\" in relative
                or relative != path.as_posix() or any(part in ("..", ".") or part.startswith(".") for part in path.parts)
                or relative == "manifest.json" or " 2" in relative
                or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)):
            raise ValueError("Native payload has an invalid relative file or digest")
        target = root / path
        if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError("Native payload file is missing or unsafe: " + relative)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Native payload file hash mismatch: " + relative)
    if not any("LICENSE" in Path(name).name.upper() or "NOTICE" in Path(name).name.upper() for name in files):
        raise ValueError("Native payload must include its licenses and notices")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native payload must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Native payload has files outside its hash inventory")
    return manifest
