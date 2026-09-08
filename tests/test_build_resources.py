"""Packaging and writable-cache safety checks; no native compilation/inference."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_resources_under_test", ROOT / "src/fast_audiovae/build_resources.py")
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


def test_source_allowlist_has_recipe_and_license_closure():
    files = resources.resource_files(ROOT)
    assert len(files) == len(set(files)) == len(resources.RESOURCE_FILES)
    assert "experiments/cpu-stage/stage/build.py" in resources.RESOURCE_FILES
    assert "experiments/intel-precision/pins/mkl.json" in resources.RESOURCE_FILES
    assert "experiments/amd-precision/pins/rebuild.json" in resources.RESOURCE_FILES
    assert "experiments/streaming-matrix/intel/paired_projection.cpp" in resources.RESOURCE_FILES
    assert "experiments/intel-precision/licenses/onemkl/LICENSE.txt" in resources.RESOURCE_FILES
    assert "experiments/amd-precision/licenses/AOCL-NOTICES" in resources.RESOURCE_FILES
    for path in files:
        relative = str(path.relative_to(ROOT))
        assert not any(part.startswith(".") for part in Path(relative).parts)
        assert " 2" not in relative and "benchmarks/" not in relative
        assert path.suffix not in {".onnx", ".data", ".npz", ".so", ".dylib", ".o", ".a"}
        assert b"/Users/" not in path.read_bytes()
    pins = json.loads((ROOT / "experiments/cpu-stage/dependency-pins.json").read_text())
    assert pins["onnxruntime"] == "1.29.0" and pins["ort_api"] == 29
    assert len(pins["ort_headers"]["sha256"]) == 6


def test_materialize_preserves_dependency_cache_and_ignores_unlisted_files(tmp_path):
    target = tmp_path / "cache"
    (target / ".deps").mkdir(parents=True)
    (target / ".build").mkdir()
    dependency = target / ".deps/download.tar"
    binary = target / ".build/kernel.so"
    dependency.write_bytes(b"downloaded dependency")
    binary.write_bytes(b"compiled kernel")
    fingerprint = resources.materialize_resources(target)
    assert fingerprint == resources.resource_fingerprint()
    assert dependency.read_bytes() == b"downloaded dependency"
    assert binary.read_bytes() == b"compiled kernel"
    assert not (target / ".env").exists()
    record = json.loads((target / ".build-resources.json").read_text())
    assert set(record["files"]) == set(resources.RESOURCE_FILES)
    native = target / "native/apple/native_kernels.c"
    stamp = native.stat().st_mtime_ns
    assert resources.materialize_resources(target) == fingerprint
    assert native.stat().st_mtime_ns == stamp
    native.write_bytes(b"modified cache source")
    assert resources.materialize_resources(target) == fingerprint
    assert native.read_bytes() == (ROOT / "native/apple/native_kernels.c").read_bytes()


def test_installed_tree_works_without_repository_or_git(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages/fast_audiovae"
    installed.mkdir(parents=True)
    resources.materialize_resources(installed / "_build_resources")
    monkeypatch.setattr(resources, "__file__", str(installed / "build_resources.py"))
    assert resources.resource_root() == installed / "_build_resources"
    work = tmp_path / "runtime-cache"
    assert resources.materialize_resources(work) == resources.resource_fingerprint()
    assert (work / "tools/build_apple.py").is_file()
    assert not (work / ".git").exists()


def test_missing_installed_resource_fails_before_modifying_destination(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages/fast_audiovae"
    installed.mkdir(parents=True)
    resources.materialize_resources(installed / "_build_resources")
    (installed / "_build_resources/native/apple/custom_ops.cpp").unlink()
    monkeypatch.setattr(resources, "__file__", str(installed / "build_resources.py"))
    target = tmp_path / "build"
    with pytest.raises(RuntimeError, match="Missing or unsafe"):
        resources.materialize_resources(target)
    assert not target.exists()


def test_destination_symlink_cannot_overwrite_unrelated_files(tmp_path):
    target = tmp_path / "build"
    outside = tmp_path / "outside"
    target.mkdir()
    outside.mkdir()
    (outside / "kept").write_text("unchanged")
    (target / "native").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe source path"):
        resources.materialize_resources(target)
    assert list(outside.iterdir()) == [outside / "kept"]
    assert (outside / "kept").read_text() == "unchanged"


def test_materializing_over_maintained_sources_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        resources.materialize_resources(ROOT)


def _payload(root, tag="linux_x86_64"):
    import hashlib
    files = {"intel/libcodec.so": b"test binary payload", "licenses/LICENSE.txt": b"test license"}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"version": 1, "wheel_platform": tag,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                "recipes": {"intel_native": {"library": "intel/libcodec.so"}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_native_payload_inventory_and_explicit_platform(tmp_path):
    payload = tmp_path / "payload"
    expected = _payload(payload)
    assert resources.validate_native_payload(payload) == expected
    assert resources.validate_native_payload(payload)["wheel_platform"] == "linux_x86_64"
    expected["wheel_platform"] = "macosx_26_0_arm64"
    (payload / "manifest.json").write_text(json.dumps(expected))
    assert resources.validate_native_payload(payload) == expected


@pytest.mark.parametrize("failure", ["hash", "unlisted", "symlink", "private_path", "platform", "mac_minor"])
def test_native_payload_fails_closed(tmp_path, failure):
    payload = tmp_path / "payload"
    manifest = _payload(payload)
    if failure == "hash":
        (payload / "intel/libcodec.so").write_bytes(b"changed")
    elif failure == "unlisted":
        (payload / ".env").write_text("private configuration")
    elif failure == "symlink":
        (payload / "linked").symlink_to(payload / "intel/libcodec.so")
    elif failure == "private_path":
        manifest["source"] = "/Users/example/private-build"
        (payload / "manifest.json").write_text(json.dumps(manifest))
    elif failure == "mac_minor":
        manifest["wheel_platform"] = "macosx_26_5_arm64"
        (payload / "manifest.json").write_text(json.dumps(manifest))
    else:
        manifest["wheel_platform"] = "manylinux_2_17_x86_64"
        (payload / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        resources.validate_native_payload(payload)
