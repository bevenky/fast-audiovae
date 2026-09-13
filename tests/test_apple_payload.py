"""Wheel closure and cache checks with inert files; no native operator execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from fast_audiovae import build_resources, native_payload

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("apple_packager", ROOT / "tools/package_apple_native.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def file(root, name, contents=b"inert fixture"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def builds(root):
    base = file(root, "base.dylib")
    dep = file(root, "libxsmm.dylib", b"dependency")
    core = file(root, "core.dylib", b"core")
    bridge = file(root, "bridge.dylib", b"bridge")
    license_file = file(root, "LIBXSMM-LICENSE.md", b"fixture license")
    base_record = dict(library=base["path"], library_sha256=base["sha256"],
        native_abi=1, ort_api_version=29, domain="venky.audio.cpu", operators=["SnakeF32"],
        commands=[["/private/compiler", "should not enter payload"]])
    runtime = [{**dep, "register": False}, {**core, "register": True, "domain": "core.v1"},
               {**bridge, "register": True, "domain": "bridge.v1"}]
    stream = dict(version="apple_stream_selected_build_v1", complete=True,
        selected_graph_sha256="5" * 64, required_cpu_features=["sme", "sme2"], onnxruntime="1.30.0",
        runtime_files=runtime, additional_libraries=[dict(library=x["path"], sha256=x["sha256"], domain=x["domain"]) for x in runtime if x["register"]],
        license_files=[license_file], commands=[["/private/compiler"]])
    paths = [root / "base.json", root / "stream.json"]
    for path, record in zip(paths, [base_record, stream]):
        path.write_text(json.dumps(record))
    return paths


def inert_audit(path, names):
    assert path.name in names
    return {"minimum_macos": "26.2" if path.name == "libxsmm.dylib" else "14.0",
            "dependencies": ["@rpath/" + path.name, "/usr/lib/libSystem.B.dylib"], "rpaths": ["@loader_path"]}


def payload(tmp_path):
    a, b = builds(tmp_path / "inputs")
    dest = tmp_path / "payload"
    manifest = packager.package(a, b, dest, inspect_library=inert_audit)
    return dest, manifest


def test_package_preserves_relative_runtime_closure_licenses_and_fallback(tmp_path):
    root, manifest = payload(tmp_path)
    assert manifest["wheel_platform"] == "macosx_26_0_arm64"
    assert manifest["minimum_macos"] == "26.2"
    assert set(manifest["recipes"]) == {"apple_native", "apple_stream_projection", "apple_stream_selected", "apple_batch_selected"}
    stream = json.loads((root / "apple/streaming-build.json").read_text())
    assert stream["complete"] and stream["required_cpu_features"] == ["sme", "sme2"]
    assert [Path(x["library"]).name for x in stream["additional_libraries"]] == ["core.dylib", "bridge.dylib"]
    assert "libxsmm.dylib" not in [Path(x["library"]).name for x in stream["additional_libraries"]]
    assert {"licenses/ONNXRUNTIME-LICENSE", "licenses/fast-audiovae-LICENSE", "licenses/LIBXSMM-LICENSE.md"} <= set(manifest["files"])
    for p in root.rglob("*.json"):
        assert "/private/compiler" not in p.read_text()
        assert str(tmp_path) not in p.read_text()
    assert build_resources.validate_native_payload(root) == manifest


def test_materialize_rebases_and_load_probes_dependencies_before_bridges(tmp_path, monkeypatch):
    root, manifest = payload(tmp_path)
    monkeypatch.setattr(native_payload, "PAYLOAD_ROOT", root)
    observed = native_payload.inspect_payload()
    dest = tmp_path / "cache/prebuilt"
    result = native_payload.materialize_payload(observed, dest, "apple_stream_selected")
    stream = json.loads(Path(result["streaming_build"]).read_text())
    assert all(Path(x["path"]).is_relative_to(dest) for x in stream["runtime_files"] + stream["license_files"])
    assert [Path(x["library"]).name for x in stream["additional_libraries"]] == ["core.dylib", "bridge.dylib"]
    calls = []
    monkeypatch.setattr(native_payload, "_load_probe", lambda paths, ids: (calls.append((paths, ids)) or (True, "")))
    assert native_payload.probe_payload(result) == (True, "")
    assert [Path(x).name for x in calls[0][0]] == ["base.dylib", "libxsmm.dylib", "core.dylib", "bridge.dylib"]
    assert len(calls[0][1]) == 4
    (dest / "apple/runtime/libxsmm.dylib").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        native_payload.probe_payload(result)
    assert len(calls) == 1


def test_batch_reuses_verified_streaming_payload_from_existing_wheel(tmp_path, monkeypatch):
    root, manifest = payload(tmp_path)
    del manifest["recipes"]["apple_batch_selected"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(native_payload, "PAYLOAD_ROOT", root)
    observed = native_payload.inspect_payload()
    result = native_payload.materialize_payload(observed, tmp_path / "cache/prebuilt", "apple_batch_selected")
    assert result is not None
    assert Path(result["base_build"]).is_file()
    assert Path(result["streaming_build"]).is_file()
    assert json.loads(Path(result["streaming_build"]).read_text())["complete"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "transitive_registered", "domain", "private", "license", "incomplete"])
def test_invalid_selected_closure_is_rejected_before_materializing(tmp_path, failure):
    root, manifest = payload(tmp_path)
    path = root / "apple/streaming-build.json"
    stream = json.loads(path.read_text())
    if failure == "missing":
        stream["additional_libraries"].pop()
    elif failure == "duplicate":
        stream["runtime_files"].append(copy.deepcopy(stream["runtime_files"][0]))
    elif failure == "transitive_registered":
        item = stream["runtime_files"][0]
        stream["additional_libraries"].append(dict(library=item["path"], sha256=item["sha256"], domain="wrong"))
    elif failure == "domain":
        stream["additional_libraries"][0]["domain"] = "wrong"
    elif failure == "private":
        stream["runtime_files"][0]["path"] = "/private/outside.dylib"
    elif failure == "license":
        stream["license_files"][0]["sha256"] = "0" * 64
    else:
        stream["complete"] = False
    path.write_text(json.dumps(stream))
    manifest["files"]["apple/streaming-build.json"] = packager.sha(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_resources.validate_native_payload(root)


def test_packager_never_overwrites_or_accepts_changed_dependency(tmp_path):
    a, b = builds(tmp_path / "inputs")
    output = tmp_path / "payload"
    output.mkdir()
    with pytest.raises(FileExistsError):
        packager.package(a, b, output, inspect_library=inert_audit)
    (a.parent / "libxsmm.dylib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        packager.package(a, b, tmp_path / "fresh", inspect_library=inert_audit)
    assert not (tmp_path / "fresh").exists()


def test_macho_reader_rejects_missing_dependency_and_reads_deployment(monkeypatch):
    commands = "cmd LC_BUILD_VERSION\n cmdsize 32\n platform 1\n minos 26.2\n sdk 26.5\ncmd LC_RPATH\n cmdsize 32\n path @loader_path (offset 12)\n"
    def read(cmd, **kw):
        if cmd[1] == "-hv":
            return "Mach header\n ARM64 ALL DYLIB"
        if cmd[1] == "-l":
            return commands
        return "file:\n\t@rpath/bridge.dylib (compatibility version1)\n\t@rpath/core.dylib (compatibility version1)\n\t/usr/lib/libSystem.B.dylib (compatibility version1)\n"
    monkeypatch.setattr(packager.subprocess, "check_output", read)
    assert packager.macho_info(Path("bridge.dylib"), {"bridge.dylib", "core.dylib"})["minimum_macos"] == "26.2"
    with pytest.raises(ValueError, match="Unbundled"):
        packager.macho_info(Path("bridge.dylib"), {"bridge.dylib"})


def test_selected_source_allowlist_is_exact_and_has_no_experiment_dependency():
    pin = json.loads((ROOT / "native/apple/streaming/sources.json").read_text())
    prefix = "native/apple/streaming/"
    shipped = {x[len(prefix):] for x in build_resources.RESOURCE_FILES if x.startswith(prefix)}
    assert shipped == set(pin["files"]) | {"sources.json"}
    assert "tools/build_apple_streaming.py" in build_resources.RESOURCE_FILES
    assert "tools/package_apple_native.py" in build_resources.RESOURCE_FILES
    for name, record in pin["files"].items():
        assert packager.sha(ROOT / prefix / name) == record["sha256"]
    assert not any("apple-streaming-v2" in x for x in build_resources.RESOURCE_FILES)
