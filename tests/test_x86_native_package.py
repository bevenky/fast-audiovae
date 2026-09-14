"""x86 platform-payload packaging using inert bytes, with no native loading."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from fast_audiovae import build_resources, native_payload

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("x86_packager", ROOT / "tools/package_x86_native.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def file(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def exported(root, vendor="amd", selected=True):
    roles = set(packager.BASE_ROLES)
    if selected:
        roles.update(packager.AMD_ROLES if vendor == "amd" else {"pair"})
    libraries = {role: file(root, "libs/" + role + ".so", (vendor + role).encode()) for role in sorted(roles)}
    runtime = [file(root, "libs/vendor.so.1", vendor.encode())]
    licenses = [file(root, "licenses/vendor/LICENSE.txt", b"fixture license")]
    base = {"native_abi": 1, "ort_api_version": 29, "domain": "venky.audio.cpu.portable",
            "tile": 256, "operators": ["SnakeF32"], "fingerprint": {"explicit_avx512_backend": 5},
            "library": libraries["native"]["path"], "library_sha256": libraries["native"]["sha256"]}
    manifest = {"schema_version": 1, "vendor": vendor, "cpu_only": True, "onnxruntime": "1.30.0",
                "canonical_math_version": 1, "required_cpu_flags": ["avx2", "avx512f", "avx512_vnni"],
                "libraries": libraries, "runtime_files": runtime, "license_files": licenses,
                "native_build": file(root, "native-build.json", json.dumps(base).encode())}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def inert_audit(path, names, *, executable=False):
    if executable:
        assert path.name == "cpu_probe" and not names
    else:
        assert path.name in names
    return {"dependencies": ["libc.so.6"], "soname": None, "rpaths": [],
            "glibc_versions": ["2.38"], "minimum_glibc": "2.38"}


def inert_probe(root):
    record = file(root, "cpu_probe", b"not executable machine code")
    (root / "cpu_probe").chmod(0o755)
    return {**record, "source_sha256": "0" * 64, "compiler": "cc", "flags": ["-O2", "-march=x86-64"]}


def package(tmp_path, *, intel=True, selected=True):
    amd, _ = exported(tmp_path / "input-amd", selected=selected)
    other = exported(tmp_path / "input-intel", "intel", selected=selected)[0] if intel else None
    output = tmp_path / "payload"
    manifest = packager.package(amd, output, intel=other, inspect_library=inert_audit, build_probe=inert_probe)
    return output, manifest


def test_combined_payload_retains_both_vendors_selected_and_legacy(tmp_path):
    output, manifest = package(tmp_path)
    assert manifest["wheel_platform"] == "linux_x86_64"
    assert manifest["minimum_glibc"] == "2.38"
    assert set(manifest["recipes"]) == {"amd_precision", "amd_stream_selected", "intel_precision", "intel_stream_projection"}
    assert manifest["recipes"]["intel_stream_projection"] == {"payload_dir": "intel", "manifest": "manifest.json"}
    assert manifest["files"]["amd/libs/native.so"] != manifest["files"]["intel/libs/native.so"]
    assert (output / "cpu_probe").stat().st_mode & 0o111
    assert build_resources.validate_native_payload(output) == manifest
    for vendor in ("amd", "intel"):
        source = tmp_path / ("input-" + vendor)
        assert (output / vendor / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
        for path in source.rglob("*"):
            if path.is_file():
                assert (output / vendor / path.relative_to(source)).read_bytes() == path.read_bytes()
    assert not any(str(tmp_path) in path.read_text() for path in output.rglob("*.json"))


@pytest.mark.parametrize("intel,selected,expected", [
    (False, True, {"amd_precision", "amd_stream_selected"}),
    (False, False, {"amd_precision"}),
    (True, False, {"amd_precision", "intel_precision"}),
])
def test_optional_intel_and_legacy_export_do_not_advertise_missing_roles(tmp_path, intel, selected, expected):
    _, manifest = package(tmp_path, intel=intel, selected=selected)
    assert set(manifest["recipes"]) == expected


def test_materialize_is_compatible_with_existing_x86_schema(tmp_path, monkeypatch):
    output, manifest = package(tmp_path)
    monkeypatch.setattr(native_payload, "PAYLOAD_ROOT", output)
    payload = native_payload.inspect_payload()
    for recipe, vendor in (("amd_stream_selected", "amd"), ("intel_stream_projection", "intel")):
        cache = tmp_path / recipe / "prebuilt"
        ready = native_payload.materialize_payload(payload, cache, recipe)
        assert Path(ready["root"]) == cache / vendor
        assert ready["manifest"] == json.loads((output / vendor / "manifest.json").read_text())
        assert ready["manifest"]["vendor"] == vendor
        for item in ready["manifest"]["libraries"].values():
            assert packager.sha(Path(ready["root"]) / item["path"]) == item["sha256"]
    assert manifest["probe"]["sha256"] == packager.sha(cache / "cpu_probe")


@pytest.mark.parametrize("failure", ["changed", "unlisted", "partial_selected", "wrong_vendor", "gpu_policy",
                                     "duplicate", "model", "traversal", "private_metadata", "base_hash", "backend_bool", "symlink"])
def test_invalid_exports_fail_before_probe_or_output(tmp_path, failure):
    root, manifest = exported(tmp_path / "input")
    if failure == "changed":
        (root / "libs/native.so").write_bytes(b"corrupted")
    elif failure == "unlisted":
        (root / "checkpoint.pt").write_bytes(b"model must never ship")
    elif failure == "partial_selected":
        record = manifest["libraries"].pop("amd_stream_phase")
        (root / record["path"]).unlink()
    elif failure == "wrong_vendor":
        manifest["vendor"] = "intel"
    elif failure == "gpu_policy":
        manifest["cpu_only"] = False
    elif failure == "duplicate":
        manifest["runtime_files"].append(manifest["libraries"]["native"])
    elif failure == "model":
        manifest["license_files"].append(file(root, "licenses/model.safetensors", b"model"))
    elif failure == "traversal":
        manifest["libraries"]["native"]["path"] = "libs/../libs/native.so"
    elif failure == "private_metadata":
        manifest["source"] = "/private/build/receipt.json"
    elif failure in ("base_hash", "backend_bool"):
        base_path = root / "native-build.json"
        base = json.loads(base_path.read_text())
        if failure == "base_hash":
            base["library_sha256"] = "0" * 64
        else:
            base["fingerprint"]["explicit_avx512_backend"] = True
        base_path.write_text(json.dumps(base))
        manifest["native_build"]["sha256"] = packager.sha(base_path)
    else:
        target = root / "libs/native.so"
        target.rename(root / "outside")
        target.symlink_to(root / "outside")
    (root / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    with pytest.raises(ValueError):
        packager.package(root, tmp_path / "output", inspect_library=inert_audit,
                         build_probe=lambda path: calls.append(path))
    assert not calls and not (tmp_path / "output").exists()


def test_existing_output_and_probe_failure_preserve_inputs(tmp_path):
    root, _ = exported(tmp_path / "input")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        packager.package(root, output)
    def failed_probe(path):
        raise RuntimeError("fixture compiler failure")
    with pytest.raises(RuntimeError, match="compiler failure"):
        packager.package(root, tmp_path / "fresh", inspect_library=inert_audit, build_probe=failed_probe)
    assert not (tmp_path / "fresh").exists()
    assert (root / "manifest.json").is_file()
    assert not list(tmp_path.glob(".x86-payload-*"))


def readelf_fixture(monkeypatch, *, needed=("core.so", "libc.so.6"), rpath="$ORIGIN", soname="op.so", machine="Advanced Micro Devices X86-64", versions=None):
    def read(command, **kwargs):
        assert command[0] == "readelf" and kwargs["text"]
        if command[1] == "-hW":
            return f"Class: ELF64\nData: 2's complement, little endian\nType: DYN (Shared object file)\nMachine: {machine}\n"
        if command[1] == "--version-info":
            return versions if versions is not None else "Version needs section '.gnu.version_r':\nName: GLIBC_2.38\n"
        assert command[1] == "-dW"
        return ("".join(f"0 (NEEDED) Shared library: [{name}]\n" for name in needed)
                + (f"0 (RUNPATH) Library runpath: [{rpath}]\n" if rpath is not None else "")
                + (f"0 (SONAME) Library soname: [{soname}]\n" if soname else ""))
    monkeypatch.setattr(packager.subprocess, "check_output", read)


def test_elf_reader_verifies_architecture_and_adjacent_dependency_closure(monkeypatch):
    readelf_fixture(monkeypatch)
    result = packager.elf_info(Path("op.so"), {"op.so", "core.so"})
    assert result == {"dependencies": ["core.so", "libc.so.6"], "soname": "op.so", "rpaths": ["$ORIGIN"],
                      "glibc_versions": ["2.38"], "minimum_glibc": "2.38"}
    with pytest.raises(ValueError, match="Unbundled"):
        packager.elf_info(Path("op.so"), {"op.so"})


@pytest.mark.parametrize("values,error", [
    ({"machine": "AArch64"}, "x86-64"),
    ({"needed": ("/private/libcore.so",)}, "Unbundled"),
    ({"needed": ("libcuda.so.1",)}, "GPU dependency"),
    ({"rpath": "/workspace/build"}, "search paths"),
    ({"rpath": "$ORIGIN:"}, "search paths"),
    ({"rpath": None}, "require \\$ORIGIN"),
    ({"soname": "other.so"}, "SONAME"),
])
def test_elf_reader_rejects_unportable_or_gpu_dependencies(monkeypatch, values, error):
    readelf_fixture(monkeypatch, **values)
    with pytest.raises(ValueError, match=error):
        packager.elf_info(Path("op.so"), {"op.so", "core.so", "libcuda.so.1"})


def test_glibc_reader_ignores_glibcxx_and_local_definitions(monkeypatch):
    readelf_fixture(monkeypatch, versions="""Version definition section '.gnu.version_d':
Name: GLIBC_9.99
Version needs section '.gnu.version_r':
Name: GLIBCXX_3.4.32
Name: CXXABI_1.3.13
Name: GLIBC_2.2.5
Name: GLIBC_2.40
Name: GLIBC_2.38
""")
    result = packager.elf_info(Path("op.so"), {"op.so", "core.so"})
    assert result["glibc_versions"] == ["2.2.5", "2.38", "2.40"]
    assert result["minimum_glibc"] == "2.40"


@pytest.mark.parametrize("newer", ["intel/core.so", "cpu_probe", None])
def test_manifest_glibc_uses_all_vendors_and_probe_with_existing_floor(tmp_path, newer):
    amd, _ = exported(tmp_path / "amd")
    intel, _ = exported(tmp_path / "intel", "intel")
    def inspect(path, names, *, executable=False):
        record = inert_audit(path, names, executable=executable)
        match = path.name == newer or (newer == "intel/core.so" and path == intel / "libs/core.so")
        record["minimum_glibc"] = "2.40" if match else "2.17"
        record["glibc_versions"] = [record["minimum_glibc"]]
        return record
    result = packager.package(amd, tmp_path / "payload", intel=intel,
                              inspect_library=inspect, build_probe=inert_probe)
    assert result["minimum_glibc"] == ("2.40" if newer else "2.38")
