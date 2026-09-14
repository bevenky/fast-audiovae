#!/usr/bin/env python3
"""Package exported Linux CPU libraries and a baseline-ISA CPU probe.

No model or native library is executed. The probe is compiled on the Linux
packaging host; relocated runtime validation remains a release prerequisite.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
resources = runpy.run_path(str(ROOT / "src/fast_audiovae/build_resources.py"))
platforms = runpy.run_path(str(ROOT / "src/fast_audiovae/platforms.py"))
BASE_ROLES = {"native", "core", "ops", "stage", "upsample", "fp32_stage"}
AMD_ROLES = {"amd_stream_pair", "amd_stream_history", "amd_stream_phase"}
INTEL_ROLES = {"intel_stream_history", "intel_stream_phase", "intel_stream_matrix",
               "intel_stream_matrix_ops", "intel_stream_onednn"}
SYSTEM_LIBRARIES = {"libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0",
                    "librt.so.1", "libgcc_s.so.1", "libstdc++.so.6", "ld-linux-x86-64.so.2"}
GPU_NAMES = re.compile(r"cuda|cudnn|cublas|hip|rocm|rocblas|opencl|sycl|level.zero|ze_loader", re.I)
DIGEST = re.compile(r"[a-f0-9]{64}")
GLIBC_FLOOR = "2.38"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version_key(value):
    require(isinstance(value, str) and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value),
            "Invalid glibc version")
    return tuple(int(part) for part in value.split("."))


def relative_file(root, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name,
            "Invalid payload path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and not any(part.startswith(".") for part in path.parts), "Unsafe payload path: " + name)
    result = root / name
    require(result.is_file() and not result.is_symlink()
            and result.resolve().is_relative_to(root), "Missing or unsafe payload file: " + name)
    require(not any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts))),
            "Symlinked payload directory: " + name)
    return result


def metadata_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            metadata_paths(key); metadata_paths(item)
    elif isinstance(value, list):
        for item in value:
            metadata_paths(item)
    elif isinstance(value, str):
        require(not value.startswith("/") and not re.match(r"^[A-Za-z]:[\\/]", value),
                "Export metadata must contain relative paths")


def elf_info(path, available_names, *, executable=False):
    """Inspect ELF headers/load commands, without loading the file."""
    header = subprocess.check_output(["readelf", "-hW", str(path)], text=True)
    require(re.search(r"Class:\s+ELF64\b", header) and re.search(r"Machine:.*X86-64", header)
            and re.search(r"Data:.*little endian", header), "Linux x86-64 ELF required: " + path.name)
    kinds = r"(?:DYN|EXEC)" if executable else "DYN"
    require(re.search(r"Type:\s+" + kinds + r"\b", header), "Unexpected ELF type: " + path.name)
    dynamic = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic)
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", dynamic)
    rpaths = [part for value in re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]*)\]", dynamic)
              for part in value.split(":")]
    require(all(value in ("$ORIGIN", "${ORIGIN}") for value in rpaths),
            "ELF search paths must be adjacent-library relative: " + path.name)
    require(len(sonames) <= 1 and all(name == path.name for name in sonames),
            "ELF SONAME must match its packaged filename: " + path.name)
    for name in [path.name, *needed]:
        require(not GPU_NAMES.search(name), "GPU dependency is forbidden in CPU payload: " + name)
    require(all("/" not in name and name in available_names | SYSTEM_LIBRARIES for name in needed),
            "Unbundled or absolute ELF dependency: " + path.name)
    adjacent = set(needed) - SYSTEM_LIBRARIES
    require(not adjacent or bool(rpaths), "Bundled ELF dependencies require $ORIGIN: " + path.name)
    versions = subprocess.check_output(["readelf", "--version-info", "--wide", str(path)], text=True)
    # Only imported version requirements count, not locally defined versions
    # or similarly named GLIBCXX/CXXABI requirements from the C++ runtime.
    requirements = versions.partition("Version needs section")[2]
    requirements = re.split(r"\nVersion (?:symbols|definition) section", requirements, maxsplit=1)[0]
    glibc = sorted(set(re.findall(r"\bName:\s+GLIBC_(\d+\.\d+(?:\.\d+)?)\b", requirements)), key=version_key)
    return {"dependencies": needed, "soname": sonames[0] if sonames else None, "rpaths": rpaths,
            "glibc_versions": glibc, "minimum_glibc": max(glibc, key=version_key, default="0.0")}


def inspect_export(root, vendor, inspect_library):
    """Require the exact weight-free inventory emitted by export_native_payload."""
    root = Path(root).resolve()
    manifest_path = relative_file(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1
            and manifest.get("vendor") == vendor and manifest.get("cpu_only") is True
            and manifest.get("canonical_math_version") == 1, "Incompatible exported CPU payload")
    metadata_paths(manifest)
    require(re.fullmatch(r"\d+\.\d+\.\d+", manifest.get("onnxruntime", "")), "Missing ONNX Runtime version")
    flags = manifest.get("required_cpu_flags")
    require(isinstance(flags, list) and all(isinstance(flag, str) for flag in flags)
            and {"avx2", "avx512f", "avx512_vnni"}.issubset(flags), "Missing CPU requirements")
    libraries = manifest.get("libraries", {})
    allowed = BASE_ROLES | (AMD_ROLES if vendor == "amd" else {"pair"} | INTEL_ROLES)
    require(isinstance(libraries, dict) and BASE_ROLES <= set(libraries) <= allowed,
            "Missing or unknown CPU library role")
    selected_roles = AMD_ROLES if vendor == "amd" else INTEL_ROLES
    selected = selected_roles & set(libraries)
    require(not selected or selected == selected_roles, "Selected vendor requires all library roles")
    require(not selected or vendor != "intel" or "pair" in libraries,
            "Selected Intel requires its retained first projection pair")
    runtime, licenses = manifest.get("runtime_files"), manifest.get("license_files")
    require(isinstance(runtime, list) and isinstance(licenses, list) and bool(licenses),
            "CPU runtime and license inventories required")
    files = {}
    def record(item, kind):
        require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "Invalid exported file record")
        name, digest = item["path"], item["sha256"]
        path = relative_file(root, name)
        require(isinstance(digest, str) and DIGEST.fullmatch(digest) and sha(path) == digest,
                "Changed exported file: " + name)
        require(name not in files, "Duplicate exported file: " + name)
        if kind == "library":
            require(len(PurePosixPath(name).parts) == 2 and name.startswith("libs/")
                    and re.fullmatch(r"[A-Za-z0-9_+.-]+\.so(?:\.\d+)*", path.name),
                    "Expected a flat ELF library inventory")
        elif kind == "license":
            require(name.startswith("licenses/") and not path.suffix.lower() in
                    {".onnx", ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".wav", ".flac", ".mp3"},
                    "Only license files may be exported under licenses/")
        else:
            require(name == "native-build.json", "Expected native-build.json")
        files[name] = path
    for item in [*libraries.values(), *runtime]:
        record(item, "library")
    for item in licenses:
        record(item, "license")
    record(manifest.get("native_build"), "metadata")
    base = json.loads(files["native-build.json"].read_text())
    metadata_paths(base)
    backend = base.get("fingerprint", {}).get("explicit_avx512_backend")
    require(base.get("native_abi") == 1 and base.get("ort_api_version") == 29
            and base.get("library") == libraries["native"]["path"]
            and base.get("library_sha256") == libraries["native"]["sha256"]
            and type(backend) is int and backend == 5,
            "Native base metadata does not match exported library")
    all_paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in all_paths), "Export contains a symlink")
    require({path.relative_to(root).as_posix() for path in all_paths if path.is_file()}
            == set(files) | {"manifest.json"}, "Export contains files outside its weight-free inventory")
    names = {path.name for name, path in files.items() if name.startswith("libs/")}
    audits = {name: inspect_library(path, names) for name, path in files.items() if name.startswith("libs/")}
    files["manifest.json"] = manifest_path
    return manifest, files, audits


def package(amd, output, *, intel=None, inspect_library=elf_info, build_probe=None):
    """Assemble a new generic Linux platform payload; never overwrite a build."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require(amd is not None or intel is not None, "At least one vendor payload is required")
    inputs = {"amd": inspect_export(amd, "amd", inspect_library)} if amd is not None else {}
    if intel is not None:
        inputs["intel"] = inspect_export(intel, "intel", inspect_library)
    # All libraries are checked before compiling even the baseline-ISA helper.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".x86-payload-", dir=output.parent) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        recipes, audits = {}, {}
        for vendor, (source_manifest, files, audit) in inputs.items():
            for name, source in files.items():
                target = staging / vendor / name
                target.parent.mkdir(parents=True, exist_ok=True)
                expected = sha(source)
                shutil.copy2(source, target)
                require(sha(target) == expected, "Export changed during copy")
            # Check recorded hashes again after copying, including base metadata.
            copied = json.loads((staging / vendor / "manifest.json").read_text())
            require(copied == source_manifest, "Export manifest changed during copy")
            for item in [*copied["libraries"].values(), *copied["runtime_files"],
                         *copied["license_files"], copied["native_build"]]:
                require(sha(staging / vendor / item["path"]) == item["sha256"], "Export changed during copy")
            entry = {"payload_dir": vendor, "manifest": "manifest.json"}
            recipes[vendor + "_precision"] = entry
            if vendor == "amd" and AMD_ROLES <= set(copied["libraries"]):
                recipes["amd_stream_selected"] = entry
            if vendor == "intel" and "pair" in copied["libraries"]:
                recipes["intel_stream_projection"] = entry
                if INTEL_ROLES <= set(copied["libraries"]):
                    recipes["intel_stream_selected"] = entry
            audits[vendor] = audit
        license_path = staging / "licenses/fast-audiovae-LICENSE"
        license_path.parent.mkdir()
        shutil.copyfile(ROOT / "LICENSE", license_path)
        probe = (build_probe or platforms["build_probe"])(staging)
        require(isinstance(probe, dict) and probe.get("path") == "cpu_probe"
                and probe.get("sha256") == sha(relative_file(staging, "cpu_probe")), "Invalid CPU probe build record")
        metadata_paths(probe)
        require((staging / "cpu_probe").stat().st_mode & 0o111, "Packaged CPU probe must be executable")
        audits["probe"] = inspect_library(staging / "cpu_probe", set(), executable=True)
        minimum_glibc = max([GLIBC_FLOOR, audits["probe"]["minimum_glibc"],
                            *(audit["minimum_glibc"] for vendor in inputs for audit in audits[vendor].values())],
                           key=version_key)
        (staging / "elf-closure.json").write_text(json.dumps(audits, indent=2) + "\n")
        files = {path.relative_to(staging).as_posix(): sha(path)
                 for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {"version": 1, "wheel_platform": "linux_x86_64", "files": files,
                    "recipes": recipes, "probe": probe, "minimum_glibc": minimum_glibc}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        resources["validate_native_payload"](staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--amd", type=Path, help="Exported AMD CPU payload directory")
    parser.add_argument("--intel", type=Path, help="Exported Intel payload; supply for a combined release wheel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = package(args.amd, args.output, intel=args.intel)
    print(json.dumps({"wheel_platform": record["wheel_platform"], "recipes": sorted(record["recipes"]),
                      "files": len(record["files"])}))
