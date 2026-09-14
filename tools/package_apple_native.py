#!/usr/bin/env python3
"""Assemble a hash-verified Apple wheel payload; no compilation or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
resources = runpy.run_path(str(ROOT / "src/fast_audiovae/build_resources.py"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def macho_info(path, available_names):
    """Read load commands without loading or executing the candidate library."""
    header = subprocess.check_output(["otool", "-hv", str(path)], text=True)
    if "ARM64" not in header or "X86_64" in header:
        raise ValueError("Apple payload requires an arm64 Mach-O: " + path.name)
    commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
    minimum = re.findall(r"\bminos\s+(\d+(?:\.\d+){0,2})", commands)
    if not minimum:
        minimum = re.findall(r"\bversion\s+(\d+(?:\.\d+){0,2})", commands)
    if len(minimum) != 1:
        raise ValueError("Unambiguous Mach-O deployment target required: " + path.name)
    rpaths = re.findall(r"cmd LC_RPATH\n.*?\n\s*path\s+(\S+)", commands)
    if any(value != "@loader_path" for value in rpaths):
        raise ValueError("Apple dependency search paths must be loader-relative: " + path.name)
    links = subprocess.check_output(["otool", "-L", str(path)], text=True).splitlines()[1:]
    links = [line.strip().split(" (", 1)[0] for line in links]
    for name in links:
        if name.startswith(("/usr/lib/", "/System/Library/")):
            continue
        if not name.startswith(("@rpath/", "@loader_path/")) or Path(name).name not in available_names:
            raise ValueError("Unbundled or absolute Apple dependency: " + name)
        if name.startswith("@rpath/") and name != links[0] and "@loader_path" not in rpaths:
            raise ValueError("Dependent @rpath library requires @loader_path search path")
        if name.split("/", 1)[1] != Path(name).name:
            raise ValueError("Apple payload uses a flat adjacent library closure")
    return {"minimum_macos": minimum[0], "dependencies": links, "rpaths": rpaths}


def package(base_build, streaming_build, output, *, int8_build=None, inspect_library=macho_info):
    """Use maintainer build receipts and copy exact bytes into a new payload."""
    base_build, streaming_build, output = map(lambda p: Path(p).resolve(), (base_build, streaming_build, output))
    if output.exists():
        raise FileExistsError(output)
    base = json.loads(base_build.read_text())
    selected = json.loads(streaming_build.read_text())
    int8_build = Path(int8_build).resolve() if int8_build is not None else None
    int8 = json.loads(int8_build.read_text()) if int8_build is not None else None
    if selected.get("version") != "apple_stream_selected_build_v1":
        raise ValueError("Selected Apple build version required")
    if (base.get("native_abi"), base.get("ort_api_version"), base.get("domain")) != (1, 29, "venky.audio.cpu"):
        raise ValueError("Original Apple base build required")
    if int8 is not None and int8.get("version") != "apple_firstpair_int8_build_v1":
        raise ValueError("Apple first-pair INT8 build version required")
    runtime = [{"path": base["library"], "sha256": base["library_sha256"]}, *selected["runtime_files"]]
    if int8 is not None:
        runtime.extend(int8["runtime_files"])
    sources, locations = {}, {}
    for item in runtime:
        path = Path(item["path"]).resolve()
        if path.name in sources or sha(path) != item["sha256"] or path.suffix != ".dylib":
            raise ValueError("Duplicate or changed native library: " + path.name)
        sources[path.name] = path
        locations[str(path)] = "apple/runtime/" + path.name
    audits = {name: inspect_library(path, set(sources)) for name, path in sources.items()}
    key = lambda version: tuple(int(x) for x in version.split("."))
    minimum = max((a["minimum_macos"] for a in audits.values()), key=key)
    major = key(minimum)[0]
    if major < 11:
        raise ValueError("Apple ARM requires macOS11 or newer")
    selected_licenses = selected.get("license_files", [])
    int8_licenses = int8.get("license_files", []) if int8 is not None else []
    if not selected_licenses:
        raise ValueError("Selected Apple dependency licenses required")
    if int8 is not None and not int8_licenses:
        raise ValueError("Apple first-pair INT8 dependency licenses required")
    licenses = [*selected_licenses, *int8_licenses]
    license_sources = {
        "licenses/fast-audiovae-LICENSE": ROOT / "LICENSE",
        "licenses/ONNXRUNTIME-LICENSE": ROOT / "experiments/apple-precision/licenses/ONNXRUNTIME-LICENSE",
    }
    for item in licenses:
        path = Path(item["path"]).resolve()
        name = "licenses/" + path.name
        if name in license_sources or sha(path) != item["sha256"]:
            raise ValueError("Duplicate or changed license file")
        license_sources[name] = path
        locations[str(path)] = name
    def rebase(path):
        name = locations.get(str(Path(path).resolve()))
        if name is None:
            raise ValueError("Build record references an unlisted file")
        return name
    base_fields = ("domain", "native_abi", "ort_api_version", "build_id", "openmp", "accelerate", "tile", "operators", "phase_simd")
    clean_base = {k: base[k] for k in base_fields if k in base}
    clean_base.update(library=rebase(base["library"]), library_sha256=base["library_sha256"], original_build_sha256=sha(base_build))
    stream = {k: selected[k] for k in ("version", "complete", "selected_graph_sha256", "required_cpu_features", "onnxruntime") if k in selected}
    stream.update(original_build_sha256=sha(streaming_build),
        runtime_files=[{**item, "path": rebase(item["path"])} for item in selected["runtime_files"]],
        additional_libraries=[{**item, "library": rebase(item["library"])} for item in selected["additional_libraries"]],
        license_files=[{**item, "path": rebase(item["path"])} for item in selected_licenses])
    int8_record = None
    if int8 is not None:
        int8_record = {k: int8[k] for k in ("version", "complete", "required_cpu_features", "onnxruntime", "native_abi", "ort_api_version", "domain") if k in int8}
        int8_record.update(original_build_sha256=sha(int8_build),
            runtime_files=[{**item, "path": rebase(item["path"])} for item in int8["runtime_files"]],
            additional_libraries=[{**item, "library": rebase(item["library"])} for item in int8["additional_libraries"]],
            license_files=[{**item, "path": rebase(item["path"])} for item in int8_licenses])
    inventory = {"apple/runtime/" + name: sha(path) for name, path in sources.items()}
    inventory.update({name: sha(path) for name, path in license_sources.items()})
    resources["validate_streaming_record"](stream, inventory)
    if int8_record is not None:
        resources["validate_int8_record"](int8_record, inventory)
    # Build all metadata and validate it before any native load; no ISA probing here.
    output.mkdir(parents=True)
    for name, path in {**{"apple/runtime/"+n:p for n,p in sources.items()}, **license_sources}.items():
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    for name, value in (("apple/base-build.json", clean_base), ("apple/streaming-build.json", stream), ("apple/macho-closure.json", audits)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    if int8_record is not None:
        (output / "apple/int8-build.json").write_text(json.dumps(int8_record, indent=2) + "\n")
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    base_entry = {"base_build": "apple/base-build.json"}
    manifest = {"version": 1, "wheel_platform": f"macosx_{major}_0_arm64", "minimum_macos": minimum,
        "files": files, "recipes": {"apple_native": base_entry, "apple_stream_projection": base_entry,
            "apple_stream_selected": {**base_entry, "streaming_build": "apple/streaming-build.json"},
            "apple_batch_selected": {**base_entry, "streaming_build": "apple/streaming-build.json"}}}
    if int8_record is not None:
        manifest["recipes"]["apple_stream_int8"] = {
            **base_entry, "streaming_build": "apple/streaming-build.json", "int8_build": "apple/int8-build.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    resources["validate_native_payload"](output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--streaming-build", type=Path, required=True)
    parser.add_argument("--int8-build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = package(args.base_build, args.streaming_build, args.output, int8_build=args.int8_build)
    print(json.dumps({"wheel_platform": record["wheel_platform"], "minimum_macos": record["minimum_macos"], "files": len(record["files"])}))
