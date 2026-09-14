"""Build the accepted AMD pair/history/phase closure; no loading or inference.

The native base and AOCL core already exist in canonical bundle staging. All
new runtime lookup is SONAME plus $ORIGIN; no experiment paths are embedded.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

LIBRARIES = {"amd_stream_pair": "libamd_int8_paired_projection.so",
             "amd_stream_history": "libaudiovae_amd_a2_rawhistory.so",
             "amd_stream_phase": "liba4_phase.so"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def elf_metadata(path):
    text = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
    for line in text.splitlines():
        if "[" not in line:
            continue
        value = line.split("[", 1)[1].split("]", 1)[0]
        if "(NEEDED)" in line or "(SONAME)" in line:
            require("/" not in value, "Nonrelocatable ELF dependency: " + value)
        if "(RPATH)" in line or "(RUNPATH)" in line:
            require(all(x in ("$ORIGIN", "${ORIGIN}") for x in value.split(":")), "Nonlocal runtime search path")
    return text


def build(work, bundle, output):
    work, bundle, output = (Path(x).resolve() for x in (work, bundle, output))
    require(platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"), "Native Linux x86-64 builder required")
    source = work / "native/amd/streaming"
    closure = json.loads((source / "sources.json").read_text())
    require(closure.get("version") == "amd_stream_selected_sources_v1", "Unknown AMD source closure")
    inputs = {source / "sources.json": sha(source / "sources.json"), Path(__file__).resolve(): sha(__file__)}
    for relative, row in closure["files"].items():
        path = (source / relative).resolve()
        require(path.is_relative_to(source), "Source leaves closure")
        inputs[path] = row["sha256"]
    for relative, value in closure["base_sources"].items():
        inputs[work / relative] = value
    spec = importlib.util.spec_from_file_location("amd_accepted_x86_build", work / "tools/build_x86.py")
    module = importlib.util.module_from_spec(spec)
    # Verify before importing any build helper from the materialized tree.
    for path, expected in inputs.items():
        require(sha(path) == expected, "Maintained source mismatch: " + str(path))
    spec.loader.exec_module(module)
    include = work / ".deps/onnxruntime/include"; module.require_headers(include)
    record_path = work / ".build/x86/build.json"
    native = json.loads(record_path.read_text()); fp = native["fingerprint"]
    manifest = json.loads((bundle / "bundle.json").read_text()); base = manifest["native"]["Linux/x86_64"]
    relative = Path(base["library"]); library = (bundle / relative).resolve()
    require(not relative.is_absolute() and library.is_relative_to(bundle), "Native library leaves bundle")
    require(sha(library) == base["library_sha256"] == native["library_sha256"], "Canonical native base changed")
    require(all(fp["source_sha256"].get(k) == v for k, v in closure["base_sources"].items()), "Base native source fingerprint changed")
    require(fp["openmp"] is False and fp["tile"] == 256 and fp["explicit_avx512_backend"] == 5, "Unexpected base arithmetic policy")
    prefix = work / ".deps/sleef"
    archives = [x for x in (prefix / "lib/libsleef.a", prefix / "lib64/libsleef.a") if x.is_file()]
    require(len(archives) == 1, "Expected one existing static SLEEF archive")
    archive = archives[0]; header = prefix / "include/sleef.h"
    require(sha(archive) == fp["sleef_library_sha256"] and sha(header) == fp["sleef_header_sha256"], "A2 must use the same SLEEF as the canonical base")
    cc, cxx, versions = module.compilers()
    novec = ["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in versions["cc"].lower() else ["-fno-tree-vectorize"]
    require(versions == fp["compilers"] and module.COMMON_FLAGS == fp["common_flags"] and novec == fp["c_novec_flags"], "Base compiler/FP policy differs")
    core = bundle / "libs/libamd_aocl_precision_core.so"
    core_elf = elf_metadata(core)
    require("mkl" not in core_elf.lower() and any("(NEEDED)" in x and "libaocl-dlp.so" in x for x in core_elf.splitlines()), "AMD AOCL core required")
    require(any("(SONAME)" in x and "[libamd_aocl_precision_core.so]" in x for x in core_elf.splitlines()), "AOCL core SONAME required")
    inputs.update({record_path: sha(record_path), library: sha(library), archive: sha(archive), header: sha(header), core: sha(core)})
    for name, expected in fp["header_sha256"].items():
        inputs[include / name] = expected
    for path, expected in inputs.items():
        require(sha(path) == expected, "AMD build input mismatch: " + str(path))
    fingerprint = {"source_closure_sha256": sha(source / "sources.json"), "builder_sha256": sha(__file__),
        "native_fingerprint": fp, "base_library_sha256": sha(library), "core_sha256": sha(core),
        "compiler": versions, "onnx_headers": fp["header_sha256"]}
    receipt = output / "build.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached.get("complete") is True:
            require(cached.get("fingerprint") == fingerprint, "Cached AMD streaming build inputs changed")
            require(set(cached["libraries"]) == set(LIBRARIES), "Cached AMD roles changed")
            for role, row in cached["libraries"].items():
                path = Path(row["path"]).resolve()
                require(path.is_relative_to(output) and path.name == LIBRARIES[role] and sha(path) == row["sha256"], "Cached AMD library changed")
                elf_metadata(path)
            return cached
    if output.exists():
        output.rename(output.with_name(output.name + ".incomplete-" + uuid.uuid4().hex[:8]))
    output.mkdir(parents=True)
    wrapper = output / "history_wrapper.c"
    wrapper.write_text('#include "native_kernels.c"\n#include "direct_history.c"\n')
    flags = module.COMMON_FLAGS
    history, phase, pair = (output / LIBRARIES[k] for k in ("amd_stream_history", "amd_stream_phase", "amd_stream_pair"))
    commands = [
        [cc, *flags, *novec, "-std=c11", "-DNCC_USE_SLEEF=1", "-I"+str(prefix / "include"),
         "-I"+str(work / "native/x86"), "-I"+str(source / "history"), "-c", str(wrapper), "-o", str(output / "history.o")],
        [cxx, *flags, "-std=c++17", "-I"+str(include), "-I"+str(work / "native/x86"),
         "-c", str(source / "history/custom_ops.cpp"), "-o", str(output / "history_bridge.o")],
        [cxx, "-shared", "-pthread", str(output / "history.o"), str(output / "history_bridge.o"), str(archive), "-lm",
         "-Wl,-Bsymbolic", "-Wl,--exclude-libs,ALL", "-Wl,-z,defs", "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+history.name, "-o", str(history)],
        [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-pthread",
         "-I"+str(include), str(source / "phase/native_ops.cpp"), "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+phase.name, "-o", str(phase)],
        [cxx, "-shared", "-fPIC", "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-fvisibility=hidden", "-Wl,-Bsymbolic-functions",
         "-I"+str(include), "-I"+str(source / "pair"), str(source / "pair/paired_projection.cpp"), str(core),
         "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+pair.name, "-Wl,-z,defs", "-o", str(pair)],
    ]
    result = {"version": "amd_stream_selected_build_v1", "complete": False, "fingerprint": fingerprint,
        "input_sha256": {str(p): h for p, h in inputs.items()}, "commands": commands, "libraries": {}, "native_execution": False}
    save = lambda: receipt.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        for index, command in enumerate(commands):
            with (output / f"compile-{index}.log").open("w") as log:
                subprocess.run(command, check=True, env=module.cpu_environment(), stdout=log, stderr=subprocess.STDOUT)
        result["runtime_dependencies"] = {name: elf_metadata(output / name) for name in LIBRARIES.values()}
        for path, expected in inputs.items():
            require(sha(path) == expected, "AMD build input changed: " + str(path))
        result["libraries"] = {role: {"path": str(output / name), "sha256": sha(output / name)} for role, name in LIBRARIES.items()}
        result["complete"] = True; save()
    except BaseException as error:
        result["error"] = repr(error); save(); raise
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "bundle", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    result = build(a.work, a.bundle, a.output)
    print(json.dumps({"complete": result["complete"], "libraries": result["libraries"]}, indent=2))


if __name__ == "__main__":
    main()
