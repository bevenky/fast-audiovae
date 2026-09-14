"""Build only the Intel history/phase transfer libraries; never run inference.

Use the existing Intel canonical native fingerprint and its exact static SLEEF
archive. The first projection and oneMKL precision libraries are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


SOURCES = ("history/direct_history.c", "history/custom_ops.cpp", "phase/native_ops.cpp")
LIBRARIES = {"history": "libintel_rawhistory.so", "phase": "libintel_phase.so"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void", ROCR_VISIBLE_DEVICES="-1",
               HIP_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    return env


def elf_metadata(path):
    value = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", value)
    require(not any(re.search(r"aocl|mkl|gomp|iomp|cuda|cudnn|rocm|hip|opencl", x, re.I) for x in needed),
            "Unexpected matrix/thread/GPU dependency in history/phase library")
    require(all("/" not in name for name in needed), "Absolute ELF dependency")
    search = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]*)\]", value)
    require(all(x in ("$ORIGIN", "${ORIGIN}") for row in search for x in row.split(":")),
            "Nonlocal ELF search path")
    return {"needed": needed, "dynamic": value}


def build(native_source, base_build, ort_include, sleef_prefix, source_root, output_dir, cc=None, cxx=None):
    require(platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"),
            "Native Linux x86-64 builder required")
    require("GenuineIntel" in Path("/proc/cpuinfo").read_text(), "Intel host required")
    native, record_path, include, prefix, source, output = [Path(p).resolve() for p in
        (native_source, base_build, ort_include, sleef_prefix, source_root, output_dir)]
    require(not output.exists(), "Build output directory already exists; use a fresh path")
    record = json.loads(record_path.read_text()); fp = record["fingerprint"]
    closure_path = source / "sources.json"; closure = json.loads(closure_path.read_text())
    require(closure.get("version") == "amd_stream_selected_sources_v1", "Unknown transferred source closure")
    require(fp.get("native_abi") == 1 and fp.get("ort_api_version") == 29
            and fp.get("openmp") is False and fp.get("tile") == 256
            and fp.get("explicit_avx512_backend") == 5, "Canonical arithmetic fingerprint required")
    flags = fp["common_flags"]; novec = fp["c_novec_flags"]
    require(flags == ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-fvisibility=hidden",
                      "-Wall", "-Wextra", "-march=x86-64", "-mtune=generic"], "Unexpected base FP/compiler flags")
    inputs = {record_path: sha(record_path), closure_path: sha(closure_path), Path(__file__).resolve(): sha(__file__)}
    for name in ("native_kernels.c", "native_kernels.h"):
        key = "native/x86/" + name
        require(fp["source_sha256"].get(key) == closure["base_sources"].get(key),
                "Transferred history expects the same canonical native source: " + key)
        inputs[native / name] = fp["source_sha256"][key]
    for name in SOURCES:
        inputs[source / name] = closure["files"][name]["sha256"]
    for name, expected in fp["header_sha256"].items():
        inputs[include / name] = expected
    archives = [path for path in (prefix / "lib/libsleef.a", prefix / "lib64/libsleef.a") if path.is_file()]
    require(len(archives) == 1, "One existing static SLEEF archive required")
    archive = archives[0]; header = prefix / "include/sleef.h"
    inputs[archive] = fp["sleef_library_sha256"]; inputs[header] = fp["sleef_header_sha256"]
    base_library = Path(record["library"]).resolve()
    inputs[base_library] = record["library_sha256"]
    for path, expected in inputs.items():
        require(path.is_file() and sha(path) == expected, "Pinned build input mismatch: " + str(path))
    cc = cc or shutil.which("gcc"); cxx = cxx or shutil.which("g++")
    require(cc and cxx, "Native C/C++ compilers required")
    versions = {name: subprocess.check_output([command, "--version"], text=True).splitlines()[0]
                for name, command in (("cc", cc), ("cxx", cxx))}
    require(versions == fp["compilers"], "Use the same compiler versions as the Intel baseline")
    expected_novec = ["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in versions["cc"].lower() else ["-fno-tree-vectorize"]
    require(novec == expected_novec, "Base scalar vectorization policy changed")
    output.mkdir(parents=True)
    wrapper = output / "history_wrapper.c"
    wrapper.write_text('#include "native_kernels.c"\n#include "direct_history.c"\n')
    history, phase = (output / LIBRARIES[k] for k in ("history", "phase"))
    commands = [
        [cc, *flags, *novec, "-std=c11", "-DNCC_USE_SLEEF=1", "-I"+str(prefix / "include"),
         "-I"+str(native), "-I"+str(source / "history"), "-c", str(wrapper), "-o", str(output / "history.o")],
        [cxx, *flags, "-std=c++17", "-I"+str(include), "-I"+str(native),
         "-c", str(source / "history/custom_ops.cpp"), "-o", str(output / "history_bridge.o")],
        [cxx, "-shared", "-pthread", str(output / "history.o"), str(output / "history_bridge.o"), str(archive), "-lm",
         "-Wl,-Bsymbolic", "-Wl,--exclude-libs,ALL", "-Wl,-z,defs", "-Wl,-rpath,$ORIGIN",
         "-Wl,-soname,"+history.name, "-o", str(history)],
        [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-pthread",
         "-I"+str(include), str(source / "phase/native_ops.cpp"), "-Wl,-rpath,$ORIGIN",
         "-Wl,-soname,"+phase.name, "-Wl,-z,defs", "-o", str(phase)],
    ]
    result = {"version": "intel_streaming_transfer_build_v1", "complete": False, "cpu_only": True,
              "inference_executed": False, "base_fingerprint": fp, "compiler": versions,
              "input_sha256": {str(p): h for p, h in inputs.items()}, "commands": commands, "libraries": {}}
    receipt = output / "build.json"
    def save(): receipt.write_text(json.dumps(result, indent=2)+"\n")
    save()
    try:
        for index, command in enumerate(commands):
            with (output / f"compile-{index}.log").open("w") as log:
                subprocess.run(command, check=True, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        result["runtime_dependencies"] = {name: elf_metadata(output / name) for name in LIBRARIES.values()}
        for path, expected in inputs.items():
            require(sha(path) == expected, "Build input changed while compiling: " + str(path))
        result["libraries"] = {role: {"path": str(output / name), "sha256": sha(output / name)}
                               for role, name in LIBRARIES.items()}
        result["complete"] = True; save()
    except BaseException as error:
        result["error"] = repr(error); save(); raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-source", "base-build", "ort-include", "sleef-prefix", "source-root", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--cc"); parser.add_argument("--cxx")
    result = build(**vars(parser.parse_args()))
    print(json.dumps({"complete": result["complete"], "libraries": result["libraries"]}, indent=2))


if __name__ == "__main__":
    main()
