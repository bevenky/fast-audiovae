"""Build the Apple ARM FP32 custom operators without linking an ORT runtime.

Requires the pinned headers in .deps/onnxruntime/include. Generated objects,
library and provenance stay under .build/apple. This script runs no models.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "venky.audio.cpu"
NATIVE_ABI = 1
ORT_API_VERSION = 29
HEADER_SHA256 = {
    "onnxruntime_c_api.h": "acc0cf4b3f28d39339c76770d76164bb7a0637dc89f5fde764b4017b632f6743",
    "onnxruntime_cxx_api.h": "9c63ed5bf0427ddf1c10a99aca0beeec77318ce77d2ea6f7ae76d7d933660f67",
    "onnxruntime_cxx_inline.h": "c508dbb7d8203003a2584b8712118cb88e5b4ef1e1c7876a9974156fb7489301",
    "onnxruntime_ep_c_api.h": "e6c986c9e98583f8113b2c6bc3864814883b806d501cf24da4d239c45753e235",
    "onnxruntime_error_code.h": "5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721",
    "onnxruntime_float16.h": "88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _output(command):
    return subprocess.check_output(command, text=True).strip()


def _write_if_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def build():
    """Return library, domain, native_abi, hashes and a build_manifest path.

    Paths returned to callers are absolute. Source paths inside the generated
    build fingerprint are relative to the repository. No previous build or
    model checkpoint is needed. Concurrent invocations serialize on a file lock.
    """
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("This build requires native Apple ARM macOS")
    includes = ROOT / ".deps/onnxruntime/include"
    for name, expected in HEADER_SHA256.items():
        path = includes / name
        if not path.is_file():
            raise RuntimeError(f"Missing pinned ORT header: .deps/onnxruntime/include/{name}")
        if sha256(path) != expected:
            raise RuntimeError(f"ORT 1.29 header hash mismatch: {name}")
    api_header = (includes / "onnxruntime_c_api.h").read_text()
    if re.search(r"^#define\s+ORT_API_VERSION\s+29\s*$", api_header, re.MULTILINE) is None:
        raise RuntimeError("Expected ORT API version 29")
    sources = ("native/apple/native_kernels.c", "native/apple/native_kernels.h",
               "native/apple/custom_ops.cpp", "tools/build_apple.py")
    source_hashes = {name: sha256(ROOT / name) for name in sources}
    if (ROOT / sources[0]).read_bytes().count(b"#define NCC_TILE 256\n") != 1:
        raise RuntimeError("The accepted Apple kernel requires NCC_TILE=256")
    cc = _output(["xcrun", "--find", "clang"])
    cxx = _output(["xcrun", "--find", "clang++"])
    sdk = _output(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    common = ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra"]
    c_flags = common + ["-std=c11", "-fno-vectorize", "-fno-slp-vectorize", "-DNCC_USE_ACCELERATE=1"]
    cpp_flags = common + ["-std=c++17", "-fvisibility=hidden"]
    # Make the deployment choice explicit instead of inheriting caller flags.
    deployment = _output(["xcrun", "--sdk", "macosx", "--show-sdk-version"])
    fingerprint = {
        "source_sha256": source_hashes, "header_sha256": HEADER_SHA256,
        "cc_version": _output([cc, "--version"]).splitlines()[0],
        "cxx_version": _output([cxx, "--version"]).splitlines()[0],
        "sdk_version": deployment, "deployment_target": deployment,
        "c_flags": c_flags, "cpp_flags": cpp_flags,
        "link_flags": ["-dynamiclib", "-framework", "Accelerate"],
        "domain": DOMAIN, "native_abi": NATIVE_ABI, "ort_api_version": ORT_API_VERSION,
        "architecture": "arm64", "tile": 256, "openmp": False,
    }
    build_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    out = ROOT / ".build/apple"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lib = out / ("libfast_audiovae_apple_" + build_id[:12] + ".dylib")
        manifest = out / "build.json"
        prior = json.loads(manifest.read_text()) if manifest.exists() else {}
        if (prior.get("fingerprint") == fingerprint and lib.is_file()
                and prior.get("library_sha256") == sha256(lib)):
            return {**prior, "library": str(lib), "build_manifest": str(manifest), "cache_hit": True}
        wrapper = out / "native_wrapper.c"
        _write_if_changed(wrapper, '#include "native_kernels.c"\n'
            'NCC_API uint32_t ncc_compiled_tile(void) { return NCC_TILE; }\n'
            'NCC_API uint32_t ncc_phase_finish_abi(void) { return 1; }\n'
            f'NCC_API const char *ncc_phase_build_id(void) {{ return "{build_id}"; }}\n')
        c_obj, cpp_obj = out / "native_kernels.o", out / "custom_ops.o"
        relative = lambda p: str(p.relative_to(ROOT))
        commands = [
            [cc, *c_flags, "-isysroot", sdk, "-Inative/apple", "-c", relative(wrapper), "-o", relative(c_obj)],
            [cxx, *cpp_flags, "-I.deps/onnxruntime/include", "-isysroot", sdk,
             "-isystem", str(Path(sdk) / "usr/include/c++/v1"), "-c",
             "native/apple/custom_ops.cpp", "-o", relative(cpp_obj)],
            [cxx, relative(c_obj), relative(cpp_obj), "-isysroot", sdk, "-dynamiclib", "-framework", "Accelerate",
             "-Wl,-install_name,@rpath/" + lib.name, "-o", relative(lib)],
        ]
        environment = dict(os.environ)
        environment["MACOSX_DEPLOYMENT_TARGET"] = deployment
        logs = []
        for command in commands:
            process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            logs.append({"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
            if process.returncode:
                raise RuntimeError("Apple library build failed:\n" + process.stdout + process.stderr)
        for name, expected in source_hashes.items():
            if sha256(ROOT / name) != expected:
                raise RuntimeError(f"Source changed during build: {name}")
        for name, expected in HEADER_SHA256.items():
            if sha256(includes / name) != expected:
                raise RuntimeError(f"Header changed during build: {name}")
        result = {
            "library": str(lib), "library_sha256": sha256(lib), "build_manifest": str(manifest),
            "domain": DOMAIN, "native_abi": NATIVE_ABI, "ort_api_version": ORT_API_VERSION,
            "build_id": build_id, "fingerprint": fingerprint, "commands": commands, "logs": logs,
            "cache_hit": False, "openmp": False, "accelerate": True, "tile": 256,
            "operators": ["SnakeF32", "CausalDW7F32", "CausalDW7SnakeF32", "PhaseSumBiasInterleaveF32",
                          "SnakeDW7SnakeF32", "BiasResidualF32"],
            "phase_simd": "NEON four-time transpose/interleave with scalar tails",
            "scope": "CPU-only library build; no model execution or timing",
        }
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(manifest)
        return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), indent=2))
