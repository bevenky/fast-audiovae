"""Build the first-pair Apple INT8 operators from bundled, pinned sources.

This offline builder never downloads dependencies, opens a model or runs kernels.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/apple/streaming/first_pair_int8"
VERSION = "apple_firstpair_int8_build_v1"
DOMAIN = "fast.audiovae.apple.firstpair.int8.v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(*, offline=False):
    """Return a relocatable single-library runtime receipt; offline by design."""
    del offline
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Apple ARM64 source build required")
    pin = json.loads((SOURCE / "sources.json").read_text())
    for name, record in pin["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError("Pinned Apple INT8 source changed: " + name)
    spec = importlib.util.spec_from_file_location("_apple_int8_base_build", ROOT / "tools/build_apple.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    headers = ROOT / ".deps/onnxruntime/include"
    for name, expected in base.HEADER_SHA256.items():
        if sha(headers / name) != expected:
            raise ValueError("Pinned ORT API29 header mismatch: " + name)
    capture = lambda command: subprocess.check_output(command, text=True).strip()
    cc = capture(["xcrun", "--find", "clang"])
    cxx = capture(["xcrun", "--find", "clang++"])
    sdk = capture(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    deployment = capture(["xcrun", "--sdk", "macosx", "--show-sdk-version"])
    fingerprint = {
        "sources": {name: record["sha256"] for name, record in pin["files"].items()},
        "source_receipt_sha256": sha(SOURCE / "sources.json"), "builder_sha256": sha(__file__),
        "headers": base.HEADER_SHA256, "compiler": capture([cxx, "--version"]).splitlines()[0],
        "sdk": deployment, "precision": "W8A8 with FP32 input/output", "fast_math": False,
        "fma_contraction": False, "upstream_commit": pin["upstream_commit"],
    }
    build_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    parent = ROOT / ".build/apple-int8"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = parent / build_id[:16]
        manifest = out / "build.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if (previous.get("complete") is True and previous.get("fingerprint") == fingerprint
                    and all(Path(r["path"]).is_file() and sha(r["path"]) == r["sha256"]
                            for r in previous["runtime_files"])):
                return {**previous, "build_manifest": str(manifest), "cache_hit": True}
        out.mkdir(parents=True, exist_ok=True)
        commands = []
        environment = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=deployment)

        def run(command):
            commands.append(command)
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            with (out / "build.log").open("a") as log:
                log.write(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("Apple INT8 build failed; inspect " + str(out / "build.log"))

        common = ["-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off",
                  "-Wall", "-Wextra", "-Werror", "-isysroot", sdk, "-I" + str(headers),
                  "-I" + str(SOURCE), "-I" + str(SOURCE / "upstream")]
        cpp = ["-std=c++17", "-isystem", str(Path(sdk) / "usr/include/c++/v1")]
        kernel = "upstream/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot"
        compilation = [
            ("adapter.cpp", cxx, cpp), ("quantize.cpp", cxx, cpp), ("ort_ops.cpp", cxx, cpp),
            (kernel + ".c", cc, ["-std=c11", "-march=armv9.2-a+sme2"]),
            (kernel + "_asm.S", cc, ["-march=armv9.2-a+sme2"]),
            ("upstream/kai/kai_common_sme_asm.S", cc, ["-march=armv9.2-a+sme2"]),
            ("upstream/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c", cc, ["-std=c11"]),
        ]
        objects = []
        for i, (name, compiler, extra) in enumerate(compilation):
            obj = out / f"object-{i}.o"
            run([compiler, *common, *extra, "-c", str(SOURCE / name), "-o", str(obj)])
            objects.append(obj)
        library = out / "libapple_firstpair_int8.dylib"
        exports = ["RegisterCustomOps", "av8_ort_calls", "av8_ort_packs", "av8_ort_quant_calls", "av8_ort_empty_calls"]
        run([cxx, "-isysroot", sdk, "-dynamiclib", *map(str, objects),
             *["-Wl,-exported_symbol,_" + name for name in exports],
             "-Wl,-install_name,@rpath/" + library.name, "-o", str(library)])
        dependencies = capture(["otool", "-L", str(library)])
        for line in dependencies.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if dependency != "@rpath/" + library.name and not dependency.startswith(("/usr/lib/", "/System/Library/")):
                raise RuntimeError("Non-system runtime dependency: " + dependency)
        for name, expected in fingerprint["sources"].items():
            if sha(SOURCE / name) != expected:
                raise RuntimeError("Source changed during build: " + name)
        runtime = {"path": str(library), "sha256": sha(library), "register": True, "domain": DOMAIN}
        result = {
            "version": VERSION, "complete": True, "build_id": build_id, "fingerprint": fingerprint,
            "library": str(library), "sha256": runtime["sha256"], "domain": DOMAIN,
            "runtime_files": [runtime],
            "additional_libraries": [{"library": str(library), "sha256": runtime["sha256"], "domain": DOMAIN}],
            "license_files": [{"path": str(p), "sha256": sha(p)} for p in sorted((SOURCE / "licenses").iterdir())],
            "commands": commands, "required_cpu_features": ["sme", "sme2"], "onnxruntime": "1.30.0",
            "custom_operator_threads": 1, "supported_ort_threads": [1], "precision": "W8A8/FP32",
            "shape": "[B,2048,T] -> [B,8192,T]; any nonnegative B/T, internal tile T<=2",
            "operator": "FirstPairInt8", "dependencies": dependencies.splitlines()[1:],
            "exported_symbols": exports, "build_manifest": str(manifest), "cache_hit": False,
        }
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(manifest)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Accepted for build-system consistency; always offline")
    args = parser.parse_args()
    print(json.dumps(build(offline=args.offline), indent=2))
