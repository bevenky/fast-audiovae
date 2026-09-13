"""Build the selected Apple streaming closure from pinned maintained sources.

This is a build only. It never opens a model or executes a numerical kernel.
KleidiAI's small source closure is shipped; LIBXSMM is fetched by immutable
archive hash when an offline source cache or prebuilt wheel is unavailable.
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
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/apple/streaming"
VERSION = "apple_stream_selected_build_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path):
    spec = importlib.util.spec_from_file_location("_apple_base_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dependency(pin, offline, supplied_archive=None):
    cache = ROOT / ".deps/apple-streaming"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "libxsmm.tar.gz"
    if supplied_archive is not None:
        supplied_archive = Path(supplied_archive).resolve()
        if sha(supplied_archive) != pin["archive_sha256"]:
            raise ValueError("LIBXSMM archive hash mismatch")
        if not archive.exists():
            shutil.copy2(supplied_archive, archive)
    if not archive.exists():
        if offline:
            raise RuntimeError("Offline Apple source build needs its pinned LIBXSMM archive or a prebuilt wheel")
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with urllib.request.urlopen(pin["url"], timeout=60) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if sha(temporary) != pin["archive_sha256"]:
                raise ValueError("Downloaded LIBXSMM archive hash mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(archive) != pin["archive_sha256"]:
        raise ValueError("Cached LIBXSMM archive changed")
    destination = cache / ("libxsmm-" + pin["commit"])
    if not destination.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(cache, filter="data")
    # Check maintained inputs on every build, including an existing source cache.
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to(destination.name)
            if not any(part in ("src", "include", "scripts") for part in relative.parts) and relative.name not in ("Makefile", "Makefile.inc", "LICENSE.md", "version.txt"):
                continue
            expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            if sha(destination / relative) != expected:
                raise ValueError("Pinned LIBXSMM source cache changed: " + str(relative))
    return destination


def build(*, offline=False, libxsmm_archive=None):
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Apple ARM64 source build required")
    pin = json.loads((SOURCE / "sources.json").read_text())
    for name, record in pin["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError("Selected native source changed without updating its receipt: " + name)
    base = _load(ROOT / "tools/build_apple.py")
    headers = ROOT / ".deps/onnxruntime/include"
    for name, digest in base.HEADER_SHA256.items():
        if sha(headers / name) != digest:
            raise ValueError("Pinned ORT API29 header mismatch")
    capture = lambda cmd: subprocess.check_output(cmd, text=True).strip()
    cc = capture(["xcrun", "--find", "clang"])
    cxx = capture(["xcrun", "--find", "clang++"])
    sdk = capture(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    deployment = capture(["xcrun", "--sdk", "macosx", "--show-sdk-version"])
    fingerprint = {"sources": {name: record["sha256"] for name, record in pin["files"].items()},
        "source_receipt_sha256": sha(SOURCE / "sources.json"), "builder_sha256": sha(__file__),
        "base_c_sha256": sha(ROOT / "native/apple/native_kernels.c"),
        "base_h_sha256": sha(ROOT / "native/apple/native_kernels.h"),
        "headers": base.HEADER_SHA256, "libxsmm": pin["libxsmm"],
        "compiler": capture([cxx, "--version"]).splitlines()[0], "sdk": deployment,
        "precision": "FP32", "fast_math": False, "fma_contraction": False}
    build_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    parent = ROOT / ".build/apple-streaming"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = parent / build_id[:16]
        manifest = out / "build.json"
        if manifest.exists():
            prior = json.loads(manifest.read_text())
            if (prior.get("fingerprint") == fingerprint and prior.get("complete") is True
                    and all(sha(r["path"]) == r["sha256"] for r in prior["runtime_files"])):
                return {**prior, "build_manifest": str(manifest), "cache_hit": True}
        out.mkdir(parents=True, exist_ok=True)
        commands = []
        environment = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=deployment)

        def run(cmd, cwd=ROOT):
            commands.append(cmd)
            result = subprocess.run(cmd, cwd=cwd, env=environment, capture_output=True, text=True)
            with (out / "build.log").open("a") as log:
                log.write(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("Apple streaming build failed; inspect " + str(out / "build.log"))

        common = ["-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-isysroot", sdk,
            "-I" + str(headers), "-I" + str(ROOT / "native/apple"), "-I" + str(SOURCE / "kleidiai")]
        cpp = [*common, "-std=c++17", "-isystem", str(Path(sdk) / "usr/include/c++/v1")]
        runtime_files, additional = [], []

        def record(path, domain=None):
            row = {"path": str(path), "sha256": sha(path), "register": domain is not None}
            if domain:
                row["domain"] = domain
                additional.append({"library": str(path), "sha256": row["sha256"], "domain": domain})
            runtime_files.append(row)

        def link(path, inputs, accelerate=False):
            run([cxx, "-isysroot", sdk, "-dynamiclib", *map(str, inputs),
                *(["-framework", "Accelerate"] if accelerate else []),
                "-Wl,-install_name,@rpath/" + path.name, "-Wl,-rpath,@loader_path", "-o", str(path)])

        base_obj = out / "state_base.o"
        run([cc, *common, "-std=c11", "-fno-vectorize", "-fno-slp-vectorize", "-DNCC_USE_ACCELERATE=1",
             "-c", str(ROOT / "native/apple/native_kernels.c"), "-o", str(base_obj)])
        obj = out / "state.o"
        run([cxx, *cpp, "-c", str(SOURCE / "state/custom_ops.cpp"), "-o", str(obj)])
        lib = out / "libapple_state_v2.dylib"
        link(lib, [base_obj, obj], True); record(lib, "fast.audiovae.apple.state.v2")
        # Keep the accepted domains and load order; each core owns independent scratch.
        family_names = {"sweep": ("matrix_sweep", "fast.audiovae.apple.matrix.sweep.v1"),
            "multitile": ("multitile", "fast.audiovae.apple.multitile.v1"),
            "multinext": ("multinext", "fast.audiovae.apple.multinext.v1")}
        for family, (basename, domain) in family_names.items():
            objects = []
            for index, relative in enumerate(pin["kleidiai"]["families"][family]):
                source = SOURCE / relative; obj = out / (family + "_upstream_" + str(index) + ".o")
                extra = ["-std=c11", "-fno-vectorize", "-fno-slp-vectorize"] if source.suffix == ".c" else []
                run([cc, *common, "-march=armv9.2-a+sme2", *extra, "-c", str(source), "-o", str(obj)])
                objects.append(obj)
            obj = out / (family + ".o")
            run([cxx, *cpp, "-c", str(SOURCE / family / "native.cpp"), "-o", str(obj)]); objects.append(obj)
            core = out / ("libapple_" + basename + "_core.dylib"); link(core, objects); record(core, domain + ".core")
            bridge = out / ("libapple_" + basename + "_ort.dylib")
            run([cxx, *cpp, str(SOURCE / family / "ort_ops.cpp"), str(core), "-dynamiclib", "-framework", "Accelerate",
                 "-Wl,-rpath,@loader_path", "-Wl,-install_name,@rpath/" + bridge.name, "-o", str(bridge)])
            record(bridge, domain)
        for family, basename, domain in (("layout", "layout_v3", "fast.audiovae.apple.layout.v3"),
                                        ("phase", "phase_state_v1", "fast.audiovae.apple.phase.state.v1")):
            lib = out / ("libapple_" + basename + ".dylib")
            run([cxx, *cpp, str(SOURCE / family / "native_ops.cpp"), "-dynamiclib", "-framework", "Accelerate",
                 "-Wl,-install_name,@rpath/" + lib.name, "-o", str(lib)])
            record(lib, domain)
        dep = _dependency(pin["libxsmm"], offline, libxsmm_archive)
        # A private source cache, no system installation, OpenMP, BLAS or Fortran.
        run(["make", "-j2", "lib/libxsmm.dylib", "STATIC=0", "BLAS=0", "FORTRAN=0", "CC=" + cc, "CXX=" + cxx,
             "CFLAGS=-O3 -fPIC -fno-fast-math -ffp-contract=off -isysroot " + sdk,
             "LDFLAGS=-isysroot " + sdk], dep)
        dependency = out / "libapple_xsmm_panel_dependency.dylib"
        shutil.copy2(dep / "lib/libxsmm.2.dylib", dependency)
        run(["install_name_tool", "-id", "@rpath/" + dependency.name, str(dependency)])
        run(["codesign", "--force", "--sign", "-", str(dependency)])
        record(dependency)
        core = out / "libapple_libxsmm_panel_core.dylib"
        run([cxx, *cpp, "-I" + str(dep / "include"), str(SOURCE / "libxsmm_panel/native.cpp"), str(dependency),
             "-dynamiclib", "-Wl,-rpath,@loader_path", "-Wl,-install_name,@rpath/" + core.name, "-o", str(core)])
        record(core, "fast.audiovae.apple.libxsmm.panel.v1.core")
        bridge = out / "libapple_libxsmm_panel_ort.dylib"
        run([cxx, *cpp, str(SOURCE / "libxsmm_panel/ort_ops.cpp"), str(core), "-dynamiclib", "-framework", "Accelerate",
             "-Wl,-rpath,@loader_path", "-Wl,-install_name,@rpath/" + bridge.name, "-o", str(bridge)])
        record(bridge, "fast.audiovae.apple.libxsmm.panel.v1")
        for name, expected in fingerprint["sources"].items():
            if sha(SOURCE / name) != expected:
                raise RuntimeError("Source changed during build")
        order = ["fast.audiovae.apple.state.v2", "fast.audiovae.apple.matrix.sweep.v1.core",
            "fast.audiovae.apple.matrix.sweep.v1", "fast.audiovae.apple.layout.v3",
            "fast.audiovae.apple.multitile.v1.core", "fast.audiovae.apple.multitile.v1",
            "fast.audiovae.apple.multinext.v1.core", "fast.audiovae.apple.multinext.v1",
            "fast.audiovae.apple.phase.state.v1", "fast.audiovae.apple.libxsmm.panel.v1.core",
            "fast.audiovae.apple.libxsmm.panel.v1"]
        additional.sort(key=lambda row: order.index(row["domain"]))
        result = {"version": VERSION, "complete": True, "build_id": build_id, "fingerprint": fingerprint,
            "selected_graph_sha256": pin["selected_graph_sha256"], "runtime_files": runtime_files, "additional_libraries": additional,
            "license_files": [{"path": str(p), "sha256": sha(p)} for p in sorted((SOURCE / "licenses").iterdir())],
            "commands": commands, "required_cpu_features": ["sme", "sme2"], "onnxruntime": "1.30.0",
            "custom_operator_threads": 1, "supported_ort_threads": [1, 4], "precision": "FP32",
            "library_rpaths": "@loader_path", "build_manifest": str(manifest), "cache_hit": False}
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n"); temporary.replace(manifest)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--libxsmm-archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(offline=args.offline, libxsmm_archive=args.libxsmm_archive), indent=2))
