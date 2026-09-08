#!/usr/bin/env python3
"""Build the supplied precision stage sources with explicit local dependencies.

This path-parameterized adapter preserves the frozen experiment's compiler
flags, object list and link order. It does not download or execute a model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError("Dependency/source hash mismatch: " + str(path))


def resolved(value, base):
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def xsmm_sources(tree, archive, pin):
    """Verify source/header files against the pinned archive, without extraction."""
    verify(archive, pin["archive_sha256"])
    checked = {}
    prefix = "libxsmm-" + pin["commit"] + "/"
    generated = {"include/libxsmm_config.h", "include/libxsmm_version.h"}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if relative in generated or not relative.startswith(("src/", "include/", "scripts/")):
                continue
            if not relative.endswith((".c", ".h", ".py", ".sh")):
                continue
            expected = hashlib.sha256(source.extractfile(member).read()).hexdigest()
            verify(tree / relative, expected)
            checked[relative] = expected
    if not checked:
        raise ValueError("Pinned LIBXSMM archive contained no matching sources")
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision-build", type=Path, required=True, help="native/build.py output build.json")
    p.add_argument("--native-build", type=Path, required=True, help="fast-audiovae tools/build_x86.py build.json")
    p.add_argument("--ort-include", type=Path, required=True)
    p.add_argument("--libxsmm", type=Path, required=True, help="Built pinned source tree containing lib/libxsmm.a")
    p.add_argument("--libxsmm-source-archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("The fused experiment requires Linux x86-64")
    out = a.output_dir.resolve()
    if out.exists():
        raise ValueError("Use a fresh build output directory")
    precision = json.loads(a.precision_build.read_text())
    native = json.loads(a.native_build.read_text())
    deps = json.loads((ROOT / "pins/dependencies.json").read_text())
    original = json.loads((ROOT / "evidence/build.json").read_text())
    if precision["sources"] != original["sources"]:
        raise ValueError("Precision build differs from the frozen native source set")
    if precision.get("mkl_library_sha256") != original["mkl_library_sha256"]:
        raise ValueError("Precision build requires pinned oneMKL 2026.1 CPU libraries")
    prec = resolved(precision["core_path"], a.precision_build.resolve().parent)
    verify(prec, precision["libraries"][prec.name])
    native_lib = resolved(native["library"], a.native_build.resolve().parent)
    verify(native_lib, native["library_sha256"])
    if native.get("native_abi") != 1 or native.get("ort_api_version") != 29 or native.get("tile") != 256:
        raise ValueError("Native FP32 dependency requires ABI1, ORT API29 and tile256")
    if native["fingerprint"]["source_sha256"]["native/x86/native_kernels.h"] != sha(ROOT / "support/native/native_kernels.h"):
        raise ValueError("Native FP32 header differs from the supplied support contract")
    include = a.ort_include.resolve()
    for name, digest in deps["ort_headers"]["sha256"].items():
        verify(include / name, digest)
    if "#define ORT_API_VERSION 29" not in (include / "onnxruntime_c_api.h").read_text():
        raise ValueError("ORT API29 headers are required")
    xsmm = a.libxsmm.resolve()
    verified_xsmm = xsmm_sources(xsmm, a.libxsmm_source_archive, deps["libxsmm"])
    source_map = json.loads((ROOT / "source-map.json").read_text())["files"]
    for name, record in source_map.items():
        if name.startswith(("fused/", "support/", "native/")):
            verify(ROOT / name, record["sha256"])
    # Recheck the actual linked oneMKL directory, not only its old build record.
    mkl_pin = json.loads((ROOT / "pins/mkl.json").read_text())
    candidates = [Path(arg) for command in precision["commands"] for arg in command
                  if Path(arg).name == mkl_pin["direct_link_libraries"][0]]
    if len(candidates) != 1:
        raise ValueError("Cannot identify the precision core's pinned oneMKL link directory")
    mkl_directory = candidates[0].parent
    for name, digest in mkl_pin["cpu_library_sha256"].items():
        verify(mkl_directory / name, digest)
    support = ROOT / "support"
    directories = [include, support / "native", support / "matrix", support / "stage",
                   support / "upsample", ROOT / "native"]
    inputs = [ROOT / "fused/stage_precision.cpp", ROOT / "fused/upsample_precision.cpp",
              ROOT / "native/precision.h", prec, native_lib, xsmm / "lib/libxsmm.a"]
    inputs += [support / name for name in ("matrix/fused_pointwise.c", "matrix/fused_pointwise.h",
               "upsample/projection.c", "upsample/projection.h", "stage/stage_dw.h", "native/native_kernels.h")]
    inputs += [include / name for name in deps["ort_headers"]["sha256"]]
    inputs += sorted((xsmm / "include").rglob("*.h"))
    inputs += [mkl_directory / name for name in mkl_pin["cpu_library_sha256"]]
    before = {str(path): sha(path) for path in inputs}
    flags = ["-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
    commands = []
    for name, source, define in (("matrix", support / "matrix/fused_pointwise.c", "FX_WITH_LIBXSMM"),
                                 ("projection", support / "upsample/projection.c", "UP_WITH_LIBXSMM")):
        commands.append([a.cc, *flags, "-std=c11", "-D" + define + "=1", "-I" + str(xsmm / "include"),
                         "-c", str(source), "-o", str(out / (name + ".o"))])
    for name in ("stage", "upsample"):
        commands.append([a.cxx, *flags, "-std=c++17", *["-I" + str(d) for d in directories],
                         "-c", str(ROOT / "fused" / (name + "_precision.cpp")), "-o", str(out / (name + ".o"))])
        objects = [out / (name + ".o"), out / "matrix.o"] + ([out / "projection.o"] if name == "upsample" else [])
        commands.append([a.cxx, "-shared", *map(str, objects), str(prec), str(native_lib), str(xsmm / "lib/libxsmm.a"),
                         "-Wl,-rpath," + str(prec.parent), "-Wl,-rpath," + str(native_lib.parent), "-ldl", "-pthread", "-lm",
                         "-o", str(out / ("libprecision_" + name + ".so"))])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void", ROCR_VISIBLE_DEVICES="-1",
               HIP_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    out.mkdir(parents=True)
    for command in commands:
        subprocess.run(command, check=True, env=env)
    if {str(path): sha(path) for path in inputs} != before:
        raise ValueError("A build input changed during compilation")
    libraries = [out / ("libprecision_" + name + ".so") for name in ("stage", "upsample")]
    record = {"gpu_used": False, "model_execution": False, "commands": commands,
              "compiler": {"cc": subprocess.check_output([a.cc, "--version"], text=True).splitlines()[0],
                           "cxx": subprocess.check_output([a.cxx, "--version"], text=True).splitlines()[0]},
              "sha256": {**before, **{str(path): sha(path) for path in libraries}},
              "libraries": list(map(str, libraries)), "libxsmm_verified_source_files": len(verified_xsmm),
              "libxsmm_archive_sha256": sha(a.libxsmm_source_archive),
              "precision_build_sha256": sha(a.precision_build), "native_build_sha256": sha(a.native_build),
              "adapter_sha256": sha(Path(__file__))}
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "built_not_executed", "libraries": record["libraries"]}))


if __name__ == "__main__":
    main()
