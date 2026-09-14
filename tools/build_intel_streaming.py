"""Build accepted Intel streaming libraries with relocatable CPU dependencies.

Does not load a shared library or run inference. Existing oneDNN builds are
accepted only at the exact previously qualified binary/header digests, together
with their real provenance. Normal installation builds the pinned source.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import uuid

LIBRARIES = {"intel_stream_history": "libintel_rawhistory.so", "intel_stream_phase": "libintel_phase.so",
             "intel_stream_matrix": "libintel_stream_matrix.so", "intel_stream_matrix_ops": "libintel_stream_matrix_ops.so",
             "intel_stream_onednn": "libdnnl.so.3"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def elf_metadata(path):
    text = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
    for line in text.splitlines():
        if "[" not in line:
            continue
        value = line.split("[", 1)[1].split("]", 1)[0]
        if "(NEEDED)" in line or "(SONAME)" in line:
            require("/" not in value, "Absolute runtime library dependency")
        if "(RPATH)" in line or "(RUNPATH)" in line:
            require(all(x in ("$ORIGIN", "${ORIGIN}") for x in value.split(":")), "Nonlocal runtime search path")
        if "(NEEDED)" in line:
            require(not re.search("cuda|cudnn|rocm|hip|opencl|gomp|iomp|aocl", value, re.I), "Unexpected GPU or threaded dependency")
    return text


def dependency(work, install=None, existing=None, source=None, provenance=None):
    pin_path = work / "native/intel/streaming/onednn.json"
    pin = json.loads(pin_path.read_text()); inputs = {pin_path: sha(pin_path)}
    if existing is not None:
        require(install is None and source is not None and provenance is not None, "Existing build needs source and real provenance")
        existing, source, provenance = [Path(x).resolve() for x in (existing, source, provenance)]
        record = json.loads(provenance.read_text())
        require(record.get("version") == pin["version"] and record.get("source_sha256") == pin["archive_sha256"]
                and record.get("configure_exit") == 0 and record.get("build_exit") == 0, "Existing oneDNN provenance differs")
        configure = record.get("configure", [])
        for key, value in pin["configuration"].items():
            if not key.startswith("CMAKE_INSTALL_"):
                require("-D"+key+"="+value in configure, "Existing oneDNN configuration differs: "+key)
        inputs[provenance] = sha(provenance)
        for relative, expected in pin["qualified_prebuilt"]["headers"].items():
            root, name = relative.split("/", 1)
            path = (source if root == "source" else existing) / name
            require(sha(path) == expected, "Qualified oneDNN header changed: "+relative)
            inputs[path] = expected
        library = (existing / "src/libdnnl.so.3").resolve(strict=True)
        require(sha(library) == pin["qualified_prebuilt"]["library_sha256"], "Existing oneDNN binary is not the qualified build")
        includes = [existing / "include", source / "include"]
        license_root = source
        receipt = {"origin": "verified existing build", "provenance": record, "provenance_sha256": sha(provenance)}
    else:
        require(install is not None and source is None and provenance is None, "Pinned oneDNN installation required")
        install = Path(install).resolve(); cache = install.parent
        record_path = cache / "dependency-receipt.json"; record = json.loads(record_path.read_text())
        expected = {"version": pin["version"], "archive_sha256": pin["archive_sha256"], "configuration": pin["configuration"]}
        require(record.get("identity") == expected and record.get("cpu_only") is True, "oneDNN source build identity changed")
        for name, digest in record["files"].items():
            path = (cache / name).resolve()
            require(path.is_relative_to(cache) and sha(path) == digest, "oneDNN cache changed")
            inputs[path] = digest
        inputs[record_path] = sha(record_path)
        libraries = [(install / d / "libdnnl.so.3").resolve() for d in ("lib", "lib64") if (install / d / "libdnnl.so.3").is_file()]
        require(len(libraries) == 1, "One CPU oneDNN runtime required")
        library = libraries[0]; includes = [install / "include"]; license_root = install / "licenses"
        receipt = {"origin": "pinned source build", "dependency_receipt_sha256": sha(record_path)}
    config = (includes[0] / "oneapi/dnnl/dnnl_config.h").read_text()
    require("#define DNNL_CPU_RUNTIME DNNL_RUNTIME_SEQ" in config and "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE" in config
            and "#define DNNL_EXPERIMENTAL_UKERNEL" in config, "oneDNN must be CPU SEQ / GPU NONE with public BRGeMM")
    for include in includes:
        inputs.update({p: sha(p) for p in sorted(include.rglob("*")) if p.is_file()})
    inputs[library] = sha(library); elf_metadata(library)
    for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
        inputs[license_root / name] = sha(license_root / name)
    return library, includes, license_root, inputs, receipt


def build(work, bundle, output, *, onednn=None, onednn_existing=None, onednn_source=None, onednn_provenance=None):
    work, bundle, output = [Path(p).resolve() for p in (work, bundle, output)]
    require(platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"), "Linux x86-64 builder required")
    source = work / "native/intel/streaming"; common = work / "native/amd/streaming"
    closure_path = source / "sources.json"; closure = json.loads(closure_path.read_text())
    require(closure.get("version") == "intel_stream_selected_sources_v1", "Accepted Intel source closure required")
    inputs = {closure_path: sha(closure_path), Path(__file__).resolve(): sha(__file__)}
    for relative, expected in closure["files"].items():
        path = (work / relative).resolve()
        require(path.is_relative_to(work) and sha(path) == expected, "Maintained source changed: "+relative)
        inputs[path] = expected
    spec = importlib.util.spec_from_file_location("intel_accepted_x86_build", work / "tools/build_x86.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    native_path = work / ".build/x86/build.json"; native = json.loads(native_path.read_text()); fp = native["fingerprint"]
    include = work / ".deps/onnxruntime/include"; module.require_headers(include)
    manifest = json.loads((bundle / "bundle.json").read_text()); base = manifest["native"]["Linux/x86_64"]
    library = (bundle / base["library"]).resolve()
    require(library.is_relative_to(bundle) and sha(library) == base["library_sha256"] == native["library_sha256"], "Canonical base library changed")
    require(fp.get("native_abi") == 1 and fp.get("ort_api_version") in (29, 30) and fp.get("openmp") is False
            and fp.get("tile") == 256 and fp.get("explicit_avx512_backend") == 5, "Canonical arithmetic policy required")
    for relative, expected in fp["source_sha256"].items():
        require(closure["files"].get(relative) == expected, "Base source fingerprint changed: "+relative)
    prefix = work / ".deps/sleef"
    archives = [p for p in (prefix / "lib/libsleef.a", prefix / "lib64/libsleef.a") if p.is_file()]
    require(len(archives) == 1, "One canonical static SLEEF archive required")
    archive = archives[0]; header = prefix / "include/sleef.h"
    require(sha(archive) == fp["sleef_library_sha256"] and sha(header) == fp["sleef_header_sha256"], "SLEEF identity differs from baseline")
    cc, cxx, versions = module.compilers()
    novec = ["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in versions["cc"].lower() else ["-fno-tree-vectorize"]
    require(versions == fp["compilers"] and module.COMMON_FLAGS == fp["common_flags"] and novec == fp["c_novec_flags"], "Base compiler/FP policy differs")
    core = bundle / "libs/libintel_precision_core.so"; core_elf = elf_metadata(core)
    require("[libintel_precision_core.so]" in core_elf and "libmkl_" in core_elf, "Canonical oneMKL precision core required")
    inputs.update({native_path: sha(native_path), library: sha(library), archive: sha(archive), header: sha(header), core: sha(core)})
    for name, expected in fp["header_sha256"].items():
        inputs[include / name] = expected
    runtime, dnnl_includes, notices, dep_inputs, dep_receipt = dependency(work, onednn, onednn_existing, onednn_source, onednn_provenance)
    inputs.update(dep_inputs)
    for path, expected in inputs.items():
        require(sha(path) == expected, "Intel build input mismatch: "+str(path))
    fingerprint = {"source_closure_sha256": sha(closure_path), "builder_sha256": sha(__file__), "native_fingerprint": fp,
        "base_library_sha256": sha(library), "core_sha256": sha(core), "onednn_sha256": sha(runtime),
        "compiler": versions, "onnx_headers": fp["header_sha256"]}
    receipt = output / "build.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached.get("complete") is True:
            require(cached.get("fingerprint") == fingerprint, "Cached Intel build identity changed")
            require(set(cached["libraries"]) == set(LIBRARIES), "Cached Intel library roles changed")
            for role, row in cached["libraries"].items():
                path = Path(row["path"]).resolve()
                require(path.is_relative_to(output) and path.name == LIBRARIES[role] and sha(path) == row["sha256"], "Cached Intel library changed")
                elf_metadata(path)
            return cached
    if output.exists():
        output.rename(output.with_name(output.name+".incomplete-"+uuid.uuid4().hex[:8]))
    output.mkdir(parents=True)
    copied_runtime = output / LIBRARIES["intel_stream_onednn"]; shutil.copy2(runtime, copied_runtime)
    (output / "licenses").mkdir()
    for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
        shutil.copy2(notices / name, output / "licenses" / name)
    wrapper = output / "history_wrapper.c"; wrapper.write_text('#include "native_kernels.c"\n#include "direct_history.c"\n')
    history, phase, matrix, ops = [output / LIBRARIES[k] for k in ("intel_stream_history", "intel_stream_phase", "intel_stream_matrix", "intel_stream_matrix_ops")]
    flags = module.COMMON_FLAGS
    commands = [
        [cc, *flags, *novec, "-std=c11", "-DNCC_USE_SLEEF=1", "-I"+str(prefix / "include"), "-I"+str(work / "native/x86"),
         "-I"+str(common / "history"), "-c", str(wrapper), "-o", str(output / "history.o")],
        [cxx, *flags, "-std=c++17", "-I"+str(include), "-I"+str(work / "native/x86"),
         "-c", str(common / "history/custom_ops.cpp"), "-o", str(output / "history_bridge.o")],
        [cxx, "-shared", "-pthread", str(output / "history.o"), str(output / "history_bridge.o"), str(archive), "-lm", "-Wl,-Bsymbolic",
         "-Wl,--exclude-libs,ALL", "-Wl,-z,defs", "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+history.name, "-o", str(history)],
        [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off", "-pthread",
         "-I"+str(include), str(common / "phase/native_ops.cpp"), "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+phase.name, "-o", str(phase)],
        [cxx, "-std=c++17", "-O3", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-fvisibility=hidden", "-Wall", "-Wextra",
         *["-I"+str(p) for p in dnnl_includes], str(source / "matrix.cpp"), str(copied_runtime), "-ldl", "-Wl,-z,defs",
         "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+matrix.name, "-o", str(matrix)],
        [cxx, "-std=c++17", "-shared", "-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-fvisibility=hidden", "-pthread",
         "-I"+str(include), str(source / "custom_ops.cpp"), str(matrix), str(core), "-Wl,-z,defs",
         "-Wl,-rpath,$ORIGIN", "-Wl,-soname,"+ops.name, "-o", str(ops)],
    ]
    result = {"version": "intel_stream_selected_build_v1", "complete": False, "fingerprint": fingerprint,
        "input_sha256": {str(p): h for p, h in inputs.items()}, "commands": commands, "libraries": {},
        "onednn_provenance": dep_receipt, "native_execution": False, "cpu_only": True}
    save = lambda: receipt.write_text(json.dumps(result, indent=2)+"\n")
    save()
    try:
        for index, command in enumerate(commands):
            with (output / f"compile-{index}.log").open("w") as log:
                subprocess.run(command, check=True, env=module.cpu_environment(), stdout=log, stderr=subprocess.STDOUT)
        result["runtime_dependencies"] = {name: elf_metadata(output / name) for name in LIBRARIES.values()}
        for path, expected in inputs.items():
            require(sha(path) == expected, "Intel build input changed: "+str(path))
        result["libraries"] = {role: {"path": str(output / name), "sha256": sha(output / name)} for role, name in LIBRARIES.items()}
        result["complete"] = True; save()
    except BaseException as error:
        result["error"] = repr(error); save(); raise
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "bundle", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--onednn", type=Path)
    for name in ("onednn-existing", "onednn-source", "onednn-provenance"):
        p.add_argument("--"+name, type=Path)
    result = build(**vars(p.parse_args()))
    print(json.dumps({"complete": result["complete"], "libraries": result["libraries"]}, indent=2))


if __name__ == "__main__":
    main()
