"""Assemble the accepted Linux x86 CPU recipes from packaged source files.

This module changes packaging and scheduling only. It never chooses new kernel
parameters, runs a timing sweep, installs system software, or executes a model.
"""
from __future__ import annotations

import copy
import errno
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from ..assets import sha256, verify_model
from . import dependencies as deps
from .common import completed_bundle


ORIGINAL_SHA256 = "34ccdc4b835d04c6a240c7cd2c8025995d22b4c9f3877e61e0b9ee7524d5de66"
ACCEPTED_SHA256 = "1ddb6dcc2b0c3ccea90d309f6ebec10eb12e844fb6320cd2d525dbfd01d4cf29"
COMPOSED_SHA256 = "e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2"
SELECTIVE_SHA256 = "26b545641a43380b2f012c44f600f9e9f4413f424402bb35d9aa758b45849304"
CANONICAL_HASHES = {
    1: ("023e40ad0fb9578fbe232942d6aeedb2ed30a426c9db59b70694927e17c67310",
        "8bb688f8606d4f07abd467de951c8554ec54fb9632bb0926db5b328670e129bd"),
    2: ("05cf6284f734ff444c9286127b277808f629f82338b2990532ba74feabb2be3b",
        "7dae8e213132ce4ce93de7a3d25e284cefddfe45d0f94d9e89db12192c306359"),
    4: ("0453d219a97d49b7a5d8ab0fdaadb68d5a2717c8660e49994ea680ebdfd48940",
        "3f9f82a5959b2f7beb46062c8c12c777ceab557ee8db5b087529d41e3b65b894"),
}
PAIR_SHA256 = "59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516"
REQUIRED_FLAGS = frozenset(("avx", "avx2", "fma", "xsave", "avx512f",
                            "avx512dq", "avx512bw", "avx512vl", "avx512_vnni"))


def _verify(path, expected):
    if sha256(path) != expected:
        raise ValueError("Recipe graph differs from the accepted artifact: " + str(path))


def _script(work, relative, args, commands):
    deps.run([sys.executable, work / relative, *args], work, commands)


def _package_source():
    return Path(__file__).resolve().parents[2]


def _vendor(info):
    value = info.get("vendor", info.get("vendor_id", ""))
    return {"GenuineIntel": "intel", "AuthenticAMD": "amd",
            "Intel": "intel", "AMD": "amd", "intel": "intel", "amd": "amd"}.get(value)


def validate_selection(platform_info, mode, threads):
    """Validate a requested recipe without loading native code."""
    vendor = _vendor(platform_info)
    if vendor is None:
        raise ValueError("The selective x86 recipes require an identified Intel or AMD CPU")
    if mode not in ("streaming", "batch"):
        raise ValueError("Mode must be streaming or batch")
    allowed = {"intel": {1, 2}, "amd": {1, 4}}[vendor]
    if vendor == "intel" and mode == "streaming":
        allowed = {1}
    if type(threads) is not int or threads not in allowed:
        raise ValueError(f"No validated {vendor} {mode} recipe for {threads} workers")
    return vendor


def _host_gate(vendor, threads):
    """Check every allowed logical CPU; native constructors also check XCR0."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The x86 recipe builder requires native Linux x86-64")
    allowed = os.sched_getaffinity(0)
    if len(allowed) < threads:
        raise RuntimeError("The selected worker count exceeds this process's CPU affinity")
    expected = "GenuineIntel" if vendor == "intel" else "AuthenticAMD"
    found = {}
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = {k.strip(): v.strip() for line in block.splitlines() if ":" in line
                  for k, v in [line.split(":", 1)]}
        if fields.get("processor", "").isdigit() and int(fields["processor"]) in allowed:
            found[int(fields["processor"])] = fields
    if set(found) != set(allowed):
        raise RuntimeError("Cannot identify all CPUs available to this process")
    for fields in found.values():
        flags = set(fields.get("flags", "").split())
        if "avx512vnni" in flags:
            flags.add("avx512_vnni")
        if fields.get("vendor_id") != expected or not REQUIRED_FLAGS.issubset(flags):
            raise RuntimeError("Selected recipe requires CPU and OS AVX512/VNNI support on every allowed CPU")


# Only these generated embedded graphs are disposable. Keep their JSON audits,
# the input model and the selected graph so retrying setup needs no new download.
_COMPOSITION_INPUTS = ("base.onnx", "c128.onnx", "c64-c32.onnx", "c256.onnx", "mkl.onnx")
_PRECISION_UNUSED = ("precision/int8_all.onnx", "precision/fp16_all.onnx")


def _discard_graphs(output, names):
    for name in names:
        (output / name).unlink(missing_ok=True)


def assemble_graphs(work, original, output, commands):
    """Reproduce accepted graphs, releasing scratch after verified checkpoints."""
    from ..graph import block_fusion
    work, original, output = Path(work), Path(original), Path(output)
    _verify(original, ORIGINAL_SHA256)
    selected = output / "precision/int8_large.onnx"
    accepted, composed = output / "accepted.onnx", output / "composed.onnx"
    precision_manifest = output / "precision/manifest.json"
    # A complete checkpoint also resumes cleanup interrupted by a previous run.
    if selected.exists() and precision_manifest.is_file():
        _verify(selected, SELECTIVE_SHA256)
        _discard_graphs(output, (*_COMPOSITION_INPUTS, "composed.onnx", "accepted.onnx", *_PRECISION_UNUSED))
        return selected
    if accepted.exists():
        _verify(accepted, ACCEPTED_SHA256)
    elif composed.exists():
        _verify(composed, COMPOSED_SHA256)
    else:
        if output.exists():
            output.rename(output.with_name(output.name + ".incomplete-" + uuid.uuid4().hex[:8]))
        output.mkdir(parents=True, exist_ok=False)
        base = output / "base.onnx"
        block_fusion.rewrite(original, base, variant="both", backend=5,
                             expected_chains=18, expected_adds=18)
        _verify(base, "d7bedc899d6374c8a7599ab1f3834ca4e97e23c563c92fac4ce882e97132ed91")
        stage_script = "experiments/cpu-stage/stage/rewrite.py"
        for name, channels, tile, mode, isa in (("c128", "128", 128, 0, 512),
                                              ("c64-c32", "64,32", 256, 0, 512),
                                              ("c256", "256", 256, 1, 0)):
            _script(work, stage_script,
                    ["--source", original, "--output", output / (name + ".onnx"),
                     "--channels", channels, "--tile-time", tile, "--segments", 2,
                     "--backend", 5, "--matrix-mode", mode, "--matrix-isa", isa,
                     "--package-source", _package_source()], commands)
        _script(work, "experiments/cpu-stage/mkl/prepare_decoder.py",
                ["--source", original, "--output", output / "mkl.onnx", "--threads", 2], commands)
        args = ["--original", original, "--base", base, "--output", composed,
                "--package-source", _package_source()]
        for name, kind, extension in (("c128", "stage", ".stage.json"),
                                      ("c64-c32", "stage", ".stage.json"),
                                      ("c256", "stage", ".stage.json"), ("mkl", "mkl", ".json")):
            args.extend(["--candidate", kind, output / (name + ".onnx"), output / (name + extension)])
        _script(work, "experiments/cpu-stage/compose_candidates.py", args, commands)
        _verify(composed, COMPOSED_SHA256)
    # Neither deleting nor resuming a checkpoint changes graph transformations.
    _discard_graphs(output, _COMPOSITION_INPUTS)
    if not accepted.exists():
        _script(work, "experiments/cpu-stage/upsample/rewrite.py",
                ["--source", composed, "--output", accepted,
                 "--tile-time", 128, "--segments", 2, "--projection-mode", 1,
                 "--projection-isa", 0, "--package-source", _package_source()], commands)
        _verify(accepted, ACCEPTED_SHA256)
    _discard_graphs(output, ("composed.onnx",))
    # The unchanged precision writer emits its audit after all three variants.
    # A failed run can leave variants but no audit; regenerate those temporary
    # graphs from the verified accepted checkpoint rather than accumulating them.
    if precision_manifest.exists():
        raise RuntimeError("The completed precision audit is missing its selected graph")
    _discard_graphs(output, (*_PRECISION_UNUSED, "precision/int8_large.onnx"))
    _script(work, "experiments/intel-precision/tools/rewrite.py",
            ["--source", accepted, "--output-dir", output / "precision"], commands)
    _verify(selected, SELECTIVE_SHA256)
    if not precision_manifest.is_file():
        raise RuntimeError("The precision writer did not publish its audit")
    _discard_graphs(output, ("accepted.onnx", *_PRECISION_UNUSED))
    return selected


def reschedule(source, destination, threads):
    """Only change the accepted eighteen scheduling attributes."""
    import onnx
    if threads not in (1, 2, 4):
        raise ValueError("Unsupported precision schedule")
    _verify(source, SELECTIVE_SHA256)
    original = onnx.load(source)
    graph = copy.deepcopy(original)
    changed = []
    domains = {"fast.audiovae.precision.stage.experimental",
               "fast.audiovae.precision.upsample.experimental", "fast.audiovae.stage.experimental"}
    for old, node in zip(original.graph.node, graph.graph.node):
        attribute = None
        if node.domain == "fast.audiovae.precision.matrix.experimental" and node.op_type == "PrecisionMatMulF32":
            attribute = "shards"
        elif node.domain in domains and node.op_type in ("StageStackF32", "UpsampleStageF32"):
            attribute = "segments"
        if attribute:
            values = [a for a in node.attribute if a.name == attribute]
            if len(values) != 1 or values[0].type != onnx.AttributeProto.INT or values[0].i != 2:
                raise ValueError("Unexpected source scheduling attribute")
            values[0].i = threads
            restored = copy.deepcopy(node)
            next(a for a in restored.attribute if a.name == attribute).i = 2
            if restored.SerializeToString() != old.SerializeToString():
                raise ValueError("Rescheduling changed mathematical attributes")
            changed.append({"node": node.name, "attribute": attribute, "before": 2, "after": threads})
        elif node.SerializeToString() != old.SerializeToString():
            raise ValueError("Unselected graph node changed")
    if len(changed) != 18 or sum(x["attribute"] == "shards" for x in changed) != 14:
        raise ValueError("Expected fourteen matrix and four stage scheduling attributes")
    if [x.SerializeToString() for x in original.graph.initializer] != [x.SerializeToString() for x in graph.graph.initializer]:
        raise ValueError("Learned tensors changed while scheduling")
    onnx.checker.check_model(graph, check_custom_domain=False)
    onnx.save(graph, destination)
    if threads == 4:
        _verify(destination, "746b0c696671ee8e8ed76e55fc35f2876f861286e8b01969e26999df0d66178d")
    return changed


def _compile_libraries(work, vendor, native, xsmm, dependency, commands):
    """Compile unchanged sources, with bundle-relative runtime library lookup."""
    output = work / ".build" / ("automatic-precision-" + vendor)
    include = work / ".deps/onnxruntime/include"
    intel = work / "experiments/intel-precision"
    package = intel if vendor == "intel" else work / "experiments/amd-precision/common"
    core_source = intel / "native" if vendor == "intel" else work / "experiments/amd-precision/aocl/source"
    input_files = {native, xsmm / "lib/libxsmm.a", Path(__file__).resolve()}
    for directory in (include, xsmm / "include", package / "fused", package / "support", core_source,
                      work / "native/x86", work / "experiments/cpu-stage/stage",
                      work / "experiments/cpu-stage/matrix", dependency / "include", dependency / "lib"):
        input_files.update(path for path in directory.rglob("*") if path.is_file() and
                           (path.suffix in (".c", ".cpp", ".h") or ".so" in path.name))
    inputs = {str(path): sha256(path) for path in sorted(input_files)}
    cache_record = output / "build.json"
    if cache_record.is_file():
        cached = json.loads(cache_record.read_text())
        if cached.get("inputs") != inputs:
            raise RuntimeError("Cached precision source or dependency changed; prepare a fresh workspace")
        for record in cached["libraries"].values():
            path = Path(record["path"]).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Cached precision library leaves its build directory")
            _verify(path, record["sha256"])
        return {name: Path(record["path"]) for name, record in cached["libraries"].items()}
    if output.exists():
        output.rename(output.with_name(output.name + ".incomplete-" + uuid.uuid4().hex[:8]))
    output.mkdir(parents=True)
    prefix = "libintel_precision" if vendor == "intel" else "libamd_aocl_precision"
    core, ops = output / (prefix + "_core.so"), output / (prefix + "_ops.so")
    flags = ["-O3", "-fPIC", "-fvisibility=hidden", "-fno-fast-math", "-ffp-contract=off"]
    strict = [*flags, "-Wall", "-Wextra", "-Werror"]
    shared = ["g++", "-shared", *flags, "-std=c++17"]
    if vendor == "intel":
        pins = json.loads((intel / "pins/mkl.json").read_text())
        links = [dependency / "lib" / name for name in pins["direct_link_libraries"]]
        defines = ["-DIP_WITH_MKL=1", "-I" + str(dependency / "include"), "-Wl,--no-as-needed"]
    else:
        links = [dependency / "lib/libaocl-dlp.so"]
        defines = ["-DIP_WITH_AOCL=1", "-I" + str(dependency / "include")]
    deps.run([*shared, *defines, core_source / "precision.cpp", *links,
              "-Wl,-rpath,$ORIGIN", "-Wl,-soname," + core.name,
              "-Wl,-Bsymbolic-functions", "-lpthread", "-lm", "-ldl", "-o", core], work, commands)
    deps.run([*shared, "-I" + str(include), core_source / "custom_op.cpp", core,
              "-Wl,-rpath,$ORIGIN", "-Wl,-Bsymbolic-functions", "-o", ops], work, commands)
    support = package / "support"
    for name, source, define in (("matrix", support / "matrix/fused_pointwise.c", "FX_WITH_LIBXSMM"),
                                ("projection", support / "upsample/projection.c", "UP_WITH_LIBXSMM")):
        deps.run(["gcc", *strict, "-std=c11", "-D" + define + "=1", "-I" + str(xsmm / "include"),
                  "-c", source, "-o", output / (name + ".o")], work, commands)
    includes = [include, support / "native", support / "matrix", support / "stage",
                support / "upsample", package / "native"]
    result = {"core": core, "ops": ops}
    for name in ("stage", "upsample"):
        obj = output / (name + ".o")
        deps.run(["g++", *strict, "-std=c++17", *["-I" + str(p) for p in includes],
                  "-c", package / "fused" / (name + "_precision.cpp"), "-o", obj], work, commands)
        library = output / (prefix + "_" + name + ".so")
        deps.run(["g++", "-shared", obj, output / "matrix.o",
                  *([output / "projection.o"] if name == "upsample" else []), core, native,
                  xsmm / "lib/libxsmm.a", "-Wl,-rpath,$ORIGIN", "-ldl", "-pthread", "-lm",
                  "-o", library], work, commands)
        result[name] = library
    # The two small FP32 residual stages use the existing stage source and
    # complete-K matrix helper, with exactly its recorded compilation flags.
    stage = work / "experiments/cpu-stage/stage"
    matrix = work / "experiments/cpu-stage/matrix"
    stage_flags = ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
    deps.run(["gcc", *stage_flags, "-std=c11", "-DFX_WITH_LIBXSMM=1", "-I" + str(xsmm / "include"),
              "-c", matrix / "fused_pointwise.c", "-o", output / "fp32_matrix.o"], work, commands)
    deps.run(["g++", *stage_flags, "-std=c++17", "-I" + str(include), "-I" + str(work / "native/x86"),
              "-I" + str(matrix), "-c", stage / "custom_op.cpp", "-o", output / "fp32_stage.o"], work, commands)
    result["fp32_stage"] = output / "libstage_pipeline.so"
    deps.run(["g++", "-shared", output / "fp32_matrix.o", output / "fp32_stage.o", native,
              xsmm / "lib/libxsmm.a", "-ldl", "-pthread", "-Wl,-rpath,$ORIGIN", "-lm",
              "-o", result["fp32_stage"]], work, commands)
    if {str(path): sha256(path) for path in sorted(input_files)} != inputs:
        raise RuntimeError("A native recipe source or dependency changed during compilation")
    cache_record.write_text(json.dumps({"inputs": inputs, "libraries": {
        name: {"path": str(path), "sha256": sha256(path)} for name, path in result.items()}}, indent=2) + "\n")
    return result


def _copy_library(source, directory):
    target = directory / source.name
    expected = sha256(source)
    if target.exists() and sha256(target) != expected:
        raise ValueError("Conflicting bundled library basename: " + source.name)
    if not target.exists():
        # These cached CPU libraries are immutable after verification. Reuse
        # their storage inside the same cache; cross-device and filesystems
        # without hardlinks retain the ordinary independent-copy behavior.
        try:
            os.link(source.resolve(), target)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK,
                                  errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            shutil.copy2(source, target)
    _verify(target, expected)
    return {"library": "libs/" + target.name, "sha256": expected}


def _libraries(work, vendor, native, libraries, dependency, bundle):
    directory = bundle / "libs"
    directory.mkdir()
    base = _copy_library(native, directory)
    records = {name: _copy_library(path, directory) for name, path in libraries.items()}
    # These are CPU-only dependencies selected by the verified provisioner.
    pattern = "libmkl_*.so*" if vendor == "intel" else "libaocl-dlp.so*"
    for source in sorted((dependency / "lib").glob(pattern)):
        if source.is_file():
            _copy_library(source, directory)
    # AOCL's accepted OpenMP implementation depends on the compiler's CPU
    # runtime. Copy that exact runtime, rather than depending on host search paths.
    if vendor == "amd":
        found = subprocess.check_output(["gcc", "-print-file-name=libgomp.so.1"], text=True).strip()
        source = Path(found)
        if not source.is_file():
            raise RuntimeError("The compiler's libgomp.so.1 CPU runtime was not found")
        _copy_library(source, directory)
    notices = bundle / "licenses"
    for source_root, name in ((work / "experiments/intel-precision/licenses", "intel"),
                              (work / "experiments/amd-precision/licenses", "amd")):
        if source_root.is_dir() and (name == vendor or name == "intel"):
            shutil.copytree(source_root, notices / name)
    return base, [records[name] for name in ("fp32_stage", "ops", "stage", "upsample")]


def _payload_file(root, record):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Invalid prebuilt file record")
    relative = Path(record["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Prebuilt artifact is missing or outside its payload")
    _verify(path, record["sha256"])
    return path


def _read_payload(value, vendor):
    root = Path(value["root"]).resolve()
    manifest = value["manifest"]
    if not isinstance(manifest, dict):
        path = Path(manifest)
        path = path if path.is_absolute() else root / path
        if not path.resolve().is_relative_to(root):
            raise ValueError("Prebuilt manifest leaves its payload")
        manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("vendor") != vendor:
        raise ValueError("Incompatible prebuilt CPU recipe payload")
    required = {"native", "core", "ops", "stage", "upsample", "fp32_stage"}
    if not required.issubset(manifest.get("libraries", {})):
        raise ValueError("Prebuilt payload lacks a required CPU library")
    for group in (manifest["libraries"].values(), manifest.get("runtime_files", []),
                  manifest.get("license_files", []), [manifest["native_build"]]):
        for record in group:
            _payload_file(root, record)
    return root, manifest


def _payload_libraries(payload, bundle):
    root, manifest = payload
    directory = bundle / "libs"
    directory.mkdir()
    records = {name: _copy_library(_payload_file(root, record), directory)
               for name, record in manifest["libraries"].items()}
    for record in manifest.get("runtime_files", []):
        _copy_library(_payload_file(root, record), directory)
    for record in manifest.get("license_files", []):
        source = _payload_file(root, record)
        relative = Path(record["path"])
        if not relative.parts or relative.parts[0] != "licenses":
            raise ValueError("License payload must live under licenses/")
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return records["native"], [records[name] for name in ("fp32_stage", "ops", "stage", "upsample")]


def _bundle(work, selected, portable, vendor, threads, native, libraries, dependency, destination, *, payload=None):
    from ..prepare_streaming import prepare_streaming
    destination.mkdir(parents=True)
    shutil.copy2(portable, destination / "decoder_portable.onnx")
    changes = reschedule(selected, destination / "decoder_accepted.onnx", threads)
    base, additional = (_payload_libraries(payload, destination) if payload else
                        _libraries(work, vendor, native, libraries, dependency, destination))
    manifest = {"onnxruntime": "1.29.0", "fallback": "decoder_portable.onnx",
                "fallback_sha256": sha256(destination / "decoder_portable.onnx"),
                "native": {"Linux/x86_64": {
                    "model": "decoder_accepted.onnx", "model_sha256": sha256(destination / "decoder_accepted.onnx"),
                    "library": base["library"], "library_sha256": base["sha256"], "math": "sleef_u10",
                    "required_backend": 5, "experiment": f"{vendor}_canonical_{threads}_worker",
                    "tested_cpu": "Intel Xeon Platinum 8280 VM" if vendor == "intel" else "AMD EPYC 9654",
                    "additional_libraries": additional}},
                "providers": ["CPUExecutionProvider"], "schedule_changes": changes}
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prepare_streaming(destination, canonical_precision=True)
    manifest = json.loads((destination / "bundle.json").read_text())
    full = manifest["native"]["Linux/x86_64"]
    stream = manifest["streaming"]["models"][full["model"]]
    expected_full, expected_stream = CANONICAL_HASHES[threads]
    _verify(destination / full["model"], expected_full)
    _verify(destination / stream["model"], expected_stream)
    return destination


def _pair_graph(bundle):
    """Exact graph-only equivalent of the accepted Intel pair recipe."""
    import onnx
    path = bundle / "bundle.json"
    manifest = json.loads(path.read_text())
    full = manifest["native"]["Linux/x86_64"]
    stream = manifest["streaming"]["models"][full["model"]]
    graph_path = bundle / stream["model"]
    _verify(graph_path, CANONICAL_HASHES[1][1])
    original = onnx.load(graph_path, load_external_data=False)
    graph = copy.deepcopy(original)
    targets = [n for n in graph.graph.node if n.op_type == "PrecisionMatMulF32" and
               {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}.get("M") == 8192]
    if len(targets) != 2:
        raise ValueError("Expected exactly two first upsampling projections")
    current = next(n for n in targets if "current" in n.name)
    previous = next(n for n in targets if "previous" in n.name)
    if current.input[1] != previous.input[1]:
        raise ValueError("First projection pair does not share an activation")
    for node in targets:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if (node.domain != "fast.audiovae.precision.matrix.experimental" or
                attrs != {"K": 2048, "M": 8192, "backend": 1, "native_abi": 1, "precision_mode": 8, "shards": 1}):
            raise ValueError("First projection pair has an unsupported arithmetic contract")
    new = onnx.helper.make_node("PackedProjectionPairF32",
        [current.input[0], previous.input[0], current.input[1]], [current.output[0], previous.output[0]],
        name="stream_first_projection_pair", domain="fast.audiovae.streaming.matrix.experimental",
        native_abi=1, M=8192, K=2048)
    names = {n.name for n in targets}
    nodes, added = [], False
    for node in graph.graph.node:
        if node.name in names:
            if not added:
                nodes.append(new)
                added = True
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    graph.opset_import.append(onnx.helper.make_opsetid(new.domain, 1))
    if [x.SerializeToString() for x in original.graph.initializer] != [x.SerializeToString() for x in graph.graph.initializer]:
        raise ValueError("First pair rewrite changed trained tensors")
    onnx.checker.check_model(graph, check_custom_domain=False)
    onnx.save(graph, graph_path)
    _verify(graph_path, PAIR_SHA256)
    stream["model_sha256"] = sha256(graph_path)
    stream["additional_libraries"] = [*full["additional_libraries"],
        {"library": "libs/libpaired_projection.so", "sha256": sha256(bundle / "libs/libpaired_projection.so")}]
    stream["projection_candidate"] = {"target": "first_pair_only", "direct_frames": [1, 2, 3, 4],
        "fallback": "accepted_INT8_core_for_larger_chunks", "trained_tensors_unchanged": True}
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _pair(work, destination, commands, *, payload=None):
    """Apply the accepted Intel pair only inside unpublished bundle staging."""
    if (destination / ".recipe-ready.json").exists():
        raise RuntimeError("Cannot rewrite a published recipe bundle")
    library = destination / "libs/libpaired_projection.so"
    if payload:
        root, manifest = payload
        if "pair" not in manifest["libraries"]:
            raise ValueError("Prebuilt Intel streaming payload lacks its paired projection library")
        pair = _payload_file(root, manifest["libraries"]["pair"])
        if not library.exists():
            _copy_library(pair, library.parent)
        _verify(library, sha256(pair))
    else:
        source = work / "experiments/streaming-matrix/intel/paired_projection.cpp"
        core = destination / "libs/libintel_precision_core.so"
        deps.run(["g++", "-shared", "-fPIC", "-O3", "-std=c++17", "-fno-fast-math",
                  "-ffp-contract=off", "-fvisibility=hidden", "-Wl,-Bsymbolic-functions",
                  "-I" + str(work / ".deps/onnxruntime/include"), source, core,
                  "-Wl,-rpath,$ORIGIN", "-o", library], work, commands)
    _pair_graph(destination)
    manifest = json.loads((destination / "bundle.json").read_text())
    _verify(destination / manifest["native"]["Linux/x86_64"]["model"], CANONICAL_HASHES[1][0])
    return destination


def _runtime_dependency_audit(bundle):
    """Reject absolute ELF dependencies and search paths before publication."""
    reports = {}
    for path in sorted((bundle / "libs").glob("*.so*")):
        text = subprocess.check_output(["readelf", "-d", str(path)], text=True)
        reports[path.name] = text
        for line in text.splitlines():
            if "(NEEDED)" in line and "[" in line:
                name = line.split("[", 1)[1].split("]", 1)[0]
                if "/" in name:
                    raise RuntimeError("Runtime dependency is not relocatable: " + name)
            if ("(RPATH)" in line or "(RUNPATH)" in line) and "[" in line:
                values = line.split("[", 1)[1].split("]", 1)[0].split(":")
                if any(value and value not in ("$ORIGIN", "${ORIGIN}") for value in values):
                    raise RuntimeError("Runtime library contains a nonlocal search path: " + path.name)
    return reports


def export_native_payload(work_dir, bundle, destination):
    """Export weight-free CPU libraries for a platform wheel, without inference.

    The caller must validate the relocated payload on its target platform before
    publishing a wheel. This copies the executed build's recorded artifacts; it
    does not declare freshly linked binaries numerically validated.
    """
    work, bundle, destination = map(lambda p: Path(p).resolve(), (work_dir, bundle, destination))
    manifest = json.loads((bundle / "bundle.json").read_text())
    recipe = manifest["automatic_recipe"]
    vendor = recipe["vendor"]
    if destination.exists():
        raise ValueError("Use a fresh native payload destination")
    # The maintainer build checks ELF metadata while it still has binutils.
    _runtime_dependency_audit(bundle)
    destination.mkdir(parents=True)
    shutil.copytree(bundle / "libs", destination / "libs")
    if (bundle / "licenses").is_dir():
        shutil.copytree(bundle / "licenses", destination / "licenses")
    full = manifest["native"]["Linux/x86_64"]
    def record(relative):
        return {"path": relative, "sha256": sha256(destination / relative)}
    native_path = "libs/" + Path(full["library"]).name
    libraries = {"native": record(native_path)}
    for item in full["additional_libraries"]:
        name = Path(item["library"]).name
        if name == "libstage_pipeline.so":
            key = "fp32_stage"
        elif name.endswith("_ops.so") or name == "precision_ops.so":
            key = "ops"
        elif name.endswith("_stage.so"):
            key = "stage"
        elif name.endswith("_upsample.so"):
            key = "upsample"
        else:
            raise ValueError("Unrecognized native payload operator: " + name)
        libraries[key] = record("libs/" + name)
    core_name = "libintel_precision_core.so" if vendor == "intel" else "libamd_aocl_precision_core.so"
    libraries["core"] = record("libs/" + core_name)
    if (destination / "libs/libpaired_projection.so").is_file():
        libraries["pair"] = record("libs/libpaired_projection.so")
    native_record = json.loads((work / ".build/x86/build.json").read_text())
    _verify(bundle / full["library"], native_record["library_sha256"])
    # Preparation needs the declared ABI/operators/backend capabilities, not
    # private compiler command paths from a maintainer's build directory.
    native_metadata = {key: native_record[key] for key in
                       ("native_abi", "ort_api_version", "domain", "operators", "tile", "library_sha256")
                       if key in native_record}
    native_metadata.update(library=native_path,
                           fingerprint={"explicit_avx512_backend": native_record["fingerprint"]["explicit_avx512_backend"]})
    (destination / "native-build.json").write_text(json.dumps(native_metadata, indent=2) + "\n")
    roles = {item["path"] for item in libraries.values()}
    payload = {"schema_version": 1, "vendor": vendor, "cpu_only": True,
               "onnxruntime": "1.29.0", "native_build": record("native-build.json"),
               "libraries": libraries,
               "runtime_files": [record(str(p.relative_to(destination))) for p in sorted((destination / "libs").iterdir())
                                 if p.is_file() and str(p.relative_to(destination)) not in roles],
               "license_files": [record(str(p.relative_to(destination))) for p in sorted((destination / "licenses").rglob("*"))
                                  if p.is_file()],
               "required_cpu_flags": sorted(REQUIRED_FLAGS),
               "canonical_math_version": 1,
               "artifact_validation": "Maintainer validation required before wheel publication"}
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_recipe(work_dir: Path, source: Path, platform_info: dict, mode: str, threads: int) -> Path:
    """Build a fresh hash-bound, relocatable CPU bundle in a materialized tree.

    The caller owns source-tree materialization and the cache publication lock.
    Fresh compiled binaries still require its process-isolated runtime check.
    """
    work, source = Path(work_dir).resolve(), Path(source).resolve()
    vendor = validate_selection(platform_info, mode, threads)
    _host_gate(vendor, threads)
    verify_model(source)
    offline = bool(platform_info.get("offline", False))
    jobs = platform_info.get("build_jobs", 2)
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError("Build jobs must be one or two")
    payload = _read_payload(platform_info["prebuilt"], vendor) if platform_info.get("prebuilt") else None
    if payload and vendor == "intel" and mode == "streaming" and "pair" not in payload[1]["libraries"]:
        raise ValueError("Prebuilt Intel streaming payload lacks its paired projection library")
    if not payload:
        for tool in ("gcc", "g++", "make", "cmake", "nm", "readelf"):
            if not shutil.which(tool):
                raise RuntimeError("CPU source setup requires " + tool)
    output = work / "bundles" / f"{vendor}-{mode}-{threads}"
    commands = []
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".x86-recipe.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (output / ".recipe-ready.json").is_file():
            return completed_bundle(output, lambda _: None)
        if payload:
            payload_root, payload_manifest = payload
            native = _payload_file(payload_root, payload_manifest["libraries"]["native"])
            native_record = json.loads(_payload_file(payload_root, payload_manifest["native_build"]).read_text())
            if native_record.get("library_sha256") != sha256(native):
                raise ValueError("Prebuilt native metadata and library disagree")
            native_record["library"] = str(native)
            native_record_path = work / ".build/prebuilt-native.json"
            native_record_path.parent.mkdir(parents=True, exist_ok=True)
            native_record_path.write_text(json.dumps(native_record, indent=2) + "\n")
            dependency, libraries = None, None
        else:
            flags = ["--jobs", jobs] + (["--offline"] if offline else [])
            _script(work, "tools/build.py", flags, commands)
            native_record_path = work / ".build/x86/build.json"
            native_record = json.loads(native_record_path.read_text())
            native = Path(native_record["library"])
            _verify(native, native_record["library_sha256"])
            xsmm = deps.libxsmm(work, offline=offline, jobs=jobs, commands=commands)
            dependency = (deps.mkl(work, offline=offline, commands=commands) if vendor == "intel"
                          else deps.aocl(work, offline=offline, jobs=jobs, commands=commands))
            libraries = _compile_libraries(work, vendor, native, xsmm, dependency, commands)
        from ..prepare import prepare
        native_bundle = completed_bundle(work / ".build/automatic-original",
            lambda destination: prepare(destination, source=source, native_build=native_record_path))
        selected = assemble_graphs(work, native_bundle / "decoder_native.onnx",
                                   work / ".build/automatic-graphs", commands)
        def create_canonical(destination):
            return _bundle(work, selected, native_bundle / "decoder_portable.onnx", vendor, threads,
                           native, libraries, dependency, destination, payload=payload)
        build_evidence = {}
        def create_output(destination):
            # The canonical full/reference graph remains in this final bundle.
            # Verify it and its stream before applying the exact Intel rewrite
            # in private staging; publication occurs only after every check.
            create_canonical(destination)
            if vendor == "intel" and mode == "streaming":
                _pair(work, destination, commands, payload=payload)
            build_evidence["runtime_dependencies"] = (
                {"verification": "Prebuilt payload hashes checked; ELF metadata checked during wheel assembly"}
                if payload else _runtime_dependency_audit(destination))
            manifest_path = destination / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["automatic_recipe"] = {
                "version": 1, "vendor": vendor, "mode": mode, "threads": threads,
                "precision": "selective_int8", "cpu_only": True, "inner_threads": 1,
                "required_cpu_flags": sorted(REQUIRED_FLAGS),
                "first_projection_pair": vendor == "intel" and mode == "streaming",
                "prebuilt": payload is not None, "source_model_sha256": sha256(source),
                "recipe_math": "Existing accepted scales, integer values, complete-K reduction and canonical streaming sine",
                "build_validation": "Graph identity checked; fresh binary runtime validation is pending"}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        completed_bundle(output, create_output)
        receipt = {"version": 1, "commands": commands, **build_evidence,
                   "compiler": (None if payload else {name: subprocess.check_output([name, "--version"], text=True).splitlines()[0]
                                for name in ("gcc", "g++")}), "bundle_files": deps.file_hashes(output),
                   "model_execution": False, "benchmark_execution": False, "cpu_only": True}
        (work / ".build" / f"recipe-{vendor}-{mode}-{threads}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return output
