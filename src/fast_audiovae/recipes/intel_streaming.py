"""Accepted Intel streaming overlay with exact INT8 projection arithmetic.

The first VNNI pair, all weights, SLEEF, state layout and batch graph stay intact.
Only the six raw histories, first phase assembly and second projection pair use
new operators. OneDNN handles the qualified 40/80ms shapes; other packet sizes
keep the existing oneMKL INT8 path.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import uuid

from ..assets import sha256
from . import amd_streaming as shared

VERSION = "intel_stream_selected_v1"
SOURCE_FULL = "023e40ad0fb9578fbe232942d6aeedb2ed30a426c9db59b70694927e17c67310"
SOURCE_STREAM = "59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516"
# Deterministic embedded rewrite of the accepted paired source graph.
SELECTED_STREAM = "7358607b6864c0fedcd1854b9315329c8ee483c6087a3ded8dd7045b6cc4f0e7"
PAIR_DOMAIN = "fast.audiovae.streaming.matrix.experimental"
MATRIX_DOMAIN = "fast.audiovae.intel.streaming.matrix.v1"
LIBRARIES = {
    "intel_stream_history": "libintel_rawhistory.so",
    "intel_stream_phase": "libintel_phase.so",
    "intel_stream_matrix": "libintel_stream_matrix.so",
    "intel_stream_matrix_ops": "libintel_stream_matrix_ops.so",
    "intel_stream_onednn": "libdnnl.so.3",
}
_require = shared._require


def _matrix(model):
    import onnx
    from onnx import helper
    weights = {w.name: w for w in model.graph.initializer}
    outputs = ("ncc_up_2_split_matmul_current", "ncc_up_2_split_matmul_previous_unshifted")
    names = ("ncc_up_2_split_matmul_current_w", "ncc_up_2_split_matmul_previous_w")
    expected = dict(M=3072, K=1024, precision_mode=8, backend=1, native_abi=1, shards=1)
    targets = []
    for output, name in zip(outputs, names):
        found = [n for n in model.graph.node if list(n.output) == [output]]
        _require(len(found) == 1, "Exactly one second-pair branch required")
        node = found[0]
        _require(node.domain == "fast.audiovae.precision.matrix.experimental" and node.op_type == "PrecisionMatMulF32"
                 and list(node.input) == [name, "view_15"] and shared._attrs(node) == expected,
                 "Second projection arithmetic or scheduling changed")
        _require(name in weights and list(weights[name].dims) == [3072, 1024]
                 and weights[name].data_type == onnx.TensorProto.FLOAT, "Second projection coefficients changed")
        targets.append(node)
    removed = {n.name for n in targets}
    _require(len(removed) == 2, "Distinct second projection branches required")
    replacement = helper.make_node("SecondProjectionPairF32", [*names, "view_15"], list(outputs),
        name="intel_second_projection_pair", domain=MATRIX_DOMAIN, native_abi=1, M=3072, K=1024, mode=4)
    _require(replacement.name not in {n.name for n in model.graph.node}, "Replacement name collision")
    nodes, added = [], False
    for node in model.graph.node:
        if node.name in removed:
            if not added:
                nodes.append(replacement); added = True
        else:
            nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid(MATRIX_DOMAIN, 1))
    return removed, {replacement.name}


def rewrite_graph(original):
    """Pure, strict rewrite of the accepted paired Intel source graph."""
    import onnx
    _require(not original.functions and not original.graph.sparse_initializer, "Flat dense graph required")
    _require(not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                     for n in original.graph.node for a in n.attribute), "Nested graph unsupported")
    _require(not any(w.data_location == onnx.TensorProto.EXTERNAL for w in original.graph.initializer), "Embedded graph required")
    _require(len({n.name for n in original.graph.node}) == len(original.graph.node), "Unique node names required")
    _require(len({w.name for w in original.graph.initializer}) == len(original.graph.initializer), "Unique initializer names required")
    _require(not {w.name for w in original.graph.initializer}.intersection(v.name for v in original.graph.input), "Overridable initializers unsupported")
    _require(not {shared.HISTORY_DOMAIN, shared.PHASE_DOMAIN, MATRIX_DOMAIN}.intersection(x.domain for x in original.opset_import), "Intel overlay already present")
    pairs = [n for n in original.graph.node if n.op_type == "PackedProjectionPairF32"]
    _require(len(pairs) == 1, "Existing Intel first projection pair required")
    pair = pairs[0]
    _require(pair.domain == PAIR_DOMAIN and pair.name == "stream_first_projection_pair"
             and shared._attrs(pair) == dict(K=2048, M=8192, native_abi=1)
             and len(pair.input) == 3 and len(pair.output) == 2, "Existing Intel pair contract changed")
    constants = {w.name: w for w in original.graph.initializer}
    _require(all(w in constants and constants[w].data_type == onnx.TensorProto.FLOAT
                 and list(constants[w].dims) == [8192, 2048] for w in pair.input[:2]), "Existing Intel pair weights changed")
    graph = copy.deepcopy(original); removed, added = set(), set()
    for operation in (shared._history, shared._phase, _matrix):
        rm, add = operation(graph)
        _require(not removed.intersection(rm), "Overlapping selected regions")
        removed.update(rm); added.update(add)
    checks = {field+"_byte_identical": shared._sequence(getattr(original.graph, field)) == shared._sequence(getattr(graph.graph, field))
              for field in ("initializer", "input", "output")}
    checks["remaining_nodes_byte_identical"] = (shared._sequence(n for n in original.graph.node if n.name not in removed) ==
                                               shared._sequence(n for n in graph.graph.node if n.name not in added))
    checks["intel_first_pair_byte_identical"] = pair.SerializeToString() == next(n for n in graph.graph.node if n.name == pair.name).SerializeToString()
    _require(all(checks.values()), "Unselected graph payload changed")
    onnx.checker.check_model(graph, check_custom_domain=False)
    return graph, {"checks": checks, "removed_nodes": sorted(removed), "inserted_nodes": sorted(added)}


def onednn(work, *, offline, jobs, commands):
    """Pinned CPU-only, sequential oneDNN source build in the private cache."""
    from . import dependencies as deps
    work = Path(work).resolve()
    pin = json.loads((work / "native/intel/streaming/onednn.json").read_text())
    _require(pin["version"] == "3.13.2", "Accepted oneDNN version required")
    _require(type(jobs) is int and jobs > 0, "Positive build parallelism required")
    root = work / ".deps/automatic-onednn-3.13.2-seq-v1"
    identity = {"version": pin["version"], "archive_sha256": pin["archive_sha256"], "configuration": pin["configuration"]}
    if deps._reuse(root, identity):
        return root / "install"
    root.mkdir(parents=True, exist_ok=True)
    archive = deps.fetch(pin["archive_url"], root / "source.tar.gz", pin["archive_sha256"], offline=offline)
    source = deps.unpack(archive, root / "source")
    install = root / "install"; local_commands = []
    deps.run(["cmake", "-S", source, "-B", root / "build", "-DCMAKE_INSTALL_PREFIX="+str(install),
              *["-D"+key+"="+value for key, value in pin["configuration"].items()]], work, local_commands)
    deps.run(["cmake", "--build", root / "build", "--parallel", jobs], work, local_commands)
    deps.run(["cmake", "--install", root / "build"], work, local_commands)
    commands.extend(local_commands)
    _require(any((install / d / "libdnnl.so.3").is_file() for d in ("lib", "lib64")), "oneDNN CPU library not produced")
    notices = install / "licenses"; notices.mkdir(exist_ok=True)
    for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
        _require((source / name).is_file(), "Missing oneDNN license notice")
        shutil.copy2(source / name, notices / name)
    files = {"source.tar.gz": sha256(archive)}
    files.update({str(p.relative_to(root)): sha256(p) for p in sorted(install.rglob("*")) if p.is_file()})
    deps._receipt(root, identity, local_commands, files)
    return install


def augment(work, destination, commands, *, payload=None, onnxruntime_version, offline=False, jobs=2):
    """Augment an unpublished Intel one-worker bundle without running inference."""
    import onnx
    from . import dependencies as deps, x86
    work, destination = Path(work).resolve(), Path(destination).resolve()
    _require(onnxruntime_version == "1.29.0", "Qualified ONNX Runtime 1.29.0 required")
    _require(not (destination / ".recipe-ready.json").exists(), "Cannot modify a published bundle")
    manifest_path = destination / "bundle.json"; before = manifest_path.read_bytes(); manifest = json.loads(before)
    native = manifest["native"]["Linux/x86_64"]
    stream = manifest["streaming"]["models"][native["model"]]
    full_path = shared._inside(destination, native["model"]); source_path = shared._inside(destination, stream["model"])
    _require(sha256(full_path) == native["model_sha256"] == SOURCE_FULL, "Canonical one-thread full graph required")
    _require(sha256(source_path) == stream["model_sha256"] == SOURCE_STREAM, "Accepted Intel paired stream graph required")
    _require(sha256(shared._inside(destination, native["library"])) == native["library_sha256"], "Base native library mismatch")
    _require(sha256(shared._inside(destination, manifest["fallback"])) == manifest["fallback_sha256"], "Portable graph mismatch")
    _require(type(SELECTED_STREAM) is str and len(SELECTED_STREAM) == 64, "Production graph has not been qualified")
    original = onnx.load(source_path, load_external_data=False)
    shared._validate_states(original, stream)
    state_before = copy.deepcopy(stream["states"])
    if payload:
        root, info = payload
        _require(info.get("vendor") == "intel" and set(LIBRARIES).issubset(info.get("libraries", {})), "Complete Intel streaming payload required")
        files = {role: x86._payload_file(Path(root).resolve(), info["libraries"][role]) for role in LIBRARIES}
    else:
        dependency = onednn(work, offline=offline, jobs=jobs, commands=commands)
        output = work / ".build/intel-streaming"
        deps.run([sys.executable, work / "tools/build_intel_streaming.py", "--work", work,
                  "--bundle", destination, "--onednn", dependency, "--output", output], work, commands)
        record = json.loads((output / "build.json").read_text())
        _require(record.get("version") == "intel_stream_selected_build_v1" and record.get("complete") is True, "Incomplete Intel streaming build")
        files = {}
        for role in LIBRARIES:
            row = record["libraries"][role]; path = Path(row["path"]).resolve()
            _require(path.is_relative_to(output.resolve()) and sha256(path) == row["sha256"], "Intel artifact mismatch")
            files[role] = path
        notices = destination / "licenses/onednn"; notices.mkdir(parents=True, exist_ok=True)
        for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
            shutil.copy2(dependency / "licenses" / name, notices / name)
    for name in ("LICENSE", "THIRD-PARTY-PROGRAMS"):
        _require((destination / "licenses/onednn" / name).is_file(), "oneDNN redistribution notice missing")
    extra = []
    for role, basename in LIBRARIES.items():
        _require(files[role].name == basename, "Intel streaming library filename differs from role")
        record = x86._copy_library(files[role], destination / "libs")
        if role not in ("intel_stream_onednn", "intel_stream_matrix"):
            extra.append(record)
    graph, audit = rewrite_graph(original)
    target = source_path.with_name("decoder_intel_selected.onnx")
    _require(not target.exists(), "Selected Intel graph already exists")
    temporary = target.with_name(target.name+".tmp-"+uuid.uuid4().hex)
    try:
        onnx.save_model(graph, temporary)
        _require(sha256(temporary) == SELECTED_STREAM, "Intel graph differs from accepted artifact")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _require(sha256(full_path) == SOURCE_FULL and sha256(source_path) == SOURCE_STREAM and manifest_path.read_bytes() == before,
             "Canonical bundle changed during preparation")
    _require(stream["states"] == state_before, "Streaming state contract changed")
    stream["model"] = str(target.relative_to(destination)); stream["model_sha256"] = SELECTED_STREAM
    # Preserve the first paired projection, which is a streaming-only library.
    stream["additional_libraries"] = [*stream["additional_libraries"], *extra]
    stream["required_streaming_operators"] = [*stream.get("required_streaming_operators", []),
        {"domain": shared.HISTORY_DOMAIN, "operator": "RawHistorySnakeDW7SnakeF32"},
        {"domain": shared.PHASE_DOMAIN, "operator": "StatefulPhaseFinishF32"},
        {"domain": MATRIX_DOMAIN, "operator": "SecondProjectionPairF32"}]
    stream["intel_stream_selected"] = {"version": VERSION, "source_full_sha256": SOURCE_FULL,
        "source_stream_sha256": SOURCE_STREAM, "model_sha256": SELECTED_STREAM, "onnxruntime": onnxruntime_version,
        "precision": "selective_int8", "threads": 1, "direct_second_pair_frames": [8, 16],
        "other_second_pair_frames": "unchanged oneMKL INT8 fallback", "first_projection_pair": "unchanged VNNI",
        "state_contract": "unchanged raw pre-Snake and dequantized previous-projection histories",
        "state_bytes": 683264, "libraries": extra, "onednn": "3.13.2 CPU SEQ / GPU NONE", **audit}
    manifest["onnxruntime"] = onnxruntime_version
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return destination
