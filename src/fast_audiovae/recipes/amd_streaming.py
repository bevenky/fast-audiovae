"""Exact accepted AMD one-worker streaming overlay, prepared in private staging.

Pair two selective INT8 projections, preserve six raw input histories directly,
and assemble the first phase from cached projections. No trained value, state
representation, precision, or full/batch model is changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid

from ..assets import sha256

VERSION = "amd_stream_selected_v1"
SOURCE_FULL = "023e40ad0fb9578fbe232942d6aeedb2ed30a426c9db59b70694927e17c67310"
SOURCE_STREAM = "8bb688f8606d4f07abd467de951c8554ec54fb9632bb0926db5b328670e129bd"
SELECTED_STREAM = "7fc62af2b701c0f65ff7adeedcd46d30b47dd775d41e0f3f7fa5cd6dcce2966c"
PAIR_DOMAIN = "fast.audiovae.amd.int8.pair.v1"
HISTORY_DOMAIN = "fast.audiovae.amd.rawhistory.a2.v1"
PHASE_DOMAIN = "fast.audiovae.amd.phase.state.v1"
LIBRARIES = {
    "amd_stream_pair": "libamd_int8_paired_projection.so",
    "amd_stream_history": "libaudiovae_amd_a2_rawhistory.so",
    "amd_stream_phase": "liba4_phase.so",
}
HISTORY_NODES = ("ncc_block_chain_18", "ncc_block_chain_26", "ncc_block_chain_34",
                 "ncc_block_chain_53", "ncc_block_chain_61", "ncc_block_chain_69")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _attrs(node):
    from onnx import helper
    _require(len({a.name for a in node.attribute}) == len(node.attribute), "Duplicate node attributes")
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _sequence(items):
    digest = hashlib.sha256()
    for item in items:
        raw = item.SerializeToString()
        digest.update(len(raw).to_bytes(8, "little")); digest.update(raw)
    return digest.hexdigest()


def _dims(value):
    import onnx
    tensor = value.type.tensor_type
    _require(tensor.elem_type == onnx.TensorProto.FLOAT, "FP32 state required")
    return [d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim]


def _pair(model):
    import onnx
    from onnx import helper
    expected = dict(K=2048, M=8192, backend=1, native_abi=1, precision_mode=8, shards=1)
    targets = [n for n in model.graph.node if n.op_type == "PrecisionMatMulF32" and _attrs(n).get("M") == 8192]
    _require(len(targets) == 2, "Expected two first projection matrices")
    for n in targets:
        _require(n.domain == "fast.audiovae.precision.matrix.experimental" and _attrs(n) == expected,
                 "Unsupported first projection arithmetic")
        _require(len(n.input) == 2 and len(n.output) == 1, "Unexpected projection arity")
    current = [n for n in targets if "current" in n.name]
    previous = [n for n in targets if "previous" in n.name]
    _require(len(current) == len(previous) == 1 and current[0].name != previous[0].name,
             "Ambiguous projection branches")
    current, previous = current[0], previous[0]
    _require(current.input[1] == previous.input[1], "Projection activations must be shared")
    constants = {w.name: w for w in model.graph.initializer}
    for n in targets:
        w = constants[n.input[0]]
        _require(list(w.dims) == [8192, 2048] and w.data_type == onnx.TensorProto.FLOAT,
                 "Unexpected fixed projection weights")
    replacement = helper.make_node("PackedProjectionPairF32",
        [current.input[0], previous.input[0], current.input[1]], [current.output[0], previous.output[0]],
        name="amd_int8_first_projection_pair_v1", domain=PAIR_DOMAIN, native_abi=1, M=8192, K=2048)
    removed = {n.name for n in targets}; added = False; nodes = []
    for n in model.graph.node:
        if n.name in removed:
            if not added:
                nodes.append(replacement); added = True
        else:
            nodes.append(n)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid(PAIR_DOMAIN, 1))
    return removed, {replacement.name}


def _history(model):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    nodes = {n.name: n for n in model.graph.node}
    constants = {w.name: w for w in model.graph.initializer}
    inputs = {v.name: v for v in model.graph.input}; outputs = {v.name: v for v in model.graph.output}
    producers = {v: n for n in model.graph.node for v in n.output}; uses = {}
    for n in model.graph.node:
        for v in n.input:
            uses.setdefault(v, []).append(n)
    removed = set(); replacements = {}; geometries = set()
    for name in HISTORY_NODES:
        n = nodes[name]; a = _attrs(n)
        _require(n.op_type == "SnakeDW7SnakeF32" and n.domain == "venky.audio.cpu.portable"
                 and len(n.input) == 7 and len(n.output) == 1, "Unexpected raw-history source node")
        _require(set(a) == {"native_abi", "channels", "dilation", "backend", "row_batches", "require_vector_sine"}
                 and a["native_abi"] == 1 and a["backend"] == 5 and a["require_vector_sine"] == 1
                 and a["row_batches"] >= 0, "Unexpected raw-history attributes")
        c, d = a["channels"], a["dilation"]; halo = 6*d
        _require(c in (512, 1024) and d in (1, 3, 9), "Unexpected history geometry")
        geometries.add((c, d))
        for i, shape in zip(range(1, 7), ([c, 1, 7], [c], [c], [c], [c], [c])):
            w = constants[n.input[i]]
            _require(w.name not in inputs and w.data_type == onnx.TensorProto.FLOAT and list(w.dims) == shape
                     and np.isfinite(numpy_helper.to_array(w)).all(), "Expected literal finite history coefficients")
        joined = producers[n.input[0]]
        _require(joined.op_type == "Concat" and not joined.domain and _attrs(joined) == {"axis": 2}
                 and len(joined.input) == 2, "Unexpected history concatenation")
        hist, x = joined.input
        _require(hist in inputs and _dims(inputs[hist]) == [1, c, halo], "Raw history input changed")
        consumers = uses[joined.output[0]]
        _require(len(consumers) == 2 and sum(z.name == name for z in consumers) == 1, "Additional history consumer")
        retain = next(z for z in consumers if z.name != name)
        _require(len(uses[n.output[0]]) == 1, "Additional depthwise output consumer")
        crop = uses[n.output[0]][0]
        for node, start in ((retain, -halo), (crop, halo)):
            _require(node.op_type == "Slice" and not node.domain and not node.attribute
                     and len(node.input) == 5 and len(node.output) == 1, "Unexpected history slice")
            _require([numpy_helper.to_array(constants[q]).tolist() for q in node.input[1:]] ==
                     [[start], [9223372036854775807], [2], [1]], "History slice constants changed")
        _require(retain.output[0] in outputs and _dims(outputs[retain.output[0]]) == [1, c, halo],
                 "Raw history output changed")
        _require(joined.output[0] not in outputs and n.output[0] not in outputs, "Intermediate is graph output")
        group = {joined.name, retain.name, n.name, crop.name}
        _require(not removed.intersection(group), "Overlapping history regions"); removed.update(group)
        attrs = {k: v for k, v in a.items() if k != "native_abi"}; attrs["candidate_abi"] = 1
        replacements[crop.name] = helper.make_node("RawHistorySnakeDW7SnakeF32",
            [x, hist, *n.input[1:]], [crop.output[0], retain.output[0]], name=n.name, domain=HISTORY_DOMAIN, **attrs)
    _require(geometries == {(c, d) for c in (512, 1024) for d in (1, 3, 9)}, "Six history regions required")
    nodes = [replacements[n.name] if n.name in replacements else n for n in model.graph.node
             if n.name in replacements or n.name not in removed]
    del model.graph.node[:]; model.graph.node.extend(nodes)
    live = {x for n in nodes for x in (*n.input, *n.output)} | {v.name for v in (*model.graph.input, *model.graph.output)}
    infos = [v for v in model.graph.value_info if v.name in live]
    del model.graph.value_info[:]; model.graph.value_info.extend(infos)
    model.opset_import.append(helper.make_opsetid(HISTORY_DOMAIN, 1))
    return removed, {n.name for n in replacements.values()}


def _phase(model):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    nodes = list(model.graph.node); constants = {w.name: w for w in model.graph.initializer}
    inputs = {v.name: v for v in model.graph.input}; outputs = {v.name: v for v in model.graph.output}
    uses = {}
    for n in nodes:
        for x in n.input: uses.setdefault(x, []).append(n.name)
    targets = [(i, n) for i, n in enumerate(nodes) if n.op_type == "PhaseSumBiasInterleaveF32" and _attrs(n).get("channels") == 1024]
    _require(len(targets) == 1, "Exactly one first phase required")
    index, phase = targets[0]; c, stride = 1024, 8
    _require(3 <= index < len(nodes)-1, "Incomplete phase region")
    previous, retain, current, _, crop = nodes[index-3:index+2]
    _require(_attrs(phase) == dict(channels=c, stride=stride, native_abi=1, previous_shift=1, row_batches=0)
             and phase.domain in ("venky.audio.cpu", "venky.audio.cpu.portable"), "First phase contract changed")
    _require([n.op_type for n in (previous, retain, current, phase, crop)] ==
             ["Concat", "Slice", "Concat", "PhaseSumBiasInterleaveF32", "Slice"], "Unexpected phase topology")
    _require(all(n.domain in ("", "ai.onnx") for n in (previous, retain, current, crop)), "Nonstandard phase wrapper")
    _require(len(previous.input) == len(current.input) == 2 and len(phase.input) == 3
             and all(len(n.output) == 1 for n in (previous, retain, current, phase, crop)), "Unexpected phase arity")
    _require(_attrs(previous) == _attrs(current) == {"axis": 2}, "Phase axes changed")
    _require(previous.input[0] in inputs and retain.output[0] in outputs, "Projected state endpoints missing")
    _require(_dims(inputs[previous.input[0]]) == _dims(outputs[retain.output[0]]) == [1, 8192, 1], "Projected state changed")
    _require(list(phase.input[:2]) == [current.output[0], previous.output[0]] and retain.input[0] == previous.output[0]
             and crop.input[0] == phase.output[0], "Phase wiring changed")
    _require(sorted(uses[previous.output[0]]) == sorted([retain.name, phase.name])
             and uses[current.output[0]] == [phase.name] and uses[phase.output[0]] == [crop.name], "Additional phase consumer")
    _require(not {previous.output[0], current.output[0], phase.output[0]}.intersection(outputs), "Intermediate phase output exposed")
    zero = numpy_helper.to_array(constants[current.input[0]])
    _require(zero.dtype == np.float32 and zero.shape == (1, 8192, 1) and not zero.view(np.uint32).any(), "Positive-zero phase padding required")
    for n, start in ((retain, -1), (crop, stride)):
        _require(len(n.input) == 5 and not n.attribute and all(constants[x].data_type == onnx.TensorProto.INT64 for x in n.input[1:]), "Phase slice changed")
        _require([numpy_helper.to_array(constants[x]).tolist() for x in n.input[1:]] ==
                 [[start], [9223372036854775807], [2], [1]], "Phase slice constants changed")
    bias = numpy_helper.to_array(constants[phase.input[2]])
    _require(bias.dtype == np.float32 and bias.shape == (c,) and np.isfinite(bias).all(), "Fixed finite phase bias required")
    replacement = helper.make_node("StatefulPhaseFinishF32", [current.input[1], previous.input[1], previous.input[0], phase.input[2]],
        [crop.output[0], retain.output[0]], name=phase.name+"_a4_direct", domain=PHASE_DOMAIN,
        channels=c, stride=stride, phase_state_abi=1, threads=1)
    _require(replacement.name not in {n.name for n in nodes}, "Phase replacement name collision")
    removed = {n.name for n in (previous, retain, current, phase, crop)}
    kept = [replacement if n.name == crop.name else n for n in nodes if n.name == crop.name or n.name not in removed]
    del model.graph.node[:]; model.graph.node.extend(kept)
    model.opset_import.append(helper.make_opsetid(PHASE_DOMAIN, 1))
    return removed, {replacement.name}


def rewrite_graph(original):
    """Pure graph rewrite. Source/final artifact hashes are checked by augment."""
    import onnx
    _require(not original.functions and not original.graph.sparse_initializer, "Flat dense graph required")
    _require(not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in original.graph.node for a in n.attribute), "Nested graph unsupported")
    _require(not any(w.data_location == onnx.TensorProto.EXTERNAL for w in original.graph.initializer), "Embedded graph required")
    _require(len({n.name for n in original.graph.node}) == len(original.graph.node), "Unique node names required")
    _require(not {w.name for w in original.graph.initializer}.intersection(v.name for v in original.graph.input), "Overridable initializers unsupported")
    _require(not {PAIR_DOMAIN, HISTORY_DOMAIN, PHASE_DOMAIN}.intersection(x.domain for x in original.opset_import), "AMD selection already present")
    graph = copy.deepcopy(original); removed = set(); added = set()
    for operation in (_pair, _history, _phase):
        rm, add = operation(graph)
        _require(not removed.intersection(rm), "Overlapping selected regions")
        removed.update(rm); added.update(add)
    checks = {field+"_byte_identical": _sequence(getattr(original.graph, field)) == _sequence(getattr(graph.graph, field))
              for field in ("initializer", "input", "output")}
    checks["remaining_nodes_byte_identical"] = (_sequence(n for n in original.graph.node if n.name not in removed) ==
                                                _sequence(n for n in graph.graph.node if n.name not in added))
    _require(all(checks.values()), "Unselected graph payload changed")
    onnx.checker.check_model(graph, check_custom_domain=False)
    return graph, {"checks": checks, "removed_nodes": sorted(removed), "inserted_nodes": sorted(added)}


def _inside(root, relative):
    root = Path(root).resolve(); relative = Path(relative)
    path = (root / relative).resolve()
    _require(not relative.is_absolute() and path.is_relative_to(root) and path.is_file(), "Bundle file missing or outside staging")
    return path


def _validate_states(graph, stream):
    inputs = {v.name: v for v in graph.graph.input}; outputs = {v.name: v for v in graph.graph.output}
    states = stream.get("states", [])
    _require(len(states) == 18 and stream.get("state_bytes") == 683264, "Accepted state inventory required")
    _require(len({s["input"] for s in states}) == len({s["output"] for s in states}) == 18, "Duplicate state endpoint")
    _require({s["input"] for s in states} == set(inputs)-{stream["latent_input"]}
             and {s["output"] for s in states} == set(outputs)-{stream["audio_output"]}, "State endpoint inventory differs from graph")
    count = 0
    for state in states:
        shape = state["shape"]
        _require(state.get("dtype") == "float32" and state.get("time_axis") == 2 and len(shape) == 3
                 and all(type(x) is int and x > 0 for x in shape), "Invalid FP32 history metadata")
        _require(_dims(inputs[state["input"]]) == _dims(outputs[state["output"]]) == shape, "State shape differs from graph")
        count += 4*shape[0]*shape[1]*shape[2]
    _require(count == 683264, "State byte count differs from graph")


def augment(work, destination, commands, *, payload=None, onnxruntime_version):
    """Augment only an unpublished canonical one-thread bundle, without inference."""
    import onnx
    from . import dependencies as deps, x86
    work, destination = Path(work).resolve(), Path(destination).resolve()
    _require(onnxruntime_version in ("1.29.0", "1.30.0"), "Explicit qualified ORT1.29.0 or1.30.0 required")
    _require(not (destination / ".recipe-ready.json").exists(), "Cannot modify a published bundle")
    manifest_path = destination / "bundle.json"; before = manifest_path.read_bytes(); manifest = json.loads(before)
    native = manifest["native"]["Linux/x86_64"]
    stream = manifest["streaming"]["models"][native["model"]]
    full_path = _inside(destination, native["model"]); source_path = _inside(destination, stream["model"])
    _require(sha256(full_path) == native["model_sha256"] == SOURCE_FULL, "Expected canonical one-thread full graph")
    _require(sha256(source_path) == stream["model_sha256"] == SOURCE_STREAM, "Expected canonical one-thread stream graph")
    _require(sha256(_inside(destination, native["library"])) == native["library_sha256"], "Base library hash mismatch")
    _require(sha256(_inside(destination, manifest["fallback"])) == manifest["fallback_sha256"], "Fallback hash mismatch")
    original = onnx.load(source_path, load_external_data=False)
    _validate_states(original, stream)
    state_before = copy.deepcopy(stream["states"])
    if payload:
        root, info = payload
        _require(info.get("vendor") == "amd", "AMD payload required")
        _require(set(LIBRARIES).issubset(info.get("libraries", {})), "Prebuilt AMD streaming payload lacks required pair/history/phase roles")
        files = {role: x86._payload_file(Path(root).resolve(), info["libraries"][role]) for role in LIBRARIES}
    else:
        output = work / ".build/amd-streaming"
        deps.run([sys.executable, work / "tools/build_amd_streaming.py", "--work", work,
                  "--bundle", destination, "--output", output], work, commands)
        build = json.loads((output / "build.json").read_text())
        _require(build.get("version") == "amd_stream_selected_build_v1" and build.get("complete") is True, "Incomplete AMD streaming build")
        files = {}
        for role in LIBRARIES:
            record = build["libraries"][role]; path = Path(record["path"]).resolve()
            _require(path.is_relative_to(output.resolve()) and sha256(path) == record["sha256"], "AMD build artifact mismatch")
            files[role] = path
    extra = []
    for role, basename in LIBRARIES.items():
        path = files[role]
        _require(path.name == basename, "AMD streaming library filename does not match role")
        extra.append(x86._copy_library(path, destination / "libs"))
    graph, audit = rewrite_graph(original)
    target = source_path.with_name("decoder_amd_selected.onnx")
    _require(not target.exists(), "Selected AMD graph already exists")
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    try:
        onnx.save_model(graph, temporary)
        _require(sha256(temporary) == SELECTED_STREAM, "AMD selected graph differs from accepted artifact")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _require(sha256(full_path) == SOURCE_FULL and sha256(source_path) == SOURCE_STREAM and manifest_path.read_bytes() == before,
             "Canonical bundle changed during preparation")
    _require(stream["states"] == state_before, "State metadata changed")
    stream["model"] = str(target.relative_to(destination)); stream["model_sha256"] = SELECTED_STREAM
    stream["additional_libraries"] = [*native["additional_libraries"], *extra]
    stream["required_streaming_operators"] = [*stream.get("required_streaming_operators", []),
        {"domain": PAIR_DOMAIN, "operator": "PackedProjectionPairF32"},
        {"domain": HISTORY_DOMAIN, "operator": "RawHistorySnakeDW7SnakeF32"},
        {"domain": PHASE_DOMAIN, "operator": "StatefulPhaseFinishF32"}]
    stream["amd_stream_selected"] = {"version": VERSION, "source_full_sha256": SOURCE_FULL,
        "source_stream_sha256": SOURCE_STREAM, "model_sha256": SELECTED_STREAM,
        "onnxruntime": onnxruntime_version, "precision": "selective_int8", "threads": 1,
        "direct_pair_frames": [1, 2, 3, 4], "larger_pair_frames": "unchanged AOCL INT8 fallback",
        "state_contract": "unchanged raw pre-Snake and dequantized previous-projection histories",
        "state_bytes": 683264, "libraries": extra, **audit}
    manifest["onnxruntime"] = onnxruntime_version
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
