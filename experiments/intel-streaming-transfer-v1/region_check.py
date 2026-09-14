"""Bounded CPU region parity checks; no audio/latents and no throughput claim."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP
import onnxruntime as ort


HISTORY = ("ncc_block_chain_18", "ncc_block_chain_26", "ncc_block_chain_34",
           "ncc_block_chain_53", "ncc_block_chain_61", "ncc_block_chain_69")
LENGTHS = (1, 8, 16, 17, 96, 257)


def attributes(node):
    return {a.name: H.get_attribute_value(a) for a in node.attribute}


def tensor(name, shape):
    return H.make_tensor_value_info(name, TP.FLOAT, shape)


def region(source, nodes, inputs, outputs, base_dir):
    wanted = {x for n in nodes for x in n.input}
    weights = []
    for item in source.graph.initializer:
        if item.name not in wanted:
            continue
        weight = copy.deepcopy(item)
        if weight.data_location == TP.EXTERNAL:
            onnx.external_data_helper.load_external_data_for_tensor(weight, str(base_dir))
            weight.ClearField("external_data"); weight.ClearField("data_location")
        weights.append(weight)
    model = H.make_model(H.make_graph(nodes, "region", inputs, outputs, weights),
                         opset_imports=source.opset_import, ir_version=source.ir_version)
    onnx.checker.check_model(model, check_custom_domain=False)
    return model.SerializeToString()


def specialize(blob, length):
    """Give the extracted standalone graph complete concrete shape metadata.

    The production graph carries intermediate shape context. A tiny extraction
    loses it; restore exact dimensions before ORT's Slice shape inference.
    """
    model = onnx.load_model_from_string(blob)
    known = {}
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        for dim in dims:
            if dim.HasField("dim_param"): dim.dim_value = length
        known[value.name] = [dim.dim_value for dim in dims]
    constants = {w.name: w for w in model.graph.initializer}
    known.update({name: list(weight.dims) for name, weight in constants.items()})
    for node in model.graph.node:
        attrs = attributes(node)
        if node.op_type == "Concat":
            shape = known[node.input[0]].copy(); axis = attrs["axis"]
            shape[axis] = sum(known[name][axis] for name in node.input)
            inferred = [shape]
        elif node.op_type == "Slice":
            shape = known[node.input[0]].copy()
            starts, ends, axes, steps = [onnx.numpy_helper.to_array(constants[name]).tolist() for name in node.input[1:]]
            for start, end, axis, step in zip(starts, ends, axes, steps):
                shape[axis] = len(range(*slice(start, end, step).indices(shape[axis])))
            inferred = [shape]
        elif node.op_type == "SnakeDW7SnakeF32": inferred = [known[node.input[0]].copy()]
        elif node.op_type == "RawHistorySnakeDW7SnakeF32":
            inferred = [known[node.input[0]].copy(), known[node.input[1]].copy()]
        elif node.op_type == "PhaseSumBiasInterleaveF32":
            inferred = [[1, attrs["channels"], known[node.input[0]][2]*attrs["stride"]]]
        elif node.op_type == "StatefulPhaseFinishF32":
            inferred = [[1, attrs["channels"], known[node.input[0]][2]*attrs["stride"]], known[node.input[2]].copy()]
        else: raise ValueError("Unexpected region operation: " + node.op_type)
        known.update(zip(node.output, inferred))
    outputs = {value.name for value in model.graph.output}
    for value in model.graph.output: value.CopyFrom(tensor(value.name, known[value.name]))
    del model.graph.value_info[:]
    for node in model.graph.node:
        model.graph.value_info.extend(tensor(name, known[name]) for name in node.output if name not in outputs)
    onnx.checker.check_model(model, check_custom_domain=False)
    return model.SerializeToString()


def session(model, library, length):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.register_custom_ops_library(str(library))
    result = ort.InferenceSession(specialize(model, length), sess_options=options, providers=["CPUExecutionProvider"])
    if result.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("CPU-only session required")
    return result


def shapes(model, length):
    result = {}
    for item in model.get_inputs():
        result[item.name] = tuple(length if isinstance(d, str) or d is None else d for d in item.shape)
    return result


def digest_inputs(values):
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in values.items()}


def check_pair(name, load_sessions, rng, result, save):
    for length in LENGTHS:
        reference, candidate = load_sessions(length)
        for pattern, amplitude in (("zero", 0.0), ("tiny", 1e-7), ("random", 0.25)):
            values = {key: (rng.standard_normal(shape).astype(np.float32) * np.float32(amplitude))
                      for key, shape in shapes(reference, length).items()}
            before = digest_inputs(values)
            result["last_call"] = {"region": name, "length": length, "pattern": pattern,
                                   "phase": "reference_run", "status": "pending"}
            save()
            start = time.perf_counter(); expected = reference.run(None, values)
            result["native_seconds"] += time.perf_counter() - start
            result["last_call"]["phase"] = "candidate_run"
            save()
            start = time.perf_counter(); actual = candidate.run(None, values)
            result["native_seconds"] += time.perf_counter() - start
            result["calls"] += 2
            result["input_checks"] += 1
            if digest_inputs(values) != before:
                result["failures"].append({"region": name, "length": length, "pattern": pattern, "kind": "input_mutation"})
            for index, (a, b) in enumerate(zip(expected, actual)):
                equal = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                equal = equal and np.array_equal(a.view(np.uint32), b.view(np.uint32))
                error = float(np.max(np.abs(a-b))) if a.size and a.shape == b.shape else 0.0
                result["maximum_absolute_error"] = max(result["maximum_absolute_error"], error)
                result["output_state_checks"] += 1
                result["output_state_checks_passed"] += int(equal)
                if not equal:
                    result["failures"].append({"region": name, "length": length, "pattern": pattern,
                                               "output_index": index, "max_abs": error})
            result["last_call"]["status"] = "complete"
            save()


def run(graphs, base_library, history_library, phase_library, output):
    graphs, output = Path(graphs).resolve(), Path(output).resolve()
    source = onnx.load(graphs / "baseline.onnx", load_external_data=False)
    history = onnx.load(graphs / "history.onnx", load_external_data=False)
    phase = onnx.load(graphs / "phase.onnx", load_external_data=False)
    nodes = {n.name: n for n in source.graph.node}
    producers = {v: n for n in source.graph.node for v in n.output}
    users = {}
    for n in source.graph.node:
        for value in n.input: users.setdefault(value, []).append(n)
    result = {"version": "intel_streaming_region_parity_v1", "complete": False, "cpu_only": True, "threads": 1,
              "runtime": ort.__version__, "lengths": LENGTHS, "patterns": ["zero", "tiny", "random"],
              "calls": 0, "native_seconds": 0.0, "input_checks": 0, "output_state_checks": 0,
              "output_state_checks_passed": 0, "maximum_absolute_error": 0.0, "failures": [],
              "scope": "Actual coefficients, deterministic synthetic region inputs; not audio quality or RTF",
              "zero_length_scope": "T=0 intentionally excluded: public streaming empty calls bypass ONNX; this region check does not establish public empty-call behavior",
              "prior_attempts": ["Initial dynamic-shape harness aborted in ORT std::clamp before reporting.",
                                 "Removing T=0 did not fix it: the second receipt locates the abort at reference_session_load for ncc_block_chain_18, before any inference. This disproves the zero-input explanation. The standalone extraction now supplies concrete input, output and intermediate shape metadata for each tested T; original kernels are unchanged."],
              "last_call": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(result, indent=2)+"\n")
    def load_pair(name, ref_model, new_model, library, length):
        result["last_call"] = {"region": name, "length": length, "phase": "reference_session_load", "status": "pending"}; save()
        ref = session(ref_model, base_library, length)
        result["last_call"]["phase"] = "candidate_session_load"; save()
        cand = session(new_model, library, length)
        return ref, cand
    save()
    rng = np.random.default_rng(20260914)
    for name in HISTORY:
        op = nodes[name]; a = attributes(op); c, halo = a["channels"], 6*a["dilation"]
        concat = producers[op.input[0]]
        retained = next(n for n in users[concat.output[0]] if n.name != name)
        crop = users[op.output[0]][0]
        history_name, current_name = concat.input
        ins = [tensor(current_name, [1,c,"T"]), tensor(history_name, [1,c,halo])]
        outs = [tensor(crop.output[0], [1,c,"T"]), tensor(retained.output[0], [1,c,halo])]
        selected = {n.name for n in (concat, retained, op, crop)}
        ref_model = region(source, [n for n in source.graph.node if n.name in selected], ins, outs, graphs)
        new = next(n for n in history.graph.node if n.name == name)
        new_model = region(history, [new], ins, outs, graphs)
        try:
            check_pair(name, lambda length: load_pair(name, ref_model, new_model, history_library, length), rng, result, save)
        except BaseException as error:
            result["exception"] = repr(error); save(); raise
    ordered = list(source.graph.node)
    index = next(i for i,n in enumerate(ordered) if n.op_type == "PhaseSumBiasInterleaveF32"
                 and attributes(n).get("channels") == 1024)
    previous, retained, current, op, crop = ordered[index-3:index+2]
    ins = [tensor(current.input[1], [1,8192,"T"]), tensor(previous.input[1], [1,8192,"T"]),
           tensor(previous.input[0], [1,8192,1])]
    outs = [tensor(crop.output[0], [1,1024,"TOUT"]), tensor(retained.output[0], [1,8192,1])]
    ref_model = region(source, ordered[index-3:index+2], ins, outs, graphs)
    new = next(n for n in phase.graph.node if n.op_type == "StatefulPhaseFinishF32")
    new_model = region(phase, [new], ins, outs, graphs)
    try:
        check_pair("first_phase", lambda length: load_pair("first_phase", ref_model, new_model, phase_library, length), rng, result, save)
    except BaseException as error:
        result["exception"] = repr(error); save(); raise
    result["complete"] = True
    result["passed"] = not result["failures"]
    save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graphs", "base-library", "history-library", "phase-library", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
