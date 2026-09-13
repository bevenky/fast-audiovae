"""Prepare the selected Apple streaming graph without changing trained values.

The 5918 recipe retains direct depthwise histories, six ordered phase/state
operators and only the 18 selected matrix nodes. No later experiment is applied.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper
from ..assets import sha256 as sha

DOMAIN = "fast.audiovae.apple.state.v2"
BASE_DOMAIN = "venky.audio.cpu"
SELECTED_GRAPH = "5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}

def _rewrite_state(model, states, *, direct_state, packed_snake, scratch_elements=4096):
    if not (direct_state or packed_snake) or scratch_elements not in (256, 512, 1024, 2048, 4096):
        raise ValueError("Choose at least one candidate and a bounded scratch tile")
    if any(n.domain == DOMAIN for n in model.graph.node):
        raise ValueError("State candidate already present; derive from the immutable baseline or matrix candidate")
    result = copy.deepcopy(model)
    nodes = list(result.graph.node)
    producers = {o: n for n in nodes for o in n.output}
    consumers = {}
    for n in nodes:
        for i in n.input: consumers.setdefault(i, []).append(n)
    constants = {t.name: numpy_helper.to_array(t) for t in result.graph.initializer}
    for n in nodes:
        if not n.domain and n.op_type == "Constant":
            a = attrs(n)
            if "value" in a: constants[n.output[0]] = numpy_helper.to_array(a["value"])
    def slice_is(node, source, start, end):
        if node.domain or node.op_type != "Slice" or node.input[0] != source or len(node.input) != 5:
            return False
        return [constants[name].reshape(-1).tolist() for name in node.input[1:]] == [[start], [end], [2], [1]]
    replacements, removed = {}, set()
    counts = {"direct_dw": 0, "direct_dw_snake": 0, "packed_standalone_snake": 0, "packed_dw_snake": 0}
    state_by_node = {s["node"]: s for s in states if s["kind"] == "native_depthwise"}
    for n in nodes:
        if n.domain != BASE_DOMAIN: continue
        a = attrs(n)
        if n.op_type in ("CausalDW7F32", "CausalDW7SnakeF32"):
            fused = n.op_type == "CausalDW7SnakeF32"
            if not direct_state and not (packed_snake and fused): continue
            if a.get("native_abi") != 1 or a.get("backend") not in (0, 2) or a.get("row_batches") != 0 or a.get("dilation") not in (1, 3, 9):
                raise ValueError("Unexpected original DW ABI")
            if fused and a.get("require_vforce") != 1: raise ValueError("Original vForce policy required")
            new_attrs = {"candidate_abi": 1, "channels": a["channels"], "dilation": a["dilation"]}
            if direct_state:
                s = state_by_node[n.name]
                halo = 6 * a["dilation"]
                if s["shape"] != [1, a["channels"], halo] or s.get("time_axis", 2) != 2:
                    raise ValueError("Activated history shape mismatch")
                joined = producers[n.input[0]]
                retain = producers[s["output"]]
                outputs = consumers.get(n.output[0], [])
                if len(outputs) != 1: raise ValueError("DW output has another use")
                crop = outputs[0]
                if joined.domain or joined.op_type != "Concat" or list(joined.input[:1]) != [s["input"]] or len(joined.input) != 2 or attrs(joined) != {"axis": 2}:
                    raise ValueError("Unexpected history concatenation")
                if not slice_is(retain, joined.output[0], -halo, np.iinfo(np.int64).max) or not slice_is(crop, n.output[0], halo, np.iinfo(np.int64).max):
                    raise ValueError("Unexpected retention/crop pattern")
                if {x.name for x in consumers[joined.output[0]]} != {n.name, retain.name}:
                    raise ValueError("History concatenation has an unaudited consumer")
                op = "StatefulDW7SnakeF32" if fused else "StatefulDW7F32"
                if fused: new_attrs["packed_snake"] = int(packed_snake)
                if fused and packed_snake: new_attrs["scratch_elements"] = scratch_elements
                replacement = helper.make_node(op, [joined.input[1], s["input"], *n.input[1:]],
                    [crop.output[0], s["output"]], name=n.name + "_direct_state_v2", domain=DOMAIN, **new_attrs)
                removed.update((joined.name, retain.name, crop.name))
                counts["direct_dw_snake" if fused else "direct_dw"] += 1
            else:
                new_attrs["scratch_elements"] = scratch_elements
                replacement = helper.make_node("PackedDW7SnakeF32", list(n.input), list(n.output),
                    name=n.name + "_packed_v2", domain=DOMAIN, **new_attrs)
                counts["packed_dw_snake"] += 1
            replacements[n.name] = replacement
        elif n.op_type == "SnakeF32" and packed_snake:
            if a.get("native_abi") != 1 or a.get("backend") not in (0, 2) or a.get("require_vforce") != 1 or a.get("row_batches") != 0:
                raise ValueError("Unexpected original Snake ABI")
            replacements[n.name] = helper.make_node("PackedSnakeF32", list(n.input), list(n.output),
                name=n.name + "_packed_v2", domain=DOMAIN, candidate_abi=1,
                channels=a["channels"], scratch_elements=scratch_elements)
            counts["packed_standalone_snake"] += 1
    if direct_state and (counts["direct_dw"], counts["direct_dw_snake"]) != (1, 18):
        raise ValueError("Expected one initial DW and eighteen residual DW+Snake operations")
    if packed_snake and counts["packed_standalone_snake"] != 25:
        raise ValueError("Expected twenty-five standalone Snake operations")
    if packed_snake and not direct_state and counts["packed_dw_snake"] != 18:
        raise ValueError("Expected eighteen fused packed-Snake operations")
    del result.graph.node[:]
    result.graph.node.extend(replacements.get(n.name, n) for n in nodes if n.name not in removed)
    live = {x for n in result.graph.node for x in (*n.input, *n.output)}
    kept = [v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:]; result.graph.value_info.extend(kept)
    result.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    if [x.SerializeToString() for x in result.graph.initializer] != [x.SerializeToString() for x in model.graph.initializer]:
        raise ValueError("Initializer bytes changed")
    onnx.checker.check_model(result)
    return result, {"counts": counts, "removed_nodes": len(removed), "direct_state": direct_state,
        "packed_snake": packed_snake, "scratch_elements": scratch_elements,
        "initializers_unchanged": True, "external_state_schema_unchanged": True,
        "history_semantics": "Each DW input, after any preceding Snake; oldest to newest."}


# Explicit allowlist. Matrix scratch assumes 80ms; larger packets retain the
# operators' checked Accelerate fallback. ORT's thread count remains independent.
MATRICES = {
    "node_conv1d_1__bct_mm": ("multinext", 64, 2048, 2),
    "ncc_up_1_split_matmul_current_node_fixed_right": ("libxsmm.panel", 2048, 8192, 2),
    "ncc_up_1_split_matmul_previous_unshifted_node_fixed_right": ("multitile", 2048, 8192, 2),
    "node_conv1d_3__bct_mm": ("matrix.sweep", 1024, 1024, 16),
    "node_conv1d_5__bct_mm": ("matrix.sweep", 1024, 1024, 16),
    "node_conv1d_7__bct_mm": ("multinext", 1024, 1024, 16),
    "ncc_up_2_split_matmul_current_node": ("matrix.sweep", 1024, 3072, 16),
    "ncc_up_2_split_matmul_previous_unshifted_node": ("matrix.sweep", 1024, 3072, 16),
    **{name: ("layout", k, k, m) for k, m, numbers, stage in (
        (256, 480, (15, 17, 19), 4), (128, 960, (21, 23, 25), 5))
        for name in [*(f"node_conv1d_{n}__bct_mm" for n in numbers),
                     f"ncc_up_{stage}_split_matmul_current_node",
                     f"ncc_up_{stage}_split_matmul_previous_unshifted_node"]},
}
OPERATORS = {"matrix.sweep": ("fast.audiovae.apple.matrix.sweep.v1", "Sme2WeightLeftF32"),
    "layout": ("fast.audiovae.apple.layout.v3", "WeightLeftSgemmF32"),
    "multitile": ("fast.audiovae.apple.multitile.v1", "MultitileWeightLeftF32"),
    "multinext": ("fast.audiovae.apple.multinext.v1", "MultinextWeightLeftF32"),
    "libxsmm.panel": ("fast.audiovae.apple.libxsmm.panel.v1", "LibxsmmPanelSmeWeightLeftF32")}


def _rewrite_matrices(model):
    result = copy.deepcopy(model)
    nodes = {node.name: node for node in result.graph.node}
    weights = {weight.name: weight for weight in result.graph.initializer}
    removed_nodes, removed_weights, new_weights = set(), set(), []
    require(set(MATRICES) <= set(nodes), "Selected Apple matrix nodes missing")
    for name, (family, k, n, maximum) in MATRICES.items():
        node = nodes[name]
        require(not node.domain and node.op_type == "MatMul" and len(node.input) == 2 and len(node.output) == 1,
                "Selected matrix source pattern changed: " + name)
        x, y, weight_name = node.input[1], node.output[0], node.input[0]
        if name.endswith("_fixed_right"):
            weight_name = node.input[1]
            original = numpy_helper.to_array(weights[weight_name])
            require(original.shape == (k, n) and original.dtype == np.float32, "Fixed-right weight changed")
            tin = nodes["stream_up1_transpose_input"]
            tout = nodes[name.removesuffix("_fixed_right") + "_restore_layout"]
            for transpose in (tin, tout):
                require(not transpose.domain and transpose.op_type == "Transpose"
                        and attrs(transpose) == {"perm": [0, 2, 1]}, "First-pair layout changed")
            require(tin.output[0] == node.input[0] and tout.input[0] == node.output[0], "First-pair endpoints changed")
            require(sum(node.output[0] in row.input for row in result.graph.node) == 1, "Unexpected matrix consumer")
            x, y = tin.input[0], tout.output[0]
            removed_nodes.update((tin.name, tout.name)); removed_weights.add(weight_name)
            left_name = weight_name + "__weight_left"
            require(left_name not in weights, "Selected weight name collision")
            new_weights.append(numpy_helper.from_array(np.ascontiguousarray(original.T), left_name))
            weight_name = left_name
        else:
            weight = weights[weight_name]
            require(weight.data_type == onnx.TensorProto.FLOAT and list(weight.dims) == [n, k], "Matrix [N,K] shape changed")
        domain, operator = OPERATORS[family]
        attributes = dict(matrix_abi=1, threads=1, k=k, n=n)
        if family != "layout":
            attributes["max_m"] = maximum
        node.CopyFrom(helper.make_node(operator, [x, weight_name], [y], name=name, domain=domain, **attributes))
    kept = [node for node in result.graph.node if node.name not in removed_nodes]
    require(not any(x == "stream_up1_shared_input_transposed" for node in kept for x in node.input), "Shared transpose still used")
    del result.graph.node[:]; result.graph.node.extend(kept)
    kept_weights = [w for w in result.graph.initializer if w.name not in removed_weights]
    del result.graph.initializer[:]; result.graph.initializer.extend(kept_weights + new_weights)
    live = {name for node in result.graph.node for name in (*node.input, *node.output)}
    vi = [v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:]; result.graph.value_info.extend(vi)
    # Preserve the selected composition's domain order, including unused old
    # sweep domains needed by the remaining selected nodes.
    for family in ("matrix.sweep", "layout", "multitile", "multinext"):
        result.opset_import.append(helper.make_opsetid(OPERATORS[family][0], 1))
    return result


def _rewrite_phases(model, states):
    result = copy.deepcopy(model)
    nodes = list(result.graph.node)
    constants = {t.name: t for t in result.graph.initializer}
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node.name)
    removed, replacements = set(), {}
    domain = "fast.audiovae.apple.phase.state.v1"
    for index, node in enumerate(nodes):
        if node.op_type != "PhaseSumBiasInterleaveF32":
            continue
        require(index >= 3 and index + 1 < len(nodes), "Incomplete phase region")
        previous, retain, current, phase, crop = nodes[index-3:index+2]
        a = attrs(phase); channels, stride = a["channels"], a["stride"]
        require(a == dict(channels=channels, native_abi=1, previous_shift=1, row_batches=0, stride=stride), "Phase ABI changed")
        require([n.op_type for n in (previous, retain, current, phase, crop)] ==
                ["Concat", "Slice", "Concat", "PhaseSumBiasInterleaveF32", "Slice"], "Phase pattern changed")
        state = next(s for s in states if s["input"] == previous.input[0])
        require(state["shape"] == [1, channels*stride, 1] and state["output"] == retain.output[0], "Phase state changed")
        require(list(phase.input[:2]) == [current.output[0], previous.output[0]]
                and crop.input[0] == phase.output[0] and retain.input[0] == previous.output[0], "Phase endpoints changed")
        require(sorted(consumers[previous.output[0]]) == sorted([retain.name, phase.name])
                and consumers[current.output[0]] == [phase.name] and consumers[phase.output[0]] == [crop.name], "Extra phase consumer")
        require(attrs(previous) == attrs(current) == {"axis": 2}, "Phase concatenation changed")
        require(np.array_equal(numpy_helper.to_array(constants[current.input[0]]), np.zeros((1, channels*stride, 1), np.float32)), "Phase padding changed")
        for sliced, start in ((crop, stride), (retain, -1)):
            require([numpy_helper.to_array(constants[x]).tolist() for x in sliced.input[1:]] ==
                    [[start], [9223372036854775807], [2], [1]], "Phase crop changed")
        bias = constants[phase.input[2]]
        require(list(bias.dims) == [channels] and bias.data_type == onnx.TensorProto.FLOAT, "Phase bias changed")
        removed.update(n.name for n in (previous, retain, current, phase, crop))
        replacements[crop.name] = helper.make_node("StatefulPhaseFinishF32",
            [current.input[1], previous.input[1], previous.input[0], phase.input[2]],
            [crop.output[0], retain.output[0]], name=phase.name + "_direct", domain=domain,
            channels=channels, stride=stride, phase_state_abi=1, threads=1)
    require(len(replacements) == 6, "Expected six phase regions")
    kept = [replacements[n.name] if n.name in replacements else n for n in nodes
            if n.name in replacements or n.name not in removed]
    del result.graph.node[:]; result.graph.node.extend(kept)
    result.opset_import.append(helper.make_opsetid(domain, 1))
    result.opset_import.append(helper.make_opsetid(OPERATORS["libxsmm.panel"][0], 1))
    return result


def rewrite(model, states):
    """Apply only the selected topology; return the graph and a structural audit."""
    original_inputs = [v.SerializeToString() for v in model.graph.input]
    original_outputs = [v.SerializeToString() for v in model.graph.output]
    result, state_audit = _rewrite_state(model, states, direct_state=True, packed_snake=False)
    result = _rewrite_phases(_rewrite_matrices(result), states)
    require(original_inputs == [v.SerializeToString() for v in result.graph.input]
            and original_outputs == [v.SerializeToString() for v in result.graph.output], "External stream interface changed")
    onnx.checker.check_model(result)
    return result, {"state": state_audit, "matrix_nodes": sorted(MATRICES), "phase_regions": 6,
                    "selected_reference_graph_sha256": SELECTED_GRAPH, "trained_values_preserved": True}


def prepare(source, destination, build_manifest):
    """Copy a prepared Apple projection bundle and bind its selected native closure."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists() and not destination.is_relative_to(source), "New bundle destination required")
    build = json.loads(Path(build_manifest).read_text())
    require(build.get("version") == "apple_stream_selected_build_v1" and build.get("complete") is True, "Incomplete Apple streaming build")
    manifest = json.loads((source / "bundle.json").read_text())
    native = manifest["native"]["Darwin/arm64"]
    spec = manifest["streaming"]["models"][native["model"]]
    stream = source / spec["model"]
    require(sha(stream) == spec["model_sha256"] and sha(source / native["model"]) == spec["source_sha256"], "Source bundle identity changed")
    rewritten, audit = rewrite(onnx.load(stream), spec["states"])
    records = build["runtime_files"]
    require(len(records) == 12 and len({Path(r["path"]).name for r in records}) == 12, "Expected unique selected closure files")
    for record in records:
        require(sha(record["path"]) == record["sha256"], "Native closure file changed")
    registered = {r["path"]: r for r in records if r["register"]}
    require(len(build["additional_libraries"]) == len(registered) == 11, "Native registration count changed")
    for record in build["additional_libraries"]:
        require(record["library"] in registered and record["sha256"] == registered[record["library"]]["sha256"], "Native registration identity mismatch")
    shutil.copytree(source, destination)
    onnx.save(rewritten, destination / spec["model"])
    for record in records:
        target = destination / "runtime" / Path(record["path"]).name
        require(not target.exists(), "Native filename collision")
        shutil.copy2(record["path"], target)
        require(sha(target) == record["sha256"], "Native copy changed")
    spec["additional_libraries"] = [{"library": "runtime/" + Path(r["library"]).name,
        "sha256": r["sha256"], "domain": r["domain"]} for r in build["additional_libraries"]]
    # Transitive files must be authenticated before the bridge is loaded too.
    spec["dependencies"] = [{"library": "runtime/" + Path(r["path"]).name, "sha256": r["sha256"]}
                            for r in records if not r["register"]]
    native["dependencies"] = copy.deepcopy(spec["dependencies"])
    spec["model_sha256"] = sha(destination / spec["model"])
    spec["apple_stream_selected"] = audit
    manifest["onnxruntime"] = "1.30.0"
    native["required_cpu_features"] = ["sme", "sme2"]
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / ".recipe-ready.json").unlink(missing_ok=True)
    require(sha(destination / native["model"]) == spec["source_sha256"], "Full decoder changed")
    return audit
