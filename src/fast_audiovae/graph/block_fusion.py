"""Opt-in, checked native residual fusions; never changes runtime defaults.

Preserve FP32 operation order and every existing initializer protobuf. Small
channel biases receive a rank-one alias initializer with identical data bytes.
External tensor files are copied without repacking. This performs no inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from .elementwise import Graph, copy_external_data, sha256

DOMAINS = ("venky.audio.cpu", "venky.audio.cpu.portable")
MODES = ("chain", "adds", "both", "none")


def _attrs(node):
    if len({a.name for a in node.attribute}) != len(node.attribute):
        raise ValueError("Duplicate attributes")
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def rewrite_model(model, base_dir=".", *, mode="both", variant=None, backend=None,
                  expected_chains=None, expected_adds=None):
    """Return a derivative and audit; mutate neither source graph nor weights.

    backend=None preserves policy. Explicit portable overrides are 0/4/5 and
    apply to every existing custom operator having a backend attribute, allowing
    whole-candidate ISA comparisons. Apple overrides are 0/2.
    """
    if variant is not None:
        mode = variant
    if mode not in MODES:
        raise ValueError("variant must be chain, adds, both, or none")
    if model.graph.sparse_initializer or model.functions or any(
        a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for n in model.graph.node for a in n.attribute
    ):
        raise ValueError("Only flat dense graphs without local functions are supported")
    domains = {n.domain for n in model.graph.node if n.domain in DOMAINS}
    if len(domains) != 1:
        raise ValueError("Expected exactly one supported native operator domain")
    domain = next(iter(domains))
    if backend is not None and (type(backend) is not int or backend not in ((0, 2) if domain == DOMAINS[0] else (0, 4, 5))):
        raise ValueError("Backend overrides: Apple 0/2, portable 0/4/5")
    versions = [v.version for v in model.opset_import if v.domain == domain]
    if versions != [1]:
        raise ValueError("Expected native domain opset 1")
    graph = Graph(model, base_dir)
    result = copy.deepcopy(model)
    replacements, removed, records, skipped = {}, set(), [], []
    occupied = {x for n in graph.nodes for x in (*n.input, *n.output)} | set(graph.initializers)
    occupied.update(n.name for n in graph.nodes)
    sine_policy = "require_vforce" if domain == DOMAINS[0] else "require_vector_sine"

    shape_cache, value_cache, value_shapes = {}, {}, {}

    def dim_mul(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        if isinstance(a, int) and isinstance(b, tuple):
            return (a * b[0], b[1])
        if isinstance(b, int) and isinstance(a, tuple):
            return (b * a[0], a[1])
        raise ValueError("Unsupported nonlinear symbolic shape")

    def shape_values(name):
        if name in value_cache:
            return value_cache[name]
        a = graph.constant(name)
        if a is not None:
            if a.dtype != np.int64:
                raise ValueError("Shape expressions must use INT64")
            out = tuple(int(v) for v in a.reshape(-1))
            out_shape = a.shape
        else:
            i = graph.producer.get(name)
            if i is None:
                raise ValueError("Unproven shape expression")
            n = graph.nodes[i]; attrs = _attrs(n)
            if n.domain:
                raise ValueError("Unsupported shape expression domain")
            if n.op_type == "Shape" and len(n.input) == 1 and not set(attrs) - {"start", "end"}:
                ss = actual_shape(n.input[0])
                out = ss[attrs.get("start", 0):attrs.get("end", len(ss))]
                out_shape = (len(out),)
            elif n.op_type in ("Reshape", "Squeeze", "Unsqueeze"):
                if set(attrs) - ({"allowzero"} if n.op_type == "Reshape" else set()):
                    raise ValueError("Unsupported shape-view attributes")
                if not n.input or len(n.output) != 1:
                    raise ValueError("Malformed shape view")
                out = shape_values(n.input[0]); before = value_shapes[n.input[0]]
                if n.op_type == "Reshape":
                    if len(n.input) != 2 or attrs.get("allowzero", 0) not in (0, 1):
                        raise ValueError("Malformed shape Reshape")
                    target = list(shape_values(n.input[1]))
                    if value_shapes[n.input[1]] != (len(target),) or any(type(v) is not int or v < -1 for v in target):
                        raise ValueError("Shape-view target must be a constant integer vector")
                    if target.count(-1) > 1:
                        raise ValueError("Ambiguous shape-view inference")
                    if attrs.get("allowzero", 0) == 1 and 0 in target and -1 in target:
                        raise ValueError("Zero and inferred shape-view dimensions cannot be combined")
                    for j, v in enumerate(target):
                        if v == 0 and attrs.get("allowzero", 0) == 0:
                            if j >= len(before):
                                raise ValueError("Invalid shape-view copied dimension")
                            target[j] = before[j]
                    known = 1
                    for v in target:
                        if v != -1:
                            known *= v
                    if -1 in target:
                        if known <= 0 or len(out) % known:
                            raise ValueError("Invalid shape-view volume")
                        target[target.index(-1)] = len(out)//known
                    elif known != len(out):
                        raise ValueError("Shape-view volume changes")
                    out_shape = tuple(target)
                else:
                    if len(n.input) not in (1, 2) or (n.op_type == "Unsqueeze" and len(n.input) != 2):
                        raise ValueError("Malformed shape squeeze/unsqueeze")
                    if len(n.input) == 2:
                        axes = list(shape_values(n.input[1]))
                        if value_shapes[n.input[1]] != (len(axes),) or any(type(v) is not int for v in axes):
                            raise ValueError("Shape-view axes must be an integer vector")
                    else:
                        axes = [j for j, v in enumerate(before) if v == 1]
                    rank = len(before) + (len(axes) if n.op_type == "Unsqueeze" else 0)
                    if any(v < -rank or v >= rank for v in axes):
                        raise ValueError("Shape-view axis out of range")
                    axes = [v+rank if v < 0 else v for v in axes]
                    if len(set(axes)) != len(axes):
                        raise ValueError("Repeated shape-view axis")
                    if n.op_type == "Squeeze":
                        if any(before[j] != 1 for j in axes):
                            raise ValueError("Cannot squeeze a nonunit shape dimension")
                        out_shape = tuple(v for j, v in enumerate(before) if j not in axes)
                    else:
                        old = iter(before)
                        out_shape = tuple(1 if j in axes else next(old) for j in range(rank))
            elif n.op_type == "Mul" and len(n.input) == 2 and not attrs:
                aa, bb = [shape_values(v) for v in n.input]
                if len(aa) != 1 or len(bb) != 1:
                    raise ValueError("Only scalar shape multiplication is supported")
                out = (dim_mul(aa[0], bb[0]),)
                ash, bsh = [value_shapes[v] for v in n.input]
                if ash not in ((), (1,)) or bsh not in ((), (1,)):
                    raise ValueError("Unsupported scalar shape broadcasting")
                out_shape = (1,) if ash or bsh else ()
            elif n.op_type == "Concat" and attrs == {"axis": 0}:
                out = tuple(v for name in n.input for v in shape_values(name))
                if any(len(value_shapes[v]) != 1 for v in n.input):
                    raise ValueError("Shape Concat requires vectors")
                out_shape = (len(out),)
            else:
                raise ValueError("Unsupported shape-expression operation")
        value_cache[name] = out
        value_shapes[name] = out_shape
        return out

    def actual_shape(name):
        if name in shape_cache:
            return shape_cache[name]
        if name in graph.inputs:
            declared = graph.shape(name)
            if declared is None or any(v is None for v in declared):
                raise ValueError("Undeclared graph input shape")
            out = tuple(v if isinstance(v, int) else (1, (name, i)) for i, v in enumerate(declared))
        elif name in graph.initializers:
            if graph.initializers[name].data_type != TensorProto.FLOAT:
                raise ValueError("Activation-side initializers must be FP32")
            out = tuple(graph.initializers[name].dims)
        else:
            i = graph.producer.get(name)
            if i is None:
                raise ValueError("Activation has no supported shape producer")
            n = graph.nodes[i]; attrs = _attrs(n)
            if n.domain == domain and n.op_type in (
                "SnakeF32", "CausalDW7F32", "CausalDW7SnakeF32", "PhaseSumBiasInterleaveF32"
            ):
                out = actual_shape(n.input[0])
                if len(out) != 3:
                    raise ValueError("Native input is not rank three")
                if n.op_type == "PhaseSumBiasInterleaveF32":
                    stride = attrs.get("stride")
                    if stride not in (2, 5, 6, 8) or out[1] != attrs.get("channels", 0) * stride:
                        raise ValueError("Invalid phase shape")
                    out = (out[0], attrs["channels"], dim_mul(out[2], stride))
                elif out[1] != attrs.get("channels"):
                    raise ValueError("Invalid native channel shape")
            elif not n.domain and n.op_type == "Reshape" and len(n.input) == 2:
                if set(attrs) - {"allowzero"} or attrs.get("allowzero", 0) not in (0, 1):
                    raise ValueError("Unsupported Reshape attributes")
                before = actual_shape(n.input[0]); target = list(shape_values(n.input[1]))
                if value_shapes[n.input[1]] != (len(target),):
                    raise ValueError("Reshape target must be a rank-one shape vector")
                if target.count(-1) > 1:
                    raise ValueError("Multiple inferred Reshape dimensions")
                if attrs.get("allowzero", 0) == 1 and 0 in target and -1 in target:
                    raise ValueError("Zero and inferred Reshape dimensions cannot be combined")
                for j, v in enumerate(target):
                    if v == 0 and attrs.get("allowzero", 0) == 0:
                        if j >= len(before):
                            raise ValueError("Invalid copied Reshape dimension")
                        target[j] = before[j]
                    elif isinstance(v, int) and v < -1:
                        raise ValueError("Invalid Reshape dimension")
                total = 1
                for v in before:
                    total = dim_mul(total, v)
                known = 1
                for v in target:
                    if v != -1:
                        known = dim_mul(known, v)
                if -1 in target:
                    if not isinstance(known, int) or known <= 0:
                        raise ValueError("Ambiguous inferred Reshape dimension")
                    coefficient = total if isinstance(total, int) else total[0]
                    if coefficient % known:
                        raise ValueError("Nonintegral inferred shape")
                    target[target.index(-1)] = coefficient // known if isinstance(total, int) else (coefficient // known, total[1])
                elif known != total:
                    raise ValueError("Reshape volume is not proven unchanged")
                out = tuple(target)
            elif not n.domain and n.op_type == "MatMul" and not attrs and len(n.input) == 2:
                left, right = [actual_shape(v) for v in n.input]
                if len(left) != 2 or len(right) != 3 or left[1] != right[1]:
                    raise ValueError("Unsupported matrix shape")
                out = (right[0], left[0], right[2])
            elif not n.domain and n.op_type in ("Add", "Mul") and not attrs and len(n.input) == 2:
                aa, bb = [actual_shape(v) for v in n.input]
                size = max(len(aa), len(bb)); aa = (1,) * (size-len(aa)) + aa; bb = (1,) * (size-len(bb)) + bb
                out = []
                for av, bv in zip(aa, bb):
                    if av != bv and av != 1 and bv != 1:
                        raise ValueError("Unproven broadcasting")
                    out.append(bv if av == 1 else av)
                out = tuple(out)
            else:
                raise ValueError("Unsupported activation-shape producer")
        shape_cache[name] = out
        return out

    def same_shape(*names):
        values = [actual_shape(name) for name in names]
        s = values[0]
        if (len(s) != 3 or s[0] != 1 or not isinstance(s[1], int)
                or s[1] <= 0 or any(v != s for v in values)):
            raise ValueError("Matching proven FP32 [1,C,T] shapes are required")
        for name in names:
            declared = graph.shape(name)
            # Derived MatMul outputs may lack value_info, but any declaration
            # that exists must not contradict the proven rank/static values.
            if name in graph.values and declared is None:
                raise ValueError("Activation declaration is not FP32")
            if declared is not None and (len(declared) != 3 or any(
                isinstance(v, int) and v != proved for v, proved in zip(declared, s))):
                raise ValueError("Activation declaration contradicts producer shape")
        if isinstance(s[2], int) and s[2] < 0:
            raise ValueError("Negative time dimension")
        return s

    def sole(name, consumer):
        if name in graph.outputs or graph.users.get(name) != [consumer]:
            raise ValueError("Intermediate has fanout or is a graph output")

    def initializer(name, dims):
        t = graph.initializers.get(name)
        if (name in graph.inputs or t is None or t.data_type != TensorProto.FLOAT
                or list(t.dims) != list(dims)):
            raise ValueError("Expected immutable FP32 coefficient initializer")
        values = graph.constant(name)
        if values is None or not np.isfinite(values).all():
            raise ValueError("Nonfinite or unreadable coefficient initializer")
        return t

    def custom(node, kind, count):
        if node.domain != domain or node.op_type != kind or len(node.input) != count or len(node.output) != 1:
            raise ValueError("Unexpected native operation")
        a = _attrs(node)
        names = {"channels", "native_abi", "backend", "row_batches", sine_policy}
        if kind == "CausalDW7SnakeF32":
            names.add("dilation")
        if set(a) != names or any(type(v) is not int for v in a.values()):
            raise ValueError("Unknown or noninteger native attributes")
        allowed = (0, 1, 2, 3, 4) if domain == DOMAINS[0] else (0, 1, 3, 4, 5)
        if a["native_abi"] != 1 or a[sine_policy] != 1 or a["backend"] not in allowed or a["row_batches"] < 0:
            raise ValueError("Unsupported native policy")
        s = same_shape(node.input[0], node.output[0])
        if a["channels"] != s[1]:
            raise ValueError("Channel attribute disagrees with activation")
        if kind == "CausalDW7SnakeF32":
            if a["dilation"] not in (1, 3, 9):
                raise ValueError("Triple fusion supports dilations 1, 3, 9")
            initializer(node.input[1], [s[1], 1, 7]); initializer(node.input[2], [s[1]])
        for name in node.input[-2:]:
            initializer(name, [s[1]])
        return a

    def identity_reshape(index, snake):
        r = graph.nodes[index]
        if r.domain or r.op_type != "Reshape" or len(r.input) != 2 or len(r.output) != 1:
            raise ValueError("Unsupported alias between native operations")
        same_shape(snake.input[0], snake.output[0], r.output[0])
        if actual_shape(r.input[0]) != actual_shape(r.output[0]):
            raise ValueError("Reshape changes the proven dimensions")
        return "symbolic shape program equals native activation dimensions"

    if mode in ("chain", "both"):
        for i, dw in enumerate(graph.nodes):
            if dw.domain != domain or dw.op_type != "CausalDW7SnakeF32":
                continue
            try:
                da = custom(dw, "CausalDW7SnakeF32", 5)
                previous_i = graph.producer.get(dw.input[0])
                if previous_i is None:
                    raise ValueError("Depthwise input has no producer")
                previous = graph.nodes[previous_i]
                alias_i = previous_i if not previous.domain and previous.op_type == "Reshape" else None
                snake_i = graph.producer.get(previous.input[0]) if alias_i is not None else previous_i
                if snake_i is None:
                    raise ValueError("No preceding Snake")
                snake = graph.nodes[snake_i]
                sa = custom(snake, "SnakeF32", 3)
                if any(sa[k] != da[k] for k in sa):
                    raise ValueError("Pre/post native policies differ")
                same_shape(snake.input[0], dw.input[0], dw.output[0])
                proof = "direct shape-preserving input"
                if alias_i is not None:
                    proof = identity_reshape(alias_i, snake)
                    sole(snake.output[0], alias_i); sole(previous.output[0], i)
                else:
                    sole(snake.output[0], i)
                name = "ncc_block_chain_" + str(i)
                if name in occupied:
                    raise ValueError("Generated name collision")
                attrs = dict(da)
                if backend is not None:
                    attrs["backend"] = backend
                replacements[i] = helper.make_node("SnakeDW7SnakeF32",
                    [snake.input[0], *dw.input[1:3], *snake.input[1:], *dw.input[3:]], list(dw.output),
                    name=name, domain=domain, **attrs)
                removed.add(snake_i)
                if alias_i is not None:
                    removed.add(alias_i)
                occupied.add(name)
                records.append({"kind": "chain", "node": name, "pre_snake": snake.name,
                                "depthwise_snake": dw.name, "alias_proof": proof, "channels": da["channels"]})
            except (ValueError, IndexError) as e:
                skipped.append({"kind": "chain", "node": dw.name, "reason": str(e)})

    if mode in ("adds", "both"):
        for i, final in enumerate(graph.nodes):
            if final.domain or final.op_type != "Add" or len(final.input) != 2 or len(final.output) != 1:
                continue
            # Preserve operand order, including signed-zero/NaN behavior.
            bias_i = graph.producer.get(final.input[1])
            if bias_i is None:
                continue
            bias_add = graph.nodes[bias_i]
            if bias_add.domain or bias_add.op_type != "Add" or len(bias_add.input) != 2 or len(bias_add.output) != 1:
                continue
            try:
                if final.attribute or bias_add.attribute:
                    raise ValueError("Unknown Add attributes")
                product, bias_name = bias_add.input
                mm_i = graph.producer.get(product)
                mm = graph.nodes[mm_i] if mm_i is not None else None
                if mm is None or mm.domain or mm.op_type != "MatMul" or mm.attribute or len(mm.input) != 2 or len(mm.output) != 1:
                    raise ValueError("Bias input is not a complete standard MatMul product")
                s = same_shape(product, bias_add.output[0], final.input[0], final.output[0])
                c = s[1]
                w = graph.initializers.get(mm.input[0])
                if (w is None or mm.input[0] in graph.inputs or w.data_type != TensorProto.FLOAT
                        or len(w.dims) != 2 or w.dims[0] != c or w.dims[1] <= 0):
                    raise ValueError("Unproven BCT pointwise matrix")
                ms = actual_shape(mm.input[1])
                if ms != (s[0], w.dims[1], s[2]):
                    raise ValueError("Unproven pointwise matrix input shape")
                t = graph.initializers.get(bias_name)
                if t is None or list(t.dims) not in ([1, c, 1], [c, 1]):
                    raise ValueError("Bias is not explicitly channel-broadcast [1,C,1] or [C,1]")
                initializer(bias_name, t.dims)
                sole(bias_add.output[0], i)
                prefix = "ncc_block_adds_" + str(i)
                if prefix in occupied or prefix + "_bias" in occupied:
                    raise ValueError("Generated name collision")
                alias = copy.deepcopy(t)
                alias.name = prefix + "_bias"
                del alias.dims[:]; alias.dims.extend([c])
                result.graph.initializer.append(alias)
                replacements[i] = helper.make_node("BiasResidualF32", [product, alias.name, final.input[0]],
                    list(final.output), name=prefix, domain=domain, channels=c, native_abi=1,
                    row_batches=0, backend=0 if backend is None else backend)
                removed.add(bias_i)
                occupied.update((prefix, alias.name))
                records.append({"kind": "adds", "node": prefix, "bias_add": bias_add.name,
                                "residual_add": final.name, "bias_source": bias_name, "channels": c})
            except (ValueError, IndexError) as e:
                skipped.append({"kind": "adds", "node": final.name, "reason": str(e)})

    counts = {kind: sum(r["kind"] == kind for r in records) for kind in ("chain", "adds")}
    for kind, expected in (("chain", expected_chains), ("adds", expected_adds)):
        if expected is not None and counts[kind] != expected:
            raise ValueError(f"Expected {expected} {kind} fusions, got {counts[kind]}; rejected: {skipped}")
    if not records and mode != "none":
        raise ValueError("No proven block fusion candidates")
    new_nodes = []
    for i, original in enumerate(graph.nodes):
        if i in removed:
            continue
        node = replacements.get(i, copy.deepcopy(original))
        if backend is not None and node.domain == domain:
            a = _attrs(node)
            if "backend" in a:
                for attr in node.attribute:
                    if attr.name == "backend":
                        attr.i = backend
        new_nodes.append(node)
    del result.graph.node[:]; result.graph.node.extend(new_nodes)
    live = {x for n in new_nodes for x in (*n.input, *n.output)} | set(graph.inputs) | graph.outputs
    infos = [v for v in result.graph.value_info if v.name in live]
    del result.graph.value_info[:]; result.graph.value_info.extend(infos)
    # Preserve every original tensor's data representation, including offsets.
    for old, new in zip(model.graph.initializer, result.graph.initializer):
        if old.SerializeToString() != new.SerializeToString():
            raise RuntimeError("Existing initializer changed")
    # In-memory checker has no external-data base directory. The file API
    # checks both source and saved derivative by path, including external data.
    if not any(t.data_location == TensorProto.EXTERNAL for t in result.graph.initializer):
        onnx.checker.check_model(result, full_check=False)
    return result, {"variant": mode, "mode": mode, "backend_override": backend, "domain": domain,
                    "required_operators": sorted({n.op_type for n in new_nodes if n.domain == domain}),
                    "required_isa": {None: "preserved native graph policy", 0: "automatic guarded selection", 2: "NEON", 4: "AVX2", 5: "AVX512F/DQ/BW/VL, AVX2, FMA and OS vector state"}[backend],
                    "counts": counts, "changes": records, "skipped": skipped,
                    "source_nodes": len(graph.nodes), "output_nodes": len(new_nodes)}


def rewrite(source, output, variant="both", backend=None, **options):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Candidate output must differ from source")
    onnx.checker.check_model(str(source), full_check=False)
    model = onnx.load(source, load_external_data=False)
    result, audit = rewrite_model(model, source.parent, variant=variant, backend=backend, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    external = copy_external_data(result, source.parent, output.parent)
    onnx.save_model(result, output)
    onnx.checker.check_model(str(output), full_check=False)
    report = {**audit, "source_sha256": sha256(source), "output_sha256": sha256(output),
              "external_files": external, "scope": "Experimental graph only; no inference or default promotion"}
    output.with_suffix(".block-fusion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True); p.add_argument("--output", required=True)
    p.add_argument("--variant", "--mode", dest="variant", choices=MODES, default="both")
    p.add_argument("--backend", type=int, choices=(0, 2, 4, 5))
    p.add_argument("--expected-chains", type=int); p.add_argument("--expected-adds", type=int)
    args = vars(p.parse_args())
    print(json.dumps(rewrite(**args), indent=2))
