"""Expose bounded, explicit causal state on the audited AudioVAE2 decoder graphs.

The graph owns no mutable state. Each caller supplies and retains its own history
inputs/outputs. Weights and nonlinearities are preserved. Existing native DW and
phase operators are reused with a bounded local halo, never a replayed prefix of
the complete model. Fused stages require their explicit streaming operator ABI.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper

from .upsampling import constant_map, validate_pair

_NATIVE = {"venky.audio.cpu", "venky.audio.cpu.portable"}
_STAGES = {"fast.audiovae.stage.experimental", "fast.audiovae.precision.stage.experimental"}
_UPSAMPLE = {"fast.audiovae.precision.upsample.experimental"}
_POINTWISE = {
    ("fast.audiovae.precision.matrix.experimental", "PrecisionMatMulF32"),
    ("venky.audio.cpu.aocl.rows", "PackedRowsMatMulF32"),
}
_STANDARD = {"Constant", "Shape", "Squeeze", "Unsqueeze", "Reshape", "Transpose",
             "Concat", "Add", "Mul", "Sub", "Div", "Pow", "Sin", "Tanh", "Identity",
             "Cast", "MatMul", "Conv", "ConvTranspose", "Pad", "Slice"}
_MAX = np.iinfo(np.int64).max
_PREFIX = "fast_stream_"


def _attrs(node):
    values = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if len(values) != len(node.attribute):
        raise ValueError(f"Duplicate attributes on {node.name}")
    return values


def _validate_layouts(model, constants):
    """Prove that views and broadcasts preserve each time column's ordering.

    Do not trust exporter value_info as a proof: an annotation can lie. Track
    dimensions from operators instead. A tuple (a,b) means a*T+b, with T the
    latent chunk length. Only the audited phase interleave can change time rate.
    """
    def plus(value, offset):
        return (value[0], value[1]+offset) if isinstance(value, tuple) else value+offset

    def product(values):
        total = 1
        for value in values:
            if isinstance(value, tuple):
                if isinstance(total, tuple):
                    raise ValueError("Nonlinear time dimensions are unsupported")
                total = (value[0]*total, value[1]*total)
            elif isinstance(total, tuple):
                total = (total[0]*value, total[1]*value)
            else:
                total *= value
        return total

    def reshape_shape(before, target, allowzero):
        target = list(target)
        if target.count(-1) > 1 or any(isinstance(d, int) and d < -1 for d in target):
            raise ValueError("Invalid reshape target")
        if allowzero and 0 in target and -1 in target:
            raise ValueError("Ambiguous zero and inferred reshape dimensions")
        for i, dim in enumerate(target):
            if dim == 0 and not allowzero:
                if i >= len(before):
                    raise ValueError("Invalid copied reshape dimension")
                target[i] = before[i]
        volume = product(before)
        known = product(d for d in target if d != -1)
        if -1 in target:
            if not isinstance(known, int) or known <= 0:
                raise ValueError("Cannot prove inferred reshape dimension")
            parts = volume if isinstance(volume, tuple) else (volume,)
            if any(v % known for v in parts):
                raise ValueError("Nonintegral reshape dimensions")
            inferred = tuple(v//known for v in parts) if isinstance(volume, tuple) else volume//known
            target[target.index(-1)] = inferred
        elif known != volume:
            raise ValueError("Reshape volume is not proven unchanged")
        return tuple(target)

    shapes = {name: tuple(a.shape) for name, a in constants.items()}
    kinds = {name: "constant" for name in constants}
    shape_values = {name: (tuple(int(v) for v in a.reshape(-1)), tuple(a.shape))
                    for name, a in constants.items() if a.dtype == np.int64}
    latent = model.graph.input[0]
    shapes[latent.name] = (1, int(latent.type.tensor_type.shape.dim[1].dim_value), (1, 0))
    kinds[latent.name] = "bct"

    def shape_input(name):
        if name not in shape_values:
            raise ValueError(f"Unproven shape expression: {name}")
        return shape_values[name]

    def data_shape(name):
        if name not in shapes:
            raise ValueError(f"Unproven feature layout: {name}")
        return shapes[name]

    def broadcast(names):
        values = [data_shape(name) for name in names]
        rank = max(map(len, values))
        out = [1]*rank
        for value in values:
            for index, dim in enumerate((1,)*(rank-len(value))+value):
                if out[index] == 1:
                    out[index] = dim
                elif dim not in (1, out[index]):
                    raise ValueError("Broadcast could change or depend on chunk time")
        live = [name for name in names if kinds.get(name) != "constant"]
        if live:
            tag = kinds[live[0]]
            if any(kinds[name] != tag or data_shape(name) != tuple(out) for name in live):
                raise ValueError("Elementwise operands have incompatible time layouts")
        else:
            tag = "constant"
        return tuple(out), tag

    for node in model.graph.node:
        attr = _attrs(node)
        standard = node.domain in ("", "ai.onnx")
        op = node.op_type
        if standard and op == "Constant":
            continue
        if standard and op == "Shape":
            dims = data_shape(node.input[0])
            values = dims[attr.get("start", 0):attr.get("end", len(dims))]
            shape_values[node.output[0]] = (values, (len(values),))
            continue
        is_shape = node.input and node.input[0] in shape_values
        if standard and is_shape and op in ("Reshape", "Squeeze", "Unsqueeze", "Mul", "Concat", "Identity", "Cast"):
            values, before = shape_input(node.input[0])
            if op == "Reshape":
                target, target_shape = shape_input(node.input[1])
                if target_shape != (len(target),):
                    raise ValueError("Shape reshape target must be a vector")
                after = reshape_shape(before, target, attr.get("allowzero", 0))
            elif op in ("Squeeze", "Unsqueeze"):
                axes = shape_input(node.input[1])[0] if len(node.input) == 2 else tuple(i for i, d in enumerate(before) if d == 1)
                rank = len(before)+(len(axes) if op == "Unsqueeze" else 0)
                if any(not isinstance(axis, int) or axis < -rank or axis >= rank for axis in axes):
                    raise ValueError("Invalid shape view axes")
                axes = tuple(axis+rank if axis < 0 else axis for axis in axes)
                if len(set(axes)) != len(axes):
                    raise ValueError("Duplicate shape view axes")
                if op == "Squeeze":
                    if any(before[axis] != 1 for axis in axes):
                        raise ValueError("Cannot squeeze a nonunit dimension")
                    after = tuple(d for i, d in enumerate(before) if i not in axes)
                else:
                    old = iter(before)
                    after = tuple(1 if i in axes else next(old) for i in range(rank))
            elif op == "Mul":
                other, other_shape = shape_input(node.input[1])
                if len(values) != 1 or len(other) != 1 or before not in ((), (1,)) or other_shape not in ((), (1,)):
                    raise ValueError("Only scalar shape multiplication is supported")
                values = (product((values[0], other[0])),)
                after = (1,) if before or other_shape else ()
            elif op == "Concat":
                parts = [shape_input(name) for name in node.input]
                if attr != {"axis": 0} or any(len(s) != 1 for _, s in parts):
                    raise ValueError("Only shape-vector concatenation is supported")
                values = tuple(d for part, _ in parts for d in part)
                after = (len(values),)
            else:
                if op == "Cast" and attr.get("to") != TP.INT64:
                    raise ValueError("Shape casts must preserve INT64")
                after = before
            shape_values[node.output[0]] = (values, after)
            continue
        before = data_shape(node.input[0]) if node.input else ()
        tag = kinds.get(node.input[0]) if node.input else None
        after = before
        if standard and op == "Reshape":
            target, vector_shape = shape_input(node.input[1])
            if vector_shape != (len(target),):
                raise ValueError("Feature reshape target must be an INT64 vector")
            after = reshape_shape(before, target, attr.get("allowzero", 0))
            if after == before:
                pass
            elif (tag == "bct" and len(after) == 4 and after[0] == 1 and
                  isinstance(after[1], int) and isinstance(after[2], int) and
                  before == (1, after[1]*after[2], after[3])):
                tag = "phase_channels"
            elif (tag == "btc" and len(after) == 4 and after[0] == 1 and
                  isinstance(after[2], int) and isinstance(after[3], int) and
                  before == (1, after[1], after[2]*after[3])):
                tag = "phase_btc"
            elif (tag == "phase_frame" and len(after) == 3 and
                  after == (before[0], before[1], product((before[2], before[3])))):
                tag = "bct"
            elif tag == "constant":
                pass
            else:
                raise ValueError(f"Reshape can reorder channels and time: {node.name}")
        elif standard and op == "Transpose":
            perm = attr.get("perm", list(reversed(range(len(before)))))
            if sorted(perm) != list(range(len(before))):
                raise ValueError("Invalid transpose permutation")
            after = tuple(before[i] for i in perm)
            if perm == list(range(len(before))) or tag == "constant":
                pass
            elif tag in ("bct", "btc") and perm == [0, 2, 1]:
                tag = "btc" if tag == "bct" else "bct"
            elif tag == "phase_channels" and perm == [0, 1, 3, 2]:
                tag = "phase_frame"
            elif tag == "phase_btc" and perm == [0, 2, 1, 3]:
                tag = "phase_frame"
            else:
                raise ValueError(f"Transpose is not an audited phase layout: {node.name}")
        elif standard and op in ("Concat", "Squeeze", "Unsqueeze"):
            raise ValueError(f"Unsupported feature layout operation: {op}")
        elif standard and op in ("Add", "Mul", "Sub", "Div", "Pow"):
            after, tag = broadcast(node.input)
        elif standard and op in ("Sin", "Tanh", "Identity", "Cast"):
            if op == "Cast" and attr.get("to") != TP.FLOAT:
                raise ValueError("Feature casts must preserve FP32")
        elif standard and op in ("Conv", "ConvTranspose"):
            weight = constants.get(node.input[1])
            if tag != "bct" or weight is None or weight.ndim != 3:
                raise ValueError("Convolutions require BCT features and fixed weights")
            group = attr.get("group", 1)
            dilation = attr.get("dilations", [1])[0]
            stride = attr.get("strides", [1])[0]
            pads = attr.get("pads", [0, 0])
            if op == "Conv":
                if before[1] != weight.shape[1]*group or stride != 1:
                    raise ValueError("Convolution channels or stride are unsupported")
                after = (1, weight.shape[0], plus(before[2], sum(pads)-dilation*(weight.shape[2]-1)))
            else:
                if before[1] != weight.shape[0] or group != 1:
                    raise ValueError("Unsupported upsampling channel layout")
                after = (1, weight.shape[1], plus(product((before[2], stride)),
                         dilation*(weight.shape[2]-1)+1-stride-sum(pads)+attr.get("output_padding", [0])[0]))
        elif standard and op == "Pad":
            pads, _ = shape_input(node.input[1])
            if len(pads) != 2*len(before):
                raise ValueError("Invalid pad rank")
            after = tuple(plus(dim, pads[i]+pads[i+len(before)]) for i, dim in enumerate(before))
        elif standard and op == "Slice":
            start, end, axes, steps = [shape_input(name)[0] for name in node.input[1:]]
            if len(start) != 1 or start != (0,) or len(end) != 1 or end[0] >= 0 or steps != (1,):
                raise ValueError("Only audited right-crop slices are supported")
            expected_axis = 2 if tag == "bct" else 1 if tag == "btc" else None
            if axes != (expected_axis,):
                raise ValueError("Slice must crop the feature time axis")
            after = tuple(plus(dim, end[0]) if i == expected_axis else dim for i, dim in enumerate(before))
        elif (standard and op == "MatMul") or (node.domain, op) in _POINTWISE:
            left, right = [data_shape(name) for name in node.input]
            if kinds[node.input[0]] == "constant" and len(left) == 2 and kinds[node.input[1]] == "bct":
                if left[1] != right[1]:
                    raise ValueError("Matrix weights do not match feature channels")
                after, tag = (1, left[0], right[2]), "bct"
            elif kinds[node.input[1]] == "constant" and len(right) == 2 and kinds[node.input[0]] == "btc":
                if left[2] != right[0]:
                    raise ValueError("Matrix weights do not match feature channels")
                after, tag = (1, left[1], right[1]), "btc"
            else:
                raise ValueError("Matrix multiplication could mix time columns")
        elif node.domain in _NATIVE:
            if tag != "bct":
                raise ValueError("Native kernels require BCT features")
            if op == "PhaseSumBiasInterleaveF32":
                if before != data_shape(node.input[1]) or before[1] != attr["channels"]*attr["stride"]:
                    raise ValueError("Phase inputs must match the projected layout")
                after = (1, attr["channels"], product((before[2], attr["stride"])))
            elif before[1] != attr.get("channels"):
                raise ValueError("Native channels disagree with feature layout")
        elif node.domain in _STAGES or node.domain in _UPSAMPLE:
            if tag != "bct":
                raise ValueError("Fused stages require BCT features")
            after = (1, attr["channels"], product((before[2], attr.get("stride", 1))))
        else:
            raise ValueError(f"Unknown feature layout operator: {node.domain}::{op}")
        for output in node.output:
            shapes[output], kinds[output] = after, tag
    if kinds.get(model.graph.output[0].name) != "bct":
        raise ValueError("Decoder output must retain BCT time order")


def rewrite_model(model):
    """Return ``(stateful_model, audit)`` without modifying ``model``.

    Supported inputs are batch-one FP32 decoder graphs in BCT layout. The audit
    records fixed state shapes and the original latent/audio names. Unsupported
    custom operators, strided/downsampling convolutions, noncausal padding and
    unknown standard operators are rejected, rather than silently reset.
    """
    if (model.functions or model.graph.sparse_initializer or
            any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for n in model.graph.node for a in n.attribute)):
        raise ValueError("Streaming conversion requires a flat dense graph")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one latent input and one audio output before conversion")
    latent, audio = model.graph.input[0], model.graph.output[0]
    ty = latent.type.tensor_type
    dims = ty.shape.dim
    if (ty.elem_type != TP.FLOAT or len(dims) != 3 or
            not dims[0].HasField("dim_value") or dims[0].dim_value != 1 or
            not dims[1].HasField("dim_value") or dims[1].dim_value <= 0 or
            dims[2].HasField("dim_value")):
        raise ValueError("Latents must be dynamic-length FP32 [1,channels,T]")
    occupied = ({n.name for n in model.graph.node} |
                {s for n in model.graph.node for s in (*n.input, *n.output)} |
                {x.name for x in model.graph.initializer})
    if any(n.startswith(_PREFIX) for n in occupied):
        raise ValueError("Graph already contains reserved streaming names")
    model = copy.deepcopy(model)
    constants = constant_map(model)
    _validate_layouts(model, constants)
    nodes = list(model.graph.node)
    consumers = defaultdict(list)
    for n in nodes:
        for name in n.input:
            consumers[name].append(n)
    # Shape inference cannot infer custom kernels, but preserved export value
    # declarations and simple projection weights identify all state channels.
    shapes = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor = value.type.tensor_type
        if tensor.elem_type == TP.FLOAT and tensor.HasField("shape"):
            shapes[value.name] = [d.dim_value if d.HasField("dim_value") else None
                                  for d in tensor.shape.dim]
    producers = {name: n for n in nodes for name in n.output}
    states, additions, result_nodes = [], [], []
    counts, required = Counter(), set()
    rewritten_crops = {}
    phase_slices = set()

    def const(name, values, dtype=np.int64):
        full = _PREFIX + name
        additions.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), full))
        return full

    def slice_node(source, start, end, output, tag, axis=2):
        return helper.make_node("Slice", [source, const(tag+"_start", [start]),
            const(tag+"_end", [end]), const(tag+"_axis", [axis]), const(tag+"_step", [1])],
            [output], name=_PREFIX+tag+"_slice")

    def state(channels, length, node, kind, suffix="", axis=2):
        if not isinstance(channels, int) or channels <= 0 or length <= 0:
            raise ValueError(f"Invalid state dimensions on {node.name}")
        tag = f"s{len(states):02d}{suffix}"
        inp, out = _PREFIX+tag+"_in", _PREFIX+tag+"_out"
        shape = [1, channels, length] if axis == 2 else [1, length, channels]
        for values, name in ((model.graph.input, inp), (model.graph.output, out)):
            values.append(helper.make_tensor_value_info(name, TP.FLOAT, shape))
        states.append({"input": inp, "output": out, "shape": shape, "dtype": "float32",
                       "node": node.name, "kind": kind, "time_axis": axis})
        return inp, out, tag

    def prefix(source, channels, length, node, kind, axis=2):
        inp, out, tag = state(channels, length, node, kind, axis=axis)
        joined = _PREFIX+tag+"_joined"
        shape = [1, channels, _PREFIX+tag+"_context_time"] if axis == 2 else [1, _PREFIX+tag+"_context_time", channels]
        model.graph.value_info.append(helper.make_tensor_value_info(joined, TP.FLOAT, shape))
        result_nodes.append(helper.make_node("Concat", [inp, source], [joined],
                                            name=_PREFIX+tag+"_concat", axis=axis))
        result_nodes.append(slice_node(joined, -length, _MAX, out, tag+"_retain", axis))
        return joined, tag

    def channels_of(name, axis=2):
        dims = shapes.get(name)
        if dims and len(dims) == 3 and dims[0] == 1:
            value = dims[1 if axis == 2 else 2]
            if value:
                return int(value)
        node = producers.get(name)
        if node and node.op_type == "MatMul" and len(node.input) == 2:
            side = 0 if axis == 2 else 1
            w = constants.get(node.input[side])
            if w is not None and w.ndim == 2:
                return int(w.shape[0 if axis == 2 else 1])
        if node and node.op_type == "PrecisionMatMulF32":
            return int(_attrs(node)["M"])
        raise ValueError(f"Cannot prove channels for history tensor {name}")

    def integer_vector(name):
        a = constants.get(name)
        if a is None or a.dtype != np.int64 or a.ndim != 1:
            raise ValueError(f"Expected immutable int64 shape/index vector: {name}")
        return a.tolist()

    for index, node in enumerate(nodes):
        attr = _attrs(node)
        standard = node.domain in ("", "ai.onnx")
        if standard and node.op_type not in _STANDARD:
            raise ValueError(f"Unsupported potential time-mixing operator: {node.op_type}")
        if id(node) in rewritten_crops:
            node.input[1] = rewritten_crops[id(node)]
            result_nodes.append(node)
            continue
        if standard and node.op_type == "Conv":
            if len(node.input) not in (2, 3) or node.input[1] not in constants:
                raise ValueError(f"Convolution weights must be immutable: {node.name}")
            w = constants[node.input[1]]
            if w.dtype != np.float32 or w.ndim != 3:
                raise ValueError(f"Expected FP32 Conv1D weights: {node.name}")
            group = attr.get("group", 1)
            dilation = attr.get("dilations", [1])
            pads = attr.get("pads", [0, 0])
            if (attr.get("strides", [1]) != [1] or len(dilation) != 1 or dilation[0] < 1 or
                    attr.get("auto_pad", b"NOTSET") not in (b"NOTSET", b"") or
                    attr.get("kernel_shape", [w.shape[2]]) != [w.shape[2]]):
                raise ValueError(f"Unsupported causal convolution parameters: {node.name}")
            halo = (w.shape[2]-1)*dilation[0]
            if pads != [halo, 0]:
                raise ValueError(f"Convolution must have exactly causal left padding: {node.name}")
            if halo:
                joined, _ = prefix(node.input[0], int(w.shape[1]*group), halo, node, "conv")
                node.input[0] = joined
                for a in node.attribute:
                    if a.name == "pads":
                        a.CopyFrom(helper.make_attribute("pads", [0, 0]))
                        break
                else:
                    node.attribute.append(helper.make_attribute("pads", [0, 0]))
                counts["conv"] += 1
            result_nodes.append(node)
        elif standard and node.op_type == "ConvTranspose":
            uses = consumers[node.output[0]]
            if len(uses) != 1:
                raise ValueError("ConvTranspose output must have exactly one right-crop consumer")
            crop = uses[0]
            cin, _, stride, _, _ = validate_pair(node, crop, constants)
            joined, tag = prefix(node.input[0], cin, 1, node, "conv_transpose")
            node.input[0] = joined
            rewritten_crops[id(crop)] = const(tag+"_left_crop", [stride])
            result_nodes.append(node)
            counts["conv_transpose"] += 1
        elif standard and node.op_type == "Pad":
            if (len(node.input) not in (2, 3) or
                    attr.get("mode", b"constant") != b"constant" or set(attr)-{"mode"}):
                raise ValueError("Only audited phase-shift zero padding is supported")
            pads = integer_vector(node.input[1])
            axis = 2 if pads == [0, 0, 1, 0, 0, 0] else 1 if pads == [0, 1, 0, 0, 0, 0] else None
            if axis is None:
                raise ValueError("Pad is not a one-frame phase shift")
            if len(node.input) == 3 and node.input[2]:
                value = constants.get(node.input[2])
                if value is None or value.size != 1 or float(value.reshape(-1)[0]) != 0:
                    raise ValueError("Phase padding must be constant zero")
            uses = consumers[node.output[0]]
            if (len(uses) != 1 or uses[0].op_type != "Slice" or
                    len(uses[0].input) != 5 or
                    [integer_vector(x) for x in uses[0].input[1:]] != [[0], [-1], [axis], [1]]):
                raise ValueError("Pad must feed the audited one-frame phase shift Slice")
            phase_slices.add(id(uses[0]))
            joined, _ = prefix(node.input[0], channels_of(node.input[0], axis), 1,
                               node, "phase_projection", axis)
            result_nodes.append(helper.make_node("Identity", [joined], list(node.output), name=node.name))
            counts["phase_projection"] += 1
        elif standard and node.op_type == "Slice":
            if id(node) not in phase_slices:
                raise ValueError(f"Slice is not an audited causal crop or phase shift: {node.name}")
            result_nodes.append(node)
        elif standard and node.op_type == "MatMul":
            constant_inputs = [i for i, name in enumerate(node.input) if name in constants]
            if len(node.input) != 2 or not constant_inputs:
                raise ValueError(f"Matrix multiplication must use immutable channel weights: {node.name}")
            if len(constant_inputs) == 1:
                side = constant_inputs[0]
                weight = constants[node.input[side]]
                if weight.dtype != np.float32 or weight.ndim != 2:
                    raise ValueError(f"Expected a channel weight matrix: {node.name}")
                # BCT uses W @ X. The alternate BTC lowering explicitly
                # transposes BCT features first and uses X @ W.
                if side == 1:
                    upstream = producers.get(node.input[0])
                    if (upstream is None or upstream.op_type != "Transpose" or
                            _attrs(upstream).get("perm") != [0, 2, 1]):
                        raise ValueError(f"Right-hand matrix requires the audited BTC transpose: {node.name}")
            result_nodes.append(node)
        elif node.domain in _NATIVE and node.op_type in ("CausalDW7F32", "CausalDW7SnakeF32", "SnakeDW7SnakeF32"):
            expected = {"CausalDW7F32": 3, "CausalDW7SnakeF32": 5, "SnakeDW7SnakeF32": 7}[node.op_type]
            if (len(node.input) != expected or len(node.output) != 1 or attr.get("native_abi") != 1 or
                    attr.get("row_batches") != 0 or attr.get("dilation") not in (1, 3, 9)):
                raise ValueError(f"Unsupported native depthwise ABI: {node.name}")
            halo = 6*attr["dilation"]
            joined, tag = prefix(node.input[0], int(attr["channels"]), halo, node, "native_depthwise")
            output = node.output[0]
            node.input[0] = joined
            node.output[0] = _PREFIX+tag+"_with_halo"
            model.graph.value_info.append(helper.make_tensor_value_info(node.output[0], TP.FLOAT,
                [1, int(attr["channels"]), _PREFIX+tag+"_depthwise_time"]))
            result_nodes.append(node)
            result_nodes.append(slice_node(node.output[0], halo, _MAX, output, tag+"_crop"))
            counts["native_depthwise"] += 1
        elif node.domain in _NATIVE and node.op_type == "PhaseSumBiasInterleaveF32":
            if (len(node.input) != 3 or len(node.output) != 1 or attr.get("previous_shift") != 1 or
                    attr.get("native_abi") != 1 or attr.get("row_batches") != 0 or
                    attr.get("stride") not in (2, 5, 6, 8)):
                raise ValueError(f"Unsupported phase ABI: {node.name}")
            width = int(attr["channels"]*attr["stride"])
            previous, tag = prefix(node.input[1], width, 1, node, "phase_projection")
            zero = const(tag+"_zero_current", np.zeros((1, width, 1), np.float32), np.float32)
            current = _PREFIX+tag+"_current"
            result_nodes.append(helper.make_node("Concat", [zero, node.input[0]], [current],
                                                name=current+"_concat", axis=2))
            model.graph.value_info.append(helper.make_tensor_value_info(current, TP.FLOAT,
                [1, width, _PREFIX+tag+"_phase_input_time"]))
            node.input[0], node.input[1] = current, previous
            output = node.output[0]
            node.output[0] = _PREFIX+tag+"_with_halo"
            model.graph.value_info.append(helper.make_tensor_value_info(node.output[0], TP.FLOAT,
                [1, int(attr["channels"]), _PREFIX+tag+"_phase_output_time"]))
            result_nodes.append(node)
            result_nodes.append(slice_node(node.output[0], attr["stride"], _MAX, output, tag+"_crop"))
            counts["native_phase"] += 1
        elif node.domain in _STAGES and node.op_type == "StageStackF32":
            if len(node.input) != 25 or len(node.output) != 1 or attr.get("native_abi") != 1:
                raise ValueError("Unsupported fused stage ABI")
            inp, out, _ = state(int(attr["channels"]), 78, node, "stage_depthwise")
            node.input.append(inp)
            node.output.append(out)
            node.op_type = "StageStackStreamingF32"
            required.add((node.domain, node.op_type))
            result_nodes.append(node)
            counts["fused_stage"] += 1
        elif node.domain in _UPSAMPLE and node.op_type == "UpsampleStageF32":
            if (len(node.input) != 28 or len(node.output) != 1 or attr.get("native_abi") != 1 or
                    attr.get("channels") != 128 or attr.get("stride") != 2):
                raise ValueError("Unsupported fused upsampling ABI")
            hist_in, hist_out, _ = state(128, 78, node, "stage_depthwise")
            prev_in, prev_out, _ = state(256, 1, node, "phase_projection")
            node.input.extend([hist_in, prev_in])
            node.output.extend([hist_out, prev_out])
            node.op_type = "UpsampleStageStreamingF32"
            required.add((node.domain, node.op_type))
            result_nodes.append(node)
            counts["fused_upsample"] += 1
        elif standard or (node.domain in _NATIVE and node.op_type in ("SnakeF32", "BiasResidualF32")) or (node.domain, node.op_type) in _POINTWISE:
            # All remaining audited custom nodes operate independently at each
            # time column. Reductions over time, attention and normalization
            # operators are intentionally absent from the standard allowlist.
            result_nodes.append(node)
        else:
            raise ValueError(f"Unsupported custom streaming operator: {node.domain}::{node.op_type}")
    if not states:
        raise ValueError("Graph has no supported causal history to expose")
    del model.graph.node[:]
    model.graph.node.extend(result_nodes)
    model.graph.initializer.extend(additions)
    # Exported symbolic names are equality assertions to ORT's memory planner.
    # A node that now consumes T+halo must not inherit an old shared T symbol
    # through custom shape inference. Give intermediate dynamic dimensions
    # independent names; the shape program still determines their real sizes.
    for index, value in enumerate(model.graph.value_info):
        for axis, dim in enumerate(value.type.tensor_type.shape.dim):
            if not dim.HasField("dim_value"):
                dim.dim_param = f"{_PREFIX}v{index}_d{axis}"
    onnx.checker.check_model(model, full_check=False)
    state_bytes = sum(int(np.prod(s["shape"]))*4 for s in states)
    audit = {"version": 1, "latent_input": latent.name, "audio_output": audio.name,
             "states": states, "state_bytes": state_bytes, "counts": dict(counts),
             "required_streaming_operators": [{"domain": d, "operator": op} for d, op in sorted(required)],
             "policy": "Explicit bounded per-layer histories; unchanged weights and nonlinearities; no whole-model prefix replay"}
    return model, audit
