"""Opt-in, bounded matrix geometry for the audited AudioVAE2 decoder stem.

MLAS uses a different reduction for a single output column. Padding only that
case to two columns makes the stem independent of latent chunk length. The
dummy column is discarded before any activation, state update or INT8 matrix.
This pass does not select or publish a new decoder default.
"""
from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper


def validate_precision_sine_backends(model):
    """Require the graph itself to select the validated AVX512 sine path.

    Library capability alone is insufficient: each native operator chooses
    its own backend. AUTO or AVX2 would bypass the consistent AVX512 tail.
    """
    native = {"venky.audio.cpu", "venky.audio.cpu.portable"}
    snake = {"SnakeF32", "CausalDW7SnakeF32", "SnakeDW7SnakeF32"}
    fused = {"fast.audiovae.stage.experimental",
             "fast.audiovae.precision.stage.experimental",
             "fast.audiovae.precision.upsample.experimental"}
    checked = 0
    for node in model.graph.node:
        if node.domain in ("", "ai.onnx") and node.op_type == "Sin":
            raise ValueError("Canonical precision requires the validated native sine path")
        if node.domain in fused or (node.domain in native and node.op_type in snake):
            attributes = [a for a in node.attribute if a.name == "backend"]
            if (len(attributes) != 1 or attributes[0].type != onnx.AttributeProto.INT
                    or attributes[0].i != 5):
                raise ValueError("Canonical precision requires explicit backend 5 on " +
                                 (node.name or node.op_type))
            checked += 1
    return checked


def canonicalize_stem(model):
    """Return a graph with the audited 64-to-2048 stem using at least two columns.

    Apply after the ordinary streaming graph transformation, or independently
    to the original full-call graph. No learned tensor is modified. The pass
    intentionally accepts only the named, fixed-shape stem of audited models.
    """
    candidates = [n for n in model.graph.node if n.name == "node_conv1d_1__bct_mm"]
    if len(candidates) != 1:
        raise ValueError("Expected one audited AudioVAE2 stem matrix")
    node = candidates[0]
    initializers = {v.name: v for v in model.graph.initializer}
    if (node.domain or node.op_type != "MatMul" or len(node.input) != 2
            or len(node.output) != 1 or node.attribute
            or node.input[0] not in initializers):
        raise ValueError("Unexpected AudioVAE2 stem matrix contract")
    weight = initializers[node.input[0]]
    if list(weight.dims) != [2048, 64] or weight.data_type != onnx.TensorProto.FLOAT:
        raise ValueError("Expected FP32 [2048,64] stem weights")
    version = next((v.version for v in model.opset_import if v.domain == ""), 0)
    if version < 18:
        raise ValueError("Stem padding requires standard ONNX opset 18 or later")
    prefix = "fast_stem_"
    names = {v.name for v in model.graph.initializer}
    for item in model.graph.node:
        names.update(item.input)
        names.update(item.output)
        names.add(item.name)
    if any(name.startswith(prefix) for name in names):
        raise ValueError("Stem was already canonicalized or uses reserved names")
    result = copy.deepcopy(model)
    constants = {"zero": np.array([0], dtype=np.int64),
                 "two": np.array([2], dtype=np.int64),
                 "axis": np.array([2], dtype=np.int64),
                 "float_zero": np.array(0, dtype=np.float32)}
    for name, value in constants.items():
        result.graph.initializer.append(numpy_helper.from_array(value, prefix + name))
    activation = node.input[1]
    matrix = copy.deepcopy(node)
    matrix.input[1] = prefix + "padded"
    matrix.output[0] = prefix + "matrix"
    replacements = [
        helper.make_node("Shape", [activation], [prefix + "width"],
                         start=2, end=3, name=prefix + "shape"),
        helper.make_node("Sub", [prefix + "two", prefix + "width"], [prefix + "needed"],
                         name=prefix + "subtract"),
        helper.make_node("Max", [prefix + "needed", prefix + "zero"], [prefix + "right"],
                         name=prefix + "maximum"),
        helper.make_node("Concat", [prefix + "zero", prefix + "right"], [prefix + "pads"],
                         axis=0, name=prefix + "padding"),
        helper.make_node("Pad", [activation, prefix + "pads", prefix + "float_zero", prefix + "axis"],
                         [prefix + "padded"], mode="constant", name=prefix + "pad"),
        matrix,
        helper.make_node("Slice", [prefix + "matrix", prefix + "zero", prefix + "width", prefix + "axis"],
                         list(node.output), name=prefix + "restore_width"),
    ]
    nodes = []
    for item in result.graph.node:
        nodes.extend(replacements if item.name == node.name else [item])
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return result, {"version": 1, "stem": node.name, "minimum_time_columns": 2,
                    "learned_tensors_unchanged": True, "extra_history": False}
