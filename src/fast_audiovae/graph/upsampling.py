"""Exact FP32 phase lowerings for AudioVAE2's causal ConvTranspose1D.

This module does not run a model. It may compose with unrelated custom graph
nodes, whose definitions are intentionally left untouched. All transformations are checked before a model is saved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys


import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

MODES = ('split-matmul', 'split-matmul-btc', 'two-tap-conv')


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def constant_map(model):
    overridable = {v.name for v in model.graph.input}
    result = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer
              if v.name not in overridable}
    for node in model.graph.node:
        if node.op_type == 'Constant' and node.domain in ('', 'ai.onnx'):
            value = attributes(node).get('value')
            if isinstance(value, onnx.TensorProto):
                result[node.output[0]] = numpy_helper.to_array(value)
    return result


def validate_pair(node, crop, constants):
    """Fail closed instead of silently changing nonmatching convolution semantics."""
    attr = attributes(node)
    stride = attr.get('strides', [1])
    if len(stride) != 1 or stride[0] <= 1:
        raise ValueError(f'{node.name}: requires one positive upsampling stride')
    stride = int(stride[0])
    requirements = {'group': 1, 'pads': [0, 0], 'dilations': [1], 'output_padding': [0]}
    for key, default in requirements.items():
        if attr.get(key, default) != default:
            raise ValueError(f'{node.name}: unsupported {key}={attr[key]}')
    if attr.get('auto_pad', b'NOTSET') not in (b'NOTSET', b''):
        raise ValueError(f'{node.name}: automatic padding is unsupported')
    if 'output_shape' in attr:
        raise ValueError(f'{node.name}: explicit output_shape is unsupported')
    if len(node.input) not in (2, 3) or node.input[1] not in constants:
        raise ValueError(f'{node.name}: requires constant weights and optional constant bias')
    weight = constants[node.input[1]]
    if weight.dtype != np.float32 or weight.ndim != 3 or weight.shape[2] != 2 * stride:
        raise ValueError(f'{node.name}: requires FP32 [Cin,Cout,2*stride] weights')
    cin, cout, kernel = weight.shape
    if attr.get('kernel_shape', [kernel]) != [kernel]:
        raise ValueError(f'{node.name}: kernel attribute disagrees with weights')
    bias = None
    if len(node.input) == 3 and node.input[2]:
        bias = constants.get(node.input[2])
        if bias is None or bias.dtype != np.float32 or bias.shape != (cout,):
            raise ValueError(f'{node.name}: bias must be constant FP32 [Cout]')
    if (crop.op_type != 'Slice' or crop.domain not in ('', 'ai.onnx')
            or len(crop.input) < 4 or not crop.input[3]
            or len(crop.output) != 1 or crop.input[0] != node.output[0]):
        raise ValueError(f'{node.name}: requires its sole consumer to be an explicit right-crop Slice')
    actual = []
    for j, default in ((1, None), (2, None), (3, None), (4, [1])):
        if j < len(crop.input) and crop.input[j]:
            if crop.input[j] not in constants:
                raise ValueError(f'{node.name}: crop bounds must be constant')
            actual.append(constants[crop.input[j]].reshape(-1).tolist())
        else:
            actual.append(default)
    if actual != [[0], [-stride], [2], [1]]:
        raise ValueError(f'{node.name}: expected crop [0:-stride] on axis2, got {actual}')
    return cin, cout, stride, weight, bias


def phase_nodes(node, crop, cin, cout, stride, weight, bias, mode, prefix):
    nodes, initializers = [], []

    def const(name, value, dtype=None):
        name = prefix + name
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        initializers.append(numpy_helper.from_array(array, name))
        return name

    def op(kind, inputs, name, output=None, **attrs):
        out = output or prefix + name
        nodes.append(helper.make_node(kind, inputs, [out], name=prefix + name + '_node', **attrs))
        return out

    x = node.input[0]
    btc = mode == 'split-matmul-btc'
    if mode in ('split-matmul', 'split-matmul-btc'):
        # Each row corresponds to (output channel, phase). Independent channel
        # dot products retain the original two-overlap structure.
        if btc:
            # ORT MatMul prepacking only accepts constant right-hand matrix B.
            # This variant explicitly trades the input-layout conversion for
            # availability of backend weight packing; speed is not presumed.
            x_btc = op('Transpose', [x], 'x_btc', perm=[0, 2, 1])
            current_w = const('current_w', weight[:, :, :stride].reshape(cin, cout * stride))
            previous_w = const('previous_w', weight[:, :, stride:].reshape(cin, cout * stride))
            current = op('MatMul', [x_btc, current_w], 'current')
            previous_unshifted = op('MatMul', [x_btc, previous_w], 'previous_unshifted')
        else:
            current_w = const('current_w', weight[:, :, :stride].transpose(1, 2, 0).reshape(cout * stride, cin))
            previous_w = const('previous_w', weight[:, :, stride:].transpose(1, 2, 0).reshape(cout * stride, cin))
            current = op('MatMul', [current_w, x], 'current')
            previous_unshifted = op('MatMul', [previous_w, x], 'previous_unshifted')
        # Shift the previous dot result by exactly one feature frame. L=1 is
        # supported: the crop leaves the single leading zero frame.
        pads = const('previous_pads', [0, 1, 0, 0, 0, 0] if btc else [0, 0, 1, 0, 0, 0], np.int64)
        previous_padded = op('Pad', [previous_unshifted, pads], 'previous_padded', mode='constant')
        starts = const('previous_starts', [0], np.int64)
        ends = const('previous_ends', [-1], np.int64)
        axes = const('time_axis', [1] if btc else [2], np.int64)
        steps = const('slice_step', [1], np.int64)
        previous = op('Slice', [previous_padded, starts, ends, axes, steps], 'previous')
        phase = op('Add', [current, previous], 'phase')
    elif mode == 'two-tap-conv':
        # Causal Conv1D tap0 reads n-1; tap1 reads n. Its channel reduction order
        # can differ from the two separate GEMMs, so full numerical parity must
        # be checked even though the real-arithmetic operator is unchanged.
        packed = np.empty((cout * stride, cin, 2), dtype=np.float32)
        packed[:, :, 0] = weight[:, :, stride:].transpose(1, 2, 0).reshape(cout * stride, cin)
        packed[:, :, 1] = weight[:, :, :stride].transpose(1, 2, 0).reshape(cout * stride, cin)
        packed_w = const('two_tap_w', packed)
        phase = op('Conv', [x, packed_w], 'phase', kernel_shape=[2], strides=[1],
                   pads=[1, 0], dilations=[1], group=1)
    else:
        raise ValueError(mode)

    # BCT: [B,Cout*s,T] -> [B,Cout,s,T] -> [B,Cout,T,s] -> [B,Cout,T*s].
    # BTC: [B,T,Cout*s] -> [B,T,Cout,s] -> [B,Cout,T,s] -> [B,Cout,T*s].
    # The batch dimension is copied with ONNX zero-shape semantics; time stays
    # dynamic throughout. No sample normalization or output delay is inserted.
    phase_shape = const('phase_shape', [0, -1, cout, stride] if btc else [0, cout, stride, -1], np.int64)
    phase4 = op('Reshape', [phase, phase_shape], 'phase4')
    interleaved = op('Transpose', [phase4], 'interleaved', perm=[0, 2, 1, 3] if btc else [0, 1, 3, 2])
    output_shape = const('output_shape', [0, cout, -1], np.int64)
    output = crop.output[0]
    if bias is None:
        op('Reshape', [interleaved, output_shape], 'output', output=output)
    else:
        unbiased = op('Reshape', [interleaved, output_shape], 'unbiased')
        bias_bct = const('bias_bct', bias.reshape(1, cout, 1))
        op('Add', [unbiased, bias_bct], 'output', output=output)
    return nodes, initializers


def rewrite(model, mode='split-matmul', stages=None):
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode}')
    model = copy.deepcopy(model)
    constants = constant_map(model)
    consumers = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node)
    graph_outputs = {v.name for v in model.graph.output}
    replacements, removed, added_init, report = {}, set(), [], []
    convs = [n for n in model.graph.node if n.op_type == 'ConvTranspose' and n.domain in ('', 'ai.onnx')]
    selected = set(range(1, len(convs) + 1)) if stages is None else set(stages)
    if not selected or not selected.issubset(set(range(1, len(convs) + 1))):
        raise ValueError(f'Stages must be a nonempty subset of 1..{len(convs)}')
    used_names = {v for n in model.graph.node for v in (*n.input, *n.output)}
    for stage, node in enumerate(convs, 1):
        if stage not in selected:
            continue
        if len(node.output) != 1 or node.output[0] in graph_outputs or len(consumers[node.output[0]]) != 1:
            raise ValueError(f'{node.name}: untrimmed output must have exactly one consumer')
        crop = consumers[node.output[0]][0]
        cin, cout, stride, weight, bias = validate_pair(node, crop, constants)
        prefix = f'ncc_up_{stage}_{mode.replace("-", "_")}_'
        if any(name.startswith(prefix) for name in used_names):
            raise ValueError(f'Generated-name collision: {prefix}')
        nodes, init = phase_nodes(node, crop, cin, cout, stride, weight, bias, mode, prefix)
        replacements[id(node)] = nodes
        removed.add(id(crop))
        added_init.extend(init)
        report.append({'stage': stage, 'source_node': node.name, 'source_crop': crop.name,
                       'mode': mode, 'input_channels': cin, 'output_channels': cout,
                       'stride': stride, 'kernel': 2 * stride, 'right_crop': stride,
                       'bias': bias is not None, 'output_name': crop.output[0],
                       'source_weight_sha256': hashlib.sha256(weight.tobytes()).hexdigest()})
    nodes = []
    for node in model.graph.node:
        if id(node) in replacements:
            nodes.extend(replacements[id(node)])
        elif id(node) not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(added_init)
    # Dead parameters must not double model storage. Other graph nodes and
    # custom domains are retained exactly; only unused initializers are pruned.
    def graph_inputs_used(graph):
        names = {v for n in graph.node for v in n.input}
        for n in graph.node:
            for attr in n.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    names.update(graph_inputs_used(attr.g))
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested in attr.graphs:
                        names.update(graph_inputs_used(nested))
        return names
    used = graph_inputs_used(model.graph) | graph_outputs
    keep = [v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    live_values = used | {v for n in model.graph.node for v in n.output}
    value_info = [v for v in model.graph.value_info if v.name in live_values]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)
    return model, report
