"""Fuse only the proven BCT split-upsample phase finishing chain.

Leaves both FP32 MatMul projections unchanged. Replaces their previous-frame
shift, overlap sum, phase interleave and bias with PhaseSumBiasInterleaveF32.
The external custom-op library must implement previous_shift=1 and ordered
(current + previous_value) + bias. No inference or benchmark is run here.
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
from .shapes import prove_input, shape_declarations, _nested_used

DOMAIN = 'venky.audio.cpu'
OP = 'PhaseSumBiasInterleaveF32'


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def rewrite_model(model, expected_count=6):
    if (model.functions or model.graph.sparse_initializer
            or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for n in model.graph.node for a in n.attribute)):
        raise ValueError('Only flat graphs without local functions or sparse initializers are supported')
    model = copy.deepcopy(model)
    initializers = {t.name: t for t in model.graph.initializer}
    overridable = {v.name for v in model.graph.input}
    declarations = shape_declarations(model)
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    graph_outputs = {v.name for v in model.graph.output}
    occupied = ({n.name for n in model.graph.node} | set(initializers)
                | {v for n in model.graph.node for v in (*n.input, *n.output)})

    def initializer(name, dtype=None):
        t = initializers.get(name)
        if t is None or name in overridable or (dtype is not None and t.data_type != dtype):
            raise ValueError('missing, overridable or incorrectly typed initializer: ' + name)
        return t

    def ints(name):
        t = initializer(name, TensorProto.INT64)
        if len(t.dims) != 1:
            raise ValueError('shape/index initializer must be a vector')
        return numpy_helper.to_array(t).tolist()

    def upstream(name, op):
        node = producers.get(name)
        if node is None or node.domain not in ('', 'ai.onnx') or node.op_type != op or len(node.output) != 1:
            raise ValueError('expected standard ' + op)
        return node

    def sole_use(node, user):
        name = node.output[0]
        if name in graph_outputs or len(consumers[name]) != 1 or consumers[name][0] is not user:
            raise ValueError('intermediate has another consumer or is a graph output: ' + name)

    def check_reshape(node, wanted):
        if len(node.input) != 2 or set(attrs(node)) - {'allowzero'} or attrs(node).get('allowzero', 0) != 0:
            raise ValueError('unsupported reshape attributes/IO')
        if ints(node.input[1]) != wanted:
            raise ValueError('reshape does not prove the requested phase layout')

    replacements, remove_ids, additions = {}, set(), []
    changes, skipped = [], []
    for index, final in enumerate(model.graph.node):
        if final.domain not in ('', 'ai.onnx') or final.op_type != 'Add' or len(final.input) != 2:
            continue
        # Only inspect the final Add in our phase-interleave pattern. This
        # excludes all residual additions and the pointwise-bias candidates.
        outreshape = producers.get(final.input[0])
        if outreshape is None or outreshape.op_type != 'Reshape':
            continue
        transpose = producers.get(outreshape.input[0]) if outreshape.input else None
        if transpose is None or transpose.op_type != 'Transpose':
            continue
        try:
            if len(final.output) != 1 or final.attribute:
                raise ValueError('unsupported final Add IO/attributes')
            bias = initializer(final.input[1], TensorProto.FLOAT)
            if len(bias.dims) != 3 or bias.dims[0] != 1 or bias.dims[2] != 1 or bias.dims[1] <= 0:
                raise ValueError('bias must be fixed FP32 [1,channels,1]')
            channels = int(bias.dims[1])
            outreshape = upstream(final.input[0], 'Reshape')
            check_reshape(outreshape, [0, channels, -1])
            transpose = upstream(outreshape.input[0], 'Transpose')
            if len(transpose.input) != 1 or attrs(transpose) != {'perm': [0, 1, 3, 2]}:
                raise ValueError('not the BCT phase interleave permutation')
            phase4 = upstream(transpose.input[0], 'Reshape')
            shape = ints(phase4.input[1]) if len(phase4.input) == 2 else []
            if len(shape) != 4 or shape[0] != 0 or shape[1] != channels or shape[3] != -1:
                raise ValueError('phase shape does not match channels')
            stride = int(shape[2])
            if stride not in (2, 5, 6, 8):
                raise ValueError('unsupported stride')
            check_reshape(phase4, [0, channels, stride, -1])
            phase = upstream(phase4.input[0], 'Add')
            if len(phase.input) != 2 or phase.attribute:
                raise ValueError('unsupported overlap Add')
            current = upstream(phase.input[0], 'MatMul')
            previous_slice = upstream(phase.input[1], 'Slice')
            if len(previous_slice.input) != 5 or previous_slice.attribute:
                raise ValueError('unsupported previous projection Slice')
            if [ints(name) for name in previous_slice.input[1:]] != [[0], [-1], [2], [1]]:
                raise ValueError('previous projection must shift by one time frame')
            previous_pad = upstream(previous_slice.input[0], 'Pad')
            if len(previous_pad.input) not in (2, 3) or set(attrs(previous_pad)) - {'mode'}:
                raise ValueError('unsupported previous projection Pad')
            if attrs(previous_pad).get('mode', b'constant') != b'constant':
                raise ValueError('previous history must be zero padded')
            if ints(previous_pad.input[1]) != [0, 0, 1, 0, 0, 0]:
                raise ValueError('Pad must add exactly one leading time frame')
            if len(previous_pad.input) == 3 and previous_pad.input[2]:
                zero = numpy_helper.to_array(initializer(previous_pad.input[2], TensorProto.FLOAT))
                if zero.size != 1 or float(zero.reshape(-1)[0]) != 0.0 or bool(np.signbit(zero.reshape(-1)[0])):
                    raise ValueError('Pad constant must be FP32 positive zero')
            previous = upstream(previous_pad.input[0], 'MatMul')
            if (len(current.input) != 2 or len(previous.input) != 2 or current.attribute or previous.attribute
                    or current.input[1] != previous.input[1]):
                raise ValueError('current/previous must be direct projections of the same input')
            w0 = initializer(current.input[0], TensorProto.FLOAT)
            w1 = initializer(previous.input[0], TensorProto.FLOAT)
            if (len(w0.dims) != 2 or list(w0.dims) != list(w1.dims)
                    or w0.dims[0] != channels * stride or w0.dims[1] <= 0):
                raise ValueError('projection weights do not prove matching FP32 phase shapes')
            shape_proof, failure = prove_input(current.input[1], int(w0.dims[1]), declarations)
            if failure:
                raise ValueError(failure)
            chain = [(previous_pad, previous_slice), (previous_slice, phase),
                     (phase, phase4), (phase4, transpose), (transpose, outreshape),
                     (outreshape, final)]
            for old, user in chain:
                sole_use(old, user)
            prefix = f'ncc_phase_finish_{index}_'
            if any(name.startswith(prefix) for name in occupied):
                raise ValueError('generated-name collision')
            bias_name = prefix + 'bias'
            fused = helper.make_node(OP, [current.output[0], previous.output[0], bias_name],
                                     list(final.output), name=prefix + 'op', domain=DOMAIN,
                                     channels=channels, stride=stride, previous_shift=1,
                                     native_abi=1, row_batches=0)
            removed = [previous_pad, previous_slice, phase4, transpose, outreshape, final]
            if any(id(n) in remove_ids or id(n) in replacements for n in removed):
                raise ValueError('overlapping rewrite patterns')
            additions.append(numpy_helper.from_array(
                np.ascontiguousarray(numpy_helper.to_array(bias).reshape(channels)), bias_name))
            replacements[id(phase)] = fused
            remove_ids.update(id(n) for n in removed)
            changes.append({'output': final.output[0], 'channels': channels, 'stride': stride,
                            'current': current.output[0], 'previous_unshifted': previous.output[0],
                            'input_shape_proof': shape_proof, 'projection_weight_shape': list(w0.dims),
                            'removed_nodes': [n.name for n in (*removed, phase)],
                            'custom_node': fused.name, 'previous_shift': 1})
        except ValueError as e:
            skipped.append({'node': final.name, 'reason': str(e)})
    if expected_count is not None and len(changes) != expected_count:
        raise ValueError(f'Expected{expected_count} phase chains, got{len(changes)}. Skips: {json.dumps(skipped)}')
    if not changes:
        raise ValueError('No proven phase-finish chains found')
    nodes = []
    for node in model.graph.node:
        if id(node) in replacements:
            nodes.append(replacements[id(node)])
        elif id(node) not in remove_ids:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(additions)
    used = _nested_used(model.graph) | overridable
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    live = used | {v for n in nodes for v in n.output}
    info = [v for v in model.graph.value_info if v.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(info)
    versions = [o for o in model.opset_import if o.domain == DOMAIN]
    if versions and (len(versions) != 1 or versions[0].version != 1):
        raise ValueError('Unexpected custom domain version')
    if not versions:
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    onnx.checker.check_model(model, full_check=False)
    return model, {'changes': changes, 'skipped': skipped}


def rewrite(source, output, expected_count=6):
    source, output = Path(source), Path(output)
    result, audit = rewrite_model(onnx.load(source, load_external_data=True), expected_count)
    if any(t.data_location == TensorProto.EXTERNAL for t in result.graph.initializer):
        onnx.external_data_helper.convert_model_from_external_data(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(result, output)
    report = {'source': str(source.resolve()), 'output': str(output.resolve()),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'domain': DOMAIN, 'op': OP, 'native_abi': 1, **audit,
              'contract': 'FP32 (cur+previous_unshifted[t-1])+bias; previous=+0 at t0; chronological phase interleave',
              'scope': 'Unchanged MatMul projections; graph generation only; custom-library parity and timing required'}
    output.with_suffix('.phase.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
