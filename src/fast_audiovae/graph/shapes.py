"""Shared rank, channel and liveness checks."""
import onnx
from onnx import TensorProto


def shape_declarations(model):
    declarations = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.type.HasField('tensor_type'):
            declarations[value.name] = value.type.tensor_type
    return declarations


def prove_input(name, channels, declarations):
    tensor_type = declarations.get(name)
    if tensor_type is None or tensor_type.elem_type != TensorProto.FLOAT:
        return None, 'activation lacks an explicit FP32 tensor declaration'
    if not tensor_type.HasField('shape'):
        return None, 'activation rank is undeclared'
    dims = tensor_type.shape.dim
    if len(dims) != 3:
        return None, f'activation rank must be3, got{len(dims)}'
    if not dims[0].HasField('dim_value') or dims[0].dim_value != 1:
        return None, 'batch dimension is not proven to be exactly1'
    if not dims[1].HasField('dim_value') or dims[1].dim_value != channels:
        return None, 'activation channels do not statically match weight input channels'
    if dims[2].HasField('dim_value') and dims[2].dim_value < 0:
        return None, 'negative time dimension'
    shape = [int(d.dim_value) if d.HasField('dim_value') else (d.dim_param or '?') for d in dims]
    return shape, None


def _nested_used(graph):
    used = {name for node in graph.node for name in node.input}
    used.update(value.name for value in graph.output)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                used.update(_nested_used(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for nested in attr.graphs:
                    used.update(_nested_used(nested))
    return used
