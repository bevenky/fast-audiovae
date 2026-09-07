"""Minimal graph-only helpers from the original fixed Intel experiment.

No runtime session, hardware probe, array benchmark or AOCL dependency is loaded.
"""
import hashlib


def libraries():
    global onnx, helper, TensorProto
    import onnx
    from onnx import helper, TensorProto


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def matrix_nodes(model):
    initializers = {t.name: t for t in model.graph.initializer}
    overridable = {v.name for v in model.graph.input}
    declarations = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    candidates = []
    for node in model.graph.node:
        if node.op_type != 'MatMul' or node.domain not in ('', 'ai.onnx'):
            continue
        if len(node.input) != 2 or len(node.output) != 1 or node.attribute:
            continue
        w = initializers.get(node.input[0])
        v = declarations.get(node.input[1])
        if (w is None or w.name in overridable or w.data_type != TensorProto.FLOAT
                or len(w.dims) != 2 or min(w.dims) <= 0 or v is None):
            continue
        tensor = v.type.tensor_type
        dims = tensor.shape.dim
        if (tensor.elem_type != TensorProto.FLOAT or len(dims) != 3
                or not dims[0].HasField('dim_value') or dims[0].dim_value != 1
                or not dims[1].HasField('dim_value') or dims[1].dim_value != w.dims[1]):
            continue
        t = int(dims[2].dim_value) if dims[2].HasField('dim_value') else dims[2].dim_param
        if not t:
            raise ValueError('Time expression must be declared: ' + node.input[1])
        candidates.append((node, w, v, (int(w.dims[0]), int(w.dims[1]), t)))
    if not candidates:
        raise ValueError('No immutable FP32 W@X batch1 matrices found')
    return candidates
