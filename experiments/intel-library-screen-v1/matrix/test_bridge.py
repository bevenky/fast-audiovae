import importlib.util
from pathlib import Path

import onnx
import pytest


SPEC = importlib.util.spec_from_file_location('matrix_bridge', Path(__file__).with_name('bridge.py'))
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def original():
    weights = []
    for name in BRIDGE.WEIGHTS:
        tensor = onnx.TensorProto(name=name, data_type=onnx.TensorProto.FLOAT, dims=[3072, 1024])
        weights.append(tensor)
    nodes = [onnx.helper.make_node('PackedProjectionPairF32', ['wa', 'wb', 'x'], ['first0', 'first1'],
                                  name='first_pair', domain='old.first')]
    for index, (weight, output) in enumerate(zip(BRIDGE.WEIGHTS, BRIDGE.OUTPUTS)):
        nodes.append(onnx.helper.make_node('PrecisionMatMulF32', [weight, 'view_15'], [output],
                     name='second_' + str(index), domain='old.matrix', native_abi=1, M=3072, K=1024,
                     precision_mode=8, backend=1, shards=2))
    graph = onnx.helper.make_graph(nodes, 'screen', [], [], initializer=weights)
    return onnx.helper.make_model(graph)


def test_rewrite_changes_only_second_pair():
    model = original()
    before = model.SerializeToString()
    candidate, audit = BRIDGE.rewrite(model, 4)
    assert model.SerializeToString() == before
    assert all(audit['checks'].values())
    assert len(candidate.graph.node) == len(model.graph.node) - 1
    assert candidate.graph.node[0].SerializeToString() == model.graph.node[0].SerializeToString()
    assert candidate.graph.node[1].input == [*BRIDGE.WEIGHTS, 'view_15']
    assert candidate.graph.node[1].output == list(BRIDGE.OUTPUTS)


def test_rewrite_rejects_changed_precision():
    model = original()
    for attribute in model.graph.node[1].attribute:
        if attribute.name == 'precision_mode':
            attribute.i = 16
    with pytest.raises(ValueError, match='contract changed'):
        BRIDGE.rewrite(model, 4)


def test_rewrite_rejects_different_pair_input():
    model = original()
    model.graph.node[2].input[1] = 'another_activation'
    with pytest.raises(ValueError, match='contract changed'):
        BRIDGE.rewrite(model, 4)


def test_rewrite_rejects_second_application():
    candidate, _ = BRIDGE.rewrite(original(), 4)
    with pytest.raises(ValueError, match='already present'):
        BRIDGE.rewrite(candidate, 4)
