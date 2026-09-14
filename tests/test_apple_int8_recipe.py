"""Structural checks only; native arithmetic is qualified separately on Apple."""
import copy

import onnx
from onnx import helper, TensorProto
import pytest

from fast_audiovae.recipes import apple_int8


def model():
    nodes = [helper.make_node(op, ["x", "w"], ["y" + str(i)], name=name, domain=domain,
                             **apple_int8.ATTRIBUTES)
             for i, (name, (domain, op)) in enumerate(apple_int8.TARGETS.items())]
    nodes.append(helper.make_node("Add", ["y0", "y1"], ["y"], name="unchanged"))
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2048, "T"]),
              helper.make_tensor_value_info("w", TensorProto.FLOAT, [8192, 2048])]
    graph = helper.make_graph(nodes, "test", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8192, "T"])])
    return helper.make_model(graph)


def test_rewrite_preserves_all_other_values_and_interfaces():
    source = model();before = source.SerializeToString()
    result = apple_int8.rewrite(source)
    assert source.SerializeToString() == before
    for a, b in zip(source.graph.node, result.graph.node):
        if a.name in apple_int8.TARGETS:
            expected = copy.deepcopy(a);expected.domain = apple_int8.DOMAIN;expected.op_type = "FirstPairInt8"
            assert b.SerializeToString() == expected.SerializeToString()
        else:
            assert b.SerializeToString() == a.SerializeToString()
    for field in ("input", "output", "initializer", "value_info"):
        assert [x.SerializeToString() for x in getattr(source.graph, field)] == [x.SerializeToString() for x in getattr(result.graph, field)]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "dimensions", "operator", "domain"])
def test_rewrite_rejects_unqualified_graph(problem):
    source = model()
    if problem == "missing":
        del source.graph.node[0]
    elif problem == "duplicate":
        source.graph.node.append(copy.deepcopy(source.graph.node[0]))
    elif problem == "dimensions":
        for a in source.graph.node[0].attribute:
            if a.name == "k":a.i = 1024
    elif problem == "operator":
        source.graph.node[0].op_type = "OtherOp"
    else:
        source.graph.node[0].domain = "unverified.domain"
    with pytest.raises(ValueError):apple_int8.rewrite(source)
