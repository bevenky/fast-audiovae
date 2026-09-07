"""Static Reshape identity proofs; no runtime or model checkpoint required."""
from __future__ import annotations

import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from fast_audiovae.graph.elementwise import Graph


def fixture(before, target, *, after=None, allowzero=None, dynamic=False,
            overridable=False, target_node=False, extra_attributes=None):
    attributes = dict(extra_attributes or {})
    if allowzero is not None:
        attributes["allowzero"] = allowzero
    reshape = helper.make_node("Reshape", ["x", "target"], ["y"], **attributes)
    tensor = numpy_helper.from_array(np.asarray(target), "target")
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, before)]
    initializers, nodes = [], []
    if dynamic or overridable:
        inputs.append(helper.make_tensor_value_info("target", tensor.data_type, list(tensor.dims)))
    if not dynamic:
        if target_node:
            nodes.append(helper.make_node("Constant", [], ["target"], value=tensor))
        else:
            initializers.append(tensor)
    nodes.append(reshape)
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, before if after is None else after)
    model = helper.make_model(helper.make_graph(nodes, "reshape_identity", inputs, [output], initializers),
                              opset_imports=[helper.make_opsetid("", 20)], ir_version=10)
    return Graph(model, "."), reshape


class IdentityReshape(unittest.TestCase):
    def prove(self, before, target, **kwargs):
        graph, reshape = fixture(before, np.asarray(target, dtype=np.int64), **kwargs)
        return graph.identity_reshape(reshape)

    def test_nonzero_targets_have_same_semantics_for_both_allowzero_modes(self):
        for allowzero in (None, 0, 1):
            for target in ([1, 3, -1], [1, -1, 7], [-1, 3, 7], [1, 3, 7]):
                with self.subTest(allowzero=allowzero, target=target):
                    self.assertTrue(self.prove([1, 3, 7], target, allowzero=allowzero))
            self.assertTrue(self.prove([1, 3, "T"], [1, 3, -1], allowzero=allowzero))
            self.assertTrue(self.prove(["B", 3, 7], [-1, 3, 7], allowzero=allowzero))

    def test_zero_copy_is_permitted_only_with_allowzero_disabled(self):
        for allowzero in (None, 0):
            self.assertTrue(self.prove([1, 3, "T"], [0, 3, -1], allowzero=allowzero))
            self.assertTrue(self.prove([1, 3, "T"], [0, 0, 0], allowzero=allowzero))
        for target in ([0, 3, -1], [1, 0, -1], [0, 3, 7], [0, 0, 0]):
            with self.subTest(target=target):
                self.assertFalse(self.prove([1, 3, 7], target, allowzero=1))
        # Even matching annotations do not turn a literal zero into a copy.
        self.assertFalse(self.prove([0, 3, 7], [0, 3, 7], allowzero=1))

    def test_annotations_do_not_prove_channel_or_batch_preservation(self):
        for target in ([1, 1, -1], [3, 1, -1], [1, 7, 3], [1, 3, 21]):
            with self.subTest(target=target):
                self.assertFalse(self.prove([1, 3, 7], target, allowzero=1))
        self.assertFalse(self.prove(["B", 3, "T"], [1, 3, -1], allowzero=1))
        self.assertFalse(self.prove([1, "C", "T"], [1, 3, -1], allowzero=1))
        self.assertFalse(self.prove([1, 3, "T"], [1, 3, -1], after=[1, 3, "Other"], allowzero=1))

    def test_invalid_shape_values_and_rank_are_rejected(self):
        for target in ([1, -1, -1], [1, 3, -2], [1, 3], [[1, 3, -1]], [1, 3, 7, 1]):
            with self.subTest(target=target):
                self.assertFalse(self.prove([1, 3, 7], target, allowzero=1))
        self.assertFalse(self.prove([3, 7], [1, 3, 7], allowzero=1))
        for dtype in (np.int32, np.float32):
            graph, node = fixture([1, 3, 7], np.array([1, 3, -1], dtype=dtype), allowzero=1)
            self.assertFalse(graph.identity_reshape(node))

    def test_dynamic_and_overridable_targets_are_not_constants(self):
        self.assertFalse(self.prove([1, 3, "T"], [1, 3, -1], allowzero=1, dynamic=True))
        self.assertFalse(self.prove([1, 3, "T"], [1, 3, -1], allowzero=1, overridable=True))
        self.assertTrue(self.prove([1, 3, "T"], [1, 3, -1], allowzero=1, target_node=True))

    def test_nonconstant_expression_target_is_not_assumed(self):
        graph, node = fixture([1, 3, "T"], np.array([1, 3, -1], np.int64), dynamic=True, allowzero=1)
        graph.model.graph.node.insert(0, helper.make_node("Identity", ["dynamic_shape"], ["target"]))
        graph.model.graph.input[1].name = "dynamic_shape"
        self.assertFalse(Graph(graph.model, ".").identity_reshape(node))

    def test_invalid_attributes_and_node_arity_are_rejected(self):
        for allowzero in (-1, 2, 1.0, "1"):
            with self.subTest(allowzero=allowzero):
                self.assertFalse(self.prove([1, 3, 7], [1, 3, -1], allowzero=allowzero))
        self.assertFalse(self.prove([1, 3, 7], [1, 3, -1], extra_attributes={"unknown": 1}))
        graph, node = fixture([1, 3, 7], np.array([1, 3, -1], np.int64), allowzero=1)
        del node.output[:]
        self.assertFalse(graph.identity_reshape(node))
        graph, node = fixture([1, 3, 7], np.array([1, 3, -1], np.int64), allowzero=1)
        node.domain = "unrelated"
        self.assertFalse(graph.identity_reshape(node))


if __name__ == "__main__":
    unittest.main()
