"""CPU-only Apple operator checks using synthetic inputs, without checkpoints.

Run: python -m unittest discover -s tests -p test_native_apple.py -v
Requires NumPy, ONNX and ONNX Runtime 1.29.0 plus staged ORT headers.
These fixtures establish neither whole-model quality nor decoding speed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib.util
from pathlib import Path
import platform
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "venky.audio.cpu"


def model(nodes, arrays, inputs, output_shape):
    value_info = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(nodes, "synthetic_cpu_operators",
        [value_info(name, shape) for name, shape in inputs.items()],
        [value_info("y", output_shape)],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()])
    return helper.make_model(graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid(DOMAIN, 1)], ir_version=10)


def elementwise_fixture(mode, seed, dilation, custom):
    rng = np.random.default_rng(seed)
    channels = 3
    shape = ["B", channels, "T"]
    arrays = {}
    if mode != "SnakeF32":
        arrays.update(weight=rng.normal(0, .15, (channels, 1, 7)).astype(np.float32),
                      bias=rng.normal(0, .1, channels).astype(np.float32))
    if mode != "CausalDW7F32":
        arrays.update(alpha=rng.uniform(-2, 2, channels).astype(np.float32),
                      reciprocal=rng.uniform(.2, 1.4, channels).astype(np.float32))
    if custom:
        inputs = ["x"]
        attributes = dict(channels=channels, native_abi=1, row_batches=0, backend=0)
        if mode != "SnakeF32":
            inputs.extend(["weight", "bias"])
            attributes["dilation"] = dilation
        if mode != "CausalDW7F32":
            inputs.extend(["alpha", "reciprocal"])
            attributes["require_vforce"] = 1
        nodes = [helper.make_node(mode, inputs, ["y"], domain=DOMAIN, **attributes)]
    else:
        nodes, current = [], "x"
        if mode != "SnakeF32":
            current = "y" if mode == "CausalDW7F32" else "dw"
            nodes.append(helper.make_node("Conv", ["x", "weight", "bias"], [current],
                group=channels, kernel_shape=[7], pads=[6*dilation, 0], dilations=[dilation], strides=[1]))
        if mode != "CausalDW7F32":
            arrays["alpha"] = arrays["alpha"].reshape(1, channels, 1)
            arrays["reciprocal"] = arrays["reciprocal"].reshape(1, channels, 1)
            arrays["two"] = np.array(2, np.float32)
            nodes += [helper.make_node("Mul", ["alpha", current], ["ax"]),
                      helper.make_node("Sin", ["ax"], ["sine"]),
                      helper.make_node("Pow", ["sine", "two"], ["square"]),
                      helper.make_node("Mul", ["reciprocal", "square"], ["correction"]),
                      helper.make_node("Add", [current, "correction"], ["y"])]
    return model(nodes, arrays, {"x": shape}, shape)


def phase_fixture(channels, stride, bias, custom):
    shape = ["B", channels * stride, "T"]
    arrays = {"bias": bias}
    if custom:
        nodes = [helper.make_node("PhaseSumBiasInterleaveF32", ["cur", "prev", "bias"], ["y"],
            domain=DOMAIN, channels=channels, stride=stride, previous_shift=1, native_abi=1, row_batches=0)]
    else:
        arrays.update(pads=np.array([0, 0, 1, 0, 0, 0], np.int64),
                      starts=np.array([0], np.int64), ends=np.array([-1], np.int64),
                      axes=np.array([2], np.int64), steps=np.array([1], np.int64),
                      phase_shape=np.array([0, channels, stride, -1], np.int64),
                      out_shape=np.array([0, channels, -1], np.int64),
                      bias_shape=np.array([1, channels, 1], np.int64))
        nodes = [helper.make_node("Pad", ["prev", "pads"], ["padded"], mode="constant"),
                 helper.make_node("Slice", ["padded", "starts", "ends", "axes", "steps"], ["shifted"]),
                 helper.make_node("Add", ["cur", "shifted"], ["sum"]),
                 helper.make_node("Reshape", ["sum", "phase_shape"], ["phases"]),
                 helper.make_node("Transpose", ["phases"], ["interleaved"], perm=[0, 1, 3, 2]),
                 helper.make_node("Reshape", ["interleaved", "out_shape"], ["flat"]),
                 helper.make_node("Reshape", ["bias", "bias_shape"], ["broadcast_bias"]),
                 helper.make_node("Add", ["flat", "broadcast_bias"], ["y"])]
    return model(nodes, arrays, {"cur": shape, "prev": shape}, ["B", channels, "T_out"])


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"),
                     "Native Apple ARM tests require Apple ARM hardware")
class AppleOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ort.__version__ != "1.29.0":
            raise RuntimeError("Operator acceptance requires ONNX Runtime 1.29.0")
        spec = importlib.util.spec_from_file_location("build_apple", ROOT / "tools/build_apple.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.build = module.build()
        cls.library = ctypes.CDLL(cls.build["library"])

    def session(self, graph, threads, custom=True):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if custom:
            options.register_custom_ops_library(self.build["library"])
        session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        return session

    def test_abi_and_vector_math(self):
        for name, expected in (("ncc_abi_version", 1), ("ncc_phase_finish_abi", 1),
                               ("ncc_compiled_tile", 256), ("ncc_selected_backend", 2)):
            function = getattr(self.library, name)
            function.argtypes, function.restype = [], ctypes.c_uint32
            self.assertEqual(function(), expected)
        self.library.ncc_capabilities.argtypes = []
        self.library.ncc_capabilities.restype = ctypes.c_uint64
        capabilities = self.library.ncc_capabilities()
        self.assertTrue(capabilities & 2)
        self.assertTrue(capabilities & 32)
        self.assertFalse(capabilities & 16)  # Only ORT owns worker parallelism.
        self.library.ncc_snake_math_name.argtypes = [ctypes.c_int32]
        self.library.ncc_snake_math_name.restype = ctypes.c_char_p
        self.assertEqual(self.library.ncc_snake_math_name(0), b"apple-vforce")
        self.library.ncc_phase_build_id.argtypes = []
        self.library.ncc_phase_build_id.restype = ctypes.c_char_p
        self.assertEqual(self.library.ncc_phase_build_id().decode(), self.build["build_id"])

    def test_snake_depthwise_dynamic_shapes_and_session_ownership(self):
        rng = np.random.default_rng(1001)
        for threads in (1, 4):
            for mode in ("SnakeF32", "CausalDW7F32", "CausalDW7SnakeF32"):
                for dilation in (1, 3, 9):
                    candidates = [self.session(elementwise_fixture(mode, seed, dilation, True), threads)
                                  for seed in (10, 20)]
                    references = [self.session(elementwise_fixture(mode, seed, dilation, False), threads, False)
                                  for seed in (10, 20)]
                    inputs = {time: rng.normal(0, .5, (2, 3, time)).astype(np.float32)
                              for time in (1, 7, 23, 257)}
                    for time in (257, 1, 7, 23, 257):
                        x = inputs[time]
                        before = x.copy()
                        first = None
                        for which in (0, 1, 0):
                            with self.subTest(threads=threads, mode=mode, dilation=dilation, time=time, session=which):
                                y = candidates[which].run(None, {"x": x})[0]
                                expected = references[which].run(None, {"x": x})[0]
                                np.testing.assert_allclose(y, expected, rtol=2e-5, atol=2e-6)
                                np.testing.assert_array_equal(x, before)
                                if which == 0:
                                    if first is not None:
                                        np.testing.assert_array_equal(y.view(np.uint32), first)
                                    first = y.view(np.uint32).copy()
                    # Fixed coefficients must be safe in simultaneously active sessions.
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        parallel = list(pool.map(lambda i: candidates[i].run(None, {"x": inputs[23]})[0], (0, 1)))
                    for which, y in enumerate(parallel):
                        np.testing.assert_array_equal(y, candidates[which].run(None, {"x": inputs[23]})[0])
                    changed = inputs[257].copy()
                    changed[..., 13:] += 7
                    initial = candidates[0].run(None, {"x": inputs[257]})[0]
                    altered = candidates[0].run(None, {"x": changed})[0]
                    np.testing.assert_array_equal(initial[..., :13], altered[..., :13])

    def test_phase_interleave_signed_zero_tails_and_session_ownership(self):
        rng = np.random.default_rng(1002)
        for threads in (1, 4):
            for channels in (1, 3):
                for stride in (2, 5, 6, 8):
                    biases = [rng.normal(0, .5, channels).astype(np.float32) for _ in (0, 1)]
                    biases[0][0], biases[1][0] = -0.0, -0.75
                    candidates = [self.session(phase_fixture(channels, stride, bias, True), threads) for bias in biases]
                    references = [self.session(phase_fixture(channels, stride, bias, False), threads, False) for bias in biases]
                    inputs = {}
                    for batch, time in ((1, 257), (2, 1), (1, 7), (2, 17)):
                        cur = rng.normal(0, 3, (batch, channels * stride, time)).astype(np.float32)
                        prev = rng.normal(0, 3, cur.shape).astype(np.float32)
                        cur[..., 0] = -0.0
                        prev[..., -1] = np.nan  # This unshifted value must never be consumed.
                        inputs[batch, time] = {"cur": cur, "prev": prev}
                    for batch, time in ((1, 257), (2, 1), (1, 7), (2, 17), (1, 257)):
                        feeds = inputs[batch, time]
                        first = None
                        for which in (0, 1, 0):
                            with self.subTest(threads=threads, channels=channels, stride=stride, batch=batch, time=time, session=which):
                                y = candidates[which].run(None, feeds)[0]
                                ref = references[which].run(None, feeds)[0]
                                shifted = np.zeros_like(feeds["prev"])
                                shifted[..., 1:] = feeds["prev"][..., :-1]
                                total = np.add(feeds["cur"], shifted, dtype=np.float32)
                                interleaved = total.reshape(batch, channels, stride, time).transpose(0, 1, 3, 2).reshape(batch, channels, time * stride)
                                expected = np.add(interleaved, biases[which][None, :, None], dtype=np.float32)
                                np.testing.assert_array_equal(y.view(np.uint32), expected.view(np.uint32))
                                np.testing.assert_array_equal(y.view(np.uint32), ref.view(np.uint32))
                                if which == 0:
                                    if first is not None:
                                        np.testing.assert_array_equal(y.view(np.uint32), first)
                                    first = y.view(np.uint32).copy()

    def test_malformed_attributes_and_nonconstant_weights_are_rejected(self):
        fixture = elementwise_fixture("SnakeF32", 10, 1, True)
        for attribute in fixture.graph.node[0].attribute:
            if attribute.name == "native_abi":
                attribute.i = 2
        with self.assertRaisesRegex(Exception, "ABI mismatch"):
            self.session(fixture, 1)
        fixture = elementwise_fixture("CausalDW7F32", 10, 1, True)
        fixture.graph.input.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [3, 1, 7]))
        with self.assertRaisesRegex(Exception, "non-overridable constant"):
            self.session(fixture, 1)
        fixture = phase_fixture(3, 5, np.zeros(3, np.float32), True)
        for attribute in fixture.graph.node[0].attribute:
            if attribute.name == "previous_shift":
                attribute.i = 0
        with self.assertRaisesRegex(Exception, "previous_shift=1"):
            self.session(fixture, 1)


if __name__ == "__main__":
    unittest.main()
