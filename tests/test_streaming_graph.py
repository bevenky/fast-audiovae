"""CPU-only chunk-boundary tests for explicit ONNX decoder state."""
import json
from pathlib import Path
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP, helper, numpy_helper

from fast_audiovae.graph.streaming import rewrite_model
from fast_audiovae.graph.upsampling import rewrite as lower_upsampling


ROOT = Path(__file__).resolve().parents[1]


def model(nodes, arrays, channels=3, out_channels=3, info=(), domains=()):
    vi = lambda name, c: helper.make_tensor_value_info(name, TP.FLOAT, [1, c, 'T'])
    graph = helper.make_graph(nodes, 'streaming_fixture', [vi('x', channels)],
        [vi('y', out_channels)], [numpy_helper.from_array(a, n) for n, a in arrays.items()],
        value_info=list(info))
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid('', 20),
                              *[helper.make_opsetid(d, 1) for d in domains]])


def conv_fixture(dilation=3, stride=2):
    rng = np.random.default_rng(42)
    arrays = {
        'w': rng.normal(0, .07, (3, 1, 7)).astype(np.float32),
        'b': rng.normal(0, .03, 3).astype(np.float32),
        'upw': rng.normal(0, .1, (3, 2, 2*stride)).astype(np.float32),
        'upb': rng.normal(0, .03, 2).astype(np.float32),
        'tailw': rng.normal(0, .1, (3, 2, 7)).astype(np.float32),
        'tailb': rng.normal(0, .03, 3).astype(np.float32),
        'starts': np.array([0], np.int64), 'ends': np.array([-stride], np.int64),
        'axes': np.array([2], np.int64), 'steps': np.array([1], np.int64),
    }
    nodes = [helper.make_node('Conv', ['x', 'w', 'b'], ['a'], name='dw',
                             pads=[6*dilation, 0], dilations=[dilation], group=3),
        helper.make_node('Sin', ['a'], ['s']),
        helper.make_node('ConvTranspose', ['s', 'upw', 'upb'], ['up'], name='up',
                         strides=[stride], pads=[0, 0]),
        helper.make_node('Slice', ['up', 'starts', 'ends', 'axes', 'steps'], ['cropped'], name='crop'),
        helper.make_node('Conv', ['cropped', 'tailw', 'tailb'], ['y'], name='tail', pads=[6, 0])]
    return model(nodes, arrays)


def sessions(source, library=None):
    streamed, audit = rewrite_model(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if library:
        options.register_custom_ops_library(str(library))
    return (ort.InferenceSession(source.SerializeToString(), options, providers=['CPUExecutionProvider']),
            ort.InferenceSession(streamed.SerializeToString(), options, providers=['CPUExecutionProvider']), audit)


def chunked(session, audit, x, sizes):
    state = {s['input']: np.zeros(s['shape'], np.float32) for s in audit['states']}
    pieces = []
    offset = 0
    for size in sizes:
        result = session.run(None, {audit['latent_input']: x[:, :, offset:offset+size], **state})
        pieces.append(result[0])
        state = {s['input']: value for s, value in zip(audit['states'], result[1:])}
        for spec in audit['states']:
            assert state[spec['input']].shape == tuple(spec['shape'])
        offset += size
    assert offset == x.shape[-1]
    return np.concatenate(pieces, axis=2), state


class StreamingGraph(unittest.TestCase):
    def test_conv_transpose_and_dilated_conv_boundaries(self):
        x = np.random.default_rng(3).normal(0, .4, (1, 3, 31)).astype(np.float32)
        for dilation in (1, 3, 9):
            for stride in (2, 5, 6, 8):
                with self.subTest(dilation=dilation, stride=stride):
                    source = conv_fixture(dilation, stride)
                    before = source.SerializeToString()
                    base, stream, audit = sessions(source)
                    self.assertEqual(source.SerializeToString(), before)
                    expected = base.run(None, {'x': x})[0]
                    for sizes in ([31], [1]*31, [1, 2, 1, 8, 3, 16]):
                        actual, _ = chunked(stream, audit, x, sizes)
                        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
                    self.assertEqual(audit['counts'], {'conv': 2, 'conv_transpose': 1})

    def test_lowered_phase_history_and_reset(self):
        x = np.random.default_rng(4).normal(0, .4, (1, 3, 31)).astype(np.float32)
        for mode in ('split-matmul', 'split-matmul-btc', 'two-tap-conv'):
            with self.subTest(mode=mode):
                source, _ = lower_upsampling(conv_fixture(), mode=mode)
                base, stream, audit = sessions(source)
                expected = base.run(None, {'x': x})[0]
                actual, _ = chunked(stream, audit, x, [1]*31)
                repeated, _ = chunked(stream, audit, x, [1, 2, 28])
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
                np.testing.assert_allclose(repeated, expected, atol=2e-7, rtol=2e-6)

    def test_state_storage_is_fixed_and_original_weights_preserved(self):
        source = conv_fixture()
        streamed, audit = rewrite_model(source)
        self.assertEqual([x.SerializeToString() for x in source.graph.initializer],
                         [x.SerializeToString() for x in streamed.graph.initializer[:len(source.graph.initializer)]])
        self.assertEqual(len(audit['states']), 3)
        self.assertEqual(audit['state_bytes'], (3*18+3+2*6)*4)
        self.assertEqual(audit['states'][0]['shape'], [1, 3, 18])
        self.assertEqual(streamed.graph.output[0].name, 'y')
        for spec in audit['states']:
            self.assertEqual(spec['dtype'], 'float32')

    def test_bad_temporal_operations_are_rejected(self):
        for mutation in ('future', 'stride', 'attention', 'unknown_custom', 'multiple_inputs', 'dynamic_weights'):
            source = conv_fixture()
            node = source.graph.node[0]
            if mutation == 'future':
                next(a for a in node.attribute if a.name == 'pads').ints[:] = [9, 9]
            elif mutation == 'stride':
                node.attribute.append(helper.make_attribute('strides', [2]))
            elif mutation == 'attention':
                source.graph.node[1].op_type = 'Softmax'
            elif mutation == 'unknown_custom':
                source.graph.node[1].op_type = 'MysteryState'
                source.graph.node[1].domain = 'unknown'
            elif mutation == 'multiple_inputs':
                source.graph.input.append(helper.make_tensor_value_info('extra', TP.FLOAT, [1]))
            elif mutation == 'dynamic_weights':
                node.input[1] = 'x'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                rewrite_model(source)

    def test_channel_time_reordering_and_time_dependent_broadcasts_are_rejected(self):
        arrays = {'flatten': np.array([1, 1, -1], np.int64),
                  'restore': np.array([1, 3, -1], np.int64),
                  'w': np.ones((1, 1, 7), np.float32),
                  'bad_bias': np.ones((1, 3, 31), np.float32)}
        nodes = [helper.make_node('Reshape', ['x', 'flatten'], ['flat']),
                 helper.make_node('Conv', ['flat', 'w'], ['filtered'], pads=[6, 0]),
                 helper.make_node('Reshape', ['filtered', 'restore'], ['y'])]
        with self.assertRaisesRegex(ValueError, 'reorder channels and time'):
            rewrite_model(model(nodes, arrays))
        for first in (helper.make_node('Transpose', ['x'], ['bad'], perm=[2, 1, 0]),
                      helper.make_node('Concat', ['x', 'x'], ['bad'], axis=2),
                      helper.make_node('Add', ['x', 'bad_bias'], ['bad'])):
            with self.subTest(op=first.op_type), self.assertRaises(ValueError):
                rewrite_model(model([first, helper.make_node('Identity', ['bad'], ['y'])], arrays))

    def test_native_depthwise_and_phase_preserve_short_chunk_audio(self):
        record = ROOT / '.build/apple/build.json'
        libraries = [Path(json.loads(record.read_text())['library'])] if record.is_file() else []
        if not libraries:
            self.skipTest('Local Apple native library not built')
        domain = 'venky.audio.cpu'
        rng = np.random.default_rng(12)
        arrays = {'w': rng.normal(0, .1, (3, 1, 7)).astype(np.float32),
                  'b': np.array([.03, .01, -.02], np.float32),
                  'a': np.array([.3, 1., 4.], np.float32),
                  'r': np.array([2., 1., .2], np.float32),
                  'mw': rng.normal(0, .1, (6, 3)).astype(np.float32),
                  'pw': rng.normal(0, .1, (6, 3)).astype(np.float32),
                  'pb': np.array([.02, -.03, .08], np.float32)}
        for op in ('CausalDW7F32', 'CausalDW7SnakeF32', 'SnakeDW7SnakeF32'):
            with self.subTest(op=op):
                attr = dict(channels=3, dilation=9, backend=0, native_abi=1, row_batches=0)
                inputs = ['x', 'w', 'b']
                if 'Snake' in op:
                    inputs += ['a', 'r']
                    attr['require_vforce'] = 1
                if op == 'SnakeDW7SnakeF32':
                    inputs += ['a', 'r']
                nodes = [helper.make_node(op, inputs, ['d'], name='depthwise', domain=domain, **attr),
                    helper.make_node('MatMul', ['mw', 'd'], ['current']),
                    helper.make_node('MatMul', ['pw', 'd'], ['previous']),
                    helper.make_node('PhaseSumBiasInterleaveF32', ['current', 'previous', 'pb'], ['y'],
                                     domain=domain, name='phase', channels=3, stride=2,
                                     previous_shift=1, native_abi=1, row_batches=0)]
                source = model(nodes, arrays, domains=[domain])
                base, stream, audit = sessions(source, libraries[-1])
                x = rng.normal(0, .4, (1, 3, 91)).astype(np.float32)
                expected = base.run(None, {'x': x})[0]
                for sizes in ([1]*91, [2, 1, 11, 3, 74]):
                    actual, _ = chunked(stream, audit, x, sizes)
                    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)

    def test_fused_stage_contracts_are_explicit(self):
        for domain in ('fast.audiovae.stage.experimental', 'fast.audiovae.precision.stage.experimental'):
            arrays = {f'c{i}': np.zeros((3,), np.float32) for i in range(24)}
            n = helper.make_node('StageStackF32', ['x', *arrays], ['y'], domain=domain,
                                 name='stage', channels=3, native_abi=1)
            streamed, audit = rewrite_model(model([n], arrays, domains=[domain]))
            self.assertEqual(streamed.graph.node[0].op_type, 'StageStackStreamingF32')
            self.assertEqual(audit['states'][0]['shape'], [1, 3, 78])
            self.assertEqual(len(streamed.graph.node[0].input), 26)
            self.assertEqual(len(streamed.graph.node[0].output), 2)
            self.assertEqual(audit['required_streaming_operators'][0]['domain'], domain)


if __name__ == '__main__':
    unittest.main()
