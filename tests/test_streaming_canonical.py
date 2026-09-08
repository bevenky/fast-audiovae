"""Focused CPU checks for canonical stem geometry and precision selection guards."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP, helper, numpy_helper
import pytest

from fast_audiovae import runtime
from fast_audiovae.assets import sha256
from fast_audiovae.graph.canonical import canonicalize_stem, validate_precision_sine_backends
from fast_audiovae.prepare_streaming import prepare_streaming


def stem_model():
    weight = np.random.default_rng(41).normal(0, .1, (2048, 64)).astype(np.float32)
    return helper.make_model(helper.make_graph([
        helper.make_node('MatMul', ['w', 'x'], ['y'], name='node_conv1d_1__bct_mm')],
        'stem', [helper.make_tensor_value_info('x', TP.FLOAT, [1, 64, 'T'])],
        [helper.make_tensor_value_info('y', TP.FLOAT, [1, 2048, 'T'])],
        [numpy_helper.from_array(weight, 'w')]), ir_version=10,
        opset_imports=[helper.make_opsetid('', 20)])


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])


def test_stem_uses_two_columns_only_for_single_frame_and_crops_dummy():
    source = stem_model()
    canonical, audit = canonicalize_stem(source)
    observed = copy.deepcopy(canonical)
    observed.graph.output.extend([
        helper.make_tensor_value_info('fast_stem_padded', TP.FLOAT, [1, 64, 'P']),
        helper.make_tensor_value_info('fast_stem_matrix', TP.FLOAT, [1, 2048, 'P'])])
    original, derived = session(source), session(observed)
    for length in (1, 2, 5, 7, 249):
        x = np.random.default_rng(length).normal(0, .3, (1, 64, length)).astype(np.float32)
        actual, padded, matrix = derived.run(None, {'x': x})
        assert actual.shape == (1, 2048, length)
        assert padded.shape == (1, 64, max(2, length))
        assert matrix.shape == (1, 2048, max(2, length))
        np.testing.assert_array_equal(padded[..., :length], x)
        if length == 1:
            np.testing.assert_array_equal(padded[..., 1:], np.zeros((1, 64, 1), np.float32))
            expected = original.run(None, {'x': np.pad(x, ((0, 0), (0, 0), (0, 1)))})[0][..., :1]
        else:
            expected = original.run(None, {'x': x})[0]
        np.testing.assert_array_equal(actual, expected)
    assert audit == {'version': 1, 'stem': 'node_conv1d_1__bct_mm', 'minimum_time_columns': 2,
                     'learned_tensors_unchanged': True, 'extra_history': False}


def test_stem_preserves_original_model_weights_and_interface():
    source = stem_model()
    before = source.SerializeToString()
    derived, _ = canonicalize_stem(source)
    assert source.SerializeToString() == before
    assert derived.graph.initializer[0].SerializeToString() == source.graph.initializer[0].SerializeToString()
    assert [x.SerializeToString() for x in derived.graph.input] == [x.SerializeToString() for x in source.graph.input]
    assert [x.SerializeToString() for x in derived.graph.output] == [x.SerializeToString() for x in source.graph.output]
    with pytest.raises(ValueError, match='already canonicalized|reserved'):
        canonicalize_stem(derived)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'dimensions', 'dtype', 'attribute', 'operator', 'opset', 'collision'])
def test_invalid_stem_contracts_are_rejected(mutation):
    source = stem_model()
    if mutation == 'missing':
        source.graph.node[0].name = 'different_stem'
    elif mutation == 'duplicate':
        source.graph.node.append(copy.deepcopy(source.graph.node[0]))
    elif mutation == 'dimensions':
        source.graph.initializer[0].dims[0] = 2047
    elif mutation == 'dtype':
        source.graph.initializer[0].data_type = TP.FLOAT16
    elif mutation == 'attribute':
        source.graph.node[0].attribute.append(helper.make_attribute('unsupported', 1))
    elif mutation == 'operator':
        source.graph.node[0].op_type = 'Gemm'
    elif mutation == 'opset':
        source.opset_import[0].version = 17
    else:
        source.graph.initializer.append(numpy_helper.from_array(np.array(1, np.float32), 'fast_stem_collision'))
    with pytest.raises(ValueError):
        canonicalize_stem(source)


def precision_bundle(root):
    # A small causal graph contains the exact audited stem plus one precision
    # projection. Preparation checks graph contracts without loading native code.
    source = stem_model()
    source.graph.input[0].name = 'z'
    nodes = [helper.make_node('Conv', ['z', 'dw'], ['x'], name='dw', pads=[6, 0], group=64),
             copy.deepcopy(source.graph.node[0]),
             helper.make_node('MatMul', ['projection', 'y'], ['audio'], name='projection')]
    del source.graph.node[:]
    source.graph.node.extend(nodes)
    source.graph.output[0].CopyFrom(helper.make_tensor_value_info('audio', TP.FLOAT, [1, 1, 'T']))
    source.graph.initializer.extend([
        numpy_helper.from_array(np.full((64, 1, 7), .1, np.float32), 'dw'),
        numpy_helper.from_array(np.full((1, 2048), 1/2048, np.float32), 'projection')])
    onnx.save(source, root/'fallback.onnx')
    precision = copy.deepcopy(source)
    node = precision.graph.node[-1]
    node.domain = 'fast.audiovae.precision.matrix.experimental'
    node.op_type = 'PrecisionMatMulF32'
    precision.opset_import.append(helper.make_opsetid(node.domain, 1))
    onnx.save(precision, root/'native.onnx')
    (root/'native.so').write_bytes(b'preparation fixture only; never loaded')
    manifest = {'onnxruntime': '1.29.0', 'fallback': 'fallback.onnx',
        'fallback_sha256': sha256(root/'fallback.onnx'), 'native': {'Linux/x86_64': {
            'library': 'native.so', 'library_sha256': sha256(root/'native.so'),
            'model': 'native.onnx', 'model_sha256': sha256(root/'native.onnx'),
            'math': 'sleef_u10', 'experiment': 'test', 'tested_cpu': 'fixture'}}}
    (root/'bundle.json').write_text(json.dumps(manifest))
    return manifest


SINE_OPERATORS = [
    ('venky.audio.cpu', 'SnakeF32'),
    ('venky.audio.cpu.portable', 'SnakeF32'),
    ('venky.audio.cpu.portable', 'CausalDW7SnakeF32'),
    ('venky.audio.cpu.portable', 'SnakeDW7SnakeF32'),
    ('fast.audiovae.stage.experimental', 'StageStackF32'),
    ('fast.audiovae.precision.stage.experimental', 'StageStackF32'),
    ('fast.audiovae.precision.upsample.experimental', 'UpsampleStageF32'),
]


@pytest.mark.parametrize('domain,operator', SINE_OPERATORS)
def test_precision_sine_guard_checks_each_operator_backend(domain, operator):
    source = stem_model()
    source.graph.node.append(helper.make_node(operator, ['y'], ['sine'],
                                             domain=domain, name='tested_sine', backend=5))
    before = source.SerializeToString()
    assert validate_precision_sine_backends(source) == 1
    assert source.SerializeToString() == before
    source.graph.node[-1].attribute[0].i = 4
    with pytest.raises(ValueError, match='explicit backend 5 on tested_sine'):
        validate_precision_sine_backends(source)


def _precision_bundle_sine(root, mutation):
    manifest = precision_bundle(root)
    source = onnx.load(root/'native.onnx')
    source.graph.node[-1].output[0] = 'before_sine'
    if mutation in ('standard_sin', 'qualified_standard_sin'):
        domain = '' if mutation == 'standard_sin' else 'ai.onnx'
        node = helper.make_node('Sin', ['before_sine'], ['audio'], domain=domain, name='tested_sine')
    else:
        node = helper.make_node('SnakeF32', ['before_sine', 'alpha', 'reciprocal'], ['audio'],
            domain='venky.audio.cpu.portable', name='tested_sine',
            backend=5, channels=1, native_abi=1, row_batches=0)
        source.graph.initializer.extend([
            numpy_helper.from_array(np.ones(1, np.float32), 'alpha'),
            numpy_helper.from_array(np.ones(1, np.float32), 'reciprocal')])
        attribute = next(a for a in node.attribute if a.name == 'backend')
        if mutation in ('auto', 'avx2'):
            attribute.i = 0 if mutation == 'auto' else 4
        elif mutation == 'missing':
            node.attribute.remove(attribute)
        elif mutation == 'float':
            attribute.CopyFrom(helper.make_attribute('backend', 5.0))
        elif mutation == 'string':
            attribute.CopyFrom(helper.make_attribute('backend', '5'))
        elif mutation == 'duplicate':
            node.attribute.append(helper.make_attribute('backend', 5))
    source.graph.node.append(node)
    if node.domain and node.domain not in {op.domain for op in source.opset_import}:
        source.opset_import.append(helper.make_opsetid(node.domain, 1))
    onnx.save(source, root/'native.onnx')
    manifest['native']['Linux/x86_64']['model_sha256'] = sha256(root/'native.onnx')
    (root/'bundle.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('mutation', [
    'auto', 'avx2', 'missing', 'float', 'string', 'duplicate',
    'standard_sin', 'qualified_standard_sin',
])
def test_precision_sine_guard_failure_leaves_bundle_unpublished(tmp_path, mutation):
    _precision_bundle_sine(tmp_path, mutation)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='Canonical precision requires'):
        prepare_streaming(tmp_path, canonical_precision=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert not (tmp_path/'streaming').exists()
    assert not list(tmp_path.glob('.streaming-*'))


def test_precision_prepare_preserves_explicit_avx512_sine_selection(tmp_path):
    _precision_bundle_sine(tmp_path, 'valid')
    original = (tmp_path/'native.onnx').read_bytes()
    result = prepare_streaming(tmp_path, canonical_precision=True)
    manifest = json.loads((tmp_path/'bundle.json').read_text())
    selected = manifest['native']['Linux/x86_64']
    streaming = result['models'][selected['model']]
    for relative in (selected['model'], streaming['model']):
        derived = onnx.load(tmp_path/relative)
        assert validate_precision_sine_backends(derived) == 1
        sine = next(n for n in derived.graph.node if n.name == 'tested_sine')
        assert next(a.i for a in sine.attribute if a.name == 'backend') == 5
    assert (tmp_path/'native.onnx').read_bytes() == original


def test_precision_prepare_requires_opt_in_and_preserves_originals(tmp_path):
    original_manifest = precision_bundle(tmp_path)
    before_manifest = (tmp_path/'bundle.json').read_bytes()
    before_native = (tmp_path/'native.onnx').read_bytes()
    with pytest.raises(ValueError, match='canonical-precision'):
        prepare_streaming(tmp_path)
    assert (tmp_path/'bundle.json').read_bytes() == before_manifest
    assert not (tmp_path/'streaming').exists()
    result = prepare_streaming(tmp_path, canonical_precision=True)
    manifest = json.loads((tmp_path/'bundle.json').read_text())
    selected = manifest['native']['Linux/x86_64']
    assert (tmp_path/'native.onnx').read_bytes() == before_native
    assert selected['model'] != 'native.onnx'
    assert selected['original_full_call_model'] == 'native.onnx'
    assert selected['required_backend'] == 5 and selected['required_math_version'] == 1
    assert selected['canonical_precision'] is True
    record = result['models'][selected['model']]
    assert record['original_source_sha256'] == original_manifest['native']['Linux/x86_64']['model_sha256']
    assert record['source_sha256'] == selected['model_sha256'] == sha256(tmp_path/selected['model'])
    assert record['required_math_version'] == 1
    assert record['canonical_stem']['minimum_time_columns'] == 2
    assert 'native.onnx' not in result['models']
    assert sha256(tmp_path/'fallback.onnx') == original_manifest['fallback_sha256']


class FakeOptions:
    def __init__(self):
        self.libraries = []
    def add_session_config_entry(self, *args):
        pass
    def register_custom_ops_library(self, path):
        self.libraries.append(path)


class FakeSession:
    def __init__(self, path, sess_options, providers):
        self.path, self.options, self.providers = path, sess_options, providers
    def disable_fallback(self):
        pass
    def get_providers(self):
        return self.providers


@pytest.mark.parametrize('math_version', [None, 0, 1, 2])
def test_full_decoder_selection_requires_matching_canonical_math(tmp_path, monkeypatch, math_version):
    manifest = precision_bundle(tmp_path)
    native = manifest['native']['Linux/x86_64']
    native.update(required_backend=5, required_math_version=1, canonical_precision=True)
    (tmp_path/'bundle.json').write_text(json.dumps(manifest))
    library = SimpleNamespace(ncc_abi_version=Mock(return_value=1), ncc_selected_backend=Mock(return_value=5),
        ncc_capabilities=Mock(return_value=64), ncc_backend_available=Mock(return_value=1),
        ncc_vector_sine_available=Mock(return_value=1), ncc_snake_math_name=Mock(return_value=b'sleef_u10'))
    if math_version is not None:
        library.ncc_streaming_math_version = Mock(return_value=math_version)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(runtime.ctypes, 'CDLL', lambda path: library)
    monkeypatch.setattr(runtime.ort, '__version__', '1.29.0')
    monkeypatch.setattr(runtime.ort, 'SessionOptions', FakeOptions)
    monkeypatch.setattr(runtime.ort, 'InferenceSession', FakeSession)
    loaded, info = runtime.load_decoder(tmp_path, threads=1)
    assert info['selected'] == ('native' if math_version == 1 else 'portable_onnx')
    assert len(loaded.options.libraries) == (1 if math_version == 1 else 0)
