"""Exercise graph dispatch with inert ORT and native handles, without inference."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fast_audiovae import runtime


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    files = {
        'decoder.onnx': b'portable source',
        'cpu.onnx': b'native source',
        'portable-stream.onnx': b'portable stream',
        'fp32-stream.onnx': b'fp32 stream',
        'int8-stream.onnx': b'int8 stream',
        'base.dylib': b'base native',
        'fp32.dylib': b'fp32 extension',
        'int8.dylib': b'int8 extension',
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    digest = lambda name: hashlib.sha256(files[name]).hexdigest()
    extension = lambda name, domain: {'library': name, 'sha256': digest(name), 'domain': domain}
    fp32 = {
        'model': 'fp32-stream.onnx', 'model_sha256': digest('fp32-stream.onnx'),
        'source_sha256': digest('cpu.onnx'), 'precision': 'FP32',
        'apple_stream_selected': {'version': 1},
        'additional_libraries': [extension('fp32.dylib', 'fp32.v1')],
    }
    int8 = copy.deepcopy(fp32)
    int8.update(model='int8-stream.onnx', model_sha256=digest('int8-stream.onnx'),
                precision='mixed_fp32_int8', apple_firstpair_int8={'threads': 1}, fp32_fallback=fp32)
    int8['additional_libraries'].append(extension('int8.dylib', 'fast.audiovae.apple.firstpair.int8.v1'))
    manifest = {
        'onnxruntime': '1.30.0', 'fallback': 'decoder.onnx',
        'native': {'Darwin/arm64': {'library': 'base.dylib', 'model': 'cpu.onnx',
                    'math': 'vforce', 'experiment': 'qualified', 'tested_cpu': 'Apple',
                    'required_cpu_features': ['sme', 'sme2']}},
        'streaming': {'version': 1, 'models': {'cpu.onnx': int8,
                      'decoder.onnx': {'model': 'portable-stream.onnx',
                                      'model_sha256': digest('portable-stream.onnx'),
                                      'source_sha256': digest('decoder.onnx')}}},
    }
    options = Mock()
    session = Mock()
    session.get_providers.return_value = ['CPUExecutionProvider']
    native = SimpleNamespace(ncc_abi_version=Mock(return_value=1),
                             ncc_selected_backend=Mock(return_value=2),
                             ncc_capabilities=Mock(return_value=32),
                             ncc_snake_math_name=Mock(return_value=b'vforce'))
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr('fast_audiovae.platforms.apple_matrix_features', lambda: {'sme': True, 'sme2': True})
    monkeypatch.setattr(runtime.ort, '__version__', '1.30.0')
    monkeypatch.setattr(runtime.ort, 'SessionOptions', lambda: options)
    create = Mock(return_value=session)
    monkeypatch.setattr(runtime.ort, 'InferenceSession', create)
    monkeypatch.setattr(runtime.ctypes, 'CDLL', Mock(return_value=native))
    retain = Mock()
    monkeypatch.setattr(runtime, '_retain_verified_library', retain)
    def load(threads=1, prefer_custom=True):
        (tmp_path / 'bundle.json').write_text(json.dumps(manifest))
        return runtime._load_session(tmp_path, threads=threads, prefer_custom=prefer_custom,
                                     prefer_packed=False, streaming=True)
    return SimpleNamespace(root=tmp_path, manifest=manifest, load=load, create=create,
                           options=options, retain=retain)


@pytest.mark.parametrize('threads,expected,precision', [
    (1, 'int8-stream.onnx', 'mixed_fp32_int8'), (2, 'fp32-stream.onnx', 'FP32'),
    (4, 'fp32-stream.onnx', 'FP32'), (8, 'fp32-stream.onnx', 'FP32'),
])
def test_worker_count_selects_graph_before_loading_extensions(bundle, threads, expected, precision):
    _, info = bundle.load(threads)
    assert info['model'] == expected
    assert info['precision'] == precision
    assert info['threads'] == threads
    assert bundle.options.intra_op_num_threads == threads
    assert bundle.create.call_args.kwargs['providers'] == ['CPUExecutionProvider']
    names = [Path(call.args[0]).name for call in bundle.retain.call_args_list]
    assert names == (['fp32.dylib', 'int8.dylib'] if threads == 1 else ['fp32.dylib'])


def test_explicit_portable_load_never_loads_int8(bundle):
    _, info = bundle.load(prefer_custom=False)
    assert info['model'] == 'portable-stream.onnx'
    assert info['selected'] == 'portable_onnx'
    assert bundle.retain.call_count == 0


@pytest.mark.parametrize('failure', ['missing', 'recursive', 'precision', 'model_hash', 'outside'])
def test_invalid_multithread_fallback_fails_before_session_creation(bundle, failure):
    entry = bundle.manifest['streaming']['models']['cpu.onnx']
    if failure == 'missing':
        del entry['fp32_fallback']
    elif failure == 'recursive':
        entry['fp32_fallback']['apple_firstpair_int8'] = {'threads': 1}
    elif failure == 'precision':
        entry['fp32_fallback']['precision'] = 'mixed_fp32_int8'
    elif failure == 'model_hash':
        entry['fp32_fallback']['model_sha256'] = '0' * 64
    else:
        entry['fp32_fallback']['model'] = '../outside.onnx'
    with pytest.raises(RuntimeError):
        bundle.load(4)
    assert bundle.create.call_count == 0
    assert bundle.retain.call_count == 0


def test_changed_int8_extension_fails_before_native_retention(bundle):
    (bundle.root / 'int8.dylib').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='differs'):
        bundle.load(1)
    assert bundle.create.call_count == 0
    assert all(Path(call.args[0]).name != 'int8.dylib' for call in bundle.retain.call_args_list)
