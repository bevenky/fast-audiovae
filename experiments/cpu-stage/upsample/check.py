"""CPU-only intermediate correctness, causal-prefix and concurrency checks."""
import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
                  HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parent
DOMAIN = 'fast.audiovae.upsample.experimental'
NATIVE = 'venky.audio.cpu.portable'


def fixture():
    spec = importlib.util.spec_from_file_location('stage_fixture', ROOT.parent / 'stage/check_stage.py')
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    model, constants = stage.fixture(128, 5)
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            if value == 'x':
                node.input[i] = 'phase'
    rng = np.random.default_rng(9821)
    initializers = [numpy_helper.from_array(rng.normal(0, .035, (256, 256)).astype(np.float32), name)
                    for name in ('wc', 'wp')]
    initializers.append(numpy_helper.from_array(rng.normal(0, .03, 128).astype(np.float32), 'phase_bias'))
    first = [helper.make_node('MatMul', ['wc', 'x'], ['current'], name='current'),
             helper.make_node('MatMul', ['wp', 'x'], ['previous'], name='previous'),
             helper.make_node('PhaseSumBiasInterleaveF32', ['current', 'previous', 'phase_bias'], ['phase'],
                              name='phase_finish', domain=NATIVE, channels=128, stride=2,
                              previous_shift=1, row_batches=0, native_abi=1)]
    nodes = first + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    model.graph.input[0].CopyFrom(helper.make_tensor_value_info('x', TensorProto.FLOAT, ['B', 256, 'T']))
    stage_outputs = [copy.deepcopy(v) for v in model.graph.output]
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info(n, TensorProto.FLOAT, ['B', 256, 'T'])
                               for n in ('current', 'previous')])
    model.graph.output.append(helper.make_tensor_value_info('phase', TensorProto.FLOAT, ['B', 128, 'T2']))
    for output in stage_outputs:
        output.type.tensor_type.shape.dim[2].dim_param = 'T2'
    model.graph.output.extend(stage_outputs)
    return model, ['wc', 'wp', 'phase_bias', *constants]


def candidate(reference, constants, q, parts, mode, debug=True):
    model = copy.deepcopy(reference)
    del model.graph.node[:]
    if not debug:
        last = copy.deepcopy(model.graph.output[-1])
        del model.graph.output[:]
        model.graph.output.append(last)
    model.graph.node.append(helper.make_node('UpsampleStageDebugF32' if debug else 'UpsampleStageF32',
        ['x', *constants], [v.name for v in model.graph.output], name='upsample_stage', domain=DOMAIN,
        channels=128, stride=2, native_abi=1, tile_time=q, segments=parts, backend=5,
        matrix_mode=0, matrix_isa=512, projection_mode=mode, projection_isa=0 if mode else 512))
    model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return model


def session(model, native, library, threads=2):
    import onnxruntime as ort
    if ort.__version__ != '1.29.0':
        raise RuntimeError('ORT 1.29.0 required')
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    so.register_custom_ops_library(str(native))
    so.register_custom_ops_library(str(library))
    result = ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])
    result.disable_fallback()
    if result.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('CPU provider required')
    return result


def compare(a, b, label, exact=False):
    if a.shape != b.shape or a.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label + ': shape, precision or finiteness')
    bits = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    if (exact and not bits) or (not exact and not np.allclose(a, b, atol=1e-5, rtol=1e-4)):
        raise AssertionError(label + ': numerical gate failed')
    return {'max_abs': float(np.abs(a.astype(np.float64) - b).max(initial=0)), 'bitwise': bits}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-library', required=True, type=Path)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--mode', type=int, choices=(0, 1), required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new evidence filename')
    os.sched_setaffinity(0, {0, 1})
    reference, constants = fixture()
    base = session(reference, a.native_library, a.library)
    rng = np.random.default_rng(6178)
    records, checks, rejected = [], [], []
    lengths = (0, 1, 2, 15, 16, 17, 31, 32, 33, 38, 39, 40, 53, 63, 64, 65, 78, 79, 80, 121, 127, 128, 129, 257)
    expected = {}
    inputs = {}
    for n in lengths:
        x = rng.normal(0, .3, (2, 256, n)).astype(np.float32)
        inputs[n] = x
        expected[n] = base.run(None, {'x': x})
    for q in (64, 128, 256):
        whole = session(candidate(reference, constants, q, 1, a.mode), a.native_library, a.library)
        uninterrupted = {n: whole.run(None, {'x': x}) for n, x in inputs.items()}
        for parts in (1, 2, 3, 5):
            model = candidate(reference, constants, q, parts, a.mode)
            split = session(model, a.native_library, a.library)
            prod = session(candidate(reference, constants, q, parts, a.mode, False), a.native_library, a.library)
            for n, x in inputs.items():
                before = x.tobytes()
                actual = split.run(None, {'x': x})
                row = {'tile': q, 'segments': parts, 'low_time': n, 'outputs': []}
                for index, y in enumerate(actual):
                    row['outputs'].append({'index': index, 'reference': compare(y, expected[n][index], 'reference'),
                                           'segmentation': compare(y, uninterrupted[n][index], 'segmentation')})
                row['production'] = compare(prod.run(None, {'x': x})[0], actual[-1], 'debug production', True)
                if before != x.tobytes():
                    raise AssertionError('Input mutated')
                records.append(row)
            x = rng.normal(0, .3, (1, 256, 257)).astype(np.float32)
            changed = x.copy()
            changed[:, :, 121:] += .25
            original = split.run(None, {'x': x})
            future = split.run(None, {'x': changed})
            for index in range(6):
                prefix = 121 if index < 2 else 242
                checks.append({'test': 'same_shape_future', 'q': q, 'segments': parts, 'output': index,
                               **compare(original[index][..., :prefix], future[index][..., :prefix], 'future prefix', True)})
            short = split.run(None, {'x': x[..., :121].copy()})
            for index in range(6):
                prefix = 121 if index < 2 else 242
                checks.append({'test': 'short_prefix', 'q': q, 'segments': parts, 'output': index,
                               **compare(original[index][..., :prefix], short[index], 'short prefix')})
            checks.append({'test': 'repeat', **compare(split.run(None, {'x': x})[-1], original[-1], 'repeat', True)})
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(lambda _: split.run(None, {'x': x})[-1], range(4)))
            for value in values:
                checks.append({'test': 'concurrency', **compare(value, original[-1], 'concurrent', True)})
            empty = prod.run(None, {'x': np.empty((0, 256, 55), np.float32)})[0]
            if empty.shape != (0, 128, 110):
                raise AssertionError('Empty batch shape')
        for name, value in [('channels', 64), ('stride', 5), ('tile_time', 65), ('segments', 0),
                            ('backend', 4), ('matrix_mode', 1), ('matrix_isa', 256),
                            ('projection_mode', 2), ('projection_isa', 1), ('native_abi', 2)]:
            bad = candidate(reference, constants, q, 2, a.mode)
            for attribute in bad.graph.node[0].attribute:
                if attribute.name == name:
                    attribute.i = value
            try:
                session(bad, a.native_library, a.library)
            except Exception:
                rejected.append({'q': q, 'attribute': name, 'value': value})
            else:
                raise AssertionError('Malformed attribute accepted: ' + name)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(status='complete', mode=a.mode, gpu_used=False, runtime='1.29.0',
                  providers=['CPUExecutionProvider'], record_count=len(records), records=records,
                  additional_checks=checks, malformed_cases_rejected=rejected,
                  native_sha256=sha(a.native_library), library_sha256=sha(a.library), script_sha256=sha(__file__),
                  gate='FP32 finite shape + atol1e-5 rtol1e-4; exact repeat/future/production/concurrency',
                  full_decoder_quality_validated=False, timing_benchmark=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'additional_checks', 'malformed_cases_rejected')}, indent=2))


if __name__ == '__main__':
    main()
