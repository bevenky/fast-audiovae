"""Tiny state scheduling edge gates. Prepared for a separately authorized CPU run."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1', ORT_DISABLE_TELEMETRY='1')

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOMAIN = 'fast.audiovae.apple.state.v2'
ABSENT = object()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(fused, dilation, coefficients, parallel=ABSENT, *, packed=False,
            standalone_packed=False):
    c = 7
    vi = lambda name, shape: h.make_tensor_value_info(name, T.FLOAT, shape)
    inputs = [vi('x', [1, c, 'time'])]
    outputs = [vi('y', [1, c, 'time'])]
    names = ['x']
    attrs = dict(candidate_abi=1, channels=c)
    if not packed:
        inputs.append(vi('history', [1, c, 6 * dilation]))
        outputs.append(vi('next', [1, c, 6 * dilation]))
        names.append('history')
    used = []
    if not standalone_packed:
        names.extend(['weight', 'bias'])
        used.extend(['weight', 'bias'])
        attrs['dilation'] = dilation
    if fused or packed:
        names.extend(['alpha', 'reciprocal'])
        used.extend(['alpha', 'reciprocal'])
    if packed:
        attrs['scratch_elements'] = 256
        op = 'PackedSnakeF32' if standalone_packed else 'PackedDW7SnakeF32'
    else:
        op = 'StatefulDW7SnakeF32' if fused else 'StatefulDW7F32'
        if fused:
            attrs['packed_snake'] = 0
    if parallel is not ABSENT:
        attrs['parallel_channels'] = parallel
    node = h.make_node(op, names, [x.name for x in outputs], domain=DOMAIN, **attrs)
    arrays = [nh.from_array(coefficients[name], name) for name in used]
    return h.make_model(h.make_graph([node], 'state_edge', inputs, outputs, arrays),
                        opset_imports=[h.make_opsetid('', 20), h.make_opsetid(DOMAIN, 1)],
                        ir_version=10)


def session(model, library, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 4
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    options.register_custom_ops_library(str(library))
    value = ort.InferenceSession(model.SerializeToString(), options,
                                 providers=['CPUExecutionProvider'])
    value.disable_fallback()
    require(value.get_providers() == ['CPUExecutionProvider'], 'CPU provider changed')
    opts = value.get_session_options()
    require(opts.intra_op_num_threads == threads and opts.inter_op_num_threads == 1
            and opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL,
            'ORT thread policy changed')
    return value


def bits_equal(a, b):
    return (a.dtype == b.dtype == np.float32 and a.shape == b.shape
            and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def run(config_path, library_path, output):
    require(not output.exists(), 'Do not overwrite an edge result')
    report = dict(version='apple_thread_state_edges_v1', status='running', cpu_only=True,
                  onnxruntime=ort.__version__, session_run_calls=0, call_seconds=0.0,
                  output_bitwise_checks=0, history_bitwise_checks=0,
                  history_oracle_checks=0, input_nonmutation_checks=0,
                  constructor_rejections=0, coverage={}, context=None,
                  protocol=dict(channels=7, dilations=[1, 3, 9],
                      modes=['StatefulDW7F32', 'StatefulDW7SnakeF32'],
                      lengths=['0', '1', 'halo-1', 'halo+1'],
                      patterns=['zero', 'quiet'], history='fixed nonzero per case',
                      threads=[1, 2, 4], attributes=['absent', 0, 3],
                      design='24 old/new boundary pairs plus12 legacy-attribute calls; '
                             'threads and input patterns rotate across edges, not full Cartesian coverage',
                      expected_session_run_calls=60, call_cap=80, native_seconds_cap=2,
                      timing='Correctness-call accounting only; no performance comparison. '
                             'Session construction and native constructor queries are excluded.'))
    artifacts = {}
    coverage = collections.Counter()

    def write():
        report['coverage'] = dict(sorted(coverage.items()))
        output.write_text(json.dumps(report, indent=2) + '\n')

    def invoke(s, feed):
        require(report['session_run_calls'] < 80 and report['call_seconds'] < 2,
                'Edge execution budget exhausted')
        copies = {name: value.copy() for name, value in feed.items()}
        report['session_run_calls'] += 1
        start = time.perf_counter()
        try:
            values = s.run(None, feed)
        finally:
            report['call_seconds'] += time.perf_counter() - start
        require(report['call_seconds'] <= 2, 'Edge execution exceeded two seconds')
        require(len(values) == 2, 'Both output slots must remain present')
        require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in values),
                'Nonfinite or non-FP32 output')
        require(values[0].shape == feed['x'].shape, 'Audio tensor shape changed')
        oracle = np.concatenate([feed['history'], feed['x']], axis=2)[..., -feed['history'].shape[2]:]
        require(bits_equal(values[1], oracle), 'History is not the exact activated-input suffix')
        report['history_oracle_checks'] += 1
        require(all(bits_equal(feed[name], value) for name, value in copies.items()),
                'Input/history was mutated')
        report['input_nonmutation_checks'] += 1
        return values

    def compare(reference, actual):
        require(bits_equal(reference[0], actual[0]), 'Y differs bitwise from accepted state operator')
        require(bits_equal(reference[1], actual[1]), 'History differs bitwise from accepted state operator')
        report['output_bitwise_checks'] += 1
        report['history_bitwise_checks'] += 1

    try:
        write()
        require(ort.__version__ == '1.30.0', 'Pinned ORT1.30 required')
        config = json.loads(config_path.read_text())
        bundle = Path(config['bundle']).resolve()
        mp = bundle / 'bundle.json'
        require(sha(mp) == config['bundle_manifest_sha256'], 'Accepted manifest changed')
        manifest = json.loads(mp.read_text())
        native = manifest['native']['Darwin/arm64']
        spec = manifest['streaming']['models'][native['model']]
        matches = [x for x in spec['additional_libraries'] if x['domain'] == DOMAIN]
        require(len(matches) == 1, 'Expected one accepted state library')
        old = (bundle / matches[0]['library']).resolve()
        require(old.is_relative_to(bundle) and sha(old) == matches[0]['sha256'],
                'Accepted state library changed or escaped bundle')
        library_path = library_path.resolve()
        bp = library_path.parent / 'receipt.json'
        build = json.loads(bp.read_text())
        require(build['cpu_only'] is True and build['graph'] == config['stream_graph_sha256'],
                'Build/reference graph mismatch')
        require(build['files'].get(str(library_path)) == sha(library_path),
                'Candidate library is not authenticated by build receipt')
        source = ROOT / 'work/fast-audiovae-streaming-baseline/native/apple/streaming/state/custom_ops.cpp'
        require(build['sources']['state/custom_ops.cpp'] == sha(source), 'Candidate source changed')
        artifacts = {str(p): sha(p) for p in [Path(__file__), config_path, mp, old, library_path, bp, source]}
        report['artifacts_before'] = artifacts
        rng = np.random.default_rng(20260914)
        coeff = dict(weight=rng.normal(0, 0.1, (7, 1, 7)).astype('f'),
                     bias=np.linspace(-0.015, 0.015, 7, dtype='f'),
                     alpha=np.linspace(0.4, 2.5, 7, dtype='f'),
                     reciprocal=np.linspace(1.7, 0.3, 7, dtype='f'))
        coeff_hash = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in coeff.items()}
        report['coefficient_sha256'] = coeff_hash
        for mi, fused in enumerate((False, True)):
            for di, dilation in enumerate((1, 3, 9)):
                group = mi * 3 + di
                mode = 'fused' if fused else 'unfused'
                old_sessions = {n: session(fixture(fused, dilation, coeff), old, n) for n in (1, 2, 4)}
                new_sessions = {n: session(fixture(fused, dilation, coeff, 3), library_path, n)
                                for n in (1, 2, 4)}
                halo = 6 * dilation
                for ti, length in enumerate((0, 1, halo - 1, halo + 1)):
                    threads = (1, 2, 4)[(group + ti) % 3]
                    pattern = 'quiet' if (group + ti) % 2 else 'zero'
                    x = np.zeros((1, 7, length), dtype='f')
                    if pattern == 'quiet':
                        x[:] = rng.normal(0, 1e-7, x.shape).astype('f')
                    history = rng.uniform(0.05, 0.3, (1, 7, halo)).astype('f')
                    history[:, ::2, :] *= -1
                    feed = {'x': x, 'history': history}
                    report['context'] = dict(mode=mode, dilation=dilation, length=length,
                                             pattern=pattern, threads=threads, parallel_channels=3)
                    reference = invoke(old_sessions[threads], feed)
                    compare(reference, invoke(new_sessions[threads], feed))
                    coverage[f'{mode}/d{dilation}/edges'] += 1
                    coverage[f'positive_grain/threads{threads}'] += 1
                    coverage[f'pattern/{pattern}'] += 1
                # Reuse the last edge/reference, without another reference call.
                for attr in (ABSENT, 0):
                    label = 'absent' if attr is ABSENT else 'zero'
                    report['context']['parallel_channels'] = label
                    control = session(fixture(fused, dilation, coeff, attr), library_path, threads)
                    compare(reference, invoke(control, feed))
                    coverage[f'{label}/threads{threads}'] += 1
                    del control
                del old_sessions, new_sessions
        for fused in (False, True):
            for bad in (-1, 8, 1.5, '3'):
                report['context'] = dict(rejection='malformed attribute', fused=fused, attribute=bad)
                try:
                    session(fixture(fused, 3, coeff, bad), library_path, 2)
                except Exception as error:
                    text = str(error)
                    require('parallel_channels' in text or "Attribute name and type don't match" in text,
                            'Construction failed for an unrelated reason: ' + text)
                    report['constructor_rejections'] += 1
                else:
                    raise RuntimeError('Invalid parallel_channels was accepted')
        # Positive scheduling is forbidden on each packed implementation.
        models = [fixture(True, 3, coeff, 3, packed=True),
                  fixture(True, 3, coeff, 3, packed=True, standalone_packed=True)]
        packed_state = fixture(True, 3, coeff, 3)
        node = packed_state.graph.node[0]
        for attribute in node.attribute:
            if attribute.name == 'packed_snake':
                attribute.i = 1
        node.attribute.append(h.make_attribute('scratch_elements', 256))
        models.append(packed_state)
        for index, model in enumerate(models):
            report['context'] = dict(rejection='packed positive attribute', fixture=index)
            try:
                session(model, library_path, 2)
            except Exception as error:
                require('Channel scheduling requires unpacked' in str(error),
                        'Packed construction failed for an unrelated reason: ' + str(error))
                report['constructor_rejections'] += 1
            else:
                raise RuntimeError('Packed channel scheduling was accepted')
        require(report['session_run_calls'] == 60 and report['constructor_rejections'] == 11,
                'Incomplete edge protocol')
        require(coeff_hash == {name: hashlib.sha256(value.tobytes()).hexdigest()
                               for name, value in coeff.items()}, 'Fixture coefficient mutation')
        report['status'] = 'passed'
        report['context'] = None
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if artifacts:
            after = {path: sha(path) for path in artifacts}
            report.update(artifacts_after=after, artifacts_unchanged=after == artifacts)
            if after != artifacts:
                report.update(status='failed', error='Artifact changed during edge qualification')
        write()
    require(report['status'] == 'passed', 'Edge qualification failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'outputs/apple-streaming-promotion-v1/config-r3.json')
    parser.add_argument('--library', type=Path, default=HERE / 'build/libthread_state.dylib')
    parser.add_argument('--output', type=Path, default=HERE / 'state-edges-results.json')
    args = parser.parse_args()
    result = run(args.config.resolve(), args.library, args.output.resolve())
    print(json.dumps({key: result[key] for key in ('status', 'session_run_calls', 'call_seconds',
                     'output_bitwise_checks', 'history_bitwise_checks', 'constructor_rejections')}))
