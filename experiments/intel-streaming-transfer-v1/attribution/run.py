#!/usr/bin/env python3
"""One short Intel stage-attribution pass; tensors remain in memory."""
import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ('abi', 'fields', 'enabled', 'active', 'prepare_calls', 'prepare_ns',
          'prepare_failures', 'row_calls', 'row_ns', 'row_failures', 'gemm_calls',
          'gemm_ns', 'nested_gemm_calls', 'nested_gemm_ns', 'outside_row_gemm_calls',
          'outside_row_gemm_ns', 'clock_errors', 'resolution_errors', 'control_errors',
          'resolved', 'snake_calls', 'snake_ns', 'snake_failures', 'fx_calls', 'fx_ns', 'fx_failures')
NAMES = ('sp_stage_c256', 'upsample_stage_c128', 'sp_stage_c64', 'sp_stage_c32')
UID = 'bn_in_00151_1818'
BUDGET_NS = 1_000_000_000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pieces(counts, enclosing):
    require(counts['nested_gemm_ns'] <= counts['row_ns'], 'Nested GEMM exceeds row time')
    result = {'prepare_ns': counts['prepare_ns'], 'gemm_ns': counts['nested_gemm_ns'],
              'row_remainder_ns': counts['row_ns'] - counts['nested_gemm_ns'],
              'snake_ns': counts['snake_ns'], 'fp32_pointwise_ns': counts['fx_ns'],
              'other_stage_and_dispatch_ns': enclosing - counts['prepare_ns'] - counts['row_ns']
              - counts['snake_ns'] - counts['fx_ns']}
    require(min(result.values()) >= 0, 'Overlapping native durations invalidate attribution')
    return result


def expected_counts(name, input_shape, attributes):
    require(attributes.get('segments', 1) == 1, 'One stage segment required')
    tile = attributes['tile_time']
    require(tile > 0 and input_shape[0] == 1, 'Expected one stream and positive tile')
    duration = input_shape[2] * (2 if name == 'upsample_stage_c128' else 1)
    tiles = math.ceil(duration / tile)
    if name == 'upsample_stage_c128':
        return dict(prepare_calls=4 * tiles, row_calls=5 * tiles, gemm_calls=5 * tiles,
                    snake_calls=6 * tiles, fx_calls=0)
    if name == 'sp_stage_c256':
        return dict(prepare_calls=3 * tiles, row_calls=3 * tiles, gemm_calls=3 * tiles,
                    snake_calls=6 * tiles, fx_calls=0)
    return dict(prepare_calls=0, row_calls=0, gemm_calls=0, snake_calls=6 * tiles, fx_calls=3 * tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.build.read_text())
    require(config['status'] == 'built', 'Successful shim build required')
    if not args.worker:
        require(not os.environ.get('LD_PRELOAD'), 'Start without an existing LD_PRELOAD')
        env = dict(os.environ)
        env.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                        'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT', 'NUMEXPR_NUM_THREADS')})
        env.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                   NVIDIA_VISIBLE_DEVICES='void', LD_PRELOAD=config['shim'], **config['resolvers'])
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                               '--build', str(args.build.resolve()), '--output', str(args.output.resolve())],
                              env=env).returncode

    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'threads': 1, 'cpu': 0, 'cpu_only': True, 'packet_ms': 80,
              'native_call_budget_seconds': 1, 'native_ns': 0, 'calls': [], 'regions': [],
              'scope': 'Four unchanged fused stages, one packet after two warm packets; '
                       'two warmups and three paired off/on calls per stage. No tensor values saved.',
              'limits': 'Diagnostic timers include wrapper overhead. GEMM is nested inside row work. '
                        'The remainder combines depthwise, copying, allocation, epilogues and ORT dispatch. '
                        'Off calls still contain disabled shim wrappers. This is not an RTF benchmark.'}
    save = lambda: (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        require(sys.platform.startswith('linux') and 'GenuineIntel' in Path('/proc/cpuinfo').read_text(),
                'Verified Intel Linux host required')
        require(0 in os.sched_getaffinity(0), 'CPU0 unavailable')
        os.sched_setaffinity(0, {0})
        require(os.environ.get('LD_PRELOAD') == config['shim'], 'Incorrect preload')
        for key, value in config['resolvers'].items():
            require(os.environ.get(key) == value, 'Incorrect resolver ' + key)
        pins = {**config['pins'], str(args.build.resolve()): sha(args.build)}
        for path, digest in pins.items():
            require(sha(path) == digest, 'Pinned file changed: ' + path)
        report['artifact_sha256'] = pins
        import numpy as np
        import onnx
        import onnxruntime as ort
        require(ort.__version__ == config['ort'] == '1.29.0', 'ORT1.29.0 required')
        report['ort'] = ort.__version__
        probe = ctypes.CDLL(config['shim'], mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        for name in ('ip_probe_initialize', 'ip_probe_reset'):
            getattr(probe, name).argtypes = []
            getattr(probe, name).restype = ctypes.c_int
        probe.ip_probe_set_enabled.argtypes = [ctypes.c_int]
        probe.ip_probe_set_enabled.restype = ctypes.c_int
        probe.ip_probe_snapshot.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        probe.ip_probe_snapshot.restype = ctypes.c_int

        def session(model):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.log_severity_level = 3
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            for library in config['libraries']:
                options.register_custom_ops_library(library)
            result = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
            result.disable_fallback()
            require(result.get_providers() == ['CPUExecutionProvider'], 'CPU provider required')
            return result

        def call(runtime, feed, phase):
            require(report['native_ns'] < BUDGET_NS, 'Native execution budget exhausted')
            start = time.perf_counter_ns()
            try:
                outputs = runtime.run(None, feed)
            finally:
                elapsed = time.perf_counter_ns() - start
                report['native_ns'] += elapsed
                report['calls'].append({'phase': phase, 'ns': elapsed})
            require(report['native_ns'] < BUDGET_NS, 'Native execution budget exceeded')
            require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in outputs),
                    'Nonfinite or non-FP32 output')
            return outputs, elapsed

        def equal(left, right):
            return len(left) == len(right) and all(a.shape == b.shape and a.dtype == b.dtype
                       and np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(left, right))

        model = onnx.load(config['graph'])
        constants = {value.name: value for value in model.graph.initializer}
        nodes = {}
        for name in NAMES:
            matches = [node for node in model.graph.node if node.name == name]
            require(len(matches) == 1, 'Missing or duplicate expected stage ' + name)
            nodes[name] = matches[0]
        captures = list(dict.fromkeys(value for node in nodes.values() for value in [*node.input, *node.output]
                                     if value and value not in constants))
        extended = copy.deepcopy(model)
        published = {value.name for value in extended.graph.output}
        for name in captures:
            if name not in published:
                extended.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
                                                                                [None, None, None]))
        full = session(extended)
        require(probe.ip_probe_initialize() == 0, 'Could not resolve original functions')
        require(probe.ip_probe_set_enabled(0) == 0, 'Could not disable probe')
        state = {value.name: np.zeros(value.shape, np.float32) for value in full.get_inputs()[1:]}
        state_names = list(state)
        with np.load(config['latents'], allow_pickle=False) as archive:
            latent = np.ascontiguousarray(archive[UID + '__z'][..., :6])
        require(latent.shape == (1, 64, 6), 'Expected six frozen latent frames')
        report['capture'] = {'uid': UID, 'warm_packets': 2, 'packet_index': 2}
        for index in range(3):
            feed = {full.get_inputs()[0].name: np.ascontiguousarray(latent[..., 2 * index:2 * index + 2]), **state}
            outputs, _ = call(full, feed, 'capture_warm' if index < 2 else 'capture_actual')
            values = dict(zip([value.name for value in full.get_outputs()], outputs))
            if index == 2:
                captured = {name: (feed[name] if name in feed else values[name]).copy() for name in captures}
            state = {name: values[name.removesuffix('_in') + '_out'] for name in state_names}
        del full, extended, latent, outputs, values, feed, state
        report['status'] = 'running'
        save()
        for name in NAMES:
            node = nodes[name]
            inputs = [value for value in node.input if value and value not in constants]
            feeds = {value: captured[value].copy() for value in inputs}
            reference = [captured[value] for value in node.output]
            graph = onnx.helper.make_graph([copy.deepcopy(node)], name,
                [onnx.helper.make_tensor_value_info(value, onnx.TensorProto.FLOAT, list(feeds[value].shape)) for value in inputs],
                [onnx.helper.make_tensor_value_info(value, onnx.TensorProto.FLOAT, list(captured[value].shape)) for value in node.output],
                [copy.deepcopy(constants[value]) for value in node.input if value in constants])
            region = onnx.helper.make_model(graph, opset_imports=list(model.opset_import), ir_version=model.ir_version)
            runtime = session(region)
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            expected = expected_counts(name, feeds[inputs[0]].shape, attrs)
            item = {'node': name, 'op': node.op_type, 'input_shapes': {key: list(value.shape) for key, value in feeds.items()},
                    'expected_calls': expected, 'rows': [], 'bitwise_checks': 0}
            report['regions'].append(item)
            for repeat in range(2):
                values, _ = call(runtime, feeds, name + '/warm')
                require(equal(values, reference), 'Extracted stage differs from complete decoder')
                item['bitwise_checks'] += 1
            for repeat in range(3):
                outputs, off_ns = call(runtime, feeds, name + '/off')
                require(equal(outputs, reference), 'Probe-off stage changed output')
                require(probe.ip_probe_reset() == 0 and probe.ip_probe_set_enabled(1) == 0, 'Cannot enable probe')
                try:
                    measured, on_ns = call(runtime, feeds, name + '/on')
                finally:
                    require(probe.ip_probe_set_enabled(0) == 0, 'Cannot disable probe')
                raw = (ctypes.c_uint64 * len(FIELDS))()
                require(probe.ip_probe_snapshot(raw, len(FIELDS)) == 0, 'Snapshot failed')
                counts = dict(zip(FIELDS, map(int, raw)))
                require(equal(measured, outputs), 'Instrumentation changed output/state bytes')
                item['bitwise_checks'] += 1
                require(counts['abi'] == 2 and counts['fields'] == len(FIELDS) and counts['resolved'] == 1, 'Bad probe ABI')
                require(all(counts[key] == 0 for key in ('enabled', 'active', 'prepare_failures', 'row_failures',
                        'clock_errors', 'resolution_errors', 'control_errors', 'outside_row_gemm_calls',
                        'outside_row_gemm_ns', 'snake_failures', 'fx_failures')), 'Attribution failure counters')
                require(all(counts[key] == value for key, value in expected.items()), 'Unexpected function interception counts')
                require(counts['gemm_calls'] == counts['nested_gemm_calls'] and counts['gemm_ns'] == counts['nested_gemm_ns'],
                        'Nested matrix accounting mismatch')
                item['rows'].append({'repeat': repeat, 'off_ns': off_ns, 'on_ns': on_ns, 'counters': counts,
                                     'exclusive': pieces(counts, on_ns)})
                save()
            total = sum(row['on_ns'] for row in item['rows'])
            item['summary'] = {'off_median_ms': statistics.median(row['off_ns'] for row in item['rows']) / 1e6,
                               'on_median_ms': statistics.median(row['on_ns'] for row in item['rows']) / 1e6,
                               'exclusive_percent': {key: 100 * sum(row['exclusive'][key] for row in item['rows']) / total
                                                     for key in item['rows'][0]['exclusive']}}
            save()
            del runtime
        report['libraries_mapped'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                             if '/' in line and '.so' in line})
        require('torch' not in sys.modules, 'Unexpected PyTorch import')
        for path, digest in pins.items():
            require(sha(path) == digest, 'Artifact changed during diagnostic: ' + path)
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'native_seconds': report['native_ns'] / 1e9,
                      'regions': [{key: row[key] for key in ('node', 'summary', 'bitwise_checks')}
                                  for row in report['regions']]}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
