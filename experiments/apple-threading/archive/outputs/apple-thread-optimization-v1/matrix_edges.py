"""Twelve matrix edge calls; imports micro without running its benchmark."""
import os
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import micro as m


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    require(not output.exists(), 'Do not overwrite an edge result')
    report = dict(version='apple_thread_matrix_edges_v1', status='running', cpu_only=True,
                  onnxruntime=m.ort.__version__, threads=4, session_run_calls=0,
                  call_seconds=0.0, constructor_rejections=[], cases=[],
                  protocol=dict(lengths=[1, 2], patterns=['zero', 'quiet'],
                      arms=['old_absent', 'new_explicit_zero', 'new_grain31'],
                      expected_calls=12, maximum_malformed_constructions=4,
                      native_session_seconds_cap=2,
                      counters='New core only; grain31 is five tasks covering 31+31+31+31+4 panels. '
                               'FPCR checks count callbacks, not unique workers.',
                      timing='Correctness-call accounting only. No latency comparison; '
                             'construction, counter queries and artifact hashing excluded.'))
    artifacts = {}

    def write():
        output.write_text(json.dumps(report, indent=2) + '\n')

    try:
        write()
        require(m.ort.__version__ == '1.30.0', 'Pinned ORT1.30 required')
        config = m.ROOT / 'outputs/apple-streaming-promotion-v1/config-r3.json'
        manifest = m.bundle / 'bundle.json'
        graph = m.bundle / m.spec['model']
        receipt = m.HERE / 'build/receipt.json'
        build = json.loads(receipt.read_text())
        require(sha(manifest) == m.cfg['bundle_manifest_sha256'], 'Accepted manifest changed')
        require(sha(graph) == m.cfg['stream_graph_sha256'] == build['graph'], 'Reference graph changed')
        model = m.onnx.load(graph)
        constants = {x.name: x for x in model.graph.initializer}
        nodes = [x for x in model.graph.node if x.op_type == 'LibxsmmPanelSmeWeightLeftF32']
        require(len(nodes) == 1, 'Expected one accepted panel matrix')
        node = nodes[0]
        old = m.library(node.domain, False).resolve()
        old_entry = next(x for x in m.spec['additional_libraries'] if x['domain'] == node.domain)
        require(sha(old) == old_entry['sha256'], 'Old bridge changed')
        paths = [Path(__file__), Path(m.__file__), config, manifest, graph, receipt, old]
        for key in ('libxsmm_panel/native.cpp', 'libxsmm_panel/ort_ops.cpp'):
            source = m.ROOT / 'work/fast-audiovae-streaming-baseline/native/apple/streaming' / key
            require(sha(source) == build['sources'][key], 'Candidate source changed')
            paths.append(source)
        for name in ('libthread_xsmm_core.dylib', 'libthread_xsmm_ort.dylib',
                     'libapple_xsmm_panel_dependency.dylib'):
            path = (m.HERE / 'build' / name).resolve()
            require(sha(path) == build['files'][str(path)], 'Candidate runtime changed')
            paths.append(path)
        artifacts = {str(path): sha(path) for path in paths}
        report['artifacts_before'] = artifacts
        report['node'] = node.name

        core = ctypes.CDLL(str(m.HERE / 'build/libthread_xsmm_core.dylib'))
        counter_names = dict(panels='av_libxsmm_panel_api_calls',
                             parallel_calls='av_libxsmm_panel_parallel_calls',
                             tasks='av_libxsmm_panel_parallel_tasks',
                             fpcr_checks='av_libxsmm_panel_parallel_fpcr_checks')
        counters = {key: getattr(core, value) for key, value in counter_names.items()}
        for function in counters.values():
            function.argtypes = []
            function.restype = ctypes.c_uint64

        def snapshot():
            return {key: int(function()) for key, function in counters.items()}

        def fixture(value):
            value_model = m.fixture(node, constants)
            value_model.graph.node[0].attribute.append(m.h.make_attribute('parallel_panels', value))
            return value_model

        sessions = dict(old_absent=m.session(m.fixture(node, constants), node.domain, 4, False),
                        new_explicit_zero=m.session(fixture(0), node.domain, 4),
                        new_grain31=m.session(fixture(31), node.domain, 4))
        for session in sessions.values():
            options = session.get_session_options()
            require(options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
                    and options.execution_mode == m.ort.ExecutionMode.ORT_SEQUENTIAL,
                    'ORT thread policy changed')

        rng = m.np.random.default_rng(20260914)
        for length in (1, 2):
            for pattern in ('zero', 'quiet'):
                x = m.np.zeros((1, 2048, length), dtype='f')
                if pattern == 'quiet':
                    x[:] = rng.normal(0, 1e-7, x.shape).astype('f')
                original = x.copy()
                row = dict(length=length, pattern=pattern, arms={})
                reference = None
                for arm, session in sessions.items():
                    require(report['session_run_calls'] < 12 and report['call_seconds'] < 2,
                            'Edge execution budget exhausted')
                    before = snapshot()
                    report['session_run_calls'] += 1
                    start = time.perf_counter()
                    try:
                        values = session.run(None, {node.input[0]: x})
                    finally:
                        report['call_seconds'] += time.perf_counter() - start
                    require(report['call_seconds'] <= 2, 'Edge calls exceeded two seconds')
                    after = snapshot()
                    delta = {key: after[key] - before[key] for key in counters}
                    expected = dict(panels=0 if arm == 'old_absent' else 128,
                                    parallel_calls=int(arm == 'new_grain31'),
                                    tasks=5 if arm == 'new_grain31' else 0,
                                    fpcr_checks=5 if arm == 'new_grain31' else 0)
                    require(delta == expected, 'Unexpected candidate core accounting: ' + str(delta))
                    require(len(values) == 1 and values[0].shape == (1, 8192, length)
                            and values[0].dtype == m.np.float32 and m.np.isfinite(values[0]).all(),
                            'Invalid output shape, dtype or values')
                    require(m.np.array_equal(x.view('u4'), original.view('u4')), 'Input mutated')
                    if reference is None:
                        reference = values[0]
                    require(m.np.array_equal(values[0].view('u4'), reference.view('u4')),
                            'Output differs bitwise from old serial operator')
                    row['arms'][arm] = dict(counter_delta=delta, bitwise_equal=True,
                        output_sha256=hashlib.sha256(values[0].tobytes()).hexdigest())
                report['cases'].append(row)
                write()

        for bad in (-1, 129, 1.5, '31'):
            try:
                m.session(fixture(bad), node.domain, 4)
            except Exception as error:
                message = str(error)
                require('parallel_panels' in message or "Attribute name and type don't match" in message,
                        'Constructor failed for unrelated reason: ' + message)
                report['constructor_rejections'].append(dict(attribute=bad, error=message))
            else:
                raise RuntimeError('Invalid parallel_panels attribute accepted: ' + repr(bad))
        require(report['session_run_calls'] == 12 and len(report['cases']) == 4
                and len(report['constructor_rejections']) == 4, 'Incomplete protocol')
        report.update(status='passed', output_bitwise_comparisons=8, input_nonmutation_checks=12)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if artifacts:
            after = {path: sha(path) for path in artifacts}
            report.update(artifacts_after=after, artifacts_unchanged=after == artifacts)
            if after != artifacts:
                report.update(status='failed', error='Artifact changed during edge run')
        write()
    require(report['status'] == 'passed', report.get('error', 'Edge checks failed'))
    print(json.dumps({key: report[key] for key in ('status', 'session_run_calls', 'call_seconds',
                                                  'output_bitwise_comparisons', 'artifacts_unchanged')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=m.HERE / 'matrix-edges-results.json')
    run(parser.parse_args().output)
