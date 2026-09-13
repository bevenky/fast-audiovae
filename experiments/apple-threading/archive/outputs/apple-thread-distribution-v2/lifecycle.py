"""Bounded lifecycle correctness, with no timing comparison or codec execution."""
import os
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import argparse
import ctypes as c
import json
from pathlib import Path
import time
import experiment as e
import micro


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(output):
    require(not output.exists(), 'Do not overwrite lifecycle evidence')
    report = dict(version='apple_paired_lifecycle_edges_v1', status='running', cpu_only=True,
                  onnxruntime=e.ort.__version__, native_calls=0, native_seconds=0.0,
                  matrix_equivalents=0.0, checks=[], cores={},
                  scope='Original constants; direct serial and grain31 lifecycle checks at M1/M2, '
                        'duplicate/missing/out-of-range tasks, abort recovery, one ORT1-worker paired call. '
                        'At most15 full-matrix equivalents and5s counted API calls; no benchmark. '
                        'Construction via create is counted; library loading/session construction is excluded.')
    artifacts = {}
    handles = []

    def write():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def call(fn, *args, units=0.0):
        require(report['matrix_equivalents'] + units <= 15 and report['native_seconds'] < 5,
                'Lifecycle call budget exhausted')
        report['native_calls'] += 1
        report['matrix_equivalents'] += units
        start = time.perf_counter()
        try:
            return fn(*args)
        finally:
            report['native_seconds'] += time.perf_counter() - start
            require(report['native_seconds'] <= 5, 'Lifecycle calls exceeded5seconds')

    def check(ok, label):
        require(ok, label)
        report['checks'].append(label)

    def equal(a, b, label):
        check(a.shape == b.shape and a.dtype == b.dtype == e.np.float32
              and e.np.isfinite(a).all() and e.np.isfinite(b).all()
              and e.np.array_equal(a.view('u4'), b.view('u4')), label)

    pointer = lambda value: value.ctypes.data_as(c.POINTER(c.c_float))
    P, Z, F = c.c_void_p, c.c_size_t, c.POINTER(c.c_float)

    def bind(lib, name, result, args):
        fn = getattr(lib, name); fn.restype = result; fn.argtypes = args
        return fn

    try:
        write()
        require(e.ort.__version__ == '1.30.0', 'ORT1.30 required')
        receipt = e.BUILD / 'receipt.json'
        build = json.loads(receipt.read_text())
        require(build['cpu_only'] is True and build['graph'] == e.CFG['stream_graph_sha256'],
                'Build baseline mismatch')
        paths = [Path(__file__), Path(e.__file__), Path(micro.__file__), receipt,
                 e.BUNDLE / 'bundle.json', e.BUNDLE / e.SPEC['model']]
        require(e.sha(e.BUNDLE / 'bundle.json') == e.CFG['bundle_manifest_sha256'],
                'Accepted manifest changed')
        for key in ('sources', 'files'):
            for path, expected in build[key].items():
                require(e.sha(path) == expected, 'Build artifact changed: ' + path)
                paths.append(Path(path))
        artifacts = {str(path): e.sha(path) for path in paths}
        report['artifacts_before'] = artifacts
        full = e.graph(); nodes = e.pair_nodes(full)
        constants = {value.name: value for value in full.graph.initializer}
        rng = e.np.random.default_rng(20260914)
        inputs = {2: rng.normal(0, .2, (1, 2048, 2)).astype('f'),
                  1: rng.normal(0, 1e-7, (1, 2048, 1)).astype('f')}
        references = {}
        for label, node, library, prefix in (
                ('current', nodes[0], 'libdist_xsmm_core.dylib', 'av_libxsmm_panel'),
                ('previous', nodes[1], 'libdist_multitile_core.dylib', 'av_multitile')):
            lib = c.CDLL(str(e.BUILD / library))
            create = bind(lib, prefix + '_create', P, [Z, Z, Z, F, Z])
            destroy = bind(lib, prefix + '_destroy', None, [P])
            run = bind(lib, prefix + '_run', c.c_int, [P, Z, F, Z, F, Z])
            error = bind(lib, prefix + '_error', c.c_char_p, [])
            weight = e.np.ascontiguousarray(e.numpy_helper.to_array(constants[node.input[1]]))
            require(weight.dtype == e.np.float32 and weight.shape == (8192, 2048), 'Weight shape changed')
            handle = call(create, 2048, 8192, 2, pointer(weight), weight.size)
            require(handle, 'Create failed: ' + str(error()))
            handles.append((destroy, handle))
            previous = label == 'previous'
            begin = bind(lib, prefix + '_begin', c.c_int if previous else P,
                         [P, Z, F, Z, F, Z] + ([Z] if previous else []))
            if previous:
                task = bind(lib, prefix + '_task', c.c_int, [P, Z])
                finish = bind(lib, prefix + '_finish', c.c_int, [P, c.c_int])
                task_count = bind(lib, prefix + '_task_count', Z, [P])
                stat = bind(lib, prefix + '_stat', c.c_uint64, [P, c.c_int])
                check(call(stat, handle, 11) == 64, label + '/expected64channelalignment')
            else:
                task = bind(lib, prefix + '_tile', c.c_int, [P, Z, Z])
                finish = bind(lib, prefix + '_finish', c.c_int, [P])
                abort = bind(lib, prefix + '_abort', None, [P])
            active = None

            def start(x, y):
                nonlocal active
                args = [handle, x.shape[-1], pointer(x), x.size, pointer(y), y.size]
                result = call(begin, *(args + ([31 * 64] if previous else [])))
                require(result == 0 if previous else bool(result), 'Begin failed: ' + str(error()))
                active = handle if previous else result
                if previous:
                    check(call(task_count, active) == 5, label + '/grain31fiveTasks')

            def tile(index, units=None):
                first = index * 31
                width = min(31, 128 - first) if index < 5 else 1
                args = [active, index] if previous else [active, first, width]
                return call(task, *args, units=width / 128 if units is None else units)

            def end(commit):
                nonlocal active
                old = active; active = None
                if previous:
                    return call(finish, old, int(commit))
                return call(finish, old) if commit else call(abort, old)

            try:
                for length, x in inputs.items():
                    ref = e.np.empty((1, 8192, length), dtype='f')
                    require(call(run, handle, length, pointer(x), x.size, pointer(ref), ref.size, units=1) == 0,
                            'Serial reference failed: ' + str(error()))
                    references[(label, length)] = ref
                x = inputs[2]; x_before = x.copy()
                sentinel = e.np.full((1, 8192, 2), .375, dtype='f')
                for failure in ('duplicate', 'missing', 'out_of_range'):
                    y = sentinel.copy(); start(x, y)
                    if failure == 'duplicate':
                        require(tile(0) == 0, 'First partial tile failed')
                        check(tile(0, units=0) != 0, label + '/duplicateRejected')
                        end(False)
                    elif failure == 'missing':
                        check(end(True) != 0, label + '/missingRejected')
                    else:
                        check(tile(5, units=0) != 0, label + '/outOfRangeRejected')
                        end(False)
                    if previous or failure != 'duplicate':
                        equal(y, sentinel, label + '/' + failure + '/noPublication')
                    else:
                        equal(y[:, 31 * 64:, :], sentinel[:, 31 * 64:, :],
                              label + '/abortLeavesUnscheduledColumnsUntouched')
                # Each valid call follows failed/aborted work on the same handle.
                for length, x in inputs.items():
                    y = e.np.empty((1, 8192, length), dtype='f'); keep = x.copy()
                    start(x, y)
                    for index in (4, 2, 0, 3, 1):
                        require(tile(index) == 0, 'Recovery tile failed: ' + str(error()))
                    check(end(True) == 0, label + '/recoveryCommit/M' + str(length))
                    equal(y, references[(label, length)], label + '/grain31Bitwise/M' + str(length))
                    equal(x, keep, label + '/inputUnchanged/M' + str(length))
                equal(inputs[2], x_before, label + '/failureInputsUnchanged')
                report['cores'][label] = dict(status='passed', failures=['duplicate', 'missing', 'out_of_range'],
                                              recovery_lengths=[2, 1], task_order=[4, 2, 0, 3, 1])
            finally:
                if active is not None:
                    # All calls here are synchronous on one caller, so no worker
                    # can still be using the lease during failure cleanup.
                    old = active; active = None
                    call(finish, old, 0) if previous else call(abort, old)
            write()
        session = e.session(micro.model(full, 'paired', grain=31), threads=1)
        opts = session.get_session_options()
        check(opts.intra_op_num_threads == 1 and opts.inter_op_num_threads == 1,
              'oneORTworkerPolicy')
        values = call(session.run, None, {'x': inputs[2]}, units=2)
        for label, value in zip(('current', 'previous'), values):
            equal(value, references[(label, 2)], label + '/oneWorkerPairedBitwise')
        check(len(values) == 2, 'bothPairedOutputsPresent')
        check(report['matrix_equivalents'] == 10 + 62 / 128, 'exactMatrixWorkAccounting')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        for destroy, handle in reversed(handles):
            # Destroy is lifecycle cleanup, outside the compute budget.
            destroy(handle)
        if artifacts:
            after = {path: e.sha(path) for path in artifacts}
            report.update(artifacts_after=after, artifacts_unchanged=after == artifacts)
            if after != artifacts:
                report.update(status='failed', error='Artifacts changed during lifecycle checks')
        write()
    require(report['status'] == 'passed', report.get('error', 'Lifecycle checks failed'))
    print(json.dumps({key: report[key] for key in ('status', 'native_calls', 'native_seconds',
                                                  'matrix_equivalents', 'artifacts_unchanged')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=e.HERE / 'lifecycle-results.json')
    main(parser.parse_args().output)
