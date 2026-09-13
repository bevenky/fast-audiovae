"""Small sweep lifecycle gates. Does not run either imported benchmark."""
import os
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
import argparse
import ctypes as ct
import json
from pathlib import Path
import time
import common as c
import micro


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(output):
    require(not output.exists(), 'Do not overwrite lifecycle evidence')
    report = dict(version='apple_sweep_lifecycle_edges_v1', status='running', cpu_only=True,
                  onnxruntime=c.ort.__version__, native_calls=0, native_seconds=0.0,
                  matrix_equivalents=0.0, checks=[],
                  scope='Current original weight via direct sweep API; M8 quiet/M16 signed, '
                        'grain320 with final192 channels, failure cleanup and recovery; '
                        'one frozen and one paired ORT1-worker M16 call cover both weights. '
                        'Cap10 full-matrix equivalents and5s counted API calls. No benchmark. '
                        'Library/session construction and final destruction are excluded; create is counted.')
    artifacts = {}; handle = None; active = False; destroy = finish = None

    def write():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def call(fn, *args, units=0.0):
        require(report['matrix_equivalents'] + units <= 10 and report['native_seconds'] < 5,
                'Lifecycle budget exhausted')
        report['matrix_equivalents'] += units; report['native_calls'] += 1
        start = time.perf_counter()
        try:
            return fn(*args)
        finally:
            report['native_seconds'] += time.perf_counter() - start
            require(report['native_seconds'] <= 5, 'Lifecycle calls exceeded5seconds')

    def check(ok, label):
        require(ok, label); report['checks'].append(label)

    def equal(a, b, label):
        check(a.shape == b.shape and a.dtype == b.dtype == c.np.float32
              and c.np.isfinite(a).all() and c.np.isfinite(b).all()
              and c.np.array_equal(a.view('u4'), b.view('u4')), label)

    ptr = lambda value: value.ctypes.data_as(ct.POINTER(ct.c_float))
    P, Z, F = ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_float)
    try:
        write(); require(c.ort.__version__ == '1.30.0', 'ORT1.30 required')
        artifacts = c.artifact_receipt()
        receipt = json.loads((c.BUILD / 'receipt.json').read_text())
        require(receipt['cpu_only'] is True and receipt['graph'] == c.e.CFG['stream_graph_sha256'],
                'Build graph mismatch')
        archive = json.loads((c.HERE / 'archive-receipt.json').read_text())
        for path, expected in receipt['sources'].items():
            resolved = archive['source_mapping'].get(path, {}).get('archived_path', path)
            require(c.e.sha(resolved) == expected, 'Build source changed: ' + resolved)
            artifacts[resolved] = expected
        for path in (Path(__file__), Path(micro.__file__)):
            artifacts[str(path)] = c.e.sha(path)
        report['artifacts_before'] = artifacts
        full = c.e.graph(); current, previous = c.pair_nodes(full)
        constants = {v.name: v for v in full.graph.initializer}
        weight = c.np.ascontiguousarray(c.e.numpy_helper.to_array(constants[current.input[1]]))
        require(weight.dtype == c.np.float32 and weight.shape == (3072, 1024), 'Original weight shape changed')
        lib = ct.CDLL(str(c.BUILD / 'libdist3_sweep_core.dylib'))

        def bind(name, result, args):
            fn = getattr(lib, 'av_sweep_' + name); fn.restype = result; fn.argtypes = args; return fn

        create = bind('create', P, [Z, Z, Z, F, Z]); destroy = bind('destroy', None, [P])
        error = bind('error', ct.c_char_p, []); stat = bind('stat', ct.c_uint64, [P, ct.c_int])
        run = bind('run', ct.c_int, [P, Z, F, Z, F, Z])
        begin = bind('begin', ct.c_int, [P, Z, F, Z, F, Z, Z])
        task_count = bind('task_count', Z, [P]); task = bind('task', ct.c_int, [P, Z])
        finish = bind('finish', ct.c_int, [P, ct.c_int])
        handle = call(create, 1024, 3072, 16, ptr(weight), weight.size)
        require(handle, 'Create failed: ' + str(error()))
        step = call(stat, handle, 11)
        check(step > 0 and 320 % step == 0 and 3072 % step == 0, 'grain320NstepAligned')
        rng = c.np.random.default_rng(20260914)
        inputs = {8: rng.normal(0, 1e-7, (1, 1024, 8)).astype('f'),
                  16: rng.normal(0, .2, (1, 1024, 16)).astype('f')}
        references = {}
        for length, x in inputs.items():
            y = c.np.empty((1, 3072, length), dtype='f')
            check(call(run, handle, length, ptr(x), x.size, ptr(y), y.size, units=1) == 0,
                  'serialReference/M' + str(length))
            references[length] = y

        def start(x, y):
            nonlocal active
            require(call(begin, handle, x.shape[-1], ptr(x), x.size, ptr(y), y.size, 320) == 0,
                    'Begin failed: ' + str(error()))
            active = True
            check(call(task_count, handle) == 10, 'grain320TenTasks')

        def end(commit):
            nonlocal active
            active = False
            return call(finish, handle, int(commit))

        x = inputs[16]; original = x.copy(); sentinel = c.np.full((1, 3072, 16), .375, dtype='f')
        for failure in ('duplicate', 'missing', 'out_of_range'):
            y = sentinel.copy(); start(x, y)
            if failure == 'duplicate':
                require(call(task, handle, 0, units=320 / 3072) == 0, 'Partial tile failed')
                check(call(task, handle, 0) != 0, 'duplicateRejected'); check(end(False) == 0, 'duplicateAbort')
            elif failure == 'missing':
                check(end(True) != 0, 'missingRejected')
            else:
                check(call(task, handle, 10) != 0, 'outOfRangeRejected'); check(end(False) == 0, 'rangeAbort')
            equal(y, sentinel, failure + '/noPublication')
        equal(x, original, 'failedLifecycleInputUnchanged')
        for length, x in inputs.items():
            y = c.np.empty((1, 3072, length), dtype='f'); keep = x.copy(); start(x, y)
            for index in reversed(range(10)):
                width = min(320, 3072 - index * 320)
                require(call(task, handle, index, units=width / 3072) == 0,
                        'Recovery tile failed: ' + str(error()))
            check(end(True) == 0, 'recoveryCommit/M' + str(length))
            equal(y, references[length], 'grain320Bitwise/M' + str(length))
            equal(x, keep, 'inputUnchanged/M' + str(length))
        counts = {str(field): call(stat, handle, field) for field in (6, 7, 9, 10)}
        check(counts == {'6': 4, '7': 7, '9': 2, '10': 21}, 'exactRunsPacksAndTileCounts')
        report['direct_counters'] = counts
        old = c.session(micro.model(full, 'frozen'), threads=1, old=True)
        paired = c.session(micro.model(full, 'paired', grain=320), threads=1)
        for session in (old, paired):
            opts = session.get_session_options()
            check(opts.intra_op_num_threads == 1 and opts.inter_op_num_threads == 1, 'oneORTworker')
        reference = call(old.run, None, {'x': inputs[16]}, units=2)
        values = call(paired.run, None, {'x': inputs[16]}, units=2)
        check(len(values) == len(reference) == 2, 'bothPairedOutputsPresent')
        equal(reference[0], references[16], 'directSerialMatchesFrozenCurrent')
        for label, actual, expected in zip(('current', 'previous'), values, reference):
            equal(actual, expected, label + '/oneWorkerPairedBitwise')
        try:
            c.session(micro.model(full, 'paired', grain=1), threads=1)
        except Exception as exc:
            check('task_channels' in str(exc) or 'grain' in str(exc).lower()
                  or 'alignment' in str(exc).lower(), 'malformedGrain1Rejected')
            report['malformed_error'] = str(exc)
        else:
            raise RuntimeError('Unaligned task_channels1 accepted')
        check(abs(report['matrix_equivalents'] - (8 + 320 / 3072)) < 1e-12, 'exactMatrixWorkAccounting')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        if active and handle:
            # Synchronous direct calls have no outstanding callbacks here.
            finish(handle, 0)
        if handle:
            destroy(handle)
        if artifacts:
            after = {path: c.e.sha(path) for path in artifacts}
            report.update(artifacts_after=after, artifacts_unchanged=after == artifacts)
            if after != artifacts:
                report.update(status='failed', error='Artifacts changed during lifecycle checks')
        write()
    require(report['status'] == 'passed', report.get('error', 'Lifecycle checks failed'))
    print(json.dumps({key: report[key] for key in ('status', 'native_calls', 'native_seconds',
                                                  'matrix_equivalents', 'artifacts_unchanged')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=c.HERE / 'lifecycle-results.json')
    main(parser.parse_args().output)
