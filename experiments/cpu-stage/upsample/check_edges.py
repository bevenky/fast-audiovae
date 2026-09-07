"""Independent CPU edge fixtures for upsampling fusion; no timing benchmark."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

# Set these before importing NumPy, ONNX, ORT, or the shared fixture helpers.
os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
                  HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from check import fixture, candidate, session, compare


def replace(model, name, value):
    for item in model.graph.initializer:
        if item.name == name:
            item.CopyFrom(numpy_helper.from_array(np.asarray(value), name))
            return
    raise AssertionError('Missing fixture initializer: ' + name)


def value(model, name):
    return np.array(numpy_helper.to_array(next(v for v in model.graph.initializer
                                             if v.name == name)), copy=True)


def phase_fixture(kind):
    model, constants = fixture()
    rows = np.arange(256)
    current_index = rows.copy()
    previous_index = (rows + 37) % 256
    current_gain = np.where(rows % 2 == 0, 1., 3.).astype(np.float32)
    previous_gain = np.where(rows % 2 == 0, 5., 7.).astype(np.float32)
    bias = ((np.arange(128) % 5) + .25).astype(np.float32)
    if kind == 'previous_impulses':
        current_gain[:] = 0
    elif kind == 'addition_order':
        current_index[:] = 0
        previous_index[:] = 0
        current_gain[:] = np.float32(2 ** 25)
        previous_gain[:] = np.float32(-(2 ** 25))
        bias[:] = 1
    wc = np.zeros((256, 256), np.float32)
    wp = np.zeros_like(wc)
    wc[rows, current_index] = current_gain
    wp[rows, previous_index] = previous_gain
    replace(model, 'wc', wc)
    replace(model, 'wp', wp)
    replace(model, 'phase_bias', bias)
    # Every residual branch is exactly zero; the skip exposes phase output.
    for u in range(3):
        for suffix in ('dw', 'db', 'pw', 'pb', 'pb3'):
            name = f'u{u}_{suffix}'
            replace(model, name, np.zeros_like(value(model, name)))
    return model, constants, (current_index, previous_index, current_gain, previous_gain, bias)


def phase_input(kind, n):
    x = np.zeros((2, 256, n), np.float32)
    if kind == 'addition_order':
        x[:, 0, :] = 1
        return x
    # Boundary impulses, including the final previous column needed by debug.
    positions = sorted({0, max(0, n // 2 - 1), n // 2, n - 1})
    for p in positions:
        x[0, :, p] = ((np.arange(256) % 7) + 1).astype(np.float32) / 8
        x[1, :, p] = -((np.arange(256) % 11) + 1).astype(np.float32) / 16
    return x


def analytic_phase(x, spec):
    ci, pi, cg, pg, bias = spec
    current = np.take(x, ci, axis=1) * cg[None, :, None]
    previous = np.take(x, pi, axis=1) * pg[None, :, None]
    shifted = np.zeros_like(previous)
    shifted[..., 1:] = previous[..., :-1]
    phase = np.add(current, shifted).reshape(x.shape[0], 128, 2, x.shape[2])
    phase = np.add(phase, bias[None, :, None, None])
    phase = phase.transpose(0, 1, 3, 2).reshape(x.shape[0], 128, 2 * x.shape[2])
    return [current, previous, phase, phase, phase, phase]


def phase_checks(args, records):
    for kind in ('distinguishable_phases', 'previous_impulses', 'addition_order'):
        reference, constants, spec = phase_fixture(kind)
        base = session(reference, args.native_library, args.library)
        expected = {}
        for q in (64, 128, 256):
            lengths = (1, 2, 81) if kind == 'addition_order' else sorted({1, q // 2 - 1, q // 2 + 1, 79, 80, 81, 121})
            whole = session(candidate(reference, constants, q, 1, args.mode),
                            args.native_library, args.library)
            split = {p: session(candidate(reference, constants, q, p, args.mode),
                                args.native_library, args.library) for p in (3, 5)}
            for n in lengths:
                x = phase_input(kind, n)
                before = x.tobytes()
                if n not in expected:
                    expected[n] = base.run(None, {'x': x})
                    analytic = analytic_phase(x, spec)
                    for index in range(6):
                        compare(expected[n][index], analytic[index], 'independent analytic phase')
                    if kind == 'addition_order' and n > 1:
                        if not np.all(expected[n][2][..., 2:] == np.float32(1)):
                            raise AssertionError('Reference addition-order fixture is not discriminating')
                uninterrupted = whole.run(None, {'x': x})
                for parts, outputs in [(1, uninterrupted), *[(p, s.run(None, {'x': x})) for p, s in split.items()]]:
                    checks = []
                    for index, output in enumerate(outputs):
                        checks.append({'output': index,
                            'reference': compare(output, expected[n][index], kind + ' reference', True),
                            'segmentation': compare(output, uninterrupted[index], kind + ' segmentation', True)})
                    records.append({'fixture': kind, 'tile': q, 'segments': parts,
                                    'low_time': n, 'outputs': checks})
                if x.tobytes() != before:
                    raise AssertionError('Phase fixture input was mutated')


def concurrency_checks(args, records):
    reference, constants = fixture()
    base = session(reference, args.native_library, args.library)
    rng = np.random.default_rng(24193)
    inputs = [rng.normal(.01 * i, .1 + .025 * i, (1 + i % 2, 256, n)).astype(np.float32)
              for i, n in enumerate((1, 81, 121, 129))]
    expected = [base.run(None, {'x': x}) for x in inputs]
    snapshots = [x.tobytes() for x in inputs]
    for q, parts in ((64, 3), (128, 5), (256, 3)):
        # New session: differing tail-plan creation may first occur concurrently.
        split = session(candidate(reference, constants, q, parts, args.mode),
                        args.native_library, args.library)
        barrier = threading.Barrier(2)

        def run(index):
            barrier.wait(timeout=30)
            return index, split.run(None, {'x': inputs[index]})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(run, (0, 1, 2, 3, 1, 0)))
        for index, actual in outputs:
            repeated = split.run(None, {'x': inputs[index]})
            records.append({'fixture': 'different_input_shape_concurrency', 'tile': q,
                'segments': parts, 'input': index, 'shape': list(inputs[index].shape),
                'outputs': [{'output': u,
                    'reference': compare(actual[u], expected[index][u], 'concurrent reference'),
                    'serial_repeat': compare(actual[u], repeated[u], 'concurrent versus repeated', True)}
                    for u in range(6)]})
        if any(x.tobytes() != before for x, before in zip(inputs, snapshots)):
            raise AssertionError('Concurrent input mutation')


def rejection_checks(args, rejected):
    reference, constants = fixture()

    def reject(model, label):
        try:
            session(model, args.native_library, args.library)
        except Exception as error:
            rejected.append({'test': label, 'exception_type': type(error).__name__})
        else:
            raise AssertionError('Malformed candidate accepted: ' + label)

    def fresh():
        return candidate(reference, constants, 128, 3, args.mode)

    for name, dims in [('wc', (256, 255)), ('wp', (65536,)), ('phase_bias', (1, 128)),
                       ('u0_dw', (128, 7)), ('u1_ap', (128, 1)), ('u2_pw', (127, 128))]:
        bad = fresh()
        replace(bad, name, np.zeros(dims, np.float32))
        reject(bad, 'constant_shape_' + name)
    for name, nonfinite in [('wc', np.nan), ('wp', np.inf), ('phase_bias', -np.inf),
                            ('u0_db', np.nan), ('u1_ap', np.inf), ('u2_pw', np.nan)]:
        bad = fresh()
        data = value(bad, name)
        data.flat[-1] = nonfinite
        replace(bad, name, data)
        reject(bad, 'nonfinite_' + name)
    for name in ('wc', 'phase_bias', 'u0_dw'):
        bad = fresh()
        data = value(bad, name)
        keep = [copy.deepcopy(v) for v in bad.graph.initializer if v.name != name]
        del bad.graph.initializer[:]
        bad.graph.initializer.extend(keep)
        bad.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, data.shape))
        reject(bad, 'nonconstant_' + name)
    for name, attribute_value in [('segments', -1), ('segments', 65), ('tile_time', 0),
                                   ('projection_mode', -1), ('channels', 0)]:
        bad = fresh()
        for attribute in bad.graph.node[0].attribute:
            if attribute.name == name:
                attribute.i = attribute_value
        reject(bad, f'attribute_{name}_{attribute_value}')
    for name in ('segments', 'projection_isa'):
        bad = fresh()
        keep = [copy.deepcopy(v) for v in bad.graph.node[0].attribute if v.name != name]
        del bad.graph.node[0].attribute[:]
        bad.graph.node[0].attribute.extend(keep)
        reject(bad, 'missing_attribute_' + name)
    bad = fresh()
    for attribute in bad.graph.node[0].attribute:
        if attribute.name == 'tile_time':
            attribute.CopyFrom(helper.make_attribute('tile_time', '128'))
    reject(bad, 'wrong_attribute_type_tile_time')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-library', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--mode', required=True, type=int, choices=(0, 1))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence filename')
    os.sched_setaffinity(0, {0, 1})
    if os.sched_getaffinity(0) != {0, 1}:
        raise RuntimeError('Expected exactly CPUs 0 and 1')
    records, rejected = [], []
    phase_checks(args, records)
    concurrency_checks(args, records)
    rejection_checks(args, rejected)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {'status': 'complete', 'scope': 'Independent synthetic edge fixtures; no timing or full-decoder waveform validation',
        'mode': args.mode, 'gpu_used': False, 'timing_benchmark': False,
        'full_decoder_quality_validated': False, 'runtime': '1.29.0',
        'providers': ['CPUExecutionProvider'], 'threads': 2, 'affinity': sorted(os.sched_getaffinity(0)),
        'record_count': len(records), 'records': records,
        'malformed_count': len(rejected), 'malformed_cases_rejected': rejected,
        'native_sha256': sha(args.native_library), 'library_sha256': sha(args.library),
        'script_sha256': sha(__file__), 'shared_check_sha256': sha(Path(__file__).with_name('check.py')),
        'gate': 'Finite FP32/shape; phase-only reference and segmentation exact; random concurrency reference atol1e-5 rtol1e-4 and same-input serial repeats exact'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'malformed_cases_rejected')}, indent=2))


if __name__ == '__main__':
    main()
