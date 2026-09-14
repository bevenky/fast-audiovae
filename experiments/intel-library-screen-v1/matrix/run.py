#!/usr/bin/env python3
"""Short pair-matrix screen using captured real second-projection activations."""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

# Set before importing any numerical library. The candidate also rejects a
# oneDNN build with a threaded runtime or a GPU runtime at compile time.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ONEDNN_VERBOSE'] = '0'
os.environ['DNNL_VERBOSE'] = '0'

import numpy as np

MODES = {0: 'oneMKL_current_separate_prepare', 1: 'oneMKL_shared_prepare',
         2: 'oneDNN_MatMul_cached_weights', 3: 'oneDNN_BRGeMM_panel32',
         4: 'oneDNN_BRGeMM_panel64'}
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ptr(array):
    return array.ctypes.data_as(FLOAT_PTR)


def comparison(actual, expected):
    if actual.shape != expected.shape:
        return {'bitwise_equal': False, 'shape_mismatch': True}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {'bitwise_equal': False, 'finite': False,
                'different_values': int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                'max_abs_error': None, 'rms_error': None}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {'bitwise_equal': bool(finite and np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
            'finite': finite, 'different_values': int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
            'max_abs_error': float(np.max(np.abs(difference))),
            'rms_error': float(np.sqrt(np.mean(difference * difference)))}


def setup_library(path):
    library = ctypes.CDLL(str(path))
    library.ims_error.argtypes = []
    library.ims_error.restype = ctypes.c_char_p
    library.ims_version.argtypes = []
    library.ims_version.restype = ctypes.c_char_p
    library.ims_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, FLOAT_PTR, FLOAT_PTR]
    library.ims_create.restype = ctypes.c_void_p
    library.ims_destroy.argtypes = [ctypes.c_void_p]
    library.ims_destroy.restype = None
    library.ims_run.argtypes = [ctypes.c_void_p, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR]
    library.ims_run.restype = ctypes.c_int
    library.ims_run_phases.argtypes = library.ims_run.argtypes + [ctypes.POINTER(ctypes.c_double)]
    library.ims_run_phases.restype = ctypes.c_int
    library.ims_info.argtypes = [ctypes.c_void_p]
    library.ims_info.restype = ctypes.c_char_p
    for name in ('ims_weight_bytes', 'ims_scratch_bytes'):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    return library


def run_shape(library, core, arrays, t, budget_seconds, warmups, rounds):
    x = arrays[f'input_{t}']
    weights = [arrays['weight0'], arrays['weight1']]
    expected = [arrays[f'expected{index}_{t}'] for index in range(2)]
    m, k = weights[0].shape
    if (m, k, t) not in ((3072, 1024, 8), (3072, 1024, 16)):
        raise ValueError('Unqualified projection shape')
    if weights[1].shape != (m, k) or x.shape != (k, t):
        raise ValueError('Capture dimensions do not match')
    if any(value.shape != (m, t) for value in expected):
        raise ValueError('Captured expected outputs have wrong shape')
    plans, reports = {}, {}
    spent = 0.
    calls = 0
    outputs = [np.empty((m, t), dtype=np.float32), np.empty((m, t), dtype=np.float32)]

    def invoke(mode, value=x, phases=None):
        nonlocal spent, calls
        if spent >= budget_seconds:
            raise TimeoutError('Completed native-call timing budget exhausted')
        begin = time.perf_counter_ns()
        if phases is None:
            status = library.ims_run(plans[mode], ptr(value), ptr(outputs[0]), ptr(outputs[1]))
        else:
            status = library.ims_run_phases(plans[mode], ptr(value), ptr(outputs[0]), ptr(outputs[1]), phases)
        elapsed = (time.perf_counter_ns() - begin) / 1e9
        spent += elapsed
        calls += 1
        if status:
            raise RuntimeError(library.ims_error().decode())
        return elapsed

    try:
        for mode, label in MODES.items():
            begin = time.perf_counter()
            plan = library.ims_create(os.fsencode(core), m, k, t, mode, ptr(weights[0]), ptr(weights[1]))
            reports[mode] = {'name': label, 'setup_seconds': time.perf_counter() - begin}
            if not plan:
                reports[mode].update(status='unavailable', reason=library.ims_error().decode())
                if mode == 0:
                    raise RuntimeError('Current precision-core baseline could not be created')
                continue
            plans[mode] = plan
            reports[mode].update(implementation=library.ims_info(plan).decode(),
                                 retained_weight_bytes=library.ims_weight_bytes(plan),
                                 onednn_scratchpad_bytes=library.ims_scratch_bytes(plan))
            invoke(mode)
            checks = [comparison(value, reference) for value, reference in zip(outputs, expected)]
            reports[mode]['captured_output_checks'] = checks
            reports[mode]['status'] = 'qualified' if all(check['bitwise_equal'] for check in checks) else 'rejected'
            if mode == 0 and reports[mode]['status'] != 'qualified':
                raise RuntimeError('Core disagrees with captured graph: refusing mismatched baseline timing')

        # Two small deterministic preparation checks, using the same real
        # weights. Keep these outside timed repetitions; their calls count in
        # the budget. They cover zero columns and ties in nearest-even rounding.
        tie = np.broadcast_to(((np.arange(k, dtype=np.int32) % 17) - 8).astype(np.float32)[:, None] + .5,
                              (k, t)).copy()
        tie[0, :] = 127.
        tie[:, 1] = 0.
        probes = {'all_zero': np.zeros_like(x), 'rounding_ties_and_zero_column': tie}
        for name, probe in probes.items():
            invoke(0, probe)
            reference = [value.copy() for value in outputs]
            for mode in plans:
                if reports[mode]['status'] != 'qualified':
                    continue
                invoke(mode, probe)
                checks = [comparison(value, target) for value, target in zip(outputs, reference)]
                reports[mode].setdefault('preparation_checks', {})[name] = checks
                if not all(check['bitwise_equal'] for check in checks):
                    reports[mode]['status'] = 'rejected'
        active = [mode for mode in plans if reports[mode]['status'] == 'qualified']
        for mode in active:
            for _ in range(warmups):
                invoke(mode)
        samples = {mode: [] for mode in active}
        for index in range(rounds):
            order = active if index % 2 == 0 else list(reversed(active))
            for mode in order:
                samples[mode].append(invoke(mode))
        base = statistics.median(samples[0])
        for mode in active:
            median = statistics.median(samples[mode])
            reports[mode].update(pair_ms_median=median * 1000,
                                 pair_ms_samples=[value * 1000 for value in samples[mode]],
                                 speedup_vs_current=base / median,
                                 reduction_percent_vs_current=(1. - median / base) * 100.)
            phases = (ctypes.c_double * 2)()
            invoke(mode, phases=phases)
            reports[mode]['one_call_attribution_us'] = {'prepare_and_transpose': phases[0],
                                                       'remaining_including_GEMM_and_scatter': phases[1]}
        return {'M': m, 'K': k, 'T': t, 'packet_ms': t * 5,
                'completed_call_seconds': spent, 'completed_calls': calls,
                'status': 'complete', 'methods': list(reports.values())}
    except Exception as exception:
        return {'M': m, 'K': k, 'T': t, 'packet_ms': t * 5,
                'completed_call_seconds': spent, 'completed_calls': calls,
                'status': 'stopped', 'reason': str(exception), 'methods': list(reports.values())}
    finally:
        for plan in plans.values():
            library.ims_destroy(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-metadata', type=Path, required=True)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--budget-seconds-per-shape', type=float, default=1.)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Intel Linux host required')
    cpuinfo = Path('/proc/cpuinfo').read_text()
    if 'GenuineIntel' not in cpuinfo or 'avx512_vnni' not in cpuinfo:
        raise RuntimeError('Verified Intel AVX512 VNNI host required')
    if not (0 < args.budget_seconds_per_shape <= 1.) or args.warmups != 2 or args.rounds != 3:
        raise ValueError('This short screen is fixed to two warmups, three rounds, at most one second per shape')
    os.sched_setaffinity(0, {args.cpu})
    receipt = json.loads(args.build.read_text())
    if receipt['status'] != 'built':
        raise ValueError('Build did not pass')
    for file, expected in receipt['pins'].items():
        if sha(file) != expected:
            raise ValueError('Build artifact changed: ' + file)
    library = setup_library(receipt['library'])
    if library.ims_version().decode() != '3.13.2':
        raise RuntimeError('This screen was prepared for oneDNN 3.13.2')
    with np.load(args.capture, allow_pickle=False) as capture:
        arrays = {name: np.ascontiguousarray(capture[name]) for name in capture.files}
    required = ['weight0', 'weight1', 'input_8', 'input_16',
                'expected0_8', 'expected1_8', 'expected0_16', 'expected1_16']
    for name in required:
        if name not in arrays or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError('Missing or invalid FP32 capture: ' + name)
    result = {'version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'onednn': library.ims_version().decode(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'cpu_model': next(line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                                if line.startswith('model name')),
              'inputs_sha256': sha(args.capture), 'metadata_sha256': sha(args.capture_metadata),
              'core_sha256': sha(args.core), 'build_sha256': sha(args.build),
              'shapes': {name: list(arrays[name].shape) for name in required},
              'scope': 'Second projection pair only. CPU, one thread. Exact existing INT8 quantization/scales. '
                       'All runtime preparation, transpose, allocations, GEMM, offset correction, output scatter '
                       'and dequantization included. Persistent weight packing and primitive creation excluded. '
                       'No decoder RTF or broad quality claim.',
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an earlier result')
    for t in (8, 16):
        result['results'].append(run_shape(library, args.core.resolve(), arrays, t,
                                          args.budget_seconds_per_shape, args.warmups, args.rounds))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'T': t, 'status': result['results'][-1]['status'],
                          'completed_call_seconds': result['results'][-1]['completed_call_seconds']}), flush=True)
        if result['results'][-1]['status'] != 'complete':
            break
    print(json.dumps({'result': str(args.output)}))


if __name__ == '__main__':
    main()
