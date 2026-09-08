"""Isolated CPU matrix diagnostics. No ORT, audio, model or GPU execution.

Parent mode runs each library in a separate process. The accepted old API
creates the fixed input/reference archive before workspace variants run.
Timings are hot, single-worker native calls and include Python/ctypes overhead.
They are not decoder RTF or predictions of multiworker throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
                  HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  MKL_DYNAMIC='FALSE', OMP_DYNAMIC='FALSE')
for key in ('LD_PRELOAD', 'MKL_CBWR', 'MKL_ENABLE_INSTRUCTIONS', 'LIBXSMM_TARGET'):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parent
# Actual repeated square matrix families. Distinct contexts preserve workspace
# capacity and output epilogue used by the current fused pipeline.
CASES = [
    ('stage256_q256', 256, 256, 256, 256, 256, 256),
    ('stage256_q128', 256, 256, 128, 256, 128, 256),
    ('up128_q128_projection', 256, 256, 64, 256, 128, 256),
    ('up128_q128_residual', 128, 128, 128, 256, 128, 256),
    ('up128_q256_projection', 256, 256, 128, 256, 256, 256),
    ('up128_q256_residual', 128, 128, 256, 256, 256, 256),
    ('residual128_t64', 128, 128, 64, 256, 128, 256),
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gpu_mappings():
    p = Path('/proc/self/maps')
    names = ('libcuda.', 'libcudart.', 'libcublas.', 'libhip', 'libamdhip', 'libnvidia', 'libonnxruntime_providers_cuda')
    return sorted({line.split()[-1] for line in p.read_text().splitlines()
                   if any(n in line.lower() for n in names)}) if p.exists() else []


def worker(args):
    require(sys.platform == 'linux' and hasattr(os, 'sched_setaffinity'), 'Linux CPU affinity required')
    require(args.cpu in os.sched_getaffinity(0), 'Requested CPU outside allowed affinity')
    os.sched_setaffinity(0, {args.cpu})
    import numpy as np
    sys.path.insert(0, str(ROOT.parent / 'core'))
    from check_workspace import API
    from check_micro import pointer
    require(not gpu_mappings(), 'GPU library loaded before diagnostic')
    lib = args.library.resolve()
    require(sha(lib) == args.library_sha256, 'Library hash changed')
    is_old = args.role == 'baseline'
    api = API(lib, 'ip_' if is_old else 'ip3_', workspace=not is_old)
    require((api.lib.ip_capabilities() & 3) == 3, 'Sequential MKL and CPU/OS AVX512 VNNI required')
    require(not gpu_mappings(), 'GPU library loaded by native dependency')
    helper_hashes = {name: sha(ROOT.parent / 'core' / name) for name in ('check_workspace.py', 'check_micro.py')}
    report = {'status': 'running', 'role': args.role, 'library_sha256': sha(lib),
              'helper_sha256': helper_hashes,
              'script_sha256': sha(__file__), 'python': sys.version, 'numpy': np.__version__,
              'affinity': sorted(os.sched_getaffinity(0)), 'native_threads': 1,
              'gpu_used': False, 'ort_used': False, 'models_executed': False,
              'protocol': {'seed': 2026090803, 'warmups': args.warmups,
                           'samples': args.samples, 'calls_per_sample': args.calls,
                           'single_worker': True, 'retained_weight_plan': True,
                           'timer_includes': 'Python closure, ctypes and native call; no subtraction',
                           'run_only': 'One fixed prepared input reused; output buffer reused',
                           'prepare_only': 'Prepare call only; old input destruction is outside each timed interval',
                           'prepare_run': 'Prepare, run and old-input destruction inside one timed interval',
                           'order': 'Fixed seeded shape shuffle and per-shape category shuffle, identical across processes',
                           'limits': 'Hot repeated microcalls, one pinned CPU; cannot infer full decoder RTF or two-worker scaling'},
              'cases': []}
    require(not args.output.exists(), 'Fresh output file required')
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    inputs = {}
    reference_before = None
    if is_old:
        require(not args.reference.exists(), 'Fresh reference archive required')
        rng = np.random.default_rng(2026090803)
        for name, m, k, t, *_ in CASES:
            inputs[name + '_w'] = rng.normal(0, .15, (m, k)).astype(np.float32)
            inputs[name + '_x'] = rng.normal(0, .3, (k, t)).astype(np.float32)
            inputs[name + '_bias'] = rng.normal(0, .03, m).astype(np.float32)
            inputs[name + '_skip'] = rng.normal(0, .3, (m, t)).astype(np.float32)
    else:
        require(args.reference_sha256 and sha(args.reference) == args.reference_sha256, 'Reference archive hash mismatch')
        reference_before = sha(args.reference)
        with np.load(args.reference, allow_pickle=False) as archive:
            inputs = {key: archive[key] for key in archive.files}

    def equal(a, b, label):
        require(a.dtype == b.dtype == np.float32 and a.shape == b.shape, label + ' shape/type')
        require(np.isfinite(a).all() and np.isfinite(b).all(), label + ' nonfinite')
        require(np.array_equal(a.view(np.uint32), b.view(np.uint32)), label + ' not uint32 equal')

    rng = np.random.default_rng(7251)
    order = list(rng.permutation(len(CASES)))
    for index in order:
        name, m, k, t, max_k, max_t, max_rows = CASES[index]
        w, x, bias, skip = [np.ascontiguousarray(inputs[name + '_' + key]) for key in ('w', 'x', 'bias', 'skip')]
        pinned = {key: hashlib.sha256(value.tobytes()).hexdigest()
                  for key, value in (('weights', w), ('input', x), ('bias', bias), ('skip', skip))}
        wp, xp, bp, sp = map(pointer, (w, x, bias, skip))
        y = np.empty((m, t), np.float32)
        yp = pointer(y)
        plan = api.lib.ip_create(m, k, wp, 8, 1)
        require(bool(plan), api.error())
        workspace = None
        active = None
        try:
            if not is_old:
                workspace = api.lib.ip_workspace_create(max_k, max_t, max_rows)
                require(bool(workspace), api.error())
            def prepare():
                if is_old:
                    value = api.lib.ip_prepare(plan, xp, t)
                    if not value:
                        raise RuntimeError(api.error())
                    return value
                status = api.lib.ip_workspace_prepare(workspace, plan, xp, t)
                if status:
                    raise RuntimeError(api.error())
                return None
            def release(value):
                if value:
                    api.lib.ip_destroy_input(value)
            def run(value, residual=False, empty=False):
                if is_old:
                    status = api.lib.ip_run_rows(plan, value, yp, 0, 0 if empty else m)
                else:
                    status = api.lib.ip_workspace_run_rows(workspace, plan, yp, 0, 0 if empty else m,
                                                          bp if residual else None, sp if residual else None)
                if status:
                    raise RuntimeError(api.error())
            active = prepare()
            run(active)
            if is_old:
                inputs[name + '_raw'] = y.copy()
            expected = inputs[name + '_raw']
            equal(y, expected, name + ' raw')
            residual_reference = np.add(skip, np.add(expected, bias[:, None], dtype=np.float32), dtype=np.float32)
            if not is_old:
                run(active, True)
                equal(y, residual_reference, name + ' ordered residual')
            release(active)
            active = None
            categories = ['prepare_only', 'raw_run_only', 'raw_prepare_run', 'empty_run_control']
            if not is_old:
                categories += ['residual_run_only', 'residual_prepare_run']
            # Seed per category avoids a different role's extra categories
            # changing order of the common four categories.
            categories.sort(key=lambda c: hashlib.sha256((name + c).encode()).hexdigest())
            row = {'case': name, 'M': m, 'K': k, 'T': t,
                   'workspace_bounds': None if is_old else [max_k, max_t, max_rows],
                   'workspace_bytes': None if is_old else api.lib.ip_workspace_bytes(workspace),
                   'input_sha256': pinned, 'raw_uint32_equal': True,
                   'ordered_residual_uint32_equal': None if is_old else True,
                   'timings': {}}
            for category in categories:
                prepared_run = category.endswith('run_only') or category == 'empty_run_control'
                residual = category.startswith('residual')
                if prepared_run:
                    active = prepare()
                def one():
                    if prepared_run:
                        start = time.perf_counter_ns()
                        run(active, residual, category == 'empty_run_control')
                        return time.perf_counter_ns() - start
                    if category == 'prepare_only':
                        start = time.perf_counter_ns()
                        value = prepare()
                        elapsed = time.perf_counter_ns() - start
                        release(value)
                        return elapsed
                    start = time.perf_counter_ns()
                    value = prepare()
                    try:
                        run(value, residual)
                    finally:
                        release(value)
                    return time.perf_counter_ns() - start
                for _ in range(args.warmups):
                    one()
                samples = [sum(one() for _ in range(args.calls)) / args.calls for _ in range(args.samples)]
                release(active)
                active = None
                if category not in ('prepare_only', 'empty_run_control'):
                    equal(y, residual_reference if residual else expected, name + ' timed ' + category)
                row['timings'][category] = {'mean_ns': float(np.mean(samples)),
                    'median_ns': float(np.median(samples)), 'minimum_ns': float(np.min(samples)),
                    'maximum_ns': float(np.max(samples)), 'sample_mean_ns': samples,
                    'cv_percent': float(np.std(samples) / np.mean(samples) * 100)}
            for key, value in (('weights', w), ('input', x), ('bias', bias), ('skip', skip)):
                require(hashlib.sha256(value.tobytes()).hexdigest() == pinned[key], 'Read buffer mutated: ' + key)
            report['cases'].append(row)
            output.write_text(json.dumps(report, indent=2) + '\n')
        finally:
            if active:
                api.lib.ip_destroy_input(active)
            if workspace:
                api.lib.ip_destroy_workspace(workspace)
            api.lib.ip_destroy_plan(plan)
    if is_old:
        np.savez(args.reference, **inputs)
    else:
        require(sha(args.reference) == reference_before, 'Reference archive changed')
    require(sha(lib) == args.library_sha256, 'Library changed during diagnostic')
    require({name: sha(ROOT.parent / 'core' / name) for name in helper_hashes} == helper_hashes,
            'Diagnostic helper source changed')
    require(not gpu_mappings(), 'GPU library loaded during diagnostic')
    report.update(status='complete', reference_sha256=sha(args.reference),
                  gpu_library_mappings=gpu_mappings())
    output.write_text(json.dumps(report, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-library', type=Path)
    p.add_argument('--mkl-workspace-library', type=Path)
    p.add_argument('--vnni-workspace-library', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--cpu', type=int, default=0)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--calls', type=int, default=20)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--role', choices=('baseline', 'workspace_mkl', 'workspace_vnni'))
    p.add_argument('--library', type=Path)
    p.add_argument('--library-sha256')
    p.add_argument('--reference', type=Path)
    p.add_argument('--reference-sha256')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    require(0 <= args.warmups <= 100 and 1 <= args.samples <= 25 and 1 <= args.calls <= 500,
            'Use bounded warmups<=100, samples<=25 and calls<=500')
    if args.worker:
        require(all((args.role, args.library, args.library_sha256, args.reference, args.output)), 'Incomplete worker arguments')
        worker(args)
        return
    require(all((args.baseline_library, args.mkl_workspace_library, args.vnni_workspace_library, args.output_dir)),
            'Supply all three libraries and a fresh output directory')
    out = args.output_dir.resolve()
    require(not out.exists(), 'Fresh output directory required')
    out.mkdir(parents=True)
    reference = out / 'fixed-reference.npz'
    reports = {}
    for role, library in (('baseline', args.baseline_library), ('workspace_mkl', args.mkl_workspace_library),
                          ('workspace_vnni', args.vnni_workspace_library)):
        library = library.resolve()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--role', role,
                   '--library', str(library), '--library-sha256', sha(library), '--reference', str(reference),
                   '--output', str(out / (role + '.json')), '--cpu', str(args.cpu),
                   '--warmups', str(args.warmups), '--samples', str(args.samples), '--calls', str(args.calls)]
        if role != 'baseline':
            command += ['--reference-sha256', sha(reference)]
        # No library is loaded in the parent. Every role gets a fresh loader.
        subprocess.run(command, check=True, env=os.environ.copy())
        result = json.loads((out / (role + '.json')).read_text())
        require(result['status'] == 'complete', 'Incomplete child diagnostic')
        reports[role] = result
    by_role = {role: {case['case']: case for case in report['cases']} for role, report in reports.items()}
    comparisons = []
    for name, *_ in CASES:
        base = by_role['baseline'][name]
        mkl = by_role['workspace_mkl'][name]
        vnni = by_role['workspace_vnni'][name]
        require(base['input_sha256'] == mkl['input_sha256'] == vnni['input_sha256'], 'Inputs differ between processes')
        values = {phase: {'vnni_over_workspace_mkl_time': vnni['timings'][phase]['median_ns'] / mkl['timings'][phase]['median_ns'],
                          'workspace_mkl_over_baseline_time': None if phase not in base['timings'] else mkl['timings'][phase]['median_ns'] / base['timings'][phase]['median_ns']}
                  for phase in mkl['timings']}
        comparisons.append({'case': name, 'ratios_of_median_sample_means': values})
    result = {'status': 'complete', 'gpu_used': False, 'models_executed': False,
              'separate_process_per_library': True, 'reference_sha256': sha(reference),
              'script_sha256': sha(__file__), 'comparisons': comparisons,
              'reports': {role: {'path': role + '.json', 'sha256': sha(out / (role + '.json'))} for role in reports},
              'limit': 'Diagnostic hot matrix ratios include common Python/ctypes overhead and separate-process drift. No decoder RTF or significance claim.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'summary': str(out / 'summary.json')}))


if __name__ == '__main__':
    main()
