"""Pinned Apple CPU candidate validation, FLOAT exports, and matched decoder timing.

This adapter retains the common campaign's numerical tolerances and case loader.
It adds all-clip state checks, a separately pinned INT8 reference, adjacent paired
timing, and read-only host observations. Importing this module runs no inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import time
import traceback


COMMON_SHA = '7f42ba2e2ae9f1c9b311e2b3b90eb1f3f6e911f0934c25b32570f36f60efb202'
MODELS = ('audio_stock', 'fast_fp32', 'int8_large', 'mimi')
SEED = 20260908


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def command(argv):
    """Read-only, bounded host observation; errors are evidence, not zero values."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=3, check=False)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout[:16000], 'stderr': result.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': repr(exc)}


def host_observation(detailed=False):
    started = time.perf_counter_ns()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'monotonic_ns': started, 'loadavg': list(os.getloadavg()),
              'process': {'pid': os.getpid(), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
                          'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
                          'max_rss_bytes': usage.ru_maxrss, 'minor_faults': usage.ru_minflt,
                          'major_faults': usage.ru_majflt, 'voluntary_switches': usage.ru_nvcsw,
                          'involuntary_switches': usage.ru_nivcsw},
              'qos': 'Unchanged; no thread QoS override or CPU affinity requested'}
    raw = command(['/bin/ps', '-axr', '-o', 'pid=,ppid=,%cpu=,%mem=,comm='])
    rows = []
    if raw.get('returncode') == 0:
        for line in raw['stdout'].splitlines():
            parts = line.split(None, 4)
            if len(parts) == 5:
                try:
                    rows.append({'pid': int(parts[0]), 'ppid': int(parts[1]),
                                 'cpu_percent': float(parts[2]), 'memory_percent': float(parts[3]),
                                 'executable': Path(parts[4]).name})
                except ValueError:
                    pass
        result['process_inventory'] = {'top_by_cpu': sorted(rows, key=lambda r: -r['cpu_percent'])[:20],
                                       'scope': 'ps CPU percentages are OS estimates, not instantaneous attribution'}
    else:
        result['process_inventory'] = raw
    result['battery'] = command(['/usr/bin/pmset', '-g', 'batt'])
    if detailed:
        result['power_settings'] = command(['/usr/bin/pmset', '-g', 'custom'])
        result['thermal'] = command(['/usr/bin/pmset', '-g', 'therm'])
        result['memory'] = command(['/usr/bin/vm_stat'])
        result['hardware'] = command(['/usr/sbin/sysctl', 'hw.model', 'hw.memsize', 'hw.ncpu',
                                      'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu'])
    result['observation_elapsed_ns'] = time.perf_counter_ns() - started
    return result


def validation_lengths(length, other_length):
    """Use real positive prefixes; never extend, repeat or pad source latents."""
    require(isinstance(length, int) and isinstance(other_length, int), 'Integer latent lengths required')
    require(length >= 2 and other_length >= 1, 'State probes need two real frames and a nonempty other clip')
    probe = min(65, length)
    short = min(33, probe)
    cut = min(32, probe - 1)
    other = min(31, other_length)
    require(0 < short <= probe <= length and 0 < cut < probe and 0 < other <= other_length,
            'Invalid real-prefix lengths')
    return {'probe': probe, 'short': short, 'future_cut': cut, 'other': other,
            'boundary_lengths': [n for n in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64) if n <= probe],
            'boundary_cuts': [n for n in (1, 8, 64) if n < probe]}


def schedule(uids, repeats=5):
    """Keep the comparison pair adjacent and balance pair order over the cohort."""
    rng = random.Random(SEED)
    pair_orders = [i % 2 for i in range(len(uids) * repeats)]
    rng.shuffle(pair_orders)
    answer = []
    index = 0
    for repeat in range(repeats):
        order = list(uids)
        rng.shuffle(order)
        for uid in order:
            pair = ['fast_fp32', 'int8_large']
            if pair_orders[index]:
                pair.reverse()
            blocks = [['audio_stock'], pair, ['mimi']]
            rng.shuffle(blocks)
            answer.append({'repeat': repeat, 'uid': uid,
                           'models': [name for block in blocks for name in block]})
            index += 1
    return answer


def summarize(measurements, uids, repeats=5):
    """All observations count; paired bootstrap keeps repeat-wide drift together."""
    grid = {}
    for row in measurements:
        key = row['model'], row['uid'], row['repeat']
        require(key not in grid and row['elapsed_ns'] > 0 and row['generated_seconds'] > 0,
                'Duplicate or invalid timing')
        grid[key] = row
    require(set(grid) == {(name, uid, repeat) for name in MODELS for uid in uids for repeat in range(repeats)},
            'Incomplete timing grid')
    result = {'models': {}, 'all_measurements_included': True,
              'primary_aggregation': 'Arithmetic mean of per-clip mean decoder RTFs; every selected clip has equal weight',
              'corpus_aggregation': 'Total elapsed / total generated duration reported separately'}
    for name in MODELS:
        rows = [r for r in measurements if r['model'] == name]
        result['models'][name] = {
            'corpus_decoder_rtf': sum(r['elapsed_ns'] for r in rows) / 1e9 / sum(r['generated_seconds'] for r in rows),
            'per_clip_rtf': {uid: statistics.mean(grid[name, uid, j]['elapsed_ns'] / 1e9 /
                                                 grid[name, uid, j]['generated_seconds'] for j in range(repeats))
                             for uid in uids},
            'repeat_rtf': [statistics.mean(grid[name, uid, j]['elapsed_ns'] / 1e9 /
                                           grid[name, uid, j]['generated_seconds'] for uid in uids) for j in range(repeats)]}
        result['models'][name]['decoder_rtf'] = statistics.mean(result['models'][name]['per_clip_rtf'].values())
        values = result['models'][name]['repeat_rtf']
        result['models'][name]['repeat_rtf_cv_percent'] = 100 * statistics.pstdev(values) / statistics.mean(values)
    for uid in uids:
        for j in range(repeats):
            require(grid['fast_fp32', uid, j]['generated_seconds'] == grid['int8_large', uid, j]['generated_seconds'],
                    'Paired duration mismatch')
    def gain(picked_uids, picked_repeats):
        old = sum(grid['fast_fp32', uid, j]['elapsed_ns'] / grid['fast_fp32', uid, j]['generated_seconds']
                  for uid in picked_uids for j in picked_repeats)
        new = sum(grid['int8_large', uid, j]['elapsed_ns'] / grid['int8_large', uid, j]['generated_seconds']
                  for uid in picked_uids for j in picked_repeats)
        return 100 * (1 - new / old)
    rng = random.Random(SEED + 4)
    fixed, hierarchical = [], []
    for _ in range(10000):
        picked = [rng.randrange(repeats) for _ in range(repeats)]
        fixed.append(gain(uids, picked))
        hierarchical.append(gain([rng.choice(uids) for _ in uids], picked))
    def interval(values):
        values = sorted(values)
        return [values[249], values[9749]]
    fixed_ci, hierarchical_ci = interval(fixed), interval(hierarchical)
    result['paired'] = {
        'time_reduction_percent': gain(uids, range(repeats)),
        'fixed_corpus_repeat_block_bootstrap_ci95_percent': fixed_ci,
        'clip_and_repeat_block_bootstrap_ci95_percent': hierarchical_ci,
        'bootstrap_draws': 10000, 'bootstrap_seed': SEED + 4,
        'ci_scope': 'Paired repeat blocks preserve shared drift across clips; hierarchical interval additionally resamples clips. Five repeats and ten selected clips give limited inference.',
        'pair_wins': sum(grid['int8_large', uid, j]['elapsed_ns'] < grid['fast_fp32', uid, j]['elapsed_ns'] for uid in uids for j in range(repeats)),
        'pair_count': len(uids) * repeats,
        'clip_mean_wins': sum(result['models']['int8_large']['per_clip_rtf'][uid] < result['models']['fast_fp32']['per_clip_rtf'][uid] for uid in uids),
        'clip_count': len(uids),
        'confirmed_minimum_10_percent': min(fixed_ci[0], hierarchical_ci[0]) >= 10,
        'target_50_percent_mean_reached': gain(uids, range(repeats)) >= 50,
        'quality_pass': None, 'promotion_approved': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--reference-config-sha256', required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--harness-sha256', required=True)
    parser.add_argument('--mode', choices=('full', 'quality', 'screen'), default='full')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--settle-seconds', type=float, default=20)
    parser.add_argument('--repeat-gap-seconds', type=float, default=3)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(args.threads == 4, 'Frozen Apple comparison requires four ORT threads')
    require(0 <= args.settle_seconds <= 60 and 0 <= args.repeat_gap_seconds <= 60, 'Invalid bounded settle interval')
    common = Path(__file__).resolve().parents[2] / 'tools' / 'platform_campaign.py'
    require(sha(common) == COMMON_SHA and sha(args.harness) == args.harness_sha256, 'Frozen source mismatch')
    spec = importlib.util.spec_from_file_location('public_decoder_harness', args.harness)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    environment = h.require_cpu_environment()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        require(os.environ.get(name) == '1', name + ' must equal 1 before numerical imports')
    require(platform.system() == 'Darwin' and platform.machine() in ('arm64', 'aarch64'), 'Apple ARM host required')
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    require(ort.__version__ == '1.29.0' and onnx.__version__ == '1.22.0', 'ORT1.29 and ONNX tooling1.22 required')
    h.np, h.ort = np, ort
    cfg, hashes = h.read_config(args.config, args.config_sha256)
    reference_cfg, reference_hashes = h.read_config(args.reference_config, args.reference_config_sha256)
    require(tuple(m['name'] for m in cfg['models']) == MODELS, 'Frozen four-model order required')
    require(len(cfg['expected_uids']) == 60 and len(cfg['timing_uids']) == 10, 'Frozen60/10 corpus required')
    for field in ('expected_uids', 'timing_uids', 'kind_contracts', 'frozen_manifest_sha256'):
        require(cfg[field] == reference_cfg[field], 'Reference cohort/contract changed: ' + field)
    require(all(m['causal'] for m in cfg['models']), 'All models must declare causal operation')
    require([m['name'] for m in cfg['models'] if m.get('approximate')] == ['int8_large'], 'Only INT8 declares approximation')
    resolve = lambda value: (args.config.parent / value).resolve()
    reference_resolve = lambda value: (args.reference_config.parent / value).resolve()
    for kind in ('audio', 'mimi'):
        require(sha(resolve(cfg[kind + '_cases'])) == sha(reference_resolve(reference_cfg[kind + '_cases'])), 'Reference case bytes differ')
    inputs = {kind: h.read_cases(resolve(cfg[kind + '_cases']), cfg['kind_contracts'][kind])[0] for kind in ('audio', 'mimi')}
    require(all(set(v) == set(cfg['expected_uids']) for v in inputs.values()), 'Corpus mismatch')
    timed_uids = [cfg['timing_uids'][i] for i in (0, 6, 9)] if args.mode == 'screen' else cfg['timing_uids']
    validate_uids = timed_uids if args.mode == 'screen' else cfg['expected_uids']
    require(not args.output_dir.exists(), 'Fresh output directory required')
    args.output_dir.mkdir(parents=True)
    record = {'status': 'running', 'platform': 'apple', 'mode': args.mode, 'gpu_used': False,
              'threads': args.threads, 'environment': environment,
              'runtime': {'onnxruntime': ort.__version__, 'onnx': onnx.__version__, 'numpy': np.__version__,
                          'soundfile': sf.__version__, 'python': platform.python_version(), 'system': platform.platform(), 'build': ort.get_build_info()},
              'config_sha256': args.config_sha256, 'reference_config_sha256': args.reference_config_sha256,
              'harness_sha256': args.harness_sha256, 'common_runner_sha256': COMMON_SHA, 'script_sha256': sha(__file__),
              'artifact_sha256': hashes, 'reference_artifact_sha256': reference_hashes,
              'checks': [], 'exports': [], 'measurements': [], 'providers': [], 'failures': [], 'host_observations': [],
              'protocol': {'validation_uids': validate_uids, 'timing_uids': [] if args.mode == 'quality' else timed_uids,
                           'warmups': 0 if args.mode == 'quality' else 2, 'repeats': 0 if args.mode == 'quality' else 5,
                           'seed': SEED, 'decoder_only': True, 'same_time_column_scales': True,
                           'rtf_denominator': 'Complete generated waveform samples / native model sample rate, including documented right padding',
                           'fp32_gate': 'Original atol1e-5/rtol1e-4 vs frozen reference and fresh stock',
                           'int8_gate': 'Original atol1e-5/rtol1e-4 vs separately pinned r3 INT8; bitwise recorded. No INT8-to-FP32 allclose gate.',
                           'state_checks': 'Every validated clip uses real lengths:base=min(65,L),prefix=min(33,base),futurecut=min(32,base-1),repeat,concurrent distinct-input length=min(31,otherL); first clip also valid boundary lengths/cuts. Actual lengths recorded.',
                           'timing': 'FP32/INT8 adjacent, balanced randomized pair order; stock/Mimi shuffled around pair. Hashes/checks/host observations outside timers.',
                           'settle_seconds': args.settle_seconds, 'repeat_gap_seconds': args.repeat_gap_seconds,
                           'no_outlier_exclusion': True, 'no_os_state_changes': True,
                           'reference_session': 'r3 held during validation and released before warmups; its native libraries may remain mapped'},
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    result_path = args.output_dir / 'results.json'
    def save():
        atomic_json(result_path, record)
    def pulse(phase, **fields):
        record['progress'] = {'phase': phase, **fields}
        save()
        print(json.dumps(record['progress']), flush=True)
    def bit_equal(x, y):
        return x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32))
    def wave_hash(x):
        return hashlib.sha256(x.tobytes()).hexdigest()
    def check(name, uid, test, passed=True, **fields):
        record['checks'].append({'model': name, 'uid': uid, 'test': test, 'passed': bool(passed), **fields})
        require(passed, name + ':' + uid + ':' + test)
    def make_session(model, resolver, label):
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
            options.add_session_config_entry(key, '0')
        libraries = model.get('custom_libraries', []) or ([model['custom_library']] if model.get('custom_library') else [])
        for library in libraries:
            options.register_custom_ops_library(str(resolver(library)))
        session = ort.InferenceSession(str(resolver(model['path'])), sess_options=options, providers=['CPUExecutionProvider'])
        session.disable_fallback()
        require(session.get_providers() == ['CPUExecutionProvider'], 'Unexpected provider')
        require(len(session.get_inputs()) == len(session.get_outputs()) == 1, 'Single input/output required')
        require(session.get_inputs()[0].type == session.get_outputs()[0].type == 'tensor(float)', 'FP32 interface required')
        record['providers'].append({'model': label, 'providers': session.get_providers()})
        return session, session.get_inputs()[0].name
    def run(session_info, z, contract):
        before = wave_hash(z)
        session, input_name = session_info
        y = session.run(None, {input_name: z})[0]
        require(y.dtype == np.float32 and y.shape == (1, 1, z.shape[-1] * contract['hop']) and np.isfinite(y).all(), 'Invalid waveform')
        require(wave_hash(z) == before, 'Latent mutation')
        return y
    def inventory():
        lib = ctypes.CDLL(None)
        lib._dyld_image_count.restype = ctypes.c_uint32
        lib._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        lib._dyld_get_image_name.restype = ctypes.c_char_p
        images = [lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())]
        forbidden = ('libcuda', 'libcublas', 'libcudnn', 'providers_cuda', 'providers_tensorrt', 'librocblas', 'libamdhip')
        bad = sorted({Path(p).name for p in images if any(s in p.lower() for s in forbidden)})
        require(not bad, 'GPU compute library loaded')
        return {'forbidden_compute_libraries': bad,
                'metal_framework_images': sorted({Path(p).name for p in images if '/Metal.framework/' in p}),
                'scope': 'Library inventory does not itself prove execution; every session is CPU-only and candidate source uses CPU kernels'}
    def verify_inputs():
        require(h.read_config(args.config, args.config_sha256)[1] == hashes, 'Candidate artifacts changed')
        require(h.read_config(args.reference_config, args.reference_config_sha256)[1] == reference_hashes, 'Reference artifacts changed')
        require(sha(args.harness) == args.harness_sha256 and sha(common) == COMMON_SHA and sha(__file__) == record['script_sha256'], 'Runner changed')
    sessions, stock, validated = {}, {}, {}
    try:
        record['host_observations'].append({'phase': 'before_validation', **host_observation(True)})
        record['gpu_mappings_before'] = inventory()
        reference_model = next(m for m in reference_cfg['models'] if m['name'] == 'int8_large')
        require(reference_model.get('approximate') is True, 'Reference must be declared INT8')
        reference_session = make_session(reference_model, reference_resolve, 'r3_int8_reference')
        for model in cfg['models']:
            pulse('session', model=model['name'])
            sessions[model['name']] = make_session(model, resolve, model['name'])
        export_bytes = sum(inputs[m['kind']][uid]['ref'].nbytes + 1024 for m in cfg['models'] for uid in validate_uids)
        if args.mode != 'screen':
            available = shutil.disk_usage(args.output_dir).free
            record['disk_preflight'] = {'expected_float_wav_bytes_upper_bound': export_bytes, 'free_bytes': available}
            require(available > export_bytes + (256 << 20), 'Insufficient export disk space')
        for model in cfg['models']:
            name, kind = model['name'], model['kind']
            contract = cfg['kind_contracts'][kind]
            hop = contract['hop']
            for index, uid in enumerate(validate_uids):
                z, ref = inputs[kind][uid]['z'], inputs[kind][uid]['ref']
                other_uid = validate_uids[(index + 1) % len(validate_uids)]
                lengths = validation_lengths(z.shape[-1], inputs[kind][other_uid]['z'].shape[-1])
                y = run(sessions[name], z, contract)
                if name == 'audio_stock':
                    stock[uid] = y.copy()
                comparison = {}
                comparison_ok = True
                if not model.get('approximate'):
                    comparison['saved_reference'] = h.compare(y, ref)
                    comparison_ok = comparison['saved_reference']['passed']
                    if kind == 'audio':
                        comparison['fresh_stock'] = h.compare(y, stock[uid])
                        comparison_ok = comparison_ok and comparison['fresh_stock']['passed']
                else:
                    old_y = run(reference_session, z, contract)
                    comparison['r3_int8'] = h.compare(y, old_y)
                    comparison['r3_int8_bitwise'] = bit_equal(y, old_y)
                    comparison_ok = comparison['r3_int8']['passed']
                    comparison['r3_int8_waveform_sha256'] = wave_hash(old_y)
                validated[name, uid] = wave_hash(y)
                check(name, uid, 'full_waveform', comparison_ok, latent_sha256=wave_hash(z), waveform_sha256=validated[name, uid],
                      fp32_parity_required=not model.get('approximate', False), comparisons=comparison)
                if args.mode != 'screen':
                    folder = args.output_dir / 'wavs' / name
                    folder.mkdir(parents=True, exist_ok=True)
                    wav = folder / (uid + '.wav')
                    temporary = folder / (uid + '.partial.wav')
                    sf.write(temporary, y.ravel(), contract['sample_rate'], subtype='FLOAT')
                    restored, rate = sf.read(temporary, dtype='float32')
                    info = sf.info(temporary)
                    require(rate == contract['sample_rate'] and info.subtype == 'FLOAT' and info.channels == 1 and bit_equal(restored, y.ravel()), 'FLOAT WAV roundtrip failed')
                    temporary.replace(wav)
                    record['exports'].append({'model': name, 'uid': uid, 'path': str(wav.resolve()), 'sha256': sha(wav),
                                              'waveform_sha256': validated[name, uid], 'rate': rate, 'samples': len(restored), 'subtype': 'FLOAT'})
                probe_length, short_length, future_cut = lengths['probe'], lengths['short'], lengths['future_cut']
                probe = np.ascontiguousarray(z[..., :probe_length])
                base = run(sessions[name], probe, contract)
                small_z = np.ascontiguousarray(probe[..., :short_length])
                small = run(sessions[name], small_z, contract)
                prefix_cmp = h.compare(small, base[..., :short_length * hop])
                check(name, uid, 'short_prefix', prefix_cmp['passed'], length=short_length, base_length=probe_length, comparison=prefix_cmp)
                full_prefix_cmp = h.compare(base, y[..., :probe_length * hop])
                check(name, uid, 'full_prefix', full_prefix_cmp['passed'], length=probe_length, source_length=z.shape[-1], comparison=full_prefix_cmp)
                future = probe.copy()
                future[..., future_cut:] += np.float32(.37)
                changed = run(sessions[name], future, contract)
                check(name, uid, 'future', bit_equal(base[..., :future_cut * hop], changed[..., :future_cut * hop]), cut=future_cut, base_length=probe_length)
                check(name, uid, 'repeat_long_short_long', bit_equal(base, run(sessions[name], probe, contract)))
                other_z = np.ascontiguousarray(inputs[kind][other_uid]['z'][..., :lengths['other']])
                other = run(sessions[name], other_z, contract)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    left = pool.submit(run, sessions[name], probe, contract)
                    right = pool.submit(run, sessions[name], other_z, contract)
                    okay = bit_equal(left.result(), base) and bit_equal(right.result(), other)
                check(name, uid, 'concurrent_distinct_input_length', okay, other_uid=other_uid, lengths=[probe_length, lengths['other']])
                if name == 'int8_large':
                    for length, actual in ((probe_length, base), (short_length, small)):
                        old = run(reference_session, np.ascontiguousarray(probe[..., :length]), contract)
                        cmp = h.compare(actual, old)
                        check(name, uid, 'r3_int8_prefix', cmp['passed'], length=length, comparison=cmp, bitwise=bit_equal(actual, old))
                if index == 0:
                    for length in lengths['boundary_lengths']:
                        edge = run(sessions[name], np.ascontiguousarray(probe[..., :length]), contract)
                        cmp = h.compare(edge, base[..., :length * hop])
                        check(name, uid, 'boundary_prefix', cmp['passed'], length=length, comparison=cmp)
                    for cut in lengths['boundary_cuts']:
                        changed_z = probe.copy()
                        changed_z[..., cut:] += np.float32(.37)
                        changed = run(sessions[name], changed_z, contract)
                        check(name, uid, 'boundary_future', bit_equal(base[..., :cut * hop], changed[..., :cut * hop]), cut=cut)
                pulse('validated', model=name, uid=uid, clip=index + 1, total=len(validate_uids))
        del reference_session
        verify_inputs()
        record['gpu_mappings_before_timing'] = inventory()
        if args.mode != 'quality':
            plan = schedule(timed_uids)
            record['timing_schedule'] = plan
            record['timing_schedule_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            models = {m['name']: m for m in cfg['models']}
            for block in plan[:len(timed_uids)]:
                for name in block['models']:
                    model = models[name]
                    for _ in range(2):
                        warm = run(sessions[name], inputs[model['kind']][block['uid']]['z'], cfg['kind_contracts'][model['kind']])
                        require(wave_hash(warm) == validated[name, block['uid']], 'Warmup waveform changed')
            pulse('settling', seconds=args.settle_seconds)
            time.sleep(args.settle_seconds)
            previous_repeat = None
            for block in plan:
                repeat, uid = block['repeat'], block['uid']
                if previous_repeat is not None and previous_repeat != repeat:
                    time.sleep(args.repeat_gap_seconds)
                record['host_observations'].append({'phase': 'before_timing_group', 'repeat': repeat, 'uid': uid,
                                                    **host_observation(previous_repeat != repeat)})
                for order, name in enumerate(block['models']):
                    model = models[name]
                    z = inputs[model['kind']][uid]['z']
                    latent_hash = wave_hash(z)
                    session, input_name = sessions[name]
                    started = time.perf_counter_ns()
                    output = session.run(None, {input_name: z})[0]
                    elapsed = time.perf_counter_ns() - started
                    require(output.dtype == np.float32 and output.shape == inputs[model['kind']][uid]['ref'].shape, 'Timed interface changed')
                    require(wave_hash(z) == latent_hash and wave_hash(output) == validated[name, uid], 'Timed input/output differs from validated bytes')
                    record['measurements'].append({'model': name, 'uid': uid, 'repeat': repeat, 'order_in_group': order,
                                                   'started_monotonic_ns': started, 'elapsed_ns': elapsed,
                                                   'latent_sha256': latent_hash, 'waveform_sha256': validated[name, uid],
                                                   'generated_seconds': output.shape[-1] / cfg['kind_contracts'][model['kind']]['sample_rate']})
                previous_repeat = repeat
                pulse('timing', repeat=repeat, uid=uid)
            record['summary'] = summarize(record['measurements'], timed_uids)
        record['host_observations'].append({'phase': 'finished', **host_observation(True)})
        verify_inputs()
        for export in record['exports']:
            require(sha(export['path']) == export['sha256'], 'Export changed')
        record['gpu_mappings_after'] = inventory()
        record['status'] = 'complete'
        record['quality_pass'] = None
        record['commit_eligible'] = False
        record['limit'] = 'Runtime correctness and speed evidence only. Perceptual quality and adoption require a separate decision.'
    except BaseException as exc:
        record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['failures'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()
    print(json.dumps({'status': record['status'], 'checks': len(record['checks']), 'exports': len(record['exports']),
                      'timings': len(record['measurements']), 'summary': record.get('summary')}), flush=True)


if __name__ == '__main__':
    main()
