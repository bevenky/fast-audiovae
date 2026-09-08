"""CPU-only exact-output comparison against the frozen selective INT8 decoder."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import threading
import time
import traceback


SEED = 20260908
THREADS = 2
AFFINITY = {0, 1}
WARMUPS = 2
REPEATS = 5
BASELINE = 'int8_large'
SINGLE_THREAD_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_hash(path, expected):
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdefABCDEF' for c in expected),
            'Expected a SHA256 for ' + str(path))
    actual = sha(path)
    require(actual == expected.lower(), 'Changed SHA256: ' + str(path))
    return actual


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_harness(path):
    spec = importlib.util.spec_from_file_location('iteration3_public_harness', path)
    require(spec is not None and spec.loader is not None, 'Cannot load harness')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def cgroup_snapshot():
    values = {}
    for name in ('cpu.stat', 'cpu.max', 'memory.current', 'memory.max'):
        try:
            values[name] = (Path('/sys/fs/cgroup') / name).read_text().strip()
        except OSError:
            values[name] = None
    return values


def usage_snapshot():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {k: getattr(r, k) for k in
            ('ru_utime', 'ru_stime', 'ru_maxrss', 'ru_minflt', 'ru_majflt',
             'ru_nvcsw', 'ru_nivcsw')}


def maps_check(h):
    value = h.gpu_library_mappings()
    require(value.get('available') is True and value.get('basenames') == [],
            'GPU-library mapping inventory unavailable or nonempty')
    return value


def intel_identity():
    text = Path('/proc/cpuinfo').read_text()
    first = text.split('\n\n', 1)[0]
    fields = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
              for line in first.splitlines() if ':' in line}
    flags = set(fields.get('flags', '').split())
    require(fields.get('vendor_id') == 'GenuineIntel' and '8280' in fields.get('model name', ''),
            'This frozen experiment requires the Xeon Platinum8280 host')
    require({'avx2', 'avx512f', 'avx512bw', 'avx512dq', 'avx512vl', 'avx512_vnni'}.issubset(flags),
            'Expected AVX512 VNNI CPU features are unavailable')
    return {'vendor_id': fields['vendor_id'], 'model_name': fields['model name'],
            'flags': sorted(flags), 'cpuinfo_sha256': hashlib.sha256(text.encode()).hexdigest()}


def find_model(config, name):
    found = [m for m in config['models'] if m['name'] == name]
    require(len(found) == 1, 'Expected exactly one model named ' + name)
    model = found[0]
    require(model['kind'] == 'audio' and model.get('causal') is True
            and model.get('approximate') is True and not model.get('extra_inputs'),
            'Expected an explicitly causal, approximate AudioVAE decoder')
    require(not model.get('custom_library') and not config.get('custom_library'),
            'Use explicit ordered custom_libraries lists')
    return model


def model_identity(config_path, config, model):
    resolve = lambda item: (config_path.parent / item).resolve()
    return {
        'graph_sha256': sha(resolve(model['path'])),
        'external_data_sha256': sorted(sha(resolve(v)) for v in model.get('external_data', [])),
        'custom_library_sha256': [sha(resolve(v)) for v in model.get('custom_libraries', [])],
        'kind': model['kind'], 'causal': model['causal'],
        'approximate': model.get('approximate'),
        'contract': config['kind_contracts'][model['kind']],
    }


def validate_campaign(report, config, config_sha, artifacts, harness_sha):
    require(report.get('status') == 'complete' and report.get('screen') is False,
            'Baseline campaign must be complete and full')
    require(report.get('failures') == []
            and report.get('timed_outputs_match_validated_waveforms') is True,
            'Baseline campaign failed or lacks validated timing-output replay')
    require(report.get('config') == config and report.get('config_sha256') == config_sha,
            'Baseline campaign configuration identity differs')
    require(report.get('verified_artifact_sha256') == artifacts
            and report.get('harness_sha256') == harness_sha
            and report.get('script_sha256') == config.get('campaign_script_sha256'),
            'Baseline campaign artifact or runner identity differs')
    require(report.get('gpu_used') is False and report.get('onnxruntime') == '1.29.0'
            and report.get('threads') == THREADS and report.get('affinity') == [0, 1],
            'Baseline campaign CPU/runtime contract differs')
    protocol = report.get('protocol', {})
    require(protocol.get('validation_uids') == config['expected_uids']
            and protocol.get('timing_uids') == config['timing_uids']
            and protocol.get('repeats') == REPEATS and protocol.get('warmups') == WARMUPS,
            'Baseline campaign cohort or timing protocol differs')
    checks = report.get('checks', [])
    require(checks and all(row.get('passed') is True for row in checks),
            'Baseline campaign contains missing or failed checks')
    names = [model['name'] for model in config['models']]
    require(names == ['audio_stock', 'fast_fp32', 'int8_all', 'int8_large', 'mimi'],
            'Expected the completed five-model INT8 campaign without FP16')
    full = [(row.get('model'), row.get('uid')) for row in checks
            if row.get('test') == 'shape_finite_input_unchanged_and_declared_fp32_gate']
    expected = {(name, uid) for name in names for uid in config['expected_uids']}
    require(len(full) == len(expected) and set(full) == expected,
            'Baseline full-waveform checks are incomplete or duplicated')
    require(all(row.get('fp32_parity_required') is False for row in checks
                if row.get('model') == BASELINE and row.get('uid') is not None),
            'Selective INT8 reference incorrectly declares FP32 parity')
    export_names = [model.get('quality_name', model['name']) for model in config['models']]
    require(len(export_names) == len(set(export_names)), 'Campaign export model names collide')
    export_ids = [(row.get('model'), row.get('uid')) for row in report.get('exports', [])]
    expected_exports = {(name, uid) for name in export_names for uid in config['expected_uids']}
    require(len(export_ids) == len(expected_exports) and set(export_ids) == expected_exports,
            'Completed campaign export inventory changed')
    boundary = [row for row in checks if row.get('uid') is None]
    expected_boundary = {(name, test, length) for name in names
                         for test, length in [('exact_repeat_and_future_invariance', None),
                                              ('long_short_long', None),
                                              *[('short_prefix', n) for n in (1, 2, 3, 7, 8, 16)]]}
    actual_boundary = [(row.get('model'), row.get('test'), row.get('length')) for row in boundary]
    require(len(checks) == len(full) + len(boundary)
            and len(actual_boundary) == len(expected_boundary)
            and set(actual_boundary) == expected_boundary,
            'Baseline boundary-check inventory changed')
    timings = report.get('measurements', [])
    timing_ids = [(r.get('model'), r.get('uid'), r.get('repeat')) for r in timings]
    expected_timings = {(n, u, i) for n in names for u in config['timing_uids'] for i in range(REPEATS)}
    require(len(timing_ids) == len(expected_timings) and set(timing_ids) == expected_timings,
            'Baseline timing inventory incomplete or duplicated')
    require(all(type(row.get('elapsed_ns')) is int and row['elapsed_ns'] > 0 for row in timings),
            'Baseline timing records invalid')
    providers = report.get('providers', [])
    require(len(providers) == len(names) and {r.get('model') for r in providers} == set(names)
            and all(r.get('providers') == ['CPUExecutionProvider'] for r in providers),
            'Baseline CPU provider inventory changed')
    for key in ('gpu_library_mappings_before_timing', 'gpu_library_mappings_after'):
        require(report.get(key, {}).get('available') is True
                and report[key].get('basenames') == [], 'Baseline GPU mapping evidence missing')


def load_references(report_path, report, model, cases, expected_uids, rate, np, sf):
    name = model.get('quality_name', model['name'])
    selected = [e for e in report.get('exports', []) if e.get('model') == name]
    require(len(selected) == len(expected_uids)
            and {e.get('uid') for e in selected} == set(expected_uids),
            'Baseline exports must cover every UID exactly once')
    refs, pinned, inventory = {}, {}, []
    for row in selected:
        uid = row['uid']
        path = Path(row['path'])
        if not path.is_absolute():
            path = report_path.parent / path
        path = path.resolve()
        pinned[path] = require_hash(path, row['sha256'])
        shape = cases[uid]['ref'].shape
        require(row.get('rate') == rate and row.get('samples') == shape[-1]
                and row.get('subtype') == 'FLOAT', 'Baseline WAV metadata mismatch: ' + uid)
        info = sf.info(path)
        require(info.format == 'WAV' and info.subtype == 'FLOAT' and info.channels == 1
                and info.samplerate == rate and info.frames == shape[-1],
                'Baseline WAV header mismatch: ' + uid)
        samples, actual_rate = sf.read(path, dtype='float32', always_2d=True)
        require(actual_rate == rate and samples.dtype == np.float32
                and samples.shape == (shape[-1], 1) and np.isfinite(samples).all(),
                'Baseline WAV samples invalid: ' + uid)
        refs[uid] = np.ascontiguousarray(samples[:, 0].reshape(shape))
        inventory.append({'uid': uid, 'wav_sha256': pinned[path],
                          'raw_waveform_sha256': sample_digest(refs[uid]),
                          'shape': list(shape), 'sample_rate': rate, 'cropped': False})
    return refs, pinned, inventory


def summarize(rows, uids, mode):
    expected = {(m, u, r) for m in ('baseline', 'candidate') for u in uids for r in range(REPEATS)}
    actual = [(r['model'], r['uid'], r['repeat']) for r in rows]
    require(len(actual) == len(expected) and set(actual) == expected,
            'Timing inventory incomplete or duplicated')
    values = {(r['model'], r['uid'], r['repeat']): r for r in rows}
    totals = {m: sum(r['elapsed_ns'] for r in rows if r['model'] == m)
              for m in ('baseline', 'candidate')}
    duration = sum(r['generated_seconds'] for r in rows if r['model'] == 'baseline')
    reduction = 1 - totals['candidate'] / totals['baseline']
    clip_rows, differences = [], []
    for uid in uids:
        times = {m: [values[m, uid, i]['elapsed_ns'] for i in range(REPEATS)]
                 for m in ('baseline', 'candidate')}
        means = {m: statistics.mean(v) for m, v in times.items()}
        differences.append((means['baseline'], means['candidate']))
        clip_rows.append({'uid': uid, 'mean_elapsed_ns': means,
                          'time_reduction': 1 - means['candidate'] / means['baseline'],
                          'stdev_elapsed_ns': {m: statistics.stdev(v) for m, v in times.items()},
                          'range_elapsed_ns': {m: [min(v), max(v)] for m, v in times.items()}})
    repeat_rows = []
    for i in range(REPEATS):
        total = {m: sum(values[m, u, i]['elapsed_ns'] for u in uids)
                 for m in ('baseline', 'candidate')}
        seconds = sum(values['baseline', u, i]['generated_seconds'] for u in uids)
        repeat_rows.append({'repeat': i, 'elapsed_ns': total,
                            'decoder_rtf': {m: v / 1e9 / seconds for m, v in total.items()},
                            'time_reduction': 1 - total['candidate'] / total['baseline']})
    rng = random.Random(SEED + 1)
    bootstrap = []
    for _ in range(5000):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        bootstrap.append(1 - sum(v[1] for v in sample) / sum(v[0] for v in sample))
    bootstrap.sort()
    all_clips = all(row['time_reduction'] >= 0 for row in clip_rows)
    return {'aggregate_decoder_rtf': {m: v / 1e9 / duration for m, v in totals.items()},
            'aggregate_time_reduction': reduction,
            'speedup': totals['baseline'] / totals['candidate'],
            'per_clip': clip_rows, 'per_repeat': repeat_rows,
            'repeat_time_reduction_stdev': statistics.stdev(v['time_reduction'] for v in repeat_rows),
            'paired_clip_bootstrap_95pct': [bootstrap[124], bootstrap[4874]],
            'uncertainty_scope': 'Descriptive paired clip bootstrap, not independent listeners or repeated VM campaigns; confidence interval does not alter the predeclared gate',
            'meets_10pct_aggregate': reduction >= .10, 'all_clip_means_not_slower': all_clips,
            'accepted': mode == 'full' and reduction >= .10 and all_clips,
            'screening_only': mode != 'full'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ('baseline-config', 'candidate-config', 'completed-campaign', 'harness'):
        parser.add_argument('--' + prefix, type=Path, required=True)
        parser.add_argument('--' + prefix + '-sha256', required=True)
    parser.add_argument('--candidate-model', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--screen', action='store_true')
    args = parser.parse_args()
    require(sys.platform == 'linux' and platform.machine() in ('x86_64', 'AMD64'),
            'This experiment requires the Intel Linux CPU host')
    paths = {name: getattr(args, name.replace('-', '_')).resolve()
             for name in ('baseline-config', 'candidate-config', 'completed-campaign', 'harness')}
    pins = {path: require_hash(path, getattr(args, name.replace('-', '_') + '_sha256'))
            for name, path in paths.items()}
    pins[Path(__file__).resolve()] = require_hash(__file__, args.script_sha256)
    h = load_harness(paths['harness'])
    env = h.require_cpu_environment()
    for key, value in SINGLE_THREAD_ENV.items():
        require(os.environ.get(key) == value, key + ' must equal ' + value + ' before process launch')
    require(os.environ.get('MKL_DYNAMIC', 'FALSE').upper() in ('FALSE', '0'), 'MKL_DYNAMIC must be false')
    require(os.environ.get('OMP_DYNAMIC', 'FALSE').upper() in ('FALSE', '0'), 'OMP_DYNAMIC must be false')
    original_affinity = h.current_affinity()
    require(original_affinity is not None and AFFINITY.issubset(original_affinity),
            'Required logical CPUs0,1 unavailable')
    os.sched_setaffinity(0, AFFINITY)
    require(h.current_affinity() == AFFINITY, 'CPU affinity was not applied')
    hardware = intel_identity()
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    h.np, h.ort = np, ort
    require(ort.__version__ == '1.29.0', 'ORT1.29.0 is required')
    runtime_binaries = sorted((Path(ort.__file__).resolve().parent / 'capi').glob('*.so*'))
    require(runtime_binaries, 'Cannot identify ORT runtime binaries for provenance')
    for binary in runtime_binaries:
        pins[binary.resolve()] = sha(binary)
    cfgs, verified = {}, {}
    for key in ('baseline', 'candidate'):
        path = paths[key + '-config']
        cfgs[key], verified[key] = h.read_config(path, pins[path])
        for relative, digest in verified[key].items():
            artifact = h.relative_path(path.parent, relative)
            require(artifact not in pins or pins[artifact] == digest, 'Conflicting artifact pins')
            pins[artifact] = digest
    baseline_cfg, candidate_cfg = cfgs['baseline'], cfgs['candidate']
    for key in ('expected_case_count', 'expected_uids', 'timing_uids', 'kind_contracts'):
        require(candidate_cfg[key] == baseline_cfg[key], 'Candidate changes frozen ' + key)
    require(baseline_cfg['expected_case_count'] == 60
            and len(baseline_cfg['timing_uids']) == 10, 'Expected the frozen60/10 cohort')
    require(baseline_cfg['baseline_harness_sha256'] == pins[paths['harness']],
            'Public harness differs from the frozen campaign')
    models = {'baseline': find_model(baseline_cfg, BASELINE),
              'candidate': find_model(candidate_cfg, args.candidate_model)}
    preserved = find_model(candidate_cfg, BASELINE)
    require(model_identity(paths['candidate-config'], candidate_cfg, preserved)
            == model_identity(paths['baseline-config'], baseline_cfg, models['baseline']),
            'Candidate config must retain unchanged int8_large baseline')
    require(args.candidate_model != BASELINE, 'Candidate needs its own distinct model name')
    case_paths = {key: h.relative_path(paths[key + '-config'].parent, cfgs[key]['audio_cases'])
                  for key in cfgs}
    require(sha(case_paths['baseline']) == sha(case_paths['candidate'])
            and sha(case_paths['baseline'].with_suffix('.json'))
            == sha(case_paths['candidate'].with_suffix('.json')), 'Candidate case bytes changed')
    contract = baseline_cfg['kind_contracts']['audio']
    require(contract == {'sample_rate': 48000, 'hop': 1920, 'latent_channels': 64},
            'Unexpected AudioVAE latent/waveform interface')
    cases, metadata = h.read_cases(case_paths['baseline'], contract)
    expected_uids = baseline_cfg['expected_uids']
    require(len(cases) == 60 and set(cases) == set(expected_uids)
            and metadata['timed_ids'] == expected_uids, 'Full case inventory or metadata order changed')
    campaign = json.loads(paths['completed-campaign'].read_text())
    validate_campaign(campaign, baseline_cfg, pins[paths['baseline-config']],
                      verified['baseline'], pins[paths['harness']])
    refs, reference_pins, reference_inventory = load_references(
        paths['completed-campaign'], campaign, models['baseline'], cases, expected_uids,
        contract['sample_rate'], np, sf)
    pins.update(reference_pins)
    mode = 'screen' if args.screen else 'full'
    timing_uids = list(baseline_cfg['timing_uids'])
    if args.screen:
        selected = []
        for prefix in ('hi_in_', 'en_us_', 'pt_br_'):
            matching = [uid for uid in timing_uids if uid.startswith(prefix)]
            require(len(matching) == 1, 'Screen requires one original timing UID for ' + prefix)
            selected.extend(matching)
        timing_uids = selected
    validation_uids = timing_uids if args.screen else list(expected_uids)
    longest_uid = max(expected_uids, key=lambda u: cases[u]['z'].shape[-1])
    probe_length = 65 if args.screen else 257
    require(cases[longest_uid]['z'].shape[-1] >= probe_length, 'Frozen corpus too short for fixed boundary probes')
    short_lengths = [v for v in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256)
                     if v < probe_length]
    cuts = [8, 32, 64] if args.screen else [1, 7, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256]
    output = args.output_dir.resolve()
    require(not output.exists(), 'Use a fresh output directory; resume is intentionally unsupported')
    require(not any(p == output or output in p.parents for p in pins), 'Output overlaps an input artifact')
    for path, digest in pins.items():
        require_hash(path, digest)
    before_maps = maps_check(h)
    output.mkdir(parents=True)
    report_path = output / 'results.json'
    report = {'status': 'running', 'mode': mode, 'accepted': False,
              'scope': 'CPU decoder-only exact-output comparison with accepted selective INT8; no encoding, TTS, loading or scoring inside timers',
              'baseline_model': BASELINE, 'candidate_model': args.candidate_model,
              'baseline_config_sha256': pins[paths['baseline-config']],
              'candidate_config_sha256': pins[paths['candidate-config']],
              'completed_campaign_sha256': pins[paths['completed-campaign']],
              'harness_sha256': pins[paths['harness']],
              'script_sha256': pins[Path(__file__).resolve()],
              'environment': {**env, **{key: os.environ.get(key) for key in SINGLE_THREAD_ENV}},
              'affinity': [0, 1], 'threads': THREADS, 'gpu_used': False,
              'hardware': hardware,
              'runtime': {'onnxruntime': ort.__version__, 'ort_build': ort.get_build_info(),
                          'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(),
                          'binary_sha256': {binary.name: pins[binary.resolve()] for binary in runtime_binaries}},
              'input_sha256': {str(path): digest for path, digest in pins.items()},
              'model_identity': {key: model_identity(paths[key + '-config'], cfgs[key], models[key]) for key in models},
              'waveform_reference': {'kind': 'frozen_selective_int8_candidate_own_validated_raw_FLOAT_WAV',
                                     'fp32_parity_asserted': False, 'cropped': False, 'inventory': reference_inventory},
              'protocol': {'validation_uids': validation_uids, 'timing_uids': timing_uids,
                           'full_frozen_uids': expected_uids, 'warmups': WARMUPS, 'repeats': REPEATS,
                           'seed': SEED, 'probe_source_uid': longest_uid, 'probe_length': probe_length,
                           'short_lengths': short_lengths, 'future_cuts': cuts,
                           'length_prefix_gate': 'Existing atol1e-5/rtol1e-4 for each model short output versus its longer output prefix',
                           'same_shape_candidate_gate': 'Exact uint32 equality to a fresh baseline call for every short shape',
                           'same_length_future_and_repeat_gate': 'Exact uint32 equality, unchanged',
                           'timing': 'Fresh adjacent A/B decoder calls, randomized pair order and clip order; all checks and logging outside timers',
                           'gate': 'Full mode, all strict checks, >=10% aggregate time reduction, every timed clip mean not slower',
                           'screen_cannot_promote': True},
              'checks': [], 'measurements': [], 'warmup_calls': 0, 'providers': [], 'failures': [],
              'decoder_calls': {'validation': 0, 'warmup': 0, 'timing': 0},
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'host_before': h.host_snapshot(), 'usage_before': usage_snapshot(),
              'gpu_library_mappings_before': before_maps}
    sessions, last_progress = {}, [0.0]
    counter_lock = threading.Lock()

    def save():
        atomic(report_path, report)

    def progress(phase, **items):
        report['progress'] = {'phase': phase, **items}
        save()
        if time.monotonic() - last_progress[0] > 20:
            print(json.dumps(report['progress']), flush=True)
            last_progress[0] = time.monotonic()

    def execute(key, latent, timed=False, phase='validation'):
        digest = sample_digest(latent)
        session, name = sessions[key]
        start = time.perf_counter_ns() if timed else None
        result = session.run(None, {name: latent})[0]
        elapsed = time.perf_counter_ns() - start if timed else None
        with counter_lock:
            report['decoder_calls']['timing' if timed else phase] += 1
        require(sample_digest(latent) == digest, key + ' modified latent bytes')
        require(result.dtype == np.float32
                and result.shape == (1, 1, latent.shape[-1] * contract['hop'])
                and np.isfinite(result).all(), key + ' produced invalid waveform')
        return result, elapsed

    def equal(actual, reference, label):
        require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32
                and np.array_equal(actual.view(np.uint32), reference.view(np.uint32)),
                'Exact uint32 waveform mismatch: ' + label)

    def checked(key, latent, reference, label, **items):
        result, _ = execute(key, latent)
        equal(result, reference, key + ':' + label)
        report['checks'].append({'model': key, 'test': label, 'passed': True,
                                 'uint32_bitwise_equal': True, **items})
        return result

    def length_prefix_gate(key, actual, reference, label, **items):
        result = h.compare(actual, reference)
        require(result['passed'], key + ': ' + label + ' failed existing numeric prefix gate')
        report['checks'].append({'model': key, 'test': label, **result, **items,
                                 'atol': 1e-5, 'rtol': 1e-4,
                                 'comparison': 'same model at different input lengths',
                                 'uint32_bitwise_required': False})

    try:
        for key in ('baseline', 'candidate'):
            progress('create_session', model=key)
            so = ort.SessionOptions()
            so.intra_op_num_threads = THREADS
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            for entry in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
                so.add_session_config_entry(entry, '0')
            base = paths[key + '-config'].parent
            for library in models[key].get('custom_libraries', []):
                so.register_custom_ops_library(str(h.relative_path(base, library)))
            session = ort.InferenceSession(str(h.relative_path(base, models[key]['path'])),
                                           sess_options=so, providers=['CPUExecutionProvider'])
            session.disable_fallback()
            require(session.get_providers() == ['CPUExecutionProvider'], 'CPU provider required')
            require(len(session.get_inputs()) == len(session.get_outputs()) == 1
                    and session.get_inputs()[0].type == session.get_outputs()[0].type == 'tensor(float)',
                    'Expected one FP32 input and one FP32 output')
            sessions[key] = session, session.get_inputs()[0].name
            report['providers'].append({'model': key, 'providers': session.get_providers()})
        maps_check(h)
        for uid in validation_uids:
            for key in ('baseline', 'candidate'):
                checked(key, cases[uid]['z'], refs[uid], 'full_waveform', uid=uid,
                        latent_sha256=sample_digest(cases[uid]['z']),
                        waveform_sha256=sample_digest(refs[uid]))
            progress('validated_waveform', uid=uid)
        z = np.ascontiguousarray(cases[longest_uid]['z'][..., :probe_length])
        baseline_probe, _ = execute('baseline', z)
        length_prefix_gate('baseline', baseline_probe,
                           refs[longest_uid][..., :probe_length * contract['hop']],
                           'probe_full_numeric_prefix', length=probe_length)
        baseline_short = {}
        for key in ('baseline', 'candidate'):
            own_probe = checked(key, z, baseline_probe, 'probe_reference')
            if key == 'candidate':
                length_prefix_gate(key, own_probe,
                                   refs[longest_uid][..., :probe_length * contract['hop']],
                                   'probe_full_numeric_prefix', length=probe_length)
            checked(key, z, baseline_probe, 'exact_repeat')
            for n in short_lengths:
                shortened = np.ascontiguousarray(z[..., :n])
                result, _ = execute(key, shortened)
                length_prefix_gate(key, result, own_probe[..., :n * contract['hop']],
                                   'short_vs_long_numeric_prefix', length=n)
                if key == 'baseline':
                    baseline_short[n] = result
                else:
                    equal(result, baseline_short[n], 'candidate versus baseline short length ' + str(n))
                    report['checks'].append({'model': key, 'test': 'short_shape_exact_baseline',
                                             'length': n, 'passed': True, 'uint32_bitwise_equal': True})
            for cut in cuts:
                altered = z.copy()
                altered[..., cut:] += np.float32(.37)
                result, _ = execute(key, altered)
                equal(result[..., :cut * contract['hop']], baseline_probe[..., :cut * contract['hop']],
                      key + ': future perturbation ' + str(cut))
                report['checks'].append({'model': key, 'test': 'strict_future_invariance',
                                         'cut': cut, 'passed': True, 'uint32_bitwise_equal': True})
            checked(key, z, baseline_probe, 'long_short_long')
            # Simultaneous independent lengths on the same session. Correctness only.
            inputs = [z.copy(), np.ascontiguousarray(z[..., :33])]
            expected_outputs = [baseline_probe, baseline_short[33]]
            for repeat in range(2):
                barrier = threading.Barrier(2)
                def concurrent(index):
                    barrier.wait(timeout=30)
                    return execute(key, inputs[index])[0]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(concurrent, i) for i in range(2)]
                    results = [f.result() for f in futures]
                for i, result in enumerate(results):
                    equal(result, expected_outputs[i], key + ': concurrent independent call')
                    report['checks'].append({'model': key, 'test': 'concurrent_independent_call',
                                             'repeat': repeat, 'slot': i, 'passed': True,
                                             'uint32_bitwise_equal': True})
            progress('validated_boundaries', model=key)
        require(h.current_affinity() == AFFINITY, 'Affinity changed during validation')
        for path, digest in pins.items():
            require_hash(path, digest)
        report['gpu_library_mappings_before_timing'] = maps_check(h)
        for uid in timing_uids:
            for key in ('baseline', 'candidate'):
                for _ in range(WARMUPS):
                    result, _ = execute(key, cases[uid]['z'], phase='warmup')
                    equal(result, refs[uid], key + ': warmup')
                    report['warmup_calls'] += 1
        report['host_before_timing'] = h.host_snapshot()
        report['cgroup_before_timing'] = cgroup_snapshot()
        report['usage_before_timing'] = usage_snapshot()
        rng = random.Random(SEED)
        orders = {}
        for uid in timing_uids:
            choices = [0, 1, 0, 1, rng.randrange(2)]
            rng.shuffle(choices)
            orders[uid] = choices
        schedule = []
        for repeat in range(REPEATS):
            shuffled = list(timing_uids)
            rng.shuffle(shuffled)
            for uid in shuffled:
                order = ['baseline', 'candidate'] if orders[uid][repeat] == 0 else ['candidate', 'baseline']
                schedule.append({'pair': len(schedule), 'repeat': repeat, 'uid': uid, 'order': order})
        report['pair_schedule'] = schedule
        save()
        for pair in schedule:
            uid = pair['uid']
            for position, key in enumerate(pair['order']):
                result, elapsed = execute(key, cases[uid]['z'], timed=True)
                equal(result, refs[uid], key + ': fresh timed call')
                require(type(elapsed) is int and elapsed > 0, 'Invalid timer result')
                seconds = result.shape[-1] / contract['sample_rate']
                report['measurements'].append({'model': key, 'uid': uid, 'repeat': pair['repeat'],
                                               'pair': pair['pair'], 'position_in_pair': position,
                                               'elapsed_ns': elapsed, 'generated_seconds': seconds,
                                               'decoder_rtf': elapsed / 1e9 / seconds,
                                               'execution': 'fresh_decoder_call',
                                               'validated_waveform_sha256': sample_digest(result)})
            progress('timing_pair', pair=pair['pair'], uid=uid, repeat=pair['repeat'])
        report['host_after_timing'] = h.host_snapshot()
        report['cgroup_after_timing'] = cgroup_snapshot()
        report['usage_after_timing'] = usage_snapshot()
        report['host_timing_delta'] = h.host_delta(report['host_before_timing'], report['host_after_timing'])
        require(h.current_affinity() == AFFINITY, 'Affinity changed during timing')
        require(report['warmup_calls'] == 2 * len(timing_uids) * WARMUPS, 'Warmup count mismatch')
        require(all(row.get('passed') is True for row in report['checks']), 'Validation failure')
        full_checks = [row for row in report['checks'] if row['test'] == 'full_waveform']
        require(len(full_checks) == 2 * len(validation_uids), 'Full waveform coverage incomplete')
        expected_checks = 2 * len(validation_uids) + 16 + 3 * len(short_lengths) + 2 * len(cuts)
        require(len(report['checks']) == expected_checks, 'Boundary-check inventory incomplete')
        expected_calls = {'validation': 2 * len(validation_uids) + 1
                          + 2 * (len(short_lengths) + len(cuts) + 7),
                          'warmup': 2 * len(timing_uids) * WARMUPS,
                          'timing': 2 * len(timing_uids) * REPEATS}
        require(report['decoder_calls'] == expected_calls, 'Fresh decoder-call count mismatch')
        for path, digest in pins.items():
            require_hash(path, digest)
        report['input_hashes_unchanged'] = True
        report['gpu_library_mappings_after'] = maps_check(h)
        report['summary'] = summarize(report['measurements'], timing_uids, mode)
        report['accepted'] = report['summary']['accepted']
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['accepted'] = False
        report['failures'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        report['usage_after'] = usage_snapshot()
        report['host_after'] = h.host_snapshot()
        save()
    print(json.dumps({'status': report['status'], 'mode': mode, 'accepted': report['accepted'],
                      'checks': len(report['checks']), 'timings': len(report['measurements']),
                      'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
