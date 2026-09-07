"""Audit the frozen Intel upsample campaign. Standard library only; no inference.

Exit 0: complete, structurally valid report (accepted may still be false).
Exit 2: evidence invalid or a completed campaign is incomplete. Exit 3: pending.
The optional output is created exclusively, never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

SOURCE_SHA = 'e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2'
SOURCE_CONFIG_SHA = '89d7c5d2135e09079c4be66d802b9ee91b04b919ef8e2aaecd65ddf259bc36a3'
GRAPH_SHA = '1ddb6dcc2b0c3ccea90d309f6ebec10eb12e844fb6320cd2d525dbfd01d4cf29'
LIB_SHA = '16e5b9ae57682cda8060a148b75efb91904aa8c0c53722db6c55de5581a2eed9'
HARNESS_SHA = '593dc1df8b7ac7334add9211da869eac931aaf61d360656b15a3196467cecf01'
LIBS = ['168f57d2e6049bcffc64d61ad0722d2f372b037ad3aabfd2b5cedd963292082a',
        '2b48508c4361a21971fce0a497ce4e9367beedd45249c2e24b6cce96ac49e106',
        '879d4a78b4412f7cefa887be6b2e3421872d09525e61d9cf760f2c30656b60ba']
MODELS = ['audio_stock', 'stage_mkl', 'upsample_stage4', 'mimi']
MODEL_SHAS = dict(zip(MODELS, [
    'fdd20e200675ab9649bf33f16bc5974101ba5cf47ab89fbfe29f3d36ff6c3c84',
    SOURCE_SHA, GRAPH_SHA,
    'e023777a2cee98a7e4a293f6af7712d4fed6364d1523cfc9a6283f112981e2f1']))
TIMING_UIDS = ['hi_in_00099_1919', 'bn_in_00633_1820', 'gu_in_00926_1827',
               'ta_in_00494_1946', 'te_in_00114_1911', 'kn_in_00584_1904',
               'en_us_00103_1779', 'es_419_00646_1781', 'fr_fr_00129_1717',
               'pt_br_00672_1745']
UIDS_SHA = 'c7a0ebecee0ac581d5586d702790a3bce34f128569b30856888303e4d295727b'
CASE_SHAS = {'audio': '9b709b787c040591de57e094a5ba5a4515cc74e256e8014530ff8c7bb54d2d24',
             'mimi': 'af2d68663daf71db954488721e8fffc5888437100772a309b13c62391df0d9c7'}
META_SHAS = {'audio': 'e1044927b8f3a3ea71ed26cd5beef43349104e00b459f3d07f8d4e7b992d7e26',
             'mimi': '457e684bd6e46cdbcc3891f84dbe79cfab6c34657774efc2311ce49ba535c3f6'}
EVIDENCE_SHAS = {
    'region-results-r1.json': '5dde4e372aa66343f02d4919b3574b8e195d9c8dac8b219e255c2a9902a785c2',
    'build.json': '9c4d98cec1491ec18a7b4b94d0ccb80f57fe5ad269576c796f187803a58e3866',
    'mode-0-r1.json': '304f3a2064cd8ef1956012aa1234476ea98f4291c6326fb3c7992d83707fc92e',
    'mode-1-r1.json': '4c3d57455d5e0b9e8feed9691633643f623c0efd355ff9630ebc8756f74ce8c3',
    'edges-0-r1.json': '171846420316c2af3e87e270fe617d8aa7dbde34959565edf6106581da63c59b',
    'edges-1-r1.json': '1942012ae7d1c16745336d0db21cc2845e96465b453cee4f76fcd72c0704f393'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def evidence(directory, region_path=None):
    """Hash-bind the already audited native reports, then recompute selection."""
    records = {}
    for name, digest in EVIDENCE_SHAS.items():
        path = Path(region_path) if region_path and name.startswith('region-') else directory / name
        require(sha(path) == digest, 'Changed frozen evidence: ' + name)
        records[name] = read(path)
    r = records['region-results-r1.json']
    require(r['status'] == 'complete' and not r['gpu_used'], 'Region incomplete or not CPU-only')
    require(r['source_model_sha256'] == SOURCE_SHA and
            r['source_config_sha256'] == SOURCE_CONFIG_SHA, 'Region source changed')
    require(r['library_sha256'] == LIB_SHA and r['harness_sha256'] == HARNESS_SHA,
            'Region implementation changed')
    samples = defaultdict(list)
    keys = []
    for item in r['observations']:
        require(number(item['milliseconds'], True), 'Invalid region time')
        keys.append((item['candidate'], item['repeat']))
        samples[item['candidate']].append(item['milliseconds'])
    names = ['reference'] + [f'p{p}_q{q}' for p in (0, 1) for q in (64, 128, 256)]
    require(Counter(keys) == Counter((n, i) for n in names for i in range(7)),
            'Incomplete or repeated region timings')
    medians = {n: statistics.median(samples[n]) for n in names}
    gains = {n: 100 * (1 - medians[n] / medians['reference']) for n in names[1:]}
    selected = max(gains, key=gains.get)
    require(selected == 'p1_q128' and gains[selected] >= 5, 'Frozen region selection no longer qualifies')
    debug = [v for rows in r['checks'].values() for v in rows]
    warmup = [v for rows in r['production_warmup_checks'].values() for v in rows]
    require(len(debug) == 36 and len(warmup) == 14 and
            all(v.get('passed') is True for v in debug + warmup), 'Region numerical checks failed')
    counts = {}
    for mode in (0, 1):
        core, edge = (records[f'{name}-{mode}-r1.json'] for name in ('mode', 'edges'))
        for d in (core, edge):
            require(d['status'] == 'complete' and d['mode'] == mode and
                    d['runtime'] == '1.29.0' and d['providers'] == ['CPUExecutionProvider'] and
                    d['library_sha256'] == LIB_SHA and d['gpu_used'] is False,
                    'Native evidence incomplete or changed')
        require(len(core['records']) == core['record_count'] == 288 and
                len(core['additional_checks']) == 204 and
                len(core['malformed_cases_rejected']) == 30 and
                len(edge['records']) == edge['record_count'] == 171 and
                len(edge['malformed_cases_rejected']) == edge['malformed_count'] == 23,
                'Native coverage changed')
        counts[str(mode)] = {'geometry_records': 288, 'additional_checks': 204,
                             'edge_records': 171, 'malformed_rejections': 53}
    return {'status': 'complete', 'source_model_sha256': SOURCE_SHA,
            'selected_region': selected, 'region_median_ms': medians[selected],
            'reference_region_median_ms': medians['reference'],
            'region_time_reduction_percent': gains[selected],
            'region_debug_checks_passed': len(debug), 'production_warmup_checks_passed': len(warmup),
            'native_checks_by_projection_mode': counts, 'evidence_sha256': EVIDENCE_SHAS}


def summarize(result_path, evidence_dir, region_path=None, hardware_path=None):
    output = {'status': 'pending', 'accepted': None, 'source_model_sha256': SOURCE_SHA,
              'selected_graph_sha256': GRAPH_SHA, 'selected_library_sha256': LIB_SHA,
              'scope': 'FP32 CPU decoder fresh full calls; saved numerical parity, not perceptual quality scores',
              'evidence': evidence(Path(evidence_dir), region_path)}
    if not Path(result_path).is_file():
        output['pending_reason'] = 'Final results file is not present'
        return output
    d = read(result_path)
    output.update(results_sha256=sha(result_path), campaign_status=d.get('status'),
                  config_sha256=d.get('config_sha256'), harness_sha256=d.get('harness_sha256'))
    rows, times = d.get('validation', []), d.get('measurements', [])
    failures = [r for r in rows if r.get('passed') is not True or
                ('vs_stock' in r and r['vs_stock'].get('passed') is not True)]
    output['validation'] = {
        'actual_records': len(rows), 'expected_records': 264,
        'real_waveform_records': sum(r.get('test') == 'real_waveform' for r in rows),
        'expected_real_waveform_records': 240,
        'actual_records_by_model': dict(Counter(r.get('model') for r in rows)),
        'failed_records': failures, 'harness_failures': d.get('failed', []),
        'actual_timing_records': len(times), 'expected_timing_records': 200}
    if d.get('status') not in ('complete', 'failed'):
        output['pending_reason'] = 'Harness has not completed; partial measurements are not an acceptance result'
        return output
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    c, protocol = d.get('config', {}), d.get('protocol', {})
    # The preparation helper writes this exact JSON format; the harness embeds
    # the loaded config without changing it. Verify the recorded config SHA too.
    config_bytes = (json.dumps(c, indent=2, allow_nan=False) + '\n').encode()
    check(hashlib.sha256(config_bytes).hexdigest() == d.get('config_sha256'),
          'Embedded config does not match the preparation-format config SHA')
    uids = c.get('expected_uids', [])
    check(c.get('expected_case_count') == 60 and canonical_sha(uids) == UIDS_SHA,
          'Expected 60-case cohort or original order changed')
    check(c.get('timing_uids') == TIMING_UIDS, 'Original ten timing clips or order changed')
    check(d.get('harness_sha256') == HARNESS_SHA, 'Public harness SHA changed')
    check(d.get('case_hashes') == CASE_SHAS, 'Saved latent/reference arrays changed')
    provenance = c.get('upsample_campaign_provenance', {})
    for key, value in [('source_config_sha256', SOURCE_CONFIG_SHA), ('source_stage_mkl_sha256', SOURCE_SHA),
                       ('selected_graph_sha256', GRAPH_SHA), ('selected_library_sha256', LIB_SHA)]:
        check(provenance.get(key) == value, 'Candidate provenance changed: ' + key)
    check(d.get('verified_artifact_sha256') == c.get('artifact_sha256'), 'Verified artifact map differs from config')
    contracts = {'audio': {'sample_rate': 48000, 'hop': 1920, 'latent_channels': 64},
                 'mimi': {'sample_rate': 24000, 'hop': 1920, 'latent_channels': 32}}
    check(c.get('kind_contracts') == contracts, 'Sample rate, hop or latent contract changed')
    meta = d.get('case_metadata', {})
    for kind in ('audio', 'mimi'):
        check(canonical_sha(meta.get(kind)) == META_SHAS[kind], 'Companion metadata changed: ' + kind)
        check(meta.get(kind, {}).get('timed_ids') == uids, 'All 60 companion metadata IDs are required')
    expected_protocol = {'threads': '2', 'precision': 'FP32', 'batch': 1,
                         'provider': 'CPUExecutionProvider', 'inter_op_threads': 1,
                         'execution': 'sequential', 'graph_optimization': 'all', 'spinning': False,
                         'warmups_per_timed_shape': 2, 'repeats': 5, 'validation_scope': 'all',
                         'timing_scope': 'frozen_config_override',
                         'timing_ids': {k: TIMING_UIDS for k in ('audio', 'mimi')}}
    for key, value in expected_protocol.items():
        check(protocol.get(key) == value, 'Protocol changed: ' + key)
    check(d.get('onnxruntime') == '1.29.0', 'Runtime changed')
    check(d.get('machine') == 'x86_64' and str(d.get('platform')).startswith('Linux'), 'Intel Linux target changed')
    affinity = d.get('effective_affinity_by_threads', {}).get('2', {})
    check(affinity.get('cpus') == [0, 1] and affinity.get('applied_before_session_creation') is True,
          'Expected two-CPU affinity was not applied before session creation')
    visibility = {'CUDA_VISIBLE_DEVICES': '-1', 'NVIDIA_VISIBLE_DEVICES': 'void',
                  'ROCR_VISIBLE_DEVICES': '-1', 'HIP_VISIBLE_DEVICES': '-1'}
    check(all(d.get('cpu_only_visibility_environment', {}).get(k) == v for k, v in visibility.items()),
          'CPU-only visibility settings changed')
    for key in ('gpu_library_mappings_start', 'gpu_library_mappings_end'):
        check(d.get(key, {}).get('available') is True and d[key].get('basenames') == [],
              'GPU library mapping evidence missing or nonempty: ' + key)
    expected_libraries = {'audio_stock': [], 'stage_mkl': LIBS,
                          'upsample_stage4': LIBS + [LIB_SHA], 'mimi': []}
    checks = d.get('session_provider_checks', [])
    check(Counter(r.get('model') for r in checks) == Counter(MODELS), 'Missing or duplicate session evidence')
    for row in checks:
        check(row.get('providers') == ['CPUExecutionProvider'] and row.get('threads') == 2 and
              row.get('profile') is False, 'Unexpected execution provider or session protocol')
        check([v.get('sha256') for v in row.get('custom_libraries', [])] ==
              expected_libraries.get(row.get('model')), 'Loaded native library sequence changed')
    check(not d.get('profile_provider_checks') and not d.get('profiles'), 'Final campaign must be unprofiled')
    for key in ('models',):
        check([r.get('name') for r in d.get(key, [])] == MODELS, 'Result model order changed')
        check([r.get('name') for r in c.get(key, [])] == MODELS, 'Config model order changed')
    artifact_map = c.get('artifact_sha256', {})
    config_models = {m.get('name'): m for m in c.get('models', [])}
    for row in d.get('models', []):
        name = row.get('name')
        check({k: v for k, v in row.items() if k != 'sha256'} == config_models.get(name),
              'Result model differs from frozen config: ' + str(name))
        check(row.get('sha256') == MODEL_SHAS.get(name) == artifact_map.get(row.get('path')),
              'Model graph SHA changed: ' + str(name))
        check(row.get('causal') is True and row.get('kind') == ('mimi' if name == 'mimi' else 'audio'),
              'Model causal or codec contract changed')
    expected_validation = Counter()
    for name in MODELS:
        expected_validation.update((name, 'real_waveform', uid) for uid in uids)
        if name != 'mimi':
            expected_validation.update((name, 'dynamic_length', n) for n in (1, 2, 7, 17))
        expected_validation.update((name, test, None) for test in
                                   ('future_and_repeated_call', 'shorter_prefix', 'long_short_long'))
    actual_validation = Counter((r.get('model'), r.get('test'),
                                 r.get('uid') if r.get('test') == 'real_waveform' else
                                 r.get('length') if r.get('test') == 'dynamic_length' else None) for r in rows)
    check(actual_validation == expected_validation, 'Validation records missing, duplicated or unexpected')
    check(all(r.get('threads') == 2 for r in rows), 'Validation thread count changed')
    for row in rows:
        test = row.get('test')
        if test == 'real_waveform' and row.get('model') != 'mimi':
            check('vs_stock' in row, 'Audio waveform check missing independent stock comparison')
        for values in (row, row.get('vs_stock', {})):
            for key in ('max_abs', 'rmse', 'prefix_max_abs', 'repeated_max_abs'):
                check(key not in values or number(values[key]), 'Invalid numerical error record')
        if test == 'future_and_repeated_call':
            check(row.get('causality_required') is True, 'Causality requirement weakened')
            if row.get('passed') is True:
                check(row.get('future_invariant') is True and row.get('prefix_max_abs') == 0 and
                      row.get('repeated_max_abs') == 0, 'Passed causal/repeat record has nonzero error')
        if test == 'long_short_long' and row.get('passed') is True:
            check(row.get('max_abs') == 0, 'Passed repeated-call record has nonzero error')
    expected_times = Counter((n, u, rep) for n in MODELS for u in TIMING_UIDS for rep in range(5))
    actual_times = Counter((r.get('model'), r.get('uid'), r.get('repeat')) for r in times)
    check(actual_times == expected_times, 'Expected 200 unique timing observations are missing or changed')
    grouped = defaultdict(list)
    for row in times:
        name, uid = row.get('model'), row.get('uid')
        kind = 'mimi' if name == 'mimi' else 'audio'
        metadata_row = next((v for v in meta.get(kind, {}).get('cases', []) if v.get('uid') == uid), {})
        check(row.get('threads') == 2 and row.get('backend') == 'ort', 'Unexpected timing backend or threads')
        check(number(row.get('seconds'), True) and number(row.get('audio_s'), True), 'Invalid timing number')
        check(type(row.get('samples')) is int and row['samples'] == metadata_row.get('reference_samples'),
              'Output sample count differs from frozen reference')
        if number(row.get('audio_s'), True) and number(row.get('samples'), True):
            check(math.isclose(row['audio_s'], row['samples'] / contracts[kind]['sample_rate'],
                               rel_tol=1e-12, abs_tol=1e-12), 'Audio duration denominator changed')
        grouped[name, uid].append(row)
    hardware = None
    if hardware_path:
        check(sha(hardware_path) == d.get('hardware_diagnostic_sha256'), 'Hardware evidence SHA changed')
        hardware = read(hardware_path)
    output['execution'] = {'runtime': d.get('onnxruntime'), 'precision': protocol.get('precision'),
                           'providers': sorted({p for r in checks for p in r.get('providers', [])}),
                           'threads': 2, 'affinity': affinity.get('cpus'),
                           'machine': d.get('machine'), 'platform': d.get('platform'),
                           'cpu_models': hardware.get('cpu', {}).get('models') if hardware else None,
                           'hardware_diagnostic_sha256': d.get('hardware_diagnostic_sha256'),
                           'cpu_model_note': None if hardware else 'Pass --hardware to verify the saved CPU model name',
                           'host_timing_delta': d.get('cpu_stat_by_threads', {}).get('2', {}).get('host_timing_delta')}
    if errors:
        output.update(status='invalid', accepted=False, errors=sorted(set(errors)))
        return output
    aggregate, clip_rows, repeat_values = {}, [], {}
    for name in MODELS:
        model_times = [r for r in times if r['model'] == name]
        aggregate[name] = sum(r['seconds'] for r in model_times) / sum(r['audio_s'] for r in model_times)
        saved = d.get('summary', {}).get(name, {}).get('2', {})
        check(saved.get('observations') == 50 and number(saved.get('decoder_rtf'), True) and
              math.isclose(saved['decoder_rtf'], aggregate[name], rel_tol=1e-12),
              'Saved aggregate differs from recomputed RTF: ' + name)
        repeat_rtfs = [sum(r['seconds'] for r in model_times if r['repeat'] == rep) /
                       sum(r['audio_s'] for r in model_times if r['repeat'] == rep) for rep in range(5)]
        repeat_values[name] = repeat_rtfs
        check(len(saved.get('repeat_rtfs', [])) == 5 and all(
            number(a, True) and math.isclose(a, b, rel_tol=1e-12)
            for a, b in zip(saved.get('repeat_rtfs', []), repeat_rtfs)), 'Saved repeat RTFs differ: ' + name)
    for uid in TIMING_UIDS:
        means = {n: statistics.mean(r['seconds'] for r in grouped[n, uid]) for n in MODELS}
        rtfs = {n: means[n] / grouped[n, uid][0]['audio_s'] for n in MODELS}
        ratio = means['stage_mkl'] / means['upsample_stage4']
        clip_rows.append({'uid': uid, 'mean_seconds': means, 'mean_rtf': rtfs,
                          'speedup_vs_stage_mkl': ratio,
                          'time_reduction_percent_vs_stage_mkl': 100 * (1 - 1 / ratio),
                          'improved': means['upsample_stage4'] < means['stage_mkl']})
    reduction = 100 * (1 - aggregate['upsample_stage4'] / aggregate['stage_mkl'])
    gates = {'campaign_complete': d['status'] == 'complete',
             'all_numerical_checks_passed': not failures and d.get('failed') == [],
             'aggregate_time_reduction_at_least_2_percent': reduction >= 2,
             'all_ten_clip_means_improved': all(r['improved'] for r in clip_rows)}
    repeat_rows = []
    for rep in range(5):
        rtfs = {name: repeat_values[name][rep] for name in MODELS}
        repeat_rows.append({'repeat': rep, 'aggregate_decoder_rtf': rtfs,
                            'time_reduction_percent_vs_stage_mkl':
                            100 * (1 - rtfs['upsample_stage4'] / rtfs['stage_mkl'])})
    reductions = [row['time_reduction_percent_vs_stage_mkl'] for row in repeat_rows]
    dispersion = {
        'scope': 'Descriptive variation across five randomized rounds, not a confidence interval or extra acceptance gate',
        'candidate_faster_round_count': sum(v > 0 for v in reductions),
        'reduction_percent_min': min(reductions), 'reduction_percent_max': max(reductions),
        'reduction_percent_sample_stdev': statistics.stdev(reductions),
        'rtf_by_model': {name: {'min': min(values), 'max': max(values),
                               'sample_stdev': statistics.stdev(values),
                               'coefficient_of_variation_percent':
                               100 * statistics.stdev(values) / statistics.mean(values)}
                         for name, values in repeat_values.items()}}
    output.update(status='invalid' if errors else 'complete',
                  accepted=not errors and all(gates.values()), acceptance_gates=gates,
                  aggregate_decoder_rtf=aggregate,
                  aggregate_time_reduction_percent_vs_stage_mkl=reduction,
                  aggregate_speedup_vs_stage_mkl=aggregate['stage_mkl'] / aggregate['upsample_stage4'],
                  per_clip=clip_rows, per_repeat=repeat_rows, repeat_dispersion=dispersion,
                  mimi_gap={'candidate_time_percent_above_mimi':
                            100 * (aggregate['upsample_stage4'] / aggregate['mimi'] - 1),
                            'further_candidate_time_reduction_percent_to_match':
                            max(0, 100 * (1 - aggregate['mimi'] / aggregate['upsample_stage4']))},
                  errors=sorted(set(errors)))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--evidence-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--region', help='Optional relocated copy of the exact frozen region JSON')
    p.add_argument('--hardware', help='Optional hardware JSON matching the result-recorded SHA')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        result = summarize(args.results, args.evidence_dir, args.region, args.hardware)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result = {'status': 'invalid', 'accepted': False, 'source_model_sha256': SOURCE_SHA,
                  'errors': [str(exc)]}
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(encoded)
    print(encoded, end='')
    return 3 if result['status'] == 'pending' else 2 if result['status'] == 'invalid' else 0


if __name__ == '__main__':
    raise SystemExit(main())
