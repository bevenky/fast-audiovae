"""Summarize saved reconstruction reports without running any model."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _change(now, before):
    if now is None or before is None:
        return None
    return 100 * (now / before - 1) if before else (0. if now == 0 else None)


def _natural_sources(report):
    return {k: v for k, v in report['recovery_metrics'].items()
            if k.startswith('source/') and not k.startswith('source/encoded_')}


def natural(report):
    sources = _natural_sources(report)
    rows = list(sources.values())
    audio_rows = [row for row in report['rows'] if 'source/' + row['source_id'] in sources]
    samples = sum(v['samples'] for v in rows)
    crops = sum(v['crops'] for v in rows)
    quiet = sum(v['quiet_samples'] for v in rows)
    if not samples or not crops or not rows:
        raise ValueError('No natural scored audio in report')
    if len(audio_rows) != crops or sum(v['samples'] for v in audio_rows) != samples:
        raise ValueError('Natural crop rows and source summaries disagree')
    def quiet_rms(key):
        return math.sqrt(sum((v[key] or 0.)**2 * v['quiet_samples'] for v in rows) / quiet) if quiet else None
    nonquiet = sum(v['nonquiet_crops'] for v in rows)
    return {
        'sources': len(rows), 'samples': samples, 'crops': crops,
        'mae': sum(v['raw_mae'] * v['samples'] for v in rows) / samples,
        'mel': sum(v['mel'] * v['crops'] for v in rows) / crops,
        'source_equal_mel': sum(v['mel'] for v in rows) / len(rows),
        'mel_linear': sum(v['teacher_mel_linear'] for v in audio_rows) / crops,
        'mel_log': sum(v['teacher_mel_log'] for v in audio_rows) / crops,
        'nonquiet_cosine_mean': sum((v['nonquiet_cosine_mean'] or 0.) * v['nonquiet_crops'] for v in rows) / nonquiet if nonquiet else None,
        'quiet_rms': quiet_rms('quiet_residual_rms'),
        'quiet_student_rms': quiet_rms('quiet_student_rms'),
        'quiet_teacher_rms': quiet_rms('quiet_teacher_rms'),
        'quiet_samples': quiet, 'quiet_windows': sum(v['quiet_windows'] for v in rows),
        'quiet_failed_windows': sum(v['quiet_failed_windows'] for v in rows),
        'peak': max(v['maximum_peak'] for v in rows),
        'teacher_peak': max(v['teacher_peak_abs'] for v in audio_rows),
        'overshoot_observations': sum(v['scored_overshoot_samples'] for v in rows),
        'teacher_overshoot_observations': sum(v['teacher_overshoot_samples'] for v in audio_rows),
        'overshoot_sources': sum(v['overshoot_sources'] for v in rows),
        'metric_target': 'Sealed AudioVAE2 post-tanh waveform; errors are student versus teacher.',
        'reduction_policy': 'MAE pools samples; mel averages crops; source_equal_mel averages sources; quiet RMS pools quiet samples. Overlapping crops remain scored observations.',
    }


def pooled_regions(report):
    """Pool saved mel sums by resolution, retaining exact region support."""
    if 'spectral_head_sources' not in report:
        return None
    config = report['criterion_config']
    bands = config['mel_bands']
    source_rows = [r for r in report['spectral_head_sources'] if not r['source_id'].startswith('encoded_')]
    result = {}
    for name in ('all', 'quiet', 'active', 'transition'):
        rows = [r['regions'][name] for r in source_rows]
        counts = [sum(r['element_counts'][i] for r in rows) for i in range(len(bands))]
        linear_sums = [sum(r['linear_sums'][i] for r in rows) for i in range(len(bands))]
        log_sums = [sum(r['log_sums'][i] for r in rows) for i in range(len(bands))]
        active = [i for i, count in enumerate(counts) if count]
        linear = sum(linear_sums[i]/counts[i] for i in active)/len(active) if active else None
        logarithmic = sum(log_sums[i]/counts[i] for i in active)/len(active) if active else None
        result[name] = {'linear': linear, 'log': logarithmic,
            'mel': config['mel_linear_weight']*linear + config['mel_log_weight']*logarithmic if active else None,
            'linear_sums': linear_sums, 'log_sums': log_sums, 'element_counts': counts,
            'frame_observations': [count//band for count, band in zip(counts, bands)],
            'sources_with_support': sum(any(r['element_counts']) for r in rows),
            'active_resolutions': len(active)}
    return result


def compare(report, baseline):
    now, before = natural(report), natural(baseline)
    if _natural_sources(report).keys() != _natural_sources(baseline).keys():
        raise ValueError('Changed natural source identities')
    for key in ('sources', 'samples', 'crops', 'quiet_samples', 'quiet_windows'):
        if now[key] != before[key]:
            raise ValueError('Changed report coverage: ' + key)
    changes = {key: _change(now[key], before[key]) for key in ('mae', 'mel', 'mel_linear', 'mel_log', 'source_equal_mel', 'quiet_rms')}
    old_sources = baseline['recovery_metrics']
    failures = []
    for key, value in _natural_sources(report).items():
        old = old_sources[key]
        if value['mel'] > old['mel'] * 1.01 + 1e-6:
            failures.append({'source': key, 'baseline_mel': old['mel'], 'candidate_mel': value['mel'],
                             'mel_change_percent': _change(value['mel'], old['mel'])})
    old_rows = {(v['source_id'], v['start_frame']): v for v in baseline['rows']}
    current_rows = {(v['source_id'], v['start_frame']): v for v in report['rows']}
    if (old_rows.keys() != current_rows.keys() or len(old_rows) != len(baseline['rows'])
            or len(current_rows) != len(report['rows'])):
        raise ValueError('Changed or repeated canonical crop identities')
    peaks = []
    for key, row in current_rows.items():
        old = old_rows[key]
        if row['student_peak_abs'] > max(1., old['student_peak_abs']) or row['student_overshoot_samples'] > old['student_overshoot_samples']:
            peaks.append({'source': row['source_id'], 'start_frame': row['start_frame'],
                          'old_peak': old['student_peak_abs'], 'new_peak': row['student_peak_abs'],
                          'teacher_peak': row['teacher_peak_abs'],
                          'old_count': old['student_overshoot_samples'], 'new_count': row['student_overshoot_samples']})
    regions = []
    available = 'spectral_head_sources' in report and 'spectral_head_sources' in baseline
    if available:
        old_spectra = {v['source_id']: v for v in baseline['spectral_head_sources']}
        new_spectra = {v['source_id']: v for v in report['spectral_head_sources']}
        if old_spectra.keys() != new_spectra.keys():
            raise ValueError('Changed spectral source identities')
        for source_id, source in new_spectra.items():
            if source_id.startswith('encoded_'):
                continue
            for region, values in source['regions'].items():
                old = old_spectra[source_id]['regions'][region]
                if values['element_counts'] != old['element_counts']:
                    raise ValueError('Changed spectral coverage')
                if (values['mel'] is None) != (old['mel'] is None):
                    raise ValueError('Changed spectral region availability')
                if values['mel'] is not None and values['mel'] > old['mel'] * 1.01 + 1e-6:
                    regions.append({'source': source_id, 'region': region,
                        'baseline_mel': old['mel'], 'candidate_mel': values['mel'],
                        'mel_change_percent': _change(values['mel'], old['mel']),
                        'baseline_linear': old['linear'], 'candidate_linear': values['linear'],
                        'linear_change_percent': _change(values['linear'], old['linear']),
                        'baseline_log': old['log'], 'candidate_log': values['log'],
                        'log_change_percent': _change(values['log'], old['log']),
                        'element_counts': values['element_counts'],
                        'frame_observations': [c//b for c, b in zip(values['element_counts'], report['criterion_config']['mel_bands'])]})
    current_regions, previous_regions = pooled_regions(report), pooled_regions(baseline)
    regional_changes = None
    if available:
        regional_changes = {name: {'baseline': previous_regions[name], 'candidate': current_regions[name],
            'changes_percent': {k: _change(current_regions[name][k], previous_regions[name][k]) for k in ('mel', 'linear', 'log')}}
            for name in current_regions}
    steady = report['encoded_zero_steady_2_to_6_seconds']['quiet_residual_rms']
    old_steady = baseline['encoded_zero_steady_2_to_6_seconds']['quiet_residual_rms']
    return {'natural': now, 'reference_natural': before, 'changes_percent': changes,
            'stationary_rms': steady, 'reference_stationary_rms': old_steady,
            'stationary_change_percent': _change(steady, old_steady),
            'stationary_teacher_rms': report['encoded_zero_steady_2_to_6_seconds'].get('quiet_teacher_rms'),
            'stationary_student_rms': report['encoded_zero_steady_2_to_6_seconds'].get('quiet_student_rms'),
            'source_mel_failures': failures, 'peak_regressions': peaks,
            'source_region_failures': regions if available else None,
            'regional_spectra_available_in_both_reports': available,
            'regional_spectral_comparison': regional_changes}


def calibration_updates(rate):
    """Retain actual isolated-update measurements, not only pass/fail labels."""
    rows = []
    for trial_index, trial in enumerate(rate.get('trials', [])):
        for outcome in trial.get('outcomes', []):
            checks = outcome.get('checks', {}).get('checks', [])
            peak_checks = [check for check in checks if check['name'].startswith('peak/')]
            by_name = {check['name']: check for check in peak_checks}
            rows.append({'trial': trial_index, 'learning_rate': trial['learning_rate'],
                'arm': outcome['arm'], 'batch': outcome.get('batch'), 'passed': outcome['passed'],
                'peak': by_name.get('peak/aggregate'), 'overshoot_count': by_name.get('peak/aggregate_count'),
                'affected_sources': by_name.get('peak/no_new_source'), 'all_peak_checks': peak_checks,
                'teacher_error_checks': [c for c in checks if not c['name'].startswith('peak/')],
                'source_spectral': outcome.get('checks', {}).get('source_spectral'),
                'update': outcome.get('update'), 'migration_parity': outcome.get('migration_parity'),
                'actual_peak_values_logged': all(k in by_name.get('peak/aggregate', {}) for k in ('baseline', 'candidate'))})
    return rows


def run(directory, original):
    read = lambda path: json.loads(path.read_text())
    baseline_path = directory / 'baseline-canonical.json'
    baseline = read(baseline_path)
    original_report = read(original)
    result = {'directory': str(directory), 'original_baseline': str(original),
              'original_baseline_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'retained_baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              'baseline': natural(baseline), 'original_8890': natural(original_report), 'arms': {}}
    for name in ('identity', 'coefficient-calibration', 'rate-calibration', 'comparisons', 'complete',
                 'preservation', 'fit-accounting', 'auxiliary-coefficient-calibration', 'auxiliary-ridge',
                 'auxiliary-identity', 'teacher-feature-accounting'):
        path = directory / (name + '.json')
        if path.exists():
            result[name] = read(path)
    if 'rate-calibration' in result:
        result['isolated_calibration_updates'] = calibration_updates(result['rate-calibration'])
    for name in ('baseline-validation', 'baseline-training-panel'):
        path = directory / (name + '.json')
        if path.exists():
            result[name] = read(path)
    for path in sorted(directory.glob('*-final-canonical.json')):
        arm = path.name.removesuffix('-final-canonical.json')
        report = read(path)
        result['arms'][arm] = {
            'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'versus_retained_candidate': compare(report, baseline),
            'versus_original_8890': compare(report, original_report)}
        for suffix in ('gradient-probes', 'training-panel', 'validation'):
            extra = directory / (arm + '-' + suffix + '.json')
            if extra.exists():
                value = read(extra)
                if suffix == 'validation':
                    value = [{'step': v['step'], 'selection': v['selection'],
                              'aggregate': v['score']['aggregate'],
                              'spectral_aggregate': v['score'].get('spectral_aggregate')} for v in value]
                result['arms'][arm][suffix] = value
    output = directory / 'audit-summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'arms': {k: {
        **v['versus_retained_candidate']['changes_percent'],
        'stationary_change_percent': v['versus_retained_candidate']['stationary_change_percent'],
        'source_mel_failures': len(v['versus_retained_candidate']['source_mel_failures']),
        'peak_regressions': len(v['versus_retained_candidate']['peak_regressions']),
        'natural': v['versus_retained_candidate']['natural']}
        for k, v in result['arms'].items()}}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.original)
