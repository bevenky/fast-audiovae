"""Fixed-panel continuation decisions with explicit optimizer-step horizons.

Review heuristics never change training losses or quiet acceptance limits.
"""
from dataclasses import replace
import hashlib
import json
import math

import joint_recovery_gates_v1 as old

VERSION = 'audiovae2_joint_recovery_gates_v2'
START_STEP, TARGET_STEP, EVERY = 5625, 10000, 500
REGIONS = (
    'all_quiet', 'near_silence', 'near_startup_first20ms',
    'source_zero_20to40ms', 'source_zero_after40ms',
    'near_after800ms', 'quiet_nonzero_reference',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def summarize_regions(windows):
    if not windows or len({w['window_id'] for w in windows}) != len(windows):
        raise ValueError('Expected unique quiet windows')
    old._check_finite_tree(windows)
    selectors = {
        'all_quiet': lambda w: True,
        'near_silence': lambda w: w['teacher_rms'] <= 1e-5,
        'near_startup_first20ms': lambda w: w['teacher_rms'] <= 1e-5 and w['source_start_sample'] < 960,
        'source_zero_20to40ms': lambda w: w['source_reference_exact_zero'] and 960 <= w['source_start_sample'] < 1920,
        'source_zero_after40ms': lambda w: w['source_reference_exact_zero'] and w['source_start_sample'] >= 1920,
        'near_after800ms': lambda w: w['teacher_rms'] <= 1e-5 and w['source_start_sample'] >= 38400,
        'quiet_nonzero_reference': lambda w: not w['source_reference_exact_zero'],
    }
    fixed = ['window_id', 'source_id', 'source_start_sample', 'source_stop_sample', 'valid_samples',
             'teacher_rms', 'residual_limit', 'output_rms_limit', 'source_reference_exact_zero']
    identities = [{k: w[k] for k in fixed} for w in windows]
    result = {'version': VERSION, 'window_identity_sha256': digest(identities), 'regions': {}}
    for name, select in selectors.items():
        chosen = [w for w in windows if select(w)]
        n = sum(w['valid_samples'] for w in chosen)
        error = sum(w['residual_square_sum'] for w in chosen)
        centered = sum(w['centered_residual_square_sum'] for w in chosen)
        pe = sum(w['student_rms'] ** 2 * w['valid_samples'] for w in chosen)
        te = sum(w['teacher_rms'] ** 2 * w['valid_samples'] for w in chosen)
        excess = sum(max(w['student_rms'] - w['output_rms_limit'], 0.) ** 2 * w['valid_samples'] for w in chosen)
        cats = {c: sum(w['failure_category'] == c for w in chosen)
                for c in ('passed', 'residual_only', 'amplitude_only', 'both')}
        if sum(cats.values()) != len(chosen) or any(not w['is_quiet'] or w['valid_samples'] <= 0 for w in chosen):
            raise ValueError('Invalid quiet-window category or extent')
        result['regions'][name] = {
            'windows': len(chosen), 'samples': n, 'window_ids_sha256': digest([w['window_id'] for w in chosen]),
            'failure_categories': cats, 'failed': len(chosen) - cats['passed'],
            'amplitude_failed': cats['amplitude_only'] + cats['both'],
            'residual_rms': math.sqrt(error/n) if n else None,
            'centered_residual_rms': math.sqrt(centered/n) if n else None,
            'teacher_rms': math.sqrt(te/n) if n else None,
            'output_rms': math.sqrt(pe/n) if n else None,
            'output_limit_excess_rms': math.sqrt(excess/n) if n else None,
        }
    return result


def _validate_regions(reports):
    baseline = reports[0]['quiet_regions']
    for report in reports:
        value = report['quiet_regions']
        old._check_finite_tree(value)
        if value['version'] != VERSION or value['window_identity_sha256'] != baseline['window_identity_sha256']:
            raise ValueError('Quiet teacher/window identities changed')
        if set(value['regions']) != set(REGIONS):
            raise ValueError('Quiet region set changed')
        for name in REGIONS:
            row, first = value['regions'][name], baseline['regions'][name]
            for key in ('windows', 'samples', 'window_ids_sha256', 'teacher_rms'):
                if row[key] != first[key]:
                    raise ValueError('Quiet region coverage/teacher changed: ' + name)
            n = old._count(row['samples'], 'region samples')
            count = old._count(row['windows'], 'region windows')
            cats = row['failure_categories']
            if set(cats) != {'passed', 'residual_only', 'amplitude_only', 'both'}:
                raise ValueError('Quiet category set changed')
            if sum(old._count(v, 'category count') for v in cats.values()) != count:
                raise ValueError('Quiet category accounting differs')
            if row['failed'] != count-cats['passed'] or row['amplitude_failed'] != cats['amplitude_only']+cats['both']:
                raise ValueError('Quiet failure accounting differs')
            for key in ('residual_rms', 'centered_residual_rms', 'output_rms', 'output_limit_excess_rms'):
                if n:
                    old._number(row[key], 'region ' + key, minimum=0)
                elif row[key] is not None:
                    raise ValueError('Nonempty metric on empty region')
        all_quiet = value['regions']['all_quiet']
        if all_quiet['windows'] != report['aggregate']['quiet_windows'] or all_quiet['failed'] != report['aggregate']['quiet_failed_windows']:
            raise ValueError('Separated regions disagree with original evaluation')


def _region_regressions(reports, index):
    now = reports[index]['quiet_regions']['regions']
    refs = [('start', reports[0]['quiet_regions']['regions'])]
    if index > 1:
        refs.append(('previous', reports[index-1]['quiet_regions']['regions']))
    flags = {}
    for name in REGIONS:
        if now[name]['samples'] < 960:
            continue
        for metric in ('residual_rms', 'output_limit_excess_rms'):
            absolute = 1e-5 if metric == 'residual_rms' and name in (
                'all_quiet', 'quiet_nonzero_reference', 'near_startup_first20ms', 'source_zero_20to40ms') else 1e-6
            for label, reference in refs:
                before, after = reference[name][metric], now[name][metric]
                if old._material(after, before, .10, absolute):
                    flag = flags.setdefault((name, metric), {'region': name, 'metric': metric, 'references': []})
                    flag['references'].append({'name': label, 'before': before, 'after': after})
    return flags


def classify_review(history_reports, steps, policy=old.ReviewPolicy()):
    try:
        if len(history_reports) != len(steps) or not steps or steps[0] != START_STEP:
            raise ValueError('Expected fixed step5625 baseline and paired report steps')
        if any(type(s) is not int for s in steps):
            raise ValueError('Optimizer steps must be integers')
        expected = list(range(START_STEP, TARGET_STEP, EVERY)) + [TARGET_STEP]
        if steps != expected[:len(steps)]:
            raise ValueError('Review cadence changed or skipped a report')
        _validate_regions(history_reports)
        # Two full500-update intervals equal the authorized1000-update horizon.
        full_interval = len(steps) > 1 and steps[-1]-steps[-2] == EVERY
        effective = replace(policy, minimum_stall_reviews=2 if full_interval else 10**9)
        result = old.classify_review(history_reports, effective)
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        return {'version': VERSION, 'action': 'pause_for_diagnosis', 'flags': [
            {'kind': 'invalid_or_invariant_failure', 'detail': str(exc)}], 'automatic_freezing': False}
    result.update(version=VERSION, optimizer_steps=list(steps), updates_since_start=steps[-1]-START_STEP,
                  region_material_regressions=[], region_repeated_regressions=[], region_stalls=[])
    if result['action'] == 'pause_for_diagnosis' and any(f['kind'] == 'invalid_or_invariant_failure' for f in result['flags']):
        return result
    if len(steps) > 1:
        latest = _region_regressions(history_reports, len(steps)-1)
        previous = _region_regressions(history_reports, len(steps)-2) if len(steps) > 2 else {}
        repeated = [latest[key] for key in sorted(set(latest)&set(previous))]
        result['region_material_regressions'] = list(latest.values())
        result['region_repeated_regressions'] = repeated
        if repeated:
            result['flags'].append({'kind': 'repeated_region_regression', 'affected': repeated})
        now, prior = (r['quiet_regions']['regions'] for r in history_reports[-2:][::-1])
        for name in REGIONS:
            if now[name]['amplitude_failed'] > prior[name]['amplitude_failed']:
                result['alerts'].append({'kind': 'region_count_alert', 'region': name,
                    'before': prior[name]['amplitude_failed'], 'after': now[name]['amplitude_failed']})
    # A component must stall across TWO1000-update segments, not merely two reviews.
    if steps[-1]-START_STEP >= 2000 and (steps[-1]-START_STEP) % 1000 == 0:
        indices = [steps.index(steps[-1]-offset) for offset in (2000, 1000, 0)]
        for name in ('near_startup_first20ms', 'source_zero_20to40ms', 'source_zero_after40ms', 'near_after800ms'):
            rows = [history_reports[i]['quiet_regions']['regions'][name] for i in indices]
            if rows[-1]['samples'] < 960 or rows[-1]['failed'] == 0:
                continue
            progress = []
            for a, b in zip(rows, rows[1:]):
                progress.append(any(a[key] > 0 and (a[key]-b[key])/a[key] >= .002
                    for key in ('residual_rms', 'output_limit_excess_rms')))
            if not any(progress):
                result['region_stalls'].append({'region': name, 'optimizer_steps': [steps[i] for i in indices]})
        if result['region_stalls']:
            result['flags'].append({'kind': 'two_segment_region_stall', 'affected': result['region_stalls']})
    if result['flags']:
        result['action'] = 'pause_for_diagnosis'
    result['step_horizons'] = {'overall_stall': 1000, 'component_stall': 2000,
                             'last_interval_is_full500': full_interval}
    return result


def log_quiet_regions(writer, value, step):
    for name, row in value['regions'].items():
        for key in ('residual_rms', 'output_rms', 'output_limit_excess_rms', 'failed', 'amplitude_failed'):
            if row[key] is not None:
                writer.add_scalar('quality/all/quiet_regions/'+name+'/'+key, row[key], step)
    for label, name in (
        ('Startup near-silence passing (%)', 'near_startup_first20ms'),
        ('Sustained source silence passing (%)', 'source_zero_after40ms'),
        ('Interior near-silence passing (%)', 'near_after800ms'),
    ):
        row = value['regions'][name]
        if row['windows']:
            writer.add_scalar('overview/'+label, 100*(row['windows']-row['failed'])/row['windows'], step)
