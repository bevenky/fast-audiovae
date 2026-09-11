"""Derived window statistics only. Reads existing evidence, does not load models."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] * (hi - at) + values[hi] * (at - lo) if lo != hi else values[lo]


def summary(rows):
    n = sum(w['valid_samples'] for w in rows)
    e = sum(w['residual_square_sum'] for w in rows)
    ac = sum(w['centered_residual_square_sum'] for w in rows)
    te = sum(w['teacher_square_sum'] for w in rows)
    pe = sum(w['prediction_square_sum'] for w in rows)
    gain_error = sum((w['dot_sum'] / w['teacher_square_sum'] - 1) ** 2 * w['teacher_square_sum']
                     for w in rows if w['teacher_square_sum'] > 0)
    stats = {'windows': len(rows), 'sources': len({w['source_id'] for w in rows}),
             'categories': dict(Counter(w['failure_category'] for w in rows)),
             'samples': n, 'error_energy': e,
             'residual_rms': math.sqrt(e / n) if n else None,
             'dc_error_energy_fraction': (e - ac) / e if e else None,
             'per_window_gain_error_energy_fraction': gain_error / e if e else None,
             'teacher_rms': math.sqrt(te / n) if n else None,
             'student_rms': math.sqrt(pe / n) if n else None,
             'output_to_teacher_rms': math.sqrt(pe / te) if te else None}
    for key in ('residual_over_limit', 'output_rms_over_limit'):
        values = [w[key] for w in rows]
        stats[key] = {'median': quantile(values, .5), 'p95': quantile(values, .95),
                      'max': max(values) if values else None}
    return stats


def category(w):
    if w['teacher_rms'] <= 1e-5:
        return 'near_startup_first20ms' if w['source_start_sample'] < 960 else 'other_near_silence'
    return 'remaining_quiet'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'results'
    completed = json.loads((out / 'completed.json').read_text())
    if completed['status'] != 'completed' or not completed['original_files_preserved'] or not all(completed['state_preserved'].values()):
        raise RuntimeError('Requires completed preserved diagnostic evidence')
    rows = {}
    for step in (0, 5000):
        with gzip.open(out / f'quiet-windows-step{step}.jsonl.gz', 'rt') as handle:
            rows[step] = [json.loads(line) for line in handle]
    by_id = {step: {w['window_id']: w for w in items} for step, items in rows.items()}
    if any(len(by_id[s]) != len(rows[s]) for s in rows) or set(by_id[0]) != set(by_id[5000]):
        raise RuntimeError('Duplicated or changed quiet window identities')
    fixed = ('source_id', 'source_start_sample', 'source_stop_sample', 'valid_samples', 'teacher_rms',
             'residual_limit', 'output_rms_limit', 'source_reference_exact_zero')
    changes = []
    for key, a in by_id[0].items():
        b = by_id[5000][key]
        if any(a[k] != b[k] for k in fixed):
            raise RuntimeError('Changed source/teacher/metric definition')
        changes.append({'source_id': a['source_id'], 'category': category(a),
                        'improved': b['residual_rms'] < a['residual_rms'],
                        'ratio': b['residual_rms'] / a['residual_rms'] if a['residual_rms'] else None,
                        'transition': a['failure_category'] + ' -> ' + b['failure_category']})
    final = rows[5000]
    result = {'windows': len(final), 'initial': summary(rows[0]), 'final': summary(final),
              'disjoint': {}, 'remaining_quiet_splits': {}, 'worst_sources_by_error_energy': [],
              'paired': {'improved': sum(c['improved'] for c in changes), 'total': len(changes),
                         'transitions': dict(Counter(c['transition'] for c in changes))}}
    for name in ('near_startup_first20ms', 'other_near_silence', 'remaining_quiet'):
        selected = [w for w in final if category(w) == name]
        old = [w for w in rows[0] if category(w) == name]
        cs = [c for c in changes if c['category'] == name]
        result['disjoint'][name] = {'initial': summary(old), 'final': summary(selected),
                                   'improved': sum(c['improved'] for c in cs),
                                   'residual_ratio_median': quantile([c['ratio'] for c in cs if c['ratio'] is not None], .5)}
    remaining = [w for w in final if category(w) == 'remaining_quiet']
    for key in ('source_reference_exact_zero', 'temporal_bin', 'adjacent_active_window', 'teacher_level_bin', 'partial_window'):
        result['remaining_quiet_splits'][key] = {
            str(value): summary([w for w in remaining if w[key] == value]) for value in sorted({w[key] for w in remaining})}
    sources = defaultdict(list)
    for w in final:
        sources[w['source_id']].append(w)
    ranked = sorted(((sid, summary(ws)) for sid, ws in sources.items()), key=lambda x: x[1]['error_energy'], reverse=True)
    result['worst_sources_by_error_energy'] = [{'source_id': sid, **stats} for sid, stats in ranked[:8]]
    result['startup_examples'] = [{k: w[k] for k in ('source_id', 'source_start_sample', 'source_stop_sample',
        'source_reference_exact_zero', 'teacher_rms', 'student_rms', 'residual_rms', 'residual_limit', 'residual_over_limit',
        'output_rms_over_limit', 'failure_category', 'dc_energy_fraction')} for w in final if category(w) == 'near_startup_first20ms']
    (args.root / 'derived-quiet-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
