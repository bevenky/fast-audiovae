"""Compact derived statistics from saved diagnostic evidence; no model calls."""
import gzip
import json
import math
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
ROOT = args.root
out = ROOT / 'results'
completed = json.loads((out / 'completed.json').read_text())
if completed['status'] != 'completed':
    raise RuntimeError('Do not interpret an incomplete or failed accounting diagnostic')
roles = {}
with gzip.open(out / 'source-diagnostics.jsonl.gz', 'rt') as handle:
    for line in handle:
        source = json.loads(line)
        for index, window in enumerate(source['windows']):
            n = window['tensor_stop'] - window['tensor_start']
            for name in window['roles']:
                role = roles.setdefault(name, {'windows': 0, 'samples': 0, 'variants': {}, 'boundaries': {}, 'operations': {}})
                role['windows'] += 1
                role['samples'] += n
                for variant, values in window['waveforms_vs_native'].items():
                    sums = role['variants'].setdefault(variant, dict.fromkeys(('error', 'dc', 'mae', 'teacher', 'student'), 0.0))
                    sums['error'] += n * values['residual_mean_square']
                    sums['dc'] += n * values['dc_mean_square']
                    sums['mae'] += n * values['mae']
                    sums['teacher'] += n * values['expected_rms'] ** 2
                    sums['student'] += n * values['actual_rms'] ** 2
                for boundary, variants in window['shared_boundaries'].items():
                    for variant, values in variants.items():
                        sums = role['boundaries'].setdefault(boundary + '/' + variant, dict.fromkeys(('error', 'teacher', 'dc'), 0.0))
                        sums['error'] += n * values['error']['rms'] ** 2
                        sums['teacher'] += n * values['teacher']['rms'] ** 2
                        sums['dc'] += n * values['error']['channel_mean_rms'] ** 2
                for op, observations in source['operations'].items():
                    kind = 'complete_residual_unit' if 'complete_residual_unit' in observations else 'linear_pre_skip'
                    values = observations[kind][str(index)]
                    energy = values['additive_energy']
                    sums = role['operations'].setdefault(op, {'samples': 0.0, 'dropped': 0.0, 'retained_drift': 0.0, 'gap': 0.0, 'skip_drift': 0.0, 'kept_bias': 0.0, 'cross': 0.0, 'teacher': 0.0})
                    mass = values['gap']['weighted_samples']
                    sums['samples'] += mass
                    for term in ('dropped', 'retained_drift', 'gap', 'skip_drift'):
                        if term in values:
                            sums[term] += mass * values[term]['rms'] ** 2
                    sums['kept_bias'] += mass * energy['kept_plus_bias_mean_square']
                    sums['cross'] += mass * energy['cross_mean_product']
                    sums['teacher'] += mass * energy['teacher_mean_square']
result = {'completion': {k: completed[k] for k in ('status', 'sources_diagnosed', 'required_restoration_contracts_passed', 'hidden_boundaries_allclose', 'hidden_boundary_failure_sources', 'original_files_preserved', 'state_preserved', 'parameter_updates')}, 'roles': {}}
for name, role in roles.items():
    n = role['samples']
    record = {'windows': role['windows'], 'samples': n, 'waveforms': {}, 'boundaries': {}, 'step0_operations': {}}
    for variant, x in role['variants'].items():
        record['waveforms'][variant] = {'residual_rms': math.sqrt(x['error'] / n), 'dc_error_energy_fraction': x['dc'] / x['error'] if x['error'] else None, 'mae': x['mae'] / n, 'rms_ratio': math.sqrt(x['student'] / x['teacher']) if x['teacher'] else None}
    for boundary, x in role['boundaries'].items():
        record['boundaries'][boundary] = {'nrmse': math.sqrt(x['error'] / x['teacher']) if x['teacher'] else None, 'error_rms': math.sqrt(x['error'] / n), 'dc_error_energy_fraction': x['dc'] / x['error'] if x['error'] else None}
    for op, x in role['operations'].items():
        mass = x['samples']
        record['step0_operations'][op] = {term + '_rms': math.sqrt(x[term] / mass) for term in ('dropped', 'retained_drift', 'skip_drift', 'gap', 'teacher')}
        record['step0_operations'][op].update(cross_mean_product=x['cross'] / mass, twice_cross_over_sum_energy=2 * x['cross'] / (x['kept_bias'] + x['dropped']) if x['kept_bias'] + x['dropped'] else None)
    result['roles'][name] = record
(ROOT / 'derived-mechanism-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
