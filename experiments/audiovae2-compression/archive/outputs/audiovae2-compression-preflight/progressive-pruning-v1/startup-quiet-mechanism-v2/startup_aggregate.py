"""CPU-only aggregate analysis; never exports audio, latent values or source IDs."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/tmp/fast-audiovae-progressive-pruning-v1/startup-quiet-mechanism-v2')
OUT = ROOT / 'results'


def main():
    files = [OUT / n for n in ('completed.json', 'source-diagnostics.jsonl.gz', 'trace-index.json',
                               'selected-waveforms.npz', 'quiet-windows-step5000.jsonl.gz')]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    done = json.loads(files[0].read_text())
    if done['status'] != 'completed' or not done['required_restoration_contracts_passed'] or not all(done['state_preserved'].values()):
        raise RuntimeError('Diagnostic prerequisite is incomplete')
    with gzip.open(OUT / 'source-diagnostics.jsonl.gz', 'rt') as f:
        sources = [json.loads(line) for line in f]
    with gzip.open(OUT / 'quiet-windows-step5000.jsonl.gz', 'rt') as f:
        limits = {w['window_id']: w for w in map(json.loads, f)}
    variants = {}; zero_ids = set(); phase_sums = {}
    for source in sources:
        for i, window in enumerate(source['windows']):
            if 'startup_near_all13' not in window['roles']:
                continue
            key = source['source_id'] + ':' + str(window['source_start_sample'])
            baseline = limits[key]
            if window['reference16k']['source_reference_exact_zero']:
                zero_ids.add(source['source_id'])
            for name, value in window['waveforms_vs_native'].items():
                row = variants.setdefault(name, {'windows': 0, 'passed': 0, 'residual_failed': 0, 'amplitude_failed': 0, 'error_sum': 0.})
                residual = value['residual_mean_square'] ** .5
                rf = residual > baseline['residual_limit']
                af = value['actual_rms'] > baseline['output_rms_limit']
                row['windows'] += 1; row['passed'] += not (rf or af)
                row['residual_failed'] += rf; row['amplitude_failed'] += af
                row['error_sum'] += residual ** 2
            phase_data = source['operations']['stage3_up']['linear_pre_skip'][str(i)]['phases']
            for values in phase_data.values():
                for name in ('dropped_current', 'dropped_previous', 'dropped', 'retained_drift'):
                    v = values[name]; s = phase_sums.setdefault(name, {'mass': 0., 'energy': 0.})
                    if v['weighted_samples']:
                        s['mass'] += v['weighted_samples']; s['energy'] += v['weighted_samples'] * v['rms'] ** 2
    for row in variants.values():
        row['residual_rms'] = (row.pop('error_sum') / row['windows']) ** .5
    if variants['step0']['windows'] != 13 or variants['step5000']['passed'] != 0 or len(zero_ids) != 10:
        raise RuntimeError('Original startup population changed')
    index = json.loads((OUT / 'trace-index.json').read_text())
    selected = [row for row in index if row['source_id'] in zero_ids and 'startup_near_all13' in row['roles']]
    patterns = {}
    with np.load(OUT / 'selected-waveforms.npz', allow_pickle=False) as waves:
        for name in ('step0', 'step5000'):
            errors = np.stack([waves[row['key'] + '_' + name].astype(np.float64) - waves[row['key'] + '_teacher'].astype(np.float64)
                               for row in selected])
            mean_pattern = errors.mean(axis=0); differences = errors - mean_pattern
            energy = float(np.mean(errors ** 2)); disagreement = float(np.mean(differences ** 2))
            norms = np.linalg.norm(errors, axis=1); mean_norm = float(np.linalg.norm(mean_pattern))
            cosines = errors @ mean_pattern / (norms * mean_norm)
            peak_times = np.abs(errors).argmax(axis=1) / 48.
            patterns[name] = {'sources': len(selected), 'samples_per_source': errors.shape[1],
                'pooled_residual_rms': energy ** .5,
                'between_source_disagreement_rms': disagreement ** .5,
                'fraction_residual_energy_shared_by_average_pattern': 1. - disagreement / energy,
                'cosine_to_average_pattern_min': float(cosines.min()),
                'cosine_to_average_pattern_mean': float(cosines.mean()),
                'residual_peak_time_ms_mean': float(peak_times.mean()),
                'residual_peak_time_ms_std': float(peak_times.std())}
    preserved = all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    if not preserved:
        raise RuntimeError('Input evidence changed')
    result = {'version': 'aggregate_startup_pattern_v1', 'input_evidence_preserved': preserved,
        'parameter_updates': 0, 'model_forwards': 0, 'audio_or_latent_values_exported': False,
        'startup_restoration': variants, 'source_zero_startup_pattern_aggregates': patterns,
        'stage3_input_contribution_rms_pooled_across_sources_and_phases':
            {name: (s['energy'] / s['mass']) ** .5 for name, s in phase_sums.items()},
        'interpretation': 'Teacher-term restoration concerns pristine step0 only. Pattern statistics describe saved outputs and do not prescribe a correction.'}
    path = ROOT / 'aggregate-startup-pattern-v1.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
