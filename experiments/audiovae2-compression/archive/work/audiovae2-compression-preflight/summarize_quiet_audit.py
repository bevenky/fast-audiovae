"""Summarize saved quiet diagnostics without another inference pass."""
from pathlib import Path
import json
import math

root = Path('outputs/audiovae2-compression-preflight/settings-screen-results-v1')
data = json.loads((root/'reference-and-quiet-addendum-v1.json').read_text())

def pooled(rows):
    samples = sum(r['valid_samples'] for r in rows)
    def energy(key):
        return sum(r['valid_samples']*r[key]**2 for r in rows)
    residual_energy = energy('residual_rms_double')
    return {'windows': len(rows), 'seconds': samples/48000,
            'passed': sum(r['failure_category']=='passed' for r in rows),
            'residual_only': sum(r['failure_category']=='residual_only' for r in rows),
            'amplitude_only': sum(r['failure_category']=='amplitude_only' for r in rows),
            'both': sum(r['failure_category']=='both' for r in rows),
            'residual_rms': math.sqrt(residual_energy/samples) if samples else None,
            'teacher_rms': math.sqrt(energy('teacher_rms')/samples) if samples else None,
            'student_rms': math.sqrt(energy('student_rms')/samples) if samples else None,
            'window_dc_energy_fraction': energy('residual_mean')/residual_energy if residual_energy else None,
            'max_abs_error': max((r['residual_abs_max'] for r in rows), default=None)}

output = {'fit_quiet_occupancy': data['fitting_quiet_occupancy']['aggregate'], 'arms': []}
for arm in data['arms']:
    quiet = arm['quiet_diagnostics']
    windows = quiet['student']['windows']
    groups = {'all': windows,
              'full_20ms_windows': [r for r in windows if r['valid_samples']==960],
              'short_tails': [r for r in windows if r['valid_samples']<960],
              'actual_source_first20ms': [r for r in windows if r['actual_startup'] and r['source_start_sample']<960],
              'actual_source_20to40ms': [r for r in windows if r['actual_startup'] and 960<=r['source_start_sample']<1920],
              'actual_source_40to800ms': [r for r in windows if r['actual_startup'] and 1920<=r['source_start_sample']<38400],
              'after800ms_or_continuation': [r for r in windows if not r['actual_startup'] or r['source_start_sample']>=38400],
              'teacher_rms_at_most1e-5': [r for r in windows if r['teacher_rms']<=1e-5],
              'teacher_rms_1e-5to1e-4': [r for r in windows if 1e-5<r['teacher_rms']<=1e-4],
              'teacher_rms_1e-4to1e-3': [r for r in windows if 1e-4<r['teacher_rms']<=1e-3]}
    result = {'arm': arm['arm'], 'spectral': arm['aggregate'],
              'common_reproduction': arm['common_reproduction'],
              'teacher_replay': quiet['teacher_replay']['aggregate'],
              'regions': {k:pooled(v) for k,v in groups.items()},
              'largest_errors': sorted(windows,key=lambda r:r['residual_abs_max'],reverse=True)[:8]}
    output['arms'].append(result)
path = root/'quiet-audit-summary-v1.json'
path.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({'path':str(path), 'fit':output['fit_quiet_occupancy'],
                  'spectral':[{k:row[k] for k in ('arm','spectral')} for row in output['arms']]}))
chosen = next(row for row in output['arms'] if row['arm']=='current_lr3e-5')
print(json.dumps({k:chosen[k] for k in ('teacher_replay','regions','largest_errors')},indent=2))
