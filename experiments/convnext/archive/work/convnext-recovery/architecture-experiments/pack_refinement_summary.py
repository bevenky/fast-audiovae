"""Compact a saved summary for transfer; all full reports remain on Runpod."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

source, destination = map(Path, sys.argv[1:])
data = json.loads(source.read_text())
for key in ('rate-calibration', 'baseline-validation', 'baseline-training-panel', 'complete'):
    data.pop(key, None)
for arm in data['arms'].values():
    arm.pop('training-panel', None)
    for validation in arm.get('validation', []):
        selection = validation['selection']
        validation['selection'] = {
            'qualified': selection['qualified'],
            'quiet_rms_ratio': selection['quiet_rms_ratio'],
            'failed_checks': [c for c in selection['checks'] if not c['passed']],
            'source_spectral_passed': selection['source_spectral']['passed'],
            'source_spectral_failed_sources': selection['source_spectral']['all_region_failed_sources'],
        }
for update in data.get('isolated_calibration_updates', []):
    update.pop('source_spectral', None)
    update.pop('migration_parity', None)
data['comparisons'] = {
    arm: {
        'selected_step': result['selected_step'],
        'seconds': result['seconds'],
        'head_sha256': result.get('head_sha256'),
        'state_sha256': result.get('state_sha256'),
        'student_sha256': result.get('student_sha256'),
        'auxiliary_adapter_preserved': result.get('auxiliary_adapter_preserved'),
        'auxiliary_exported': result.get('auxiliary_exported'),
        'promoted': result['promoted'],
        'folding': result.get('folding'),
        'migration_max_abs': result.get('migration', {}).get('max_abs_difference'),
        'final_scalar_flags': {k: v for k, v in result['final'].items() if not isinstance(v, (dict, list))},
    }
    for arm, result in data['comparisons'].items()
}
data['full_remote_summary_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
warmup = source.parent / 'teacher-reference-warmup.json'
if warmup.exists():
    data['teacher-reference-warmup'] = json.loads(warmup.read_text())
payload = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
encoded = base64.b64encode(zlib.compress(payload)).decode()
destination.write_text(encoded)
print(json.dumps({'bytes': len(payload), 'encoded_characters': len(encoded),
                  'sha256': hashlib.sha256(payload).hexdigest(), 'destination': str(destination)}))
