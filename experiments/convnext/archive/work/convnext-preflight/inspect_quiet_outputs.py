"""Measure saved diagnostic WAVs without generating or modifying audio."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf

root = Path('/workspace/fast-audiovae-convnext-20260909-r3/training-runs/objective-comparison-v1')
result = {'source': 'saved_float48k_wavs', 'inference_calls': 0, 'window_ms': 20, 'arms': {}}
for arm in ('waveform-mel', 'waveform-only'):
    base = root / arm
    metadata = json.loads((base / 'audio.json').read_text())
    selected = []
    for item in metadata['items']:
        teacher, sr = sf.read(base / item['teacher'], dtype='float64')
        prediction, sr2 = sf.read(base / item['student'], dtype='float64')
        assert sr == sr2 == 48000 and teacher.shape == prediction.shape
        teacher_rms = float(np.sqrt(np.mean(teacher**2)))
        if teacher_rms >= .003 and item['source_id'] not in ('6549-71115-0026',):
            continue
        rms = lambda x: float(np.sqrt(np.mean(x*x)))
        error = prediction - teacher
        pmean, tmean = float(prediction.mean()), float(teacher.mean())
        n = teacher.size // 960
        tr = np.sqrt(np.mean(teacher[:n*960].reshape(n, 960)**2, axis=1))
        pr = np.sqrt(np.mean(prediction[:n*960].reshape(n, 960)**2, axis=1))
        er = np.sqrt(np.mean(error[:n*960].reshape(n, 960)**2, axis=1))
        intervals = []
        for lo, hi, name in ((0, 1e-4, 'below_minus80dbfs'), (1e-4, 1e-3, 'minus80_to_minus60dbfs'),
                             (1e-3, 1e-2, 'minus60_to_minus40dbfs'), (1e-2, float('inf'), 'above_minus40dbfs')):
            mask = (tr >= lo) & (tr < hi)
            if not mask.any():
                continue
            intervals.append({'teacher_level': name, 'windows': int(mask.sum()),
                              'teacher_rms_mean': float(tr[mask].mean()),
                              'student_rms_mean': float(pr[mask].mean()),
                              'error_rms_mean': float(er[mask].mean()),
                              'student_rms_max': float(pr[mask].max())})
        selected.append({'source_id': item['source_id'], 'role': item['role'], 'start_frame': item['start_frame'],
                         'teacher_rms': teacher_rms, 'student_rms': rms(prediction),
                         'teacher_dbfs': 20*np.log10(max(teacher_rms, 1e-15)),
                         'student_dbfs': 20*np.log10(max(rms(prediction), 1e-15)),
                         'teacher_dc': tmean, 'student_dc': pmean,
                         'student_dc_power_fraction': pmean**2/max(rms(prediction)**2, 1e-30),
                         'waveform_error_rms': rms(error),
                         'error_after_dc_removal_rms': rms(error-error.mean()),
                         'student_peak': float(np.max(np.abs(prediction))), 'level_windows': intervals})
    result['arms'][arm] = selected
print(json.dumps(result, indent=2, allow_nan=False))
