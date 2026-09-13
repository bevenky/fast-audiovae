"""Record aggregate tolerance crossings; retain unchanged fail-fast qualification."""
from pathlib import Path
import json
import os
import sys
import hashlib

HERE = Path(__file__).resolve().parent
os.environ.pop('PYTORCH_MPS_PREFER_METAL', None)
sys.path.insert(0, str(HERE.parent / 'apple-gpu-v2'))
import qualify_candidates as qual
import qualify_transpose_only as wrapper
import numpy as np

if __name__ == '__main__':
    output = HERE / 'state-error-aggregate-r1.json'
    assert not output.exists()
    original = qual.compare
    failures = []

    def compare(value, reference):
        result = original(value, reference)
        if not result['passed']:
            diff = np.abs(value.astype(np.float64)-reference.astype(np.float64))
            threshold = 1e-5+1e-4*np.abs(reference.astype(np.float64))
            failures.append(dict(shape=list(value.shape), elements=value.size,
                failing_elements=int(np.count_nonzero(diff>threshold)),
                max_abs=float(diff.max()), rms=float(np.sqrt(np.mean(diff**2))),
                max_tolerance_ratio=float((diff/threshold).max()),
                maximum_excess_over_tolerance=float(np.maximum(0,diff-threshold).max())))
        return result

    qual.compare = compare
    try:
        wrapper.main('qualify-transpose-only-errorprobe-r1')
    finally:
        qual.compare = original
        output.write_text(json.dumps(dict(status='diagnostic', failures=failures,
            tolerance_unchanged=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            raw_audio_or_state_values_exported=False), indent=2)+'\n')
