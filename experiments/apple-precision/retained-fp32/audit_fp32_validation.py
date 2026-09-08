"""Audit completed retained FP32 exports. Reads waveforms only; no inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
                  HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    report, cfg = json.loads(a.report.read_text()), json.loads(a.config.read_text())
    assert report['status'] == 'complete' and not report['failures']
    assert report['config_sha256'] == sha(a.config)
    assert report['runtime']['onnxruntime'] == '1.29.0'
    assert report['runtime']['onnx'] == '1.22.0'
    assert report['gpu_used'] is False and not report['measurements']
    assert report['protocol']['warmups'] == report['protocol']['repeats'] == 0
    assert report['protocol']['timing_uids'] == []
    assert len(report['providers']) == 2
    assert all(v['providers'] == ['CPUExecutionProvider'] for v in report['providers'])
    assert len(report['checks']) == 156 and all(v['passed'] for v in report['checks'])
    uids, names = cfg['expected_uids'], ['audio_stock', 'fast_fp32']
    expected = {(name, uid) for name in names for uid in uids}
    exports = {(row['model'], row['uid']): row for row in report['exports']}
    assert len(uids) == len(set(uids)) == 60
    assert len(exports) == len(report['exports']) == 120 and set(exports) == expected
    checks = {(r['model'], r['uid']): r for r in report['checks'] if r['test'] == 'full_waveform'}
    assert set(checks) == expected and all(r['fp32_parity_required'] for r in checks.values())
    archive_path = (a.config.parent / cfg['audio_cases']).resolve()
    archive = np.load(archive_path, allow_pickle=False)
    rows = []

    def difference(actual, reference):
        assert actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape
        delta = actual.astype(np.float64) - reference
        return {'passed': bool(np.allclose(actual, reference, atol=1e-5, rtol=1e-4)),
                'bitwise_equal': bool(np.array_equal(actual.view(np.uint32), reference.view(np.uint32))),
                'max_abs': float(np.abs(delta).max()), 'rmse': float(np.sqrt(np.mean(delta * delta)))}

    for uid in uids:
        waves = {}
        z, golden = archive[uid + '__z'], archive[uid + '__ref'].ravel()
        latent_hash = hashlib.sha256(z.tobytes()).hexdigest()
        for name in names:
            row, check = exports[name, uid], checks[name, uid]
            path = Path(row['path']).resolve()
            assert path.is_relative_to(a.report.parent.resolve()) and sha(path) == row['sha256']
            info = sf.info(path)
            wave, rate = sf.read(path, dtype='float32')
            assert info.subtype == row['subtype'] == 'FLOAT' and info.channels == 1
            assert rate == row['rate'] == 48000 and wave.shape == (row['samples'],)
            assert wave.size == z.shape[-1] * 1920 and np.isfinite(wave).all()
            assert hashlib.sha256(wave.tobytes()).hexdigest() == check['waveform_sha256']
            assert check['latent_sha256'] == latent_hash
            waves[name] = wave
        row = {'uid': uid, 'samples': len(golden),
               'stock_vs_saved': difference(waves['audio_stock'], golden),
               'fast_vs_saved': difference(waves['fast_fp32'], golden),
               'fast_vs_fresh_stock': difference(waves['fast_fp32'], waves['audio_stock'])}
        assert all(row[key]['passed'] for key in ('stock_vs_saved', 'fast_vs_saved', 'fast_vs_fresh_stock'))
        rows.append(row)
    summaries = {}
    for key in ('stock_vs_saved', 'fast_vs_saved', 'fast_vs_fresh_stock'):
        worst = max(rows, key=lambda r: r[key]['max_abs'])
        summaries[key] = {'passed': len(rows), 'failed': 0,
                          'bitwise_equal_clips': sum(r[key]['bitwise_equal'] for r in rows),
                          'max_abs': worst[key]['max_abs'], 'worst_uid': worst['uid'],
                          'max_clip_rmse': max(r[key]['rmse'] for r in rows)}
    result = {'status': 'complete', 'scope': 'Untimed CPU waveform audit. No perceptual model scoring.',
              'report_sha256': sha(a.report), 'config_sha256': sha(a.config),
              'archive_sha256': sha(archive_path), 'audit_script_sha256': sha(__file__),
              'runtime': report['runtime'], 'checks': len(report['checks']),
              'check_categories': dict(Counter(r['test'] for r in report['checks'])),
              'exports': len(exports), 'failures': 0, 'measurements': 0,
              'comparison_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
              'summary': summaries, 'clips': rows}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': summaries, 'checks': 156, 'exports': 120}))


if __name__ == '__main__':
    main()
