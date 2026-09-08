"""Summarize the completed, non-promoting Intel decoder screens."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    variants = [
        ('screen-r1', 'Row fusion and reusable INT8 workspace'),
        ('screen-upsample256-r1', 'Above plus larger upsampling tiles'),
        ('screen-both-r1', 'Above plus both tile changes'),
        ('screen-vnni-r1', 'Row fusion plus original direct VNNI'),
        ('screen-hybrid-r1', 'Row fusion plus shape-selected wider VNNI'),
        ('screen-inline-r3', 'Above plus exact inline sine'),
        ('screen-fused-inline-r1', 'Above plus fused DW and post-Snake'),
    ]
    records = []
    for name, description in variants:
        path = args.evidence/name/'results.json'
        data = json.loads(path.read_text())
        if (data['status'] != 'complete' or data['mode'] != 'screen' or data['accepted']
                or data['failures'] or not all(c['passed'] for c in data['checks'])
                or data['gpu_used'] or data['runtime']['onnxruntime'] != '1.29.0'):
            raise ValueError('Summary expects complete CPU-only screens with no accuracy failures')
        records.append({'variant': name, 'description': description,
                        'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'checks': len(data['checks']), 'timings': len(data['measurements']),
                        'summary': data['summary']})
    if any(r['summary']['meets_10pct_aggregate'] for r in records):
        raise ValueError('A promising screen requires full validation before this rejection report')
    lines = [
        '# Intel combined-kernel results', '',
        'September 8, 2026. None of this batch met the required 10% reduction in complete decoder time. The repository and runtime defaults were not changed.', '',
        'The earlier 4.5% result already included the first 3.9% candidate. It was a tile variant of the same pipeline, not an independent VNNI gain. Subsequent VNNI, accurate-sine and row-fusion variants were implemented and measured as complete combinations.', '',
        '| Complete candidate | Paired baseline RTF | Candidate RTF | Less decoder time | Exact decoder checks |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in records:
        s = r['summary']
        rtfs = s['aggregate_decoder_rtf']
        delta = s['aggregate_time_reduction']*100
        change = f'{delta:.2f}%' if delta >= 0 else f'{-delta:.2f}% slower'
        lines.append(f"| {r['description']} | {rtfs['baseline']:.5f} | {rtfs['candidate']:.5f} | {change} | {r['checks']}/{r['checks']} |")
    lines += [
        '',
        'Each row is a separate randomized, adjacent A/B screen on the Intel Xeon Platinum 8280 VM, two vCPUs, CPU affinity 0/1, ONNX Runtime 1.29.0 and single-threaded oneMKL. Three fixed full clips covered Hindi, English and Brazilian Portuguese, with two warmups and five timed calls per model per clip. Each run contains 30 timed decoder calls. Lower RTF is better. Absolute RTF varies between runs, so compare each candidate with its own paired baseline.', '',
        'These are decoder-only measurements. They exclude encoder, TTS generation, session loading and warmup. No GPU was used. The 67 checks include complete waveform equality, short inputs, future-input causality, repeatability, history reuse and concurrent calls; they are not 67 different audio clips. Candidate outputs must match the accepted selective INT8 baseline bit for bit at the same input shape. The existing short-versus-long prefix tolerance stays unchanged.', '',
        'The wider direct VNNI kernel improved some small matrix shapes, but oneMKL remained faster on larger tiles. Selecting between them prevented the original broad VNNI regression; it did not produce a large complete-decoder gain. Matrix-only timings cannot be added to the decoder percentages.', '',
        'Two guarded reciprocal quantizers each passed 2,650,368 serial INT8-value checks and 132,384 concurrent checks. Both were rejected before decoder timing because preparation became slower on the tested decoder shapes. Their guarded multiplication did not change accepted INT8 values, but the guard overhead erased the saved division cost.', '',
        'The exact sine experiment used the same pinned SLEEF 3.9.0 u10 implementation, generated as an inline AVX512 header. It passed 3,066,624 sine-value comparisons and 6,366 row, activation and history checks. The final DW/post-Snake fusion also passed all 6,366 row checks. Disassembly verified removal of external 16-lane sine calls. The final fusion removes an intermediate vector DW store/reload; it does not remove the fallback stack array.', '',
        'No candidate qualified for the subsequent 60-clip full validation and ten-clip timing gate, so none was promoted. There is no new full-cohort quality or Mimi comparison claim. The checked waveforms were identical to the accepted INT8 outputs, so this batch did not rerun perceptual scoring on identical samples.', '',
        'The remaining substantial matrix experiment is a compact persistent weight layout with activation tiles produced directly in the layout consumed by the matrix kernel, across the remaining matrix regions. That is a larger implementation change. The current evidence does not establish that it will meet 10%; another small VNNI tile adjustment is not a demonstrated route.', '',
        'Raw paired measurements, artifact hashes, native checks, diagnostics and candidate sources are preserved in the accompanying experiment archive. These sources are isolated research code with machine-specific build paths, not a packaged production backend.', '',
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'iteration3-results.md').write_text('\n'.join(lines))
    (args.output/'iteration3-summary.json').write_text(json.dumps({
        'status': 'complete', 'promoted': False, 'gpu_used': False,
        'threshold_time_reduction': 0.10, 'screens': records,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2)+'\n')
    print(json.dumps({'status': 'written', 'screens': len(records)}))


if __name__ == '__main__':
    main()
