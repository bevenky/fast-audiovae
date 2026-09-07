"""Measure CPU decoder RTF with precomputed FP32 latents and no file I/O in timing."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from fast_audiovae import load_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='artifacts')
    parser.add_argument('--cases', required=True, help='NPZ with z or clip-name__z arrays')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--portable', action='store_true')
    parser.add_argument('--amd-packed', action='store_true')
    parser.add_argument('--output', default='benchmark-results/decoder.json')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error('repeats and warmup must be positive')
    if args.portable and args.amd_packed:
        parser.error('choose portable or AMD packing')
    with np.load(args.cases, allow_pickle=False) as arrays:
        cases = [arrays[k] for k in sorted(arrays.files) if k == 'z' or k.endswith('__z')]
    if not cases:
        parser.error('no latent arrays found')
    for z in cases:
        if z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (1, 64) or z.shape[2] < 1:
            parser.error('each latent array must be FP32 [1,64,L] with L >= 1')
    session, backend = load_decoder(args.bundle, threads=args.threads,
                                   prefer_custom=not args.portable, prefer_packed=args.amd_packed)
    name = session.get_inputs()[0].name
    feeds = [{name: z} for z in cases]
    for feed in feeds:
        for _ in range(args.warmup):
            session.run(None, feed)
    rng = random.Random(args.seed)
    results = []
    for repeat in range(args.repeats):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for index in order:
            start = time.perf_counter()
            waveform = session.run(None, feeds[index])[0]
            elapsed = time.perf_counter() - start
            if waveform.shape != (1, 1, 1920 * cases[index].shape[2]) or not np.isfinite(waveform).all():
                raise RuntimeError('Invalid decoder output')
            seconds = waveform.shape[-1] / 48000
            results.append({'case': index, 'repeat': repeat, 'decode_seconds': elapsed,
                            'audio_seconds': seconds, 'rtf': elapsed / seconds})
    total_time = sum(r['decode_seconds'] for r in results)
    total_audio = sum(r['audio_seconds'] for r in results)
    report = {'backend': backend, 'precision': 'FP32', 'case_count': len(cases),
              'warmup_per_case': args.warmup, 'repeats': args.repeats, 'seed': args.seed,
              'rtf': total_time / total_audio, 'results': results,
              'scope': 'Fresh decoder calls only; excludes encoding, loading and warmup; no cached streaming state'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
