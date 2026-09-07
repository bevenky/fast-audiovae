"""Compare prepared and stock decoders on user-supplied latents, without timing."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fast_audiovae import load_decoder


def validate(bundle, source, cases, threads=(1, 4), packed=False):
    threads = tuple(threads)
    if not threads or any(type(t) is not int or t < 1 for t in threads):
        raise ValueError('At least one positive integer thread count is required')
    if ort.__version__ != '1.29.0':
        raise RuntimeError('This validation requires ONNX Runtime 1.29.0')
    results = []
    with np.load(cases, allow_pickle=False) as data:
        keys = sorted(k for k in data.files if k.endswith('__z') or k == 'z')
        if not keys:
            raise ValueError('NPZ needs z or clip-name__z arrays with FP32 shape [1,64,L]')
        for count in threads:
            candidate, backend = load_decoder(bundle, threads=count, prefer_packed=packed)
            if backend['selected'] != 'native' or (packed and not backend.get('packed_weights')):
                raise RuntimeError('Requested native backend was not selected: ' + str(backend))
            options = ort.SessionOptions()
            options.intra_op_num_threads = count
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            reference = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
            reference.disable_fallback()
            for index, key in enumerate(keys):
                z = data[key]
                if z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (1, 64) or z.shape[2] < 1:
                    raise ValueError('Invalid FP32 latent array: ' + key)
                original = reference.run(None, {reference.get_inputs()[0].name: z})[0]
                actual = candidate.run(None, {candidate.get_inputs()[0].name: z})[0]
                same_shape = actual.shape == original.shape == (1, 1, 1920 * z.shape[2])
                finite = bool(np.isfinite(actual).all() and np.isfinite(original).all())
                passed = same_shape and finite and np.allclose(actual, original, atol=1e-5, rtol=1e-4)
                error = np.abs(actual.astype(np.float64) - original) if same_shape else np.array([np.inf])
                results.append({'case': index, 'threads': count, 'latent_frames': z.shape[2],
                                'passed': bool(passed), 'max_abs': float(error.max()),
                                'rmse': float(np.sqrt(np.mean(error * error))),
                                'backend': backend['selected'], 'packed': backend.get('packed_weights', False)})
            del candidate, reference
    return {'passed': all(r['passed'] for r in results), 'comparisons': len(results),
            'atol': 1e-5, 'rtol': 1e-4, 'providers': ['CPUExecutionProvider'],
            'precision': 'FP32', 'scope': 'Numerical parity only; no timing or perceptual quality claims',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--amd-packed', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = validate(args.bundle, args.source, args.cases, args.threads, args.amd_packed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
