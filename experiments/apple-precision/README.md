# Apple CPU precision experiments

Use the existing Apple FP32 runtime. The final SME2 experiment was 15.4% faster on average in a small, noisy screen, but did not establish a reliable 10% improvement. Its correctness checks passed; full INT8 perceptual testing was not run. No runtime default changes.

| Implementation | Matched FP32 RTF | INT8 RTF | Status |
| --- | ---: | ---: | --- |
| Initial SDOT | 0.09384 | 0.17095 | Slower, retained for reference |
| Final SME2 with parallel preparation | 0.06400 | 0.05412 | Promising but unproven |

Each row is a separate three-clip screen on Apple M5 Max, four ORT threads, two warmups and five repetitions. Compare within a row. The final timing interval spans a 3.2% regression to a 29.5% improvement. See [results and limits](../../docs/apple-precision.md).

`sdot/` and `sme2/` preserve the tested sources. Both replace 22 selected large matrix products with symmetric INT8, retaining per-row weight scales, per-time-column activation scales and full-K INT32 accumulation. Inputs, outputs, nonlinearities and nine smaller products remain FP32. These are explicit experiments, not general Apple CPU dispatch paths.

To rebuild the final candidate from the repository root, use native ARM macOS, Python 3.11+, ONNX Runtime 1.29.0, ONNX 1.22.0, NumPy and the pinned ORT headers:

```sh
python experiments/apple-precision/sme2/build.py \
  --output-dir .build/apple-sme2 \
  --ort-include /path/to/onnxruntime/include \
  --ort-pins experiments/apple-precision/pins/ort.json

CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void \
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python experiments/apple-precision/sme2/check.py \
  --build .build/apple-sme2/build.json --output .build/apple-sme2-checks.json
```

The builder downloads nothing and requires a fresh output directory. SME2 must be supported at runtime. The C++ wrapper is compiled for baseline ARM; SME2 flags are confined to the upstream kernel objects. The checked streaming vector length and floating-point mode must match the packing contract.

`rewrite.py --source /path/to/accepted/apple/decoder.onnx --output /path/to/new/decoder.onnx --shards 4` accepts only the recorded FP32 graph hash. Register its existing Apple native library and the new bridge identified by `build.json`. Prepared model/case files, weights, audio and binaries are not included. The pinned [ORT header hashes](pins/ort.json) and [upstream source/license](sme2/native/sme/upstream-provenance.json) are included.

The shared `tools/platform_campaign.py` accepts explicit local config and harness hashes; `tools/compare_decoders.py` and `tools/diagnose_cpu.py` provide its measured helper closure. Saved [evidence](../../benchmarks/apple-precision/publication.json) uses normalized paths and is not a ready-to-run model configuration. Profiling in `diagnostics/` explains the initial SDOT bottleneck and is separate from RTF.

For a fresh quality comparison of the retained FP32 exports, supply the paired source manifest, the local export manifest produced by the waveform audit, and the metric assets pinned in `quality/scorer/metric_assets.json`:

```sh
python experiments/apple-precision/quality/score_quality.py \
  --scorer-dir experiments/apple-precision/quality/scorer \
  --sources /path/to/source-manifest.json \
  --variant-manifest /path/to/local/quality-inputs.json \
  --baseline audio_stock --trim-audiovae2-right-padding \
  --assets /path/to/pinned-metric-assets \
  --output /path/to/new/comparison.json
```

Keep the explicit `--scorer-dir`: the frozen wrapper's original default points to its former workspace layout. The wrapper verifies scorer and asset hashes and forces CPU scoring. It requires the recorded scoring dependencies; no predictor weights or audio are bundled here.
