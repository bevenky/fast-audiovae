# Streaming projection kernels

These recipes speed up the first streaming upsampling projection pair. They keep the existing trained weights, audio format and causal history. Full-clip graphs remain unchanged.

One CPU thread, ONNX Runtime 1.29.0. Lower RTF is better.

| CPU | Chunk | Previous optimized | New | Less decoding time |
| --- | ---: | ---: | ---: | ---: |
| Apple M5 Max | 80 ms | 0.24715 | 0.14975 | 39.4% |
| Apple M5 Max | 160 ms | 0.13772 | 0.09077 | 34.1% |
| Intel Xeon Platinum 8280 VM | 80 ms | 0.37609 | 0.31152 | 17.2% |
| Intel Xeon Platinum 8280 VM | 160 ms | 0.27842 | 0.24781 | 11.0% |

Each row averages the per-clip median over three fixed multilingual clips, with two warmups and five measured repetitions. Baseline and candidate calls were paired in randomized order. Every timed output passed the existing waveform checks. Full-call controls varied despite identical graph bytes; these are streaming gains.

Both CPUs also passed 180 complete streams over all 60 multilingual clips at 40, 80 and 160 ms. Intel outputs were bitwise identical to the accepted decoder. Apple passed its existing tolerance and stored upstream comparisons. No extra quantization or retraining was introduced. [Results and provenance](../../benchmarks/streaming/projection.json).

Apple puts fixed weights on the right of the matrix multiplication so ORT can prepare them once. Intel uses a compact VNNI layout across output channels, shares activation preparation, and retains the accepted integer arithmetic. Its direct kernel handles one through four frames; larger chunks use the existing matrix core.

## Prepare a bundle

Start from an already prepared native streaming bundle. Keep the source bundle for comparison.

On Apple:

```sh
python experiments/streaming-matrix/apple/prepare.py \
  --source artifacts --output work/apple-projection/bundle
```

On the validated Intel configuration, start with its accepted one-worker precision bundle and use the matching ORT headers and precision core:

```sh
python experiments/streaming-matrix/intel/build_candidate.py \
  --baseline artifacts --output work/intel-projection/bundle \
  --ort-include /path/to/onnxruntime/include \
  --core /path/to/libintel_precision_core.so
```

The Intel output must be on the same filesystem as its source because unchanged files are hardlinked. Its compiler needs AVX512-VNNI intrinsics; runtime CPU and OS capability checks prevent unsupported execution. Intel validation does not establish AMD performance.

Load the resulting bundle with `load_streaming_decoder(bundle_path, threads=1)`. These are explicit recipes; the automatic setup path does not yet select them.

## Reproduce the checks

Set the two bundle paths and frozen latent files in `common/config-template.json`, then run:

```sh
python experiments/streaming-matrix/common/paired_benchmark.py \
  --config run.json --output timing.json
python experiments/streaming-matrix/common/qualify_corpus.py \
  --config run.json --output quality.json
```

On Intel, prefix each command with `taskset -c 0`. The runner enforces CPU execution, one thread, matching sample counts and unchanged artifact hashes. Timing covers decoder calls and flush, with initialization and input preparation excluded. The corpus check uses all 60 clips at 40, 80 and 160 ms; it requires bitwise agreement on Intel and the existing strict tolerance on Apple. It does not calculate new perceptual MOS scores.
