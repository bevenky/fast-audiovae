# Apple batch validation

The new batch path reuses the selected Apple kernels for independent inputs. This is a short check of an **unreleased source checkout using the unchanged v0.3.0 native payload**, not a new release or a replacement for the [streaming qualification](streaming-baseline.md). Ordinary `load()` still selects streaming; independent calls use `load(mode="batch")`.

The batch graph accepts one complete `[1, 64, L]` input and returns all `1,920 * L` samples at 48 kHz. Each call starts with zero history. It preserves the trained tensors and native kernels, replacing the 26 external history inputs with constants. It does not carry state between calls or prepend latent frames.

## Matched short measurements

Apple M5 Max, macOS 26.5.1, FP32 and ONNX Runtime 1.30.0 CPU execution. Thread counts are ORT intra-op threads; inter-op and nested BLAS/OpenMP workers are limited to one. Existing custom kernels retain their one-worker policies.

RTF is API time divided by returned audio duration; lower is faster. Each row compares methods from the same run.

| ORT threads | Input | Stock batch | Released batch | New batch | Fresh-stream control |
|---:|---:|---:|---:|---:|---:|
| 1 | 40 ms | 0.4782 | 0.3595 | 0.2121 | 0.2094 |
| 1 | 80 ms | 0.4565 | 0.3542 | 0.1340 | 0.1430 |
| 4 | 40 ms | 0.4253 | 0.3194 | 0.2441 | 0.2402 |
| 4 | 80 ms | 0.2995 | 0.1677 | 0.1507 | 0.1626 |

Three fixed speech inputs cover Bengali, English and Spanish. Each method received two warmup groups and five measured groups of five independent calls per input. The table averages the three per-input medians equally. Loading, preparation and external output checks are excluded; API validation and allocation are included. The fresh-stream control also includes stream creation, zero-state initialization, empty flush and close, so these values are not continuous-stream RTF.

The [benchmark record](../benchmarks/batch/apple-20260913.json) retains ranges, paired comparisons and artifact hashes. Variability was visible, and the thread settings ran sequentially. These results do not isolate thread scaling or support comparison with historical continuous-stream numbers.

## Validation scope

At each thread setting, 25 inputs covered 1, 2, 3, 4 and 13 latent frames from speech, expressive audio and encoded silence. All 100 comparisons across the four methods passed against stock full decoding of the exact same short input, using `atol=1e-5, rtol=1e-4`. Maximum absolute difference was **2.135e-6**. Every output retained its expected sample count.

A/B/A repeats were bitwise stable. Empty input, future-prefix checks and two concurrent candidate callers passed; the shared native session serialized those callers. All recorded artifact hashes remained unchanged and the accepted processes exited cleanly.

A separate public API smoke confirmed automatic batch selection at one and four threads, and unchanged default streaming with graph `5918e523…`. The final source suite passed **314 tests and 1,959 subtests**, with a clean exit. No native kernel was edited or rebuilt. No long-corpus benchmark or perceptual-quality scores were rerun for this batch change.
