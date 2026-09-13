# Initial Apple GPU results

This is a short qualification and timing baseline from September 13, 2026, on an Apple M5 Max running macOS 26.5.1. AudioVAE2 uses the original verified ONNX coefficients in a FP32 PyTorch MPS implementation. It does not use the optimized CPU kernels. The public GPU path has not demonstrated a speed advantage over the selected CPU implementation.

PyTorch was 2.14.0 and ONNX Runtime was 1.30.0. MPS CPU fallback, fast math and autocast were disabled. The CPU comparator was the selected `5918e523` streaming graph with one worker; the GPU process also used one host thread. GPU work is not restricted to one execution lane.

## Streaming timing

Lower real-time factor (RTF) is faster. These are pooled completed packet and flush times divided by emitted audio duration, retaining every measured repetition.

| Decoder | Output rate | 40 ms RTF | 80 ms RTF |
| --- | ---: | ---: | ---: |
| Selected AudioVAE2 CPU | 48 kHz | 0.11322 | 0.07716 |
| AudioVAE2 GPU, direct model boundary | 48 kHz | 0.13902 | 0.07438 |
| AudioVAE2 GPU, public API | 48 kHz | 0.14809 | 0.07765 |
| Pocket continuous Mimi GPU | 24 kHz | Not supported | 0.06381 |

At 80 ms the public AudioVAE2 GPU path was approximately level with CPU, with 0.63% higher pooled time. At 40 ms it took 30.80% more time. The direct GPU boundary was 3.61% faster than CPU at 80 ms, but it omits the public API's validation of every history tensor. This does not establish a public API speedup.

Both direct GPU models include the same NumPy input validation and ownership copy, upload, model execution, blocking audio return, explicit MPS synchronization, owned NumPy output and output finite check. The public AudioVAE2 API additionally validates all 26 updated histories before committing state. Its arithmetic and weights are identical to the direct AudioVAE2 GPU arm.

The workload was three aligned 960 ms middle segments of Bengali, English and Spanish speech, with fresh decoder state at each segment's start. Each route had two warmup sweeps and three measured sweeps, giving nine measured streams per table entry. Arm order reversed on alternating repetitions and clips; three repetitions leave a 2/1 direction imbalance. Session creation, stream construction and close were excluded. Mimi's KV initialization occurs during excluded stream construction; AudioVAE2's lazy zero-history initialization remains inside the first timed packet.

This is short-stream evidence, not long-utterance throughput, an all-audio corpus test or an energy measurement. The codecs have different output sample rates and latent rates: AudioVAE2 is 64D at 25 Hz; Pocket Mimi is continuous 32D at 12.5 Hz. Mimi cannot emit a genuine 40 ms latent packet. Its stored BF16 weights widen exactly to FP32, matching the existing CPU comparison; no new quantization was introduced.

## Numerical qualification

- AudioVAE2: 31 checks passed, covering three speech languages, two expressive samples, zero and quiet latents, 40/80 ms streaming, resets, independent streams and a public batch call. Maximum absolute error was `2.0433e-6`.
- Pocket Mimi: 22 checks passed against its frozen CPU ONNX references, including speech, zero latents, 80/160 ms streaming, resets and independent streams. Maximum absolute error was `4.6752e-7`.
- All 21 active route/clip combinations passed before timing. All 126 qualification, warmup and measured streams also passed their exact-crop full CPU reference comparison; maximum absolute error was `5.0664e-7`. Tolerances were `atol=1e-5`, `rtol=1e-4`, with exact sample-count and finite-output checks. These are numerical comparisons, not a bitwise or perceptual-quality claim.
- The source test suite recorded 358 passed tests and 1,959 passed subtests, with one pytest deprecation warning. This is separate from the model qualification above.

The benchmark accounted for 2,292 packet, flush and reference calls, totaling 13.027 seconds inside those call boundaries. Sources and input artifacts passed the recorded identity checks. An initial Mimi attempt failed before any model call because MPS was unavailable in that process; the successful qualification is explicitly the retained `r2` receipt.

Local evidence receipts are `outputs/apple-gpu-v1/benchmark-results.json`, `audio-qualification.json`, `mimi-qualification-r2.json` and `source-tests-final.log`. This table describes the initial eager MPS baseline only; it does not include subsequent compiled-kernel experiments.

The subsequent [short GPU experiments](apple-gpu-experiments.md) reduced decoding time with full-step compilation and include a new matched CPU/Mimi comparison. The table above remains the original eager baseline.
