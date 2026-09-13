# Apple GPU optimization experiments

These September 13, 2026 experiments use the original AudioVAE2 decoder weights in FP32 on Apple M5 Max, with PyTorch 2.14.0. They keep causal streaming and the existing 40/80 ms packet contracts. CPU remains the default. No quantization or retraining was used.

Each paired timing test uses three 960 ms speech crops, one warmup sweep and two measured sweeps. The public AudioVAE2 API includes owned input upload, validation of audio and all 26 histories, completed GPU execution and owned CPU-ready output. First compilation is reported separately. These are short measurements, not corpus or long-stream qualification.

The latest [numerical repair](../experiments/apple-gpu-v4/report.md) resolves the subsequent matrix candidate's internal-state failure. The selected hybrid with pointwise matrix calls passed 60 waveform checks and 1,014 state comparisons at unchanged tolerances. Its matched short tests reduced back-to-back decoding time by 53.95% at 40 ms and 56.44% at 80 ms; a separate paced 80 ms probe improved mean service time by 17.84%. This candidate is now integrated in the optional public GPU loader; the historical V2 results below and [V3 failure](../experiments/apple-gpu-v3/report.md) remain retained. CPU defaults remain unchanged. See the [integrated implementation](apple-gpu.md) for current loading and validation.

The subsequent [direct upstream GPU comparison](../experiments/apple-gpu-v5/report.md) uses the actual published AudioVAE2 class, original checkpoint and official streaming API on MPS, retaining weight normalization. All 80 waveform comparisons passed. In that matched short run, upstream RTF was 0.16657/0.09148 at 40/80 ms; the repaired candidate was 0.05832/0.03020. This reference is distinct from the original-weight port used in V4, and these back-to-back timings are distinct from paced-arrival measurements.

## Historical V2 matched comparison

All rows below were measured in the same session on the same three speech segments. Lower RTF is faster.

| Decoder | 40 ms RTF | 80 ms RTF |
| --- | ---: | ---: |
| Original AudioVAE2 GPU | 0.15392 | 0.08342 |
| Compiled AudioVAE2 GPU, original histories | 0.12619 | 0.06811 |
| Compiled AudioVAE2 GPU, projected overlap | 0.12702 | 0.06996 |
| Selected AudioVAE2 CPU, one thread | 0.12299 | 0.08398 |
| Pocket continuous Mimi GPU | Not supported | 0.06597 |

The preferred candidate is compilation with the original history representation: 18.02% lower time at 40 ms and 18.35% at 80 ms than the original GPU implementation in this session. Every paired comparison improved. At 80 ms its pooled time was 3.25% above Pocket Mimi GPU and 18.89% below the selected CPU route. At 40 ms it remained 2.60% slower than CPU. These short results do not establish equivalence or statistical significance; individual streams varied.

AudioVAE2 emits 48 kHz audio from 25 Hz latents; Pocket Mimi emits 24 kHz from 12.5 Hz latents. Both return completed CPU-ready output. AudioVAE2 additionally checks all retained history values on every call; the Mimi adapter validates input/output but not every history value. These are matched packet workloads, with this known API difference.

All 108 streams passed the original CPU ONNX references and consumed/emitted exactly the expected frames and samples. Maximum waveform absolute difference in this run was 5.06639e-7. There were no new compiled graphs during the qualification, warmup or measured sweeps. The run took 16.49 seconds overall, including 8.44 seconds in ordinary GPU calls. Initial compiler calls were excluded from RTF and recorded separately: approximately 1.09 seconds for 40 ms and 0.73 seconds for 80 ms for the preferred candidate with populated compiler caches. These are not cold-install compilation-time promises.

At the end of V2, original-history compilation was the preferred candidate. Projected-overlap reuse and standalone Snake remained experiments. V4 subsequently added the qualified hybrid matrix form; V6 integrates it in the public loader. CPU kernel selection is unchanged.

## Candidate results

Positive percentages mean less decoding time than the paired control. Each row uses its own contemporaneous control; isolated improvements must not be added together.

| Change | 40 ms time reduction | 80 ms time reduction | Finding |
| --- | ---: | ---: | --- |
| Reuse projected upsampler overlap | -1.16% | 1.81% | No clear timing gain alone |
| Compile the full step, with overlap reuse | 18.18% | 16.77% | Clear gain in the first paired comparison |
| Compile Snake separately, with overlap reuse | -5.57% | -2.38% | Slower than eager overlap control |
| Precise Metal Snake, with overlap reuse | 5.52% | 5.75% | Modest gain; less than full-step compilation |

A subsequent direct comparison of full-step compilation with and without overlap reuse found less than a 1% difference. The evidence supports compilation; it does not establish an additional overlap-reuse benefit.

## Numerical checks

Both compiled variants passed 120 waveform comparisons and 2,028 comparisons of retained state tensors. Maximum waveform absolute difference was 1.68011e-6; maximum retained-state difference was 9.29833e-6. The unchanged per-element tolerance was atol=1e-5, rtol=1e-4. This establishes bounded numerical parity, not bitwise identity or a new perceptual-quality benchmark.

Checks covered Bengali, English, Spanish, two expressive fixtures, zero and attenuated latents, mixed 40/80 ms packets, prefix causality, reset, empty input/flush, independent streams, input/state nonmutation and exact sample counts. Zero or scaled latents are synthetic numerical fixtures, not a replacement for evaluating encoded natural silence. No compiler graphs were added during qualification after preparation.

The focused source checks passed 72 tests and 7 subtests.

## Compiler detail

The original-history compiled variant initially returned a strided history view that violated the next-call contiguity check. The failed attempts were retained. An explicit contiguous materialization outside the compiled graph fixes that contract, with its cost included in timing. Both fixed shapes then passed the full state checks.

Compilation combines convolution bias, affine transforms and Snake, plus some overlap and history operations. The inspected generated wrappers still call 39 regular convolutions and six transposed convolutions externally. Static generated-kernel call counts do not measure all internal GPU dispatches.

The standalone shader uses the stored Snake coefficients and precise sine, with fast math and contraction disabled. It is an experimental alternative, not an additional optimization layered over the full-step compiler.

## Evidence

The retained workspace receipts are under outputs/apple-gpu-v2 and archived in experiments/apple-gpu-v2: overlap-r1.json, compile-r1.json, snake-r1.json, combined-r1.json, combined-r2.json, layout-probe.log, combined-r3.json, qualification-r1.json source-tests.log and final-compare-r1.json. The failed combined-r1/r2 attempts are explicitly excluded from performance conclusions. The original GPU model, candidate harness, shader and qualification code are saved alongside the receipts.
