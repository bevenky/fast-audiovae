# Intel upsampling iteration

The candidate passed the predeclared experiment gate: at least 2% aggregate time reduction, all ten clip means improved and all numerical checks passed.

CPU-only, Intel Xeon Platinum 8280 VM, two threads pinned to CPUs 0 and 1, ONNX Runtime 1.29.0 and FP32. The same 60 recordings were validated; the original 10 timing clips used two warmups and five shuffled repetitions. These are fresh complete decoder calls, excluding loading, encoding, TTS and profiling. Lower RTF is better.

| Decoder | RTF |
|---|---:|
| Stock AudioVAE2 | 0.67068 |
| Accepted optimized AudioVAE2 | 0.25398 |
| New upsampling experiment | 0.24424 |
| Mimi | 0.17327 |

The new candidate reduces decoding time by **3.83%** against the accepted implementation in this same run. **10/10** clip averages improve. It still needs **29.06%** less time to match Mimi.

All **264 validation records pass**, including sixty clips per decoder, AudioVAE2 comparisons against both saved reference waveforms and stock ONNX, short lengths, causal prefixes and exact repeated calls. There are **200 timing observations**. Original weights, source clips, FP32 and tolerances remain unchanged. Learned MOS scores were not rerun. Numerical parity on this corpus does not prove universal perceptual equivalence.

Across five rounds the candidate was faster in 5/5. The reduction ranges from 3.07% to 4.18%. These are descriptive repeat measurements, not a confidence interval.

## What changed

A new operator combines the fourth upsampling projection pair, phase merge and C128 residual stack. It uses LIBXSMM for separate complete-K projections, direct AVX512 for residual matrices, ordered additions and private causal history. It preserves all coefficients and replaces four graph nodes with one. Runtime defaults are unchanged.

The isolated region screen improved from 301.249 ms to 267.188 ms, an 11.31% reduction, on one 6.8-second real activation. Direct projection variants were slower. This stage result is separate from the complete-decoder result above.

## Remaining work

A separate 6.8-second profile of the new candidate locates the remaining work. It excludes two warmup intervals and sums kernel durations over three calls. Profiling overhead means these values must not replace unprofiled RTF.

| Operator group | Mean ms/call | Kernel-time share |
|---|---:|---:|
| IntelPlainMatMulF32 | 597.43 | 35.13% |
| StageStackF32 | 502.03 | 29.52% |
| UpsampleStageF32 | 268.68 | 15.80% |
| MatMul | 102.04 | 6.00% |
| SnakeF32 | 61.03 | 3.59% |
| SnakeDW7SnakeF32 | 44.67 | 2.63% |

Standalone matrix operators account for 41.13% of this profile; fused residual and upsampling stages account for another 45.32%. Those fused stages contain matrix operations, Snake and depthwise work, so this profile does not separate their internal costs. The standalone Snake share of 3.59% is not the total Snake cost. Further internal attribution should precede choosing another kernel target.

The next kernel hypothesis is a row-local Snake/depthwise pipeline that removes a repeated tile-to-row copy and reduces live activation buffers. It needs internal cost attribution and separate validation before any performance claim. AMD and Apple performance were not changed or benchmarked in this iteration.

For a larger speed improvement, prioritize a bounded Supertonic-style causal student experiment against real native 48 kHz audio, with DAC-VAE as additional supervision. Keeping the noncausal teacher encoder would violate the full-codec requirement. A new native 48→48 kHz causal encoder is required. Quality and Intel RTF remain unproven for that student. See [student strategy](intel-student-strategy.md).

## Evidence

[Complete measurements and source hashes](../benchmarks/intel-upsampling/publication.json) include the frozen configuration, every decoder timing and validation record, region measurements, native checks and build metadata. Paths are aliased for portability; original and published hashes are recorded separately. The independent evidence audit was recorded before the full campaign and retains its original pending note; final-summary.json records the completed acceptance decision.
