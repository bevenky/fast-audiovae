# Apple INT8 results

The Apple FP32 implementation remains the supported choice. The final SME2 candidate passed its execution checks and improved average decoder time, but the screen was too noisy to establish the required 10% improvement. It remains an opt-in experiment with no further tuning or timing campaign planned.

## Final screen

Apple M5 Max, native ARM macOS, ONNX Runtime 1.29.0 CPUExecutionProvider, four ORT threads and one inner library worker. The same Hindi, English and Portuguese clips were compared with two warmups and five measured repetitions, using randomized adjacent model groups. These are complete decoder calls; encoding, loading, warmup, validation and quality scoring are excluded.

| Decoder | RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.21840 |
| Existing Apple FP32 | 0.06400 |
| Selective SME2 INT8 | 0.05412 |
| Mimi | 0.06736 |

Selective INT8 reduced aggregate time by 15.4%, but won only 9 of 15 matched clip/repetition pairs. Its timing-repeat bootstrap interval was **-3.2% to +29.5%**. This interval resamples repetitions within three fixed clips; it describes timing noise, not generalization across languages or devices. FP32 repeat RTF varied substantially, with a coefficient of variation of 32.7%. The result does not support a reliable minimum 10% improvement.

The screen passed all 84 execution checks with zero failures and collected 60 timing observations across four models. INT8 waveform repeats and causal behavior were checked. These are not a claim of bitwise equality to FP32 or perceptual equivalence. Full perceptual testing of the Apple INT8 candidate was not run. See the [raw screen](../benchmarks/apple-precision/sme2/screen-results.json) and [summary](../benchmarks/apple-precision/sme2/screen-summary.json).

## What changed

The first Apple port used NEON SDOT but prepared activations serially and then repacked them in another pass. It took 82.2% longer than its matched FP32 control: RTF 0.17095 versus 0.09384. These earlier absolute RTF values come from a separate screen and must not be compared directly with the final screen as a controlled speedup.

A separate diagnostic profile found 251.7 ms in serial quantization/allocation and packing, or 51.2% of the 491.3 ms spent in precision matrix nodes. Quantization/allocation accounted for 116.2 ms and packing for 135.6 ms. Its 893.6 ms of dot/dequantization worker time overlaps across threads and is not additional decoder wall time. The [profile evidence](../benchmarks/apple-precision/diagnostics/results.json) retains node shapes and raw events. Instrumentation adds overhead, so these figures are not headline RTF.

The final candidate writes quantized activation panels directly in parallel and uses the pinned Arm KleidiAI SME2 complete-K integer matrix kernel. It keeps symmetric INT8 bytes, per-time-column activation scales and ordered FP32 dequantization. Weights occupy the fixed left operand; activations occupy the dynamic right operand. The original FP32 nonlinearities and smaller matrices remain unchanged.

The upstream kernel always applies a packed bias. The wrapper supplies negative zero to preserve the signed-zero result of the original dequantization expression. It also checks CPU support, streaming vector length, floating-point mode, bounds and aliasing. Each invocation owns activation state; immutable weight packing may be shared. The official streaming ABI helper is retained.

## Native validation and reproducibility

The final native suite passed 20 cases covering arbitrary tails, full-range integer values, rounding ties, subnormals, unchanged inputs, different row partitions and causal prefixes. It also passed 17 invalid-input cases, six concurrent shared-plan cases and eight dynamic ORT cases. A separate protocol suite checked incomplete, failed and duplicate preparation jobs plus shared prepared inputs.

The SME wrapper's independent checks passed 27,147 bitwise outputs, five parallel row partitions and nine zero/subnormal fixtures. Additional K=16384 checks passed all four full-range sign combinations and four concurrent callers, plus overflow, alias, scale and vector-length guards. These check exact INT8 arithmetic, not FP32 reconstruction quality.

The [source package](../experiments/apple-precision/README.md) retains tested native source bytes, parameterized builders, rewrite/check tools, the exact pinned SME2 closure and its license. Runtime sources and both original build manifests were checked for equality during publication. The original graph and binary hashes remain in the evidence; model weights, audio, fixtures and prebuilt libraries are omitted.

The [publication manifest](../benchmarks/apple-precision/publication.json) records unchanged source digests and both original and normalized JSON digests. Private paths and background application labels were generalized. Historical freeze records and the interrupted original full validation remain clearly separate from the completed final screen.

## Retained FP32 validation

The chosen FP32 path was freshly validated against stock AudioVAE2 and the saved references on all 60 multilingual clips. All 156 execution checks passed: 120 full waveforms, eight future-input checks, 26 short prefixes and two replay/history/concurrency groups. There were zero failures, 120 FLOAT waveform exports at 48 kHz, and no timing or warmup phase.

Maximum absolute error was 0.000001669 against fresh stock and 0.000002295 against the saved reference, within the original `atol=1e-5, rtol=1e-4` gate. No clip was bitwise identical, so the actual new exports were scored again. See the [FP32 audit](../benchmarks/apple-precision/retained-fp32/audit.json) and [validation summary](../benchmarks/apple-precision/retained-fp32/summary.json).

Fresh CPU quality scoring completed all 60 pairs and 12 metrics with zero errors. The retained FP32 path matches stock at the reported precision:

| Metric | Stock | Retained FP32 |
| --- | ---: | ---: |
| PESQ-WB | 3.74155 | 3.74155 |
| STOI | 0.935986 | 0.935986 |
| UTMOS | 2.25690 | 2.25690 |
| DNSMOS overall | 2.76535 | 2.76535 |
| DNSMOS P.808 | 3.40393 | 3.40393 |

Means give each of ten languages equal weight. All metrics operate on audio resampled to 16 kHz, so they assess at most the 0-8 kHz band. UTMOS here is the pinned SpeechMOS strong learner, not a full ensemble. These objective results support the retained FP32 path on this cohort; they are not human listening results or evidence for Apple INT8 quality. Exact paired differences, remaining metrics and model/code pins are in the [quality comparison](../benchmarks/apple-precision/retained-fp32/quality/comparison.json).
