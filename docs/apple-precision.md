# Apple INT8 results

Use the Apple FP32 implementation. The final SME2 panels experiment passed correctness checks but reduced decoder time by only 3.0%, below the required 10% improvement. Its automated quality scores were also slightly lower than FP32. It remains an explicit experiment; runtime defaults are unchanged.

## Final matched comparison

Apple M5 Max, native ARM macOS, ONNX Runtime 1.29.0 CPUExecutionProvider, four ORT threads and one inner library worker. The fixed ten clips were compared with two warmups and five measured repetitions. FP32 and INT8 calls were adjacent with balanced randomized order. The other models were shuffled around that pair, and all 200 observations were retained.

| Decoder | RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.092772 |
| Recommended Apple FP32 | 0.024685 |
| Selective SME2 panels INT8 | 0.023938 |
| Mimi | 0.028114 |

RTF is the arithmetic mean of the per-clip mean RTFs, so each selected clip has equal weight. It measures complete decoder calls and uses complete generated waveform duration, including documented right padding. Loading, encoding, warmup, validation, profiling and quality scoring are excluded. There is no cached streaming API.

INT8 saved **3.026%** of decoder time. Its 95% bootstrap interval, resampling clips and repeat blocks, was **0.86% to 5.86%**. This fails the 10% adoption threshold. FP32 already ran faster than Mimi in this matched comparison. These numbers do not establish performance on other Apple CPUs or under other system loads. See the [compact evidence](../benchmarks/apple-precision/sme2-panels.json).

## Correctness and quality

All **1,620 execution checks passed with zero failures** across the four models and 60 multilingual clips. The new INT8 implementation was bitwise equal to the previous Apple INT8 implementation on all 60 full waveforms and 120 prefix comparisons. Checks also covered causal prefixes, future-input perturbations, repeated calls and concurrent calls with different inputs and lengths. FP32 retained the original `atol=1e-5, rtol=1e-4` reference gates.

Equality to the earlier INT8 implementation verifies that these kernel changes preserved its output. It does not make INT8 equal to FP32. Fresh automated scoring on all 60 clips found small decreases for INT8:

| Metric | Apple FP32 | SME2 panels INT8 |
| --- | ---: | ---: |
| PESQ-WB | 3.741549 | 3.716975 |
| STOI | 0.935986 | 0.934413 |
| UTMOS22 | 2.256900 | 2.251456 |
| DNSMOS overall | 2.765348 | 2.759253 |
| DNSMOS P.808 | 3.403932 | 3.397418 |

Higher is better. The common 16 kHz source comparison covers at most the 0-8 kHz band, although AudioVAE2 produces 48 kHz output. Means cover the same balanced ten-language cohort. UTMOS uses the pinned SpeechMOS strong learner; UTMOS and DNSMOS are predictions, not human listening-panel results. These means do not establish an absence of audible loss, and no new standardized MUSHRA panel was conducted.

## What changed and what remains expensive

The candidate replaces 22 large matrix products with symmetric INT8 while preserving per-row weight scales, per-time-column activation scales and complete-K INT32 accumulation. Nine smaller products, nonlinearities and depthwise convolution remain FP32. Learned initializer bytes and the existing encoder interface are unchanged.

Compared with the earlier Apple experiment, it prepares bounded time panels directly into the SME2 layout, avoids the production copy in the original tensor layout and shares preparation for the paired C128 upsample projections. Four fused late regions reuse intermediate buffers. The pinned Arm KleidiAI kernel, its official streaming ABI helper and guarded floating-point contract are retained. CPU support and streaming vector length are checked before execution.

A separate profile attributes **62.55% of recorded kernel time** to the four fused regions. The C64 and C32 stages, whose matrix products remain FP32, account for **31.18% of the same total**, already included in that 62.55%. These percentages overlap and must not be added. A fused operator includes multiple operations, so this trace does not independently separate matrix, Snake, depthwise or residual costs inside it. It is neither a causal-latency breakdown nor a replacement for unprofiled decoder RTF.

## Reproduction and historical evidence

The [SME2 panels source package](../experiments/apple-precision/sme2-panels/README.md) includes exact native/fused sources, pinned upstream code and licenses, graph rewrites, builders, checks and portable campaign/config entrypoints. It does not bundle model weights, audio, case archives or compiled libraries. The portable runner differs from the executed corrected runner only in the shared helper's repository path; its hash and diff are recorded separately.

The first full run stopped before timing because its validation script assumed every Mimi clip contained at least 65 latent frames. Four clips were shorter. The corrected runner uses each real clip's valid prefix length, without padding or repeating source data, and keeps the timing logic and numerical tolerances unchanged. The failed run remains separate from the completed results.

Earlier [SDOT and SME2 screens](../benchmarks/apple-precision/publication.json) and [retained FP32 validation](../benchmarks/apple-precision/retained-fp32/audit.json) remain archived. Their controls and system conditions differ from this campaign. Do not compare their absolute RTFs with the new row as a controlled speedup, or mix the historical Meta DAC-VAE timing into this comparison.
