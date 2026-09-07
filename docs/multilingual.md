# Multilingual decoder and reconstruction benchmark

The default native AudioVAE2 decoder is 4.19 times faster than stock ONNX on Apple M5 Max and 3.42 times faster on AMD EPYC 9654 at four threads. It is 1.92 times faster on the two-vCPU Intel VM. Its reconstruction metrics agree closely with stock. Meta DACVAE leads the aggregate reference-based quality metrics; Mimi leads UTMOS and DNSMOS P.835 predictions. These results describe different tradeoffs, not a universal codec ranking.

## Corpus and model contracts

The [pinned FLEURS test split](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) supplies 60 recordings, six each in Hindi, Bengali, Gujarati, Tamil, Telugu, Kannada, US English, Latin American Spanish, French and Brazilian Portuguese. Selection used seed 20260907 before reconstruction or scoring: full 4 to 12 second utterances, totaling 534.58 seconds. Two splits lacked a second gender label and used the declared seeded fallback. Speaker balancing was unavailable.

| Codec checkpoint | Continuous latent | Output | Contract retained |
|---|---|---|---|
| [AudioVAE2 / VoxCPM2](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915) | 64 dimensions, 25 Hz | 48 kHz | Causal; posterior mean; encoder input 16 kHz |
| [Pocket continuous Mimi](https://huggingface.co/kyutai/pocket-tts/tree/39592ff23c9ef80098bb74895d104c26275fe2c9) | 32 dimensions, 12.5 Hz | 24 kHz | Causal English checkpoint; not RVQ Mimi |
| [Meta DACVAE](https://huggingface.co/facebook/dacvae-watermarked/tree/8680102d141858a21bd533543966a2eb2e569f92) | 128 dimensions, 25 Hz | 48 kHz | Noncausal; posterior sample seed 0; full watermark retained |

The checkpoints have different latent capacities and output rates. Meta's decoder timing includes its integrated watermark. These are not equal-capacity, equal-bandwidth or equal-causality comparisons.

## CPU decoder timing

All rows use FP32, batch one, ONNX Runtime 1.29.0, CPUExecutionProvider only and one inter-op thread. Intra-op threads are four on Apple/AMD and two on Intel. This campaign used only CPUs. The optimized rows use the default native path, without optional packed-matrix acceleration.

| CPU and threads | Stock AudioVAE2 RTF | Default native RTF | Stock/native speedup | Mimi RTF | Meta DACVAE RTF |
|---|---:|---:|---:|---:|---:|
| Apple M5 Max, 4 | 0.11351 | 0.02706 | 4.19x | 0.03285 | 0.54146 |
| AMD EPYC 9654, 4 | 0.26078 | 0.07629 | 3.42x | 0.05400 | 0.66065 |
| Intel Xeon Platinum 8280 VM, 2 | 0.67968 | 0.35355 | 1.92x | 0.17709 | 2.06885 |

Lower RTF is faster. It is total measured decode time divided by the full generated audio duration, including right hop padding. The timing subset was fixed as the first selected clip per language: ten clips, 94.02 source seconds, one separate warmup per shape and three measured repetitions. Apple and AMD each completed 120 measurements and all 260 numerical/state/causal gates. Intel completed 150 measurements and all 327 gates, including a fifth experimental matrix route. There were no failed gates. Default native AudioVAE2 improved every one of the ten clip means versus stock on all three hosts. Numerical gates use `atol=1e-5, rtol=1e-4`; future-prefix checks apply only to the causal codecs.

These are fresh full-clip decoder calls, including the Python session wrapper and output-shape checks. Encoding, loading, file I/O, warmup, validation, scoring and profiling are excluded. They do not measure cached streaming, first-audio latency or whole TTS. All 60 clips were validated, although only ten were timed. The same ten inputs were used on all three hosts.

AMD execution was pinned to CPUs 0 through 3. Its visible CPU-time quota was 16.15 CPU equivalents; observed throttling-counter increments and host steal time were zero during timing. Apple used the normal macOS scheduler without affinity. Intel used both visible guest CPUs, with zero additional cgroup throttling, 0.00935% host steal and 0.00640% iowait. Its five comparison sessions remained loaded; available RAM was approximately 311 MiB before timing and 279 MiB after, with no swap configured. This VM had limited memory headroom and was not an isolated single-model deployment. Frequency was not fixed or measured. These are recorded host conditions, not cross-architecture performance guarantees. See the [machine-readable measurements](../benchmarks/multilingual/cpu-results.json); earlier pilot measurements in [performance.md](performance.md) use a different sample and should not be mixed with this table.

## Reconstruction quality

All 300 primary signals scored successfully: 60 originals and 60 outputs from each of stock AudioVAE2, native AudioVAE2, Mimi and Meta. There were no recorded metric errors or warnings. Means give equal weight to the ten language groups; because every group has six clips, they equal the pooled utterance mean.

| Reconstruction | PESQ-WB | STOI | ESTOI | SI-SDR, dB | Spectral convergence | Log spectral error |
|---|---:|---:|---:|---:|---:|---:|
| Native AudioVAE2 | 3.7415 | 0.9360 | 0.8846 | 12.9323 | 0.1435 | 0.7680 |
| Stock AudioVAE2 | 3.7415 | 0.9360 | 0.8846 | 12.9323 | 0.1435 | 0.7680 |
| Pocket continuous Mimi | 2.1300 | 0.8074 | 0.6707 | 3.6693 | 0.3284 | 1.0591 |
| Meta DACVAE | 4.2838 | 0.9731 | 0.9504 | 16.3054 | 0.1042 | 0.7026 |

Meta beats Mimi on PESQ and STOI for all 60 clips. Against AudioVAE2, it wins 58 of 60 on PESQ and 59 of 60 on STOI; it leads both language-level means in all ten groups.

Higher is better except for the two spectral errors. All metrics use the same FP32 16 kHz resampling, without external gain normalization, silence removal, fitted alignment or time warping. Standard metric-internal normalization remains active. Sources are 16 kHz, so this cannot establish fidelity above 8 kHz. Spectral errors average FFT sizes 256, 512, 1024 and 2048; they are not calibrated listening scores.

| Signal | UTMOS22 strong | DNS SIG | DNS BAK | DNS OVRL | DNS P.808 |
|---|---:|---:|---:|---:|---:|
| Original reference | 2.3207 | 3.2587 | 3.5834 | 2.7750 | 3.4284 |
| Native AudioVAE2 | 2.2569 | 3.2349 | 3.6134 | 2.7653 | 3.4039 |
| Stock AudioVAE2 | 2.2569 | 3.2349 | 3.6134 | 2.7653 | 3.4039 |
| Pocket continuous Mimi | 2.5168 | 3.3164 | 3.8202 | 2.8940 | 3.3387 |
| Meta DACVAE | 2.2217 | 3.2553 | 3.6034 | 2.7790 | 3.4313 |

UTMOS is the pinned single strong learner, not the ensemble or UTMOSv2. DNSMOS uses the official non-personalized P.835/P.808 windowing and calibration. These are learned predictions, not human ratings or source-fidelity tests. Their English synthetic-speech and noise-suppression domains limit interpretation on multilingual codec reconstruction.

The unexpected MOS ranking was [audited](../benchmarks/multilingual/mos-audit.json): all 225 converted UTMOS tensors match the official checkpoint bitwise. A separate torchaudio frontend plus the unchanged author head showed exact numerical agreement on 12 fixed signals; raw-bit equality was not tested. The public author API has no omitted input RMS normalization. A separate [300-signal RMS intervention](../benchmarks/multilingual/level-sensitivity.json) preserved Mimi's lead: adjusted UTMOS was 2.2573 AudioVAE2, 2.5077 Mimi and 2.2221 Meta. Attenuating the original to Mimi's level lowered its mean UTMOS. Scalar level alone does not explain the result; no denoising or human-preference mechanism is established.

All 60 optimized encoder latents matched upstream bitwise in the separate Apple encoder check. The cross-host campaign measures and validates decoders. Direct native-versus-stock decoder comparisons passed on all 60 clips, with maximum absolute difference 1.669e-6. Mean quality differences were tiny, including +9.74e-8 UTMOS and +1.06e-6 DNS OVRL. This supports numerical agreement under the checked tolerance; it is not proof of perceptual equivalence or a claim that every waveform is bitwise identical.


## Intel matrix experiment

A separate oneMKL 2026.1.0 CPU prototype replaces 12 heavy FP32 matrix nodes, keeping late small products in ONNX Runtime MLAS. Sequential MKL uses ORT-owned row workers, with no GPU or additional BLAS thread pool. At two threads it reduced decoder RTF from 0.35355 to 0.32621, a 7.73% reduction. All ten clip means improved, by 3.77% to 11.33%; 29 of 30 individual observations improved, with one Hindi repeat 2.67% slower. All numerical gates passed under the unchanged tolerances.

The prototype is not a shipped backend or an automatic loader choice. A separate fresh-process memory check measured about 211 MiB more resident memory, including 172 MiB of copied FP32 weights. Its selected libraries and headers occupy about 436 MB on disk. That is a material cost for this gain. The public timing record labels it `fast_mkl` and preserves its separate graph/library hashes; the README table uses the existing default native implementation.

## Remaining CPU cost

Separate instrumented profiles of the default native decoder attribute 52.04% of Apple operator time and 73.08% of AMD operator time to MatMul. Fused depthwise/Snake plus standalone Snake account for 36.48% and 17.91%, respectively. These profiles cover one representative shape per host and were excluded from the RTF measurements; their percentages do not decompose every timed recording.

The native depthwise kernels accumulate seven taps directly; Snake uses vector sine, and adjacent depthwise/Snake operations are fused. Dense matrix multiplication is the largest family in that profile, but the same-clip audit also finds substantial activation and residual-addition costs relative to Mimi. The existing optional AMD packing is a separate layout optimization with a memory cost, not lower-precision compression. Further matrix or epilogue changes need the same waveform and per-clip timing gates before becoming defaults. This campaign does not establish a further speedup from a proposed kernel.

Intel's [separate 6.8-second technical profile](../benchmarks/intel-profile.json), using two threads and three profiled calls, identifies the following costs. It uses an earlier English clip, separate from the ten multilingual timing clips.

| Native Intel operation | Share of profiled operator time |
|---|---:|
| Matrix multiplication | 56.52% |
| Standalone Snake | 15.42% |
| Fused depthwise + Snake | 14.13% |
| Bias and residual additions | 10.13% |
| Phase finishing | 1.84% |
| Other operations | 1.96% |

The deeper [CPU optimization plan](optimization-plan.md) maps those costs to exact matrix shapes, tensor sizes and native instructions. It prioritizes pre-Snake/DW/post-Snake fusion, ordered residual additions, and local time tiles across complete residual stages, with an additional guarded AVX512 path for Intel. These are proposed changes, not newly measured speedups. The plan also distinguishes the existing matrix-library experiments from new work that removes full intermediate tensors.

The two-vCPU VM already uses two workers; a four-thread technical screen was 3.47% slower. Matching Mimi would require 49.91% less time than the default native decoder, or 45.71% less than the MKL prototype. No tested change closes that remaining gap. New kernels must pass the unchanged waveform, causal-prefix, changing-length and concurrency checks, then improve the fixed per-clip timing comparison before promotion.

## Inspect or repeat the measurements

The [CPU record](../benchmarks/multilingual/cpu-results.json) contains individual timing repetitions, model and library hashes, provider checks, host counters and separate profiles. Numerical gates are stored in [validation.jsonl](../benchmarks/multilingual/validation.jsonl). The [corpus manifest](../benchmarks/multilingual/manifest.json), [quality summary](../benchmarks/multilingual/quality-summary.json) and [per-clip quality records](../benchmarks/multilingual/quality-per-clip.jsonl) keep the sample and scoring provenance explicit. No model weights or recordings are redistributed here.

`benchmarks/compare_decoders.py` accepts prepared ONNX models and latent/reference NPZ archives. Copy [example-comparison.json](../benchmarks/example-comparison.json), replace its placeholders with actual paths, hashes and fixed UIDs, then pass the frozen config hash. The harness performs numerical, causal-prefix and repeated-call checks before randomized timing. `benchmarks/diagnose_cpu.py` records the visible CPU environment. Preparation of the Mimi and Meta model exports, encoding and MOS scoring are separate steps; this is not a one-command reproduction of the full quality study.

The public harness is a portable refactor of the measured campaign, with stricter artifact and input validation. The raw result records retain the exact hash of the harness used for each measurement. Protocol tests cover the refactor; it was not substituted into completed timing runs.
