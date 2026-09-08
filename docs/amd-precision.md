# AMD selective INT8 results

The completed AMD AOCL screen reduced decoder time by 43.3% against the matched optimized FP32 control. Runtime checks and fresh quality scoring are complete, with the INT8 recipe's small measured score declines retained. This is a tested opt-in recipe and does not change public defaults.

## Matched screen

The single final screen used the fixed Hindi, English and Portuguese clips, five measured repetitions after two warmups, ONNX Runtime 1.29.0 CPUExecutionProvider, four outer workers pinned to four logical CPUs and one inner integer-library worker. Model groups were randomized within each clip/repetition. RTF includes complete decoder calls and excludes encoding, loading, warmup, validation and quality scoring.

| Decoder | RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.25556 |
| Optimized FP32 | 0.06056 |
| AOCL selective INT8 | 0.03435 |
| Mimi | 0.05201 |

INT8 won all 15 paired observations. The paired repeat bootstrap interval was 43.02% to 43.57% less time; per-clip repeat coefficients of variation ranged from 0.19% to 1.41%. This interval describes repeat noise within three fixed clips, not performance across all languages or AMD processors. The screen passed all 84 execution checks and collected 60 timing observations across four models. [Raw screen](../benchmarks/amd-precision/screen-aocl4-r1-results.json), [statistics](../benchmarks/amd-precision/screen-aocl4-r1-statistics.json).

## Why the original port was slow

The initial unchanged oneMKL port ran at RTF 0.22967 against FP32 0.06071 in its own matched screen. The process loaded `libmkl_def` and `libmkl_vml_def`; these mappings are a dispatch clue, not a trace of every GEMM instruction used. It also used only two native matrix shards/stage segments inside a four-thread ORT budget. It is not the recommended AMD recipe.

The final derivative changes the integer GEMM backend to AOCL-DLP and raises 14 matrix shard counts plus four stage segment counts to four. It preserves weight quantization, activation scales computed across channels separately for each time column, complete-K integer products, signedness compensation and ordered FP32 dequantization. Nonlinearities remain FP32. The result measures the backend and scheduling changes together, not a library-only comparison.

A separate diagnostic profile supports the backend finding. Summed GEMM worker time, including internal packing, was 3808.8 ms for the original port and 400.0 ms for the final configuration on the same 255-frame latent case. Preparation was approximately 52 ms in both. Worker times overlap; they are not decoder wall time. The separately profiled decoder wall times were 2342.6 ms and 351.1 ms, and instrumentation adds overhead. These diagnostic numbers must not replace the matched RTF table. [Attribution and pins](../benchmarks/amd-precision/diagnostic-summary.json).

## Correctness and quality scope

The original-stage and two-versus-four scheduling suite passed all 299 checks, including all three outputs from retained C64/C32 residual stacks, short tails, exact future perturbations and the original FP32 tolerance. Full-range integer native tests also passed. The final full 60-clip run completed 234 execution checks and 180 waveform exports with zero failures and no timing phase.

The exact-waveform audit found all 60 stock outputs and 52 of 60 INT8 outputs equal to the earlier Intel outputs; none of the 60 optimized FP32 outputs were byte-identical. The actual AMD exports were therefore scored afresh. Native INT8 parity verifies implementation arithmetic; it does not establish equality to FP32 or unchanged perceived quality. [Full runtime report](../benchmarks/amd-precision/quality-aocl4-r1-results.json), [identity audit](../benchmarks/amd-precision/quality-identity.json).

## Quality

Fresh CPU scoring completed all 60 paired clips and 12 metrics without errors. Means give each of ten languages equal weight. The most familiar metrics are:

| Metric | Optimized FP32 | AOCL INT8 | INT8 minus FP32 |
| --- | ---: | ---: | ---: |
| PESQ-WB | 3.74155 | 3.71737 | -0.02418 |
| STOI | 0.935986 | 0.934448 | -0.001538 |
| UTMOS | 2.25690 | 2.25161 | -0.005295 |
| DNSMOS overall | 2.76535 | 2.76034 | -0.005009 |
| DNSMOS P.808 | 3.40393 | 3.39706 | -0.006875 |

Paired bootstrap intervals for PESQ, STOI and UTMOS remain below zero. The DNSMOS overall and P.808 intervals include zero. Full differences, confidence intervals and the remaining metrics are in the [quality comparison](../benchmarks/amd-precision/quality/comparison.json).

All metrics use 16 kHz resampled audio and assess at most the 0-8 kHz band. UTMOS uses the pinned SpeechMOS strong learner. These are objective predictions on this fixed cohort, not an AMD listening-panel result or a guarantee for unseen audio. The speed gain comes with a measured approximation tradeoff; use FP32 when that tradeoff is unsuitable.

## Rebuilding

The [source package](../experiments/amd-precision/README.md) retains exact measured native/fused source, dependency pins, licenses, graph/test tools and original build records. A new packaging-only adapter compiles the four prerequisite objects before linking against supplied AOCL, LIBXSMM and native CPU libraries. It preserves the recorded flags and order and has offline command checks; it has not been compiled or timed as an additional candidate.

The measured dependencies are AOCL-DLP commit `c577191304a3db0029f2f12fcacbc8ad296a645d`, LIBXSMM commit `55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1`, ORT API29 headers and the recorded SLEEF-backed native CPU library. Exact supplied source/header/library digests are checked before and after compilation. Rebuilt outputs are always unvalidated, even when dependency hashes match. Different compiler paths or inputs do not inherit historical binary identity.

This specific backend requires AMD AVX512 VNNI plus OS vector-state support. It is not a generic all-AMD acceleration claim. The public FP32 runtime is unchanged. [Publication metadata](../benchmarks/amd-precision/publication.json) distinguishes original evidence digests from normalized private-path copies. Models, weights, audio, binaries, installed dependencies and object files are omitted.
