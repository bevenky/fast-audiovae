# Four-thread distribution of one streaming packet

The experiment distributes both independent first-stage projection branches over one ONNX Runtime four-thread pool. It retains the original FP32 matrix kernels, weight packing, phase assembly and streaming state semantics. It does not create four decoder instances or parallelize dependent packets.

The short complete-decoder screen showed a promising **9.00% median additional time reduction at 80 ms** over the previous combined four-thread candidate. The 40 ms result was inconclusive. This is retained experimental code, not a new default or a published production performance baseline.

## Work distribution

Both projections consume the same ready tensor. Each branch has 128 independent N64 output panels. The chosen grain of 16 panels produces eight tasks per branch, 16 tasks total, interleaved in one synchronous ORT dispatch. Each worker executes unchanged single-thread math for its assigned tiles. There is no nested BLAS pool.

The current projection uses the existing LIBXSMM JIT and immutable packed weights. The previous projection uses the existing KleidiAI kernel, original packed-weight strides and full destination row stride. Its input is packed once before dispatch; disjoint output columns are filled by workers; transpose and phase/state assembly occur after the join. Region guards prevent concurrent reuse while allowing independent work inside a call.

Optional traces confirmed four distinct worker threads and four overlapping callbacks in every traced call. With the chosen 16-task split, average callback overlap was 3.80 across the dispatch envelope. Median preparation plus finish cost was 13.42 microseconds; first-callback delay was 5.13 microseconds, and the gap after the final callback was 0.54 microseconds. These nine instrumented calls observe callback overlap, not hardware core residency, memory bandwidth or four cores continuously executing instructions. Trace clocks were disabled in all speed measurements.

## Short results

Apple M5 Max, CPU only, ONNX Runtime 1.30, four intra-op workers, one inter-op worker, sequential graph, spinning disabled. Nested math-library worker settings remain one. Original trained constants and unchanged stored latent inputs were used.

The isolated pair screen tried 16, 8 and 4 total tasks. The finer 16-task split improved the M2/80 ms projection pair by 12.65% median relative to separate current-parallel/previous-serial operators, winning five of six randomized-order repetitions. Its M1/40 ms result was 22.32%, winning six of six. These are small operator samples, not complete-decoder gains, and do not establish a universally optimal task size.

The decoder screen compared the previous combined candidate against this paired scheduler using the same newly built libraries. Both retained the prior channel-parallel depthwise/Snake changes. Three Bengali, English and Spanish prefixes each supplied 960 ms. Timing used two warmups and four alternating-order paired repetitions per prefix. Per-packet diagnostic state copies were excluded from timing; state correctness was checked separately.

| Packet | Median paired time reduction | Faster pairs | Separate-path median RTF | Paired-path median RTF |
|---|---:|---:|---:|---:|
| 40 ms | 1.44% | 7/12 | 0.17319 | 0.17194 |
| 80 ms | 9.00% | 8/12 | 0.11745 | 0.10882 |

The RTF columns are medians across timing observations; median paired reductions are calculated pair by pair. These aggregations need not yield the same percentage. Short prefixes include stream startup and are not the README's long-recording protocol, so these values must not replace or be compared directly with its approximately 0.071 observation. Nor should this 9% be multiplied by the previous experiment's 12.5% into a claimed measured total.

At 80 ms, each of the three recording groups had a positive median paired reduction (4.65%, 9.56%, 2.13%). At 40 ms, the English prefix was slower. The 80 ms result is worth retaining; broad 40 ms benefit is unproven. Total process CPU time divided by wall time rose from 1.88 to 1.93 for the complete 80 ms decoder. This aggregate is consistent with substantial work outside this newly parallelized region, but is not a per-operator CPU-occupancy measurement.

## Correctness and reproducibility

- The isolated projection screen passed 266 exact tensor comparisons, including quiet and zero input, empty time/batch axes, odd M3 fallback and B2. Maximum difference was zero.
- Complete-decoder qualification passed 8,100 waveform and all-26-history-state comparisons against the original released graph/libraries. Every float matched bit-for-bit. It covered the three ordinary prefixes plus expressive, digital-silence, quiet and zero-latent edges, with 40/80/160 ms calls and odd tails. Output sample counts, decoded frame counts and empty flushes were checked.
- Lifecycle checks passed 37 assertions: duplicate, missing and out-of-range tiles were rejected; aborted work recovered on the same handles; a nondividing grain of 31 panels preserved bitwise output; and the paired operator also matched with one ORT worker. The 59 API calls used 10.484375 full-matrix equivalents and 0.108 seconds of counted API time. Source/build/input identities remained unchanged through the check.
- The speed comparison used the same rebuilt native libraries on both sides. The original released libraries were used separately as the numerical reference. No perceptual scorers or long-corpus codec benchmark were rerun for this scheduling experiment.
- Package/default/source tests passed: 72 tests and 46 subtests. The selected default graph hash remains `5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841`. Streaming and one thread remain the public defaults.
- The isolated build records source, library and header hashes. The added sibling interface header was included in the receipt after compilation, with that bookkeeping correction recorded explicitly. Existing legacy CBLAS fallback declarations emitted deprecation warnings; their interface and arithmetic were retained.

Total recorded execution envelopes across the operator, trace, decoder and lifecycle scripts were about 14.70 seconds. These include Python streaming overhead and the separate state-capture checks where applicable; they are not a pure native CPU-time measure. Compilation, model loading, session construction and untimed checks took additional wall time. No other agent ran native workloads concurrently.

Evidence: [operator results](micro-results.json), [worker traces](trace-results.json), [decoder results](decoder-results.json), [lifecycle checks](lifecycle-results.json), [build receipt](build/receipt.json), [source receipt changes](source-receipt-update.json). Experiment scripts and the paired bridge are saved in this directory; native lifecycle additions remain uncommitted in the repository.

## DNSMOS publication

The documentation correction was committed and pushed separately to `origin/main` as `5f04330`. The README now explicitly shows P.835 overall and P.808, including AudioVAE2's P.808 advantage: **3.404 versus Pocket Mimi's 3.339**. Mimi still leads P.835 overall, **2.894 versus 2.765**. These verified predictor values were not swapped or selectively relabeled. The detailed documentation now records the prior saved-waveform replay and its provenance. No experimental kernel code was included in that documentation commit.

## Decision

Retain the paired scheduler as an 80 ms candidate. Do not promote it for 40 ms or change the public loader based on this short screen. Further four-thread work should expand ready-work distribution in the next projection pair, with one worker budget and explicit joins at data dependencies. Single-thread arithmetic optimization is outside this experiment's objective.
