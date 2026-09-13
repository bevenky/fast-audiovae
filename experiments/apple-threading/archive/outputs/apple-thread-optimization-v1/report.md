# Apple CPU threading experiment

The combined matrix and convolution scheduling candidate passed numerical checks and improved the short 80 ms decoder comparison. It is retained as an opt-in experiment. It is not enabled by the public loader, and it does not replace the published single-thread baseline. Streaming and one thread remain the public defaults; no automatic all-core mode was added.

## What changed

- `StatefulDW7F32` / `StatefulDW7SnakeF32` accept an optional `parallel_channels` task size. ORT distributes independent channel ranges; the existing NEON/vForce routine still computes each range with one inner worker and unchanged arithmetic, sine tiles and history semantics.
- The first current-projection `LibxsmmPanelSmeWeightLeftF32` accepts an optional `parallel_panels` task size. Each callback handles complete N64 output panels using the same packed weights and JIT code. Callbacks have private parameters and outputs, verify FPCR, and publish accounting after joining. The whole-operation concurrency guard remains.
- Missing or zero attributes preserve the serial routes. Native `threads=1` attributes remain distinct from the explicit ORT worker budget. There is no nested BLAS worker pool.
- Maintained source hashes were updated, retaining the original source hashes. Selected default graph `5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841`, weights, package defaults and installed release artifacts were not changed.

Only these scheduling wrappers and their source receipt changed. No quantization, model retraining, activation approximation, temporal convolution change or GPU execution was introduced.

## Measurements

Apple M5 Max, ONNX Runtime 1.30 CPUExecutionProvider, sequential graph, one inter-op worker, spinning disabled. Operator checks used one, two and four intra-op threads. Decoder comparisons used four threads. Results are short diagnostic screens, not a long-corpus or paced-streaming performance claim.

The initial operator screen used original trained constants on the actual 40/80 ms matrix and late convolution shapes. It passed 1,184 output-list comparisons, including zero-length and longer-packet matrix fallback cases. Most four-thread convolution cases improved substantially. Matrix grain selection was noisy: candidate times for grains 16 and 32 were similar while the serial controls shifted. Consequently, the isolated approximately 50% matrix figure is not treated as a proven gain or proof that grain 32 is optimal.

The first decoder screen compared the frozen released library, the rebuilt serial library, matrix-only, convolution-only and combined candidates. Three Bengali/English/Spanish prefixes each supplied 960 ms, with 40/80 ms stateful packets, one warmup and three paired repetitions. Separate short checks included encoded silence, quiet speech, expressive audio, zero latents, an odd tail and a 160 ms packet that exercises the existing BLAS fallback. Every waveform and all 26 state tensors matched bit-for-bit in 62,316 tensor comparisons.

Because the rebuilt serial control itself timed differently from the frozen library, the useful scheduling comparison is against that rebuilt serial control:

| Candidate | 40 ms median reduction | Wins | 80 ms median reduction | Wins |
|---|---:|---:|---:|---:|
| Matrix only | -0.07% | 4/9 | -3.89% | 3/9 |
| Convolution only | 5.55% | 5/9 | -10.71% | 4/9 |
| Combined | 2.07% | 5/9 | 11.33% | 7/9 |

A single bounded 80 ms confirmation used only rebuilt-serial and combined sessions, two warmups and four balanced paired repetitions on each of the same three prefixes. The combined candidate used 32 N64 panels per task and eight channel ranges for the 12 later fused convolution/Snake nodes.

| Confirmation | Result |
|---|---:|
| Median paired time reduction | 12.50% |
| Faster pairs | 10/12 |
| Maximum waveform/state difference | 0, bit-for-bit |
| Additional tensor comparisons | 11,664 |
| Total process CPU time versus serial control | 1.11x |

The confirmation's equal-recording mean of median RTFs was 0.14374 serial and 0.11771 combined. These aggregate RTFs and the median paired reduction use different aggregation and therefore need not give the same percentage. They include short-prefix startup and decoder checks. State snapshots are taken outside timed calls between packets, which can affect cache state. They must not be compared directly with the README's approximately 0.071 single-thread full-recording observation or used to claim a new production RTF.

The combined 80 ms result was positive in both short screens. Forty-millisecond benefit and each component's standalone whole-decoder benefit are not established. We have not measured eight workers, concurrent stream throughput, sustained pacing, power consumption or long-corpus quality in this experiment.

## Correctness and scope

- State edge checks: 60 operator calls, 36 bitwise output and 36 bitwise history comparisons, exact independent history-tail oracle, unchanged inputs, odd channel counts, dilations 1/3/9, zero and quiet inputs, empty and short tails. Eleven malformed/unsupported attribute configurations were correctly rejected.
- Matrix edge checks: 12 operator calls, eight bitwise output comparisons, zero and quiet input, explicit serial zero, nondividing grain31, and four malformed attribute rejections. Parallel counters confirmed exactly 128 panels, five completed tasks and five FPCR checks per grain31 call. Old-library calls did not increment new-core counters, checking that the isolated libraries actually remained distinct.
- The combined and single-candidate decoder outputs and histories were bitwise equal to the frozen same-model reference. This is a focused numerical-equivalence result, not a fresh perceptual-score run across the full audio corpus.
- Focused package/default/payload tests passed: 72 tests and 46 subtests. Initial attempts failed at test collection because the runtime-only environment had no pytest, and the separate test environment needed `PYTHONPATH=src`; the final source-targeted invocation passed.
- Counted native/decoder execution across micro, state edges, matrix edges, first decoder screen and confirmation totaled about 26.58 seconds. Model loading, compilation, fixture preparation and Python checks are additional wall time. No other agent executed benchmarks concurrently.

## Decision and next work

The user's clarified objective is four-thread packet latency: distribute the work of one causal streaming packet across a shared budget of four CPU workers. Further work should improve parallel coverage, load balance and synchronization, while leaving the accepted one-thread implementation unchanged. The current first-projection candidate already creates four tasks (128 panels divided into groups of 32); task count alone does not establish four simultaneous active workers.

For each large matrix, partition complete output channels across workers, retaining the original reduction within each output. For fused depthwise/Snake work, partition channel ranges with independent history/output ownership. Inspect whether the independent current/previous projection branches can share a task dispatch to reduce joins, but verify graph dependencies before combining them. Use the same ORT worker budget throughout; library calls inside a scheduled task remain single-threaded to prevent nested pools. Dependent stages and streaming state advance in order.

The first current/previous projection branches share a ready input and join at the phase/state assembly. A candidate region can therefore schedule disjoint (branch, output-panel) tasks in one flat ORT dispatch, then join before the unchanged assembly. This requires exposing safe tiles from the current Multitile wrapper; invoking its shared-scratch handle concurrently is not safe. Two whole serial branches alone would expose only two tasks. The outer decoder lock continues to protect packet ordering and does not prevent internal parallel work.

The next short diagnostic should record task execution overlap, per-worker work duration, and the serial work between dispatches. Acceptance is reduced complete-packet latency at an explicit four-thread budget, with matched 40/80 ms inputs and numerical/state parity. Faster isolated single-thread kernels are outside this work item's objective.

Keep the combined candidate and all component results. Do not promote the standalone components, claim a 40 ms win, replace the README's performance table or change the one-thread default from these results. The existing public `load(threads=N)` remains an explicit request, with current recipe validation limits unchanged.

Next, qualify the combined80 path through the normal streaming API without per-packet diagnostic state snapshots, and check first-packet and steady packet latency separately. Preserve the accepted one-thread graph. Only after that integration check should the loader select a measured multithread variant automatically for an explicit larger thread request. If broader worker counts are added, test those counts and honor the requested upper bound rather than replacing it with all available cores. Further matrix work should consider the still-serial first previous-projection region, preserving private mutable workspace and independent output partitions.

## Why the gain is smaller than Mimi's thread scaling

These percentages have different baselines: Mimi's earlier 30.2% is one versus four ORT workers; this candidate's 12.5% is an additional improvement over the existing four-worker path. A new matched one-worker versus final four-worker comparison has not been run, so these figures must not be treated as the same comparison or multiplied into a claimed measured total.

Mapping the 13 changed node names onto the saved four-thread profile shows that they occupied 3540.125 of 12836.75 profiled microseconds per packet, or 27.58%. Most decoder work was outside this experiment. As an illustrative Amdahl calculation only, accelerating 27.58% of work by exactly 4x reduces total time by 20.68% before scheduling overhead. The profile is instrumented and from a separate run; this is a scope estimate, not a measured hard performance ceiling.

The largest untouched first-stage previous projection alone was 11.42% of that profile. The next upsampling projection pair was another 8.11%. Those direct SME wrappers still run serially and have their own mutable packing/output scratch. The later custom Apple SGEMM family also explicitly requests single-thread BLAS. Threading the ORT session does not distribute those custom routines automatically.

Mimi's measured graph places a larger share in the standard ORT convolution families: ordinary and transposed convolutions were 59.44% of its one-worker profile. AudioVAE2 instead places more work in our specialized matrix routes, depthwise convolution and Snake. Standard-runtime kernels can use the runtime's existing work scheduler; our direct native wrappers must explicitly hand independent work to it. This is a concrete remaining implementation difference, not evidence of a numerical bug.

Current official ORT CPU source passes its operator thread pool into MLAS convolution and GEMM: [Conv](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/conv.cc), [ConvTranspose](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/conv_transpose.cc), [GEMM](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/util/math_cpu.cc). This verifies the upstream scheduling design. Exact v1.30 source retrieval was unavailable, so it does not identify the installed wheel's precise selected microkernel.

The first AudioVAE2 projection has 8192x2048 FP32 weights, 64 MiB, multiplied by only one or two time positions for 40/80 ms packets. Weight-only arithmetic intensity is about 0.5/1 FLOP per byte respectively, so there is much less temporal weight reuse than in a long dense convolution. Shared bandwidth and scheduling overhead are plausible further limits, particularly at 40 ms. Neither a bandwidth-counter measurement nor a complete per-kernel hardware attribution has been performed; do not present memory bandwidth as the proven root cause.

The remaining optimization should therefore expand efficient pool usage into the untouched expensive projections and reuse their prepared weights, while keeping packet order and scratch ownership correct. Merely increasing worker count, enabling an unrestricted BLAS pool, or assuming every operation scales 4x does not address that gap.

Evidence: [micro results](micro-results.json), [state edges](state-edges-results.json), [matrix edges](matrix-edges-results.json), [decoder screen](decoder-results.json), [80 ms confirmation](confirmation-results.json), [build and source identities](build/receipt.json). Scripts are retained beside these reports; native source changes are in the repository and saved in [native.patch](native.patch). No commit or push was made.
