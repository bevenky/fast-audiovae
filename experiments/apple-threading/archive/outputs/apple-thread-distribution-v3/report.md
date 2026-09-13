# Second projection pair across four Apple CPU workers

The extension is rejected. It distributed work across all four workers correctly, but did not reduce complete streaming decoder latency. The earlier first-pair and depthwise/Snake improvements remain intact. Default streaming mode and the default single thread are unchanged. No kernel changes were committed or pushed.

| Output packet | Change in decoder time | Faster paired trials | Decision |
| --- | ---: | ---: | --- |
| 40 ms | 0.06% slower | 6/12 | No demonstrated gain |
| 80 ms | 4.01% slower | 0/12 | Reject extension |

Changes are medians of paired timing ratios. At 80 ms, every trial regressed, by 3.40% to 5.26%. This compares the retained earlier candidate against that same candidate plus second-pair scheduling, in one measurement session. Absolute RTFs from different experiment sessions should not be compared as cumulative gains.

The complete decoder median RTFs in this session were 0.091591 versus 0.091543 at 40 ms, and 0.060711 versus 0.063015 at 80 ms. Ratios of separate medians differ slightly from medians of paired ratios; the latter determine the decision above.

## What changed

The second upsampling projection pair shares an input with 1,024 channels. Each branch produces 3,072 channels. Instead of executing the two original matrix operators sequentially, the experiment split each branch into four 768-channel tasks and scheduled all eight tasks through the same four-worker ONNX Runtime pool. Packing and final output layout conversion remained outside those tasks. State and phase assembly were unchanged.

The experiment retained the original matrix arithmetic, weights, reduction order and CPU precision. It used ONNX Runtime 1.30.0, four intra-op workers, one inter-op worker, sequential graph execution, no spinning, and one inner native/BLAS thread. Only CPUExecutionProvider was enabled.

## Verification and scope

- 8,100 complete decoder waveform/state tensor comparisons passed bitwise, with maximum absolute difference zero.
- 266 isolated projection comparisons passed bitwise, including empty inputs, quiet values, longer fallback lengths and multiple batch elements.
- Lifecycle checks passed for duplicate, missing and out-of-range task rejection, recovery after abort, partial final tiles, and a one-worker ORT call. The logged constructor error for misaligned task size is an expected rejection test.
- All three instrumented calls used four distinct worker threads with four overlapping callbacks. The average overlapping callbacks during dispatch was 3.49 to 3.63. This measures callback overlap, not hardware utilization or core residency.
- Timing used three 960 ms language prefixes, two warmups per candidate/input, and four alternating-order paired repetitions per recording at both packet sizes. Numerical qualification additionally covered expressive audio, digital zero, quiet input, zero latents, 160 ms calls and odd tails. No long corpus benchmark was run.
- Counted call envelopes totaled about 7.66 seconds across isolated tests, traces, complete decoder calls and lifecycle checks. This excludes compilation, model/session setup and script overhead.

The isolated 80 ms projection test gave only a 0.90% median gain with three wins out of six for the selected tile size. That was inconclusive and did not carry through to the decoder. The trace verifies actual parallel execution, but does not establish the exact cause of the slowdown. Packing, output conversion, scheduling, or contention could matter; the experiment does not isolate those contributions.

## Retained evidence

The prior first-pair candidate remains the useful result: its own short paired experiment measured approximately 9% lower 80 ms decoding time. This new extension adds no demonstrated benefit. Further worker splitting is not justified by these results.

All raw results, traces, binaries and source remain in this directory:

- `decoder-results.json`, `micro-results.json`, `trace-results.json`, `lifecycle-results.json`
- `build/receipt.json`, containing the original build commands and hashes
- `archived-source/native.cpp`, the exact tested experimental sweep implementation
- `sweep-native.patch` and `archive-receipt.json`, documenting the source archive and restoration

Only the second-pair experimental addition to the previously clean package `sweep/native.cpp` was restored. The five preexisting modified package files were verified unchanged, and all 35 pinned native source hashes match their source receipt. `build.py` now uses the archived experimental source, and `lifecycle.py` resolves its old build path through the archive receipt. Their original validated versions are preserved under `archived-source/validated-*.py`. These path-only archival adjustments were checked for Python syntax; no additional performance run was performed after archival.
