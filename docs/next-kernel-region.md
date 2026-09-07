# Remaining upsampling work after the combined Intel candidate

This is static graph and source analysis for the next experiment. The proposed
fusion has not been implemented or timed.

The current 109-node combined model was reconstructed in memory and matched its
recorded SHA256 exactly: `e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2`.
Its source hash is `34ccdc4b835d04c6a240c7cd2c8025995d22b4c9f3877e61e0b9ee7524d5de66`.
No reconstructed model or weights were saved. [analysis.json](../benchmarks/kernel-experiments/next-steps/analysis.json) contains the exact
shapes, node names, byte counts, consumers, and provenance.
[proposal.json](../benchmarks/kernel-experiments/next-steps/proposal.json) records the bounded next-region design separately from the
computed graph accounting.

## What remains

The 12 MKL replacements are six upsampling projections in stages 1–3 and six
residual pointwise matrices in stages 1–2. Four `StageStackF32` nodes absorb the
12 residual pointwise matrices in stages 3–6. All six phase-finish nodes remain.

Only seven ordinary `MatMul` nodes remain: the small latent projection and the
six upsampling projections in stages 4–6. Their dense arithmetic totals 18.717
GFLOP at L170; the latent projection adds only 0.0446 GFLOP. Replaced matrices
still execute their original dense work inside MKL or the stage operators.

L170 produces 326,400 samples, or 6.8 seconds at 48 kHz. A multiply-accumulate
counts as two FLOPs. Bias, Snake, depthwise and layout arithmetic are excluded
from the dense counts below.

| Destination stage | Two projections, each M × K × T | Current operator | Pair GFLOP | Following residual GFLOP inside StageStack | Each projection or phase tensor, MB |
|---|---|---|---:|---:|---:|
| 1, C1024 | 8192 × 2048 × 170 | MKL | 11.409 | Outside StageStack | 5.571 |
| 2, C512 | 3072 × 1024 × 1360 | MKL | 17.113 | Outside StageStack | 16.712 |
| 3, C256 | 1280 × 512 × 8160 | MKL | 21.391 | 16.043 | 41.779 |
| 4, C128 | 256 × 256 × 40800 | ORT MatMul | 10.695 | 8.022 | 41.779 |
| 5, C64 | 128 × 128 × 81600 | ORT MatMul | 5.348 | 4.011 | 41.779 |
| 6, C32 | 64 × 64 × 163200 | ORT MatMul | 2.674 | 2.005 | 41.779 |

The 68.629 GFLOP across all six upsampling pairs is still mandatory under the
current weights and FP32 recipe. Fusion targets how intermediate results are
stored and consumed. It does not remove these dot products.

The inspected x86 `PhaseKernel::FinishRow` still processes four time positions
with SSE2. Selecting backend 5 for Snake does not select a wider phase kernel;
this source has no AVX2 or AVX512 phase specialization. The completed Intel
profile puts all phase-finish nodes at 43.44 ms, only 2.55% of summed kernel
time. Widening interleaving alone retains both large projection tensors unless
the surrounding operators are also fused.

## Removable boundaries versus required output

For each late stage, let D = 41,779,200 bytes. The current graph materializes two
projection tensors and one interleaved phase tensor before `StageStackF32`.
Each projection has exactly one consumer, its phase-finish node. In stages 3–6,
the phase output has exactly one consumer, the corresponding StageStack. None
of these intermediates is a graph output, so the graph permits the proposed
fusion without changing other consumers.

| Change for one late stage | Full-tensor data eliminated | Logical producer-write plus consumer-read boundaries eliminated |
|---|---:|---:|
| Produce both projections as local tiles and finish phases locally | 2D = 83.558 MB | Approximately 4D = 167.117 MB |
| Feed those phases directly into the local residual stage buffer | Another D = 41.779 MB | Another 2D = 83.558 MB |
| Complete fused region | 3D = 125.338 MB | Approximately 6D = 250.675 MB |

The exact three-boundary totals are 250,670,080 bytes for stage 3;
250,674,176 for stage 4; 250,674,688 for stage 5; and 250,674,944 for stage 6.
They subtract the unread final previous-projection column. Existing segmented
StageStack overlap reads are listed separately in the JSON. These are logical
operator-boundary bytes, not measured DRAM traffic or peak RSS. Local tile
loads/stores, matrix packing, weights and warmup work remain.
The precise counts count unique semantically consumed tensor values; they do
not count repeated loads or the unused lane in the initial SSE vector load.

The fused region must still publish its final StageStack output, another D-byte
tensor needed by the following operator. That required output is not counted
as removable. Combining the three late stride-2 regions could remove about
752 MB of logical boundaries per L170 call. Including stage 3 raises that to
1.003 GB, with more difficult matrix and phase geometry. Neither number is an
RTF prediction.
The upstream pre-upsampling Snake output is also left outside this bounded
proposal, so its tensor is not included in these savings.

## Smallest next experiment to consider after profiling

The completed Intel profile measures 97.70 ms in the two stage-4 projections
and 184.12 ms in the following C128 residual stage, before phase merging. These
measurements support a bounded first experiment at stage 4: `view_43 [1,256,40800]` through its two projections, phase finish, and
`sp_stage_c128`, producing `add_901 [1,128,81600]`. This absorbs four current
nodes. Its stride is 2, so the existing high-rate Q64/128/256 stage tiles align
with complete pairs of output phases. Stage 3 uses stride 5 and M1280/K512;
retaining its efficient large-K matrix policy is a separate constraint.

For low-rate frame n and phase p, preserve exactly:

```text
current  = complete_K_dot(Wcurrent[c,p,:], X[:,n])
previous = complete_K_dot(Wprevious[c,p,:], X[:,n-1]) if n>0 else +0.0f
U[c, stride*n+p] = (current + previous) + bias[c]
```

Keep the two full K reductions distinct and add bias after their overlap sum.
The existing residual matrix helper always adds an epilogue bias, so raw
upsampling projections need a no-epilogue path. Passing a zero bias would add an
operation that the original projections do not contain. Merely stacking both
weight matrices into a larger MatMul retains the full projection intermediates;
that earlier AMD experiment already failed to improve the measured decoder.

One worker owns a contiguous output segment. Align its warmup start down to a
complete phase group at or before `max(0, first_output - 78)`. For stride 2,
that requires at most 79 warmup output frames. Initialize the previous-projection
column from the real preceding low-rate input when this aligned start is
nonzero; only the global beginning gets explicit positive zero. Do not generate
negative-time upsampling output using the learned bias. Carry the last previous
projection column and the three transformed Snake histories of 6, 18 and 54
high-rate frames across tiles. Discard the aligned warmup outputs.

With C128 and Qoutput128, Qinput is 64. The existing four C×Q scratch tiles can
temporarily hold phase output, current projection, previous projection and
gathered X. The gathered input fits because 256×64 = 128×128. After phase
interleaving, preserve one previous column and reuse the same scratch for the
three residual units. Four tiles plus histories and the small DW row buffer
total approximately 304 KB per worker. Projection weights add 512 KiB and the
three residual matrix weights add 192 KiB, before other data and code. Cache
residency therefore needs measurement even with buffer reuse.

Keep the current MKL/XSMM and direct-matrix alternatives available. Smaller
projection tiles can lower GEMM efficiency or increase weight reloads. Use ORT
segment workers with sequential matrix calls so the fusion does not introduce
nested thread pools.

The first check should expose optional debug projections, phases and all three
residual outputs. Validate them against the current graph, including tails,
short inputs, segmentation, causal prefixes and global zero history. Then pass
the existing strict whole-waveform gate before comparing whole-decoder CPU RTF
and memory. The final combined profile, which this analysis does not contain,
should determine whether stage 4, stage 3, or a different remaining operation
gets priority.
