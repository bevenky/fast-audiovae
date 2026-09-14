# Intel library screen, 14 September 2026

Release follow-up: the exact history, phase and matrix changes are integrated in
the [Intel serving recipe](../../docs/intel-serving.md). The report below preserves
the experiment-time decisions; Snake remains separate.

The second projection pair is the clear exact-output winner. Its oneDNN replacement
reduces complete decoder time by 7.2% at 40 ms and 3.7% at 80 ms. Combining it with
oneMKL sine improves the paired timings further, with small numerical differences.
The existing serving package and automatic selection remain unchanged.

## Complete decoder results

All comparisons use the new combined Intel INT8 graph, with one CPU thread and
ONNX Runtime 1.29.0. These are causal streaming decoder measurements.

| Change | 40 ms time reduction | 80 ms time reduction | Numerical result |
|---|---:|---:|---|
| oneDNN BRGeMM64 matrix pair | 7.22% | 3.75% | Exact waveform and state checks |
| oneMKL high-accuracy Snake | 1.66% | 2.54% | Small floating-point differences |
| Matrix pair + Snake | 8.67% | 5.26% | Small floating-point differences |

Each change has its own alternating control measurements. Absolute timings vary
on this shared VM, so compare each candidate with its paired control. These
numbers do not replace the earlier 0.24475 RTF reference with an unmatched run.

| Change | Packet | Paired baseline RTF | Candidate RTF | Improved pairs |
|---|---:|---:|---:|---:|
| oneDNN BRGeMM64 matrix pair | 40 ms | 0.42428 | 0.39365 | 9/9 |
| oneDNN BRGeMM64 matrix pair | 80 ms | 0.28291 | 0.27231 | 9/9 |
| oneMKL high-accuracy Snake | 40 ms | 0.39381 | 0.38727 | 6/9 |
| oneMKL high-accuracy Snake | 80 ms | 0.26050 | 0.25387 | 8/9 |
| Matrix pair + Snake | 40 ms | 0.42916 | 0.39193 | 9/9 |
| Matrix pair + Snake | 80 ms | 0.27860 | 0.26396 | 9/9 |

Three frozen Bengali, English and Spanish FLEURS prefixes were used, each 1.6 seconds.
Each variant received one warm pass and three measured repetitions. CPU affinity
was CPU 0 on the two-vCPU Xeon Platinum 8280 VM. No GPU was used. Each report
records completed native-call time separately; no long recordings or quality-model
scoring runs were added. Loading, weight packing, comparisons and hashing are excluded.

## Isolated matrix and Snake tests

Both matrices have M=3072 and K=1024. T=8 represents 40 ms and T=16 represents
80 ms. Actual intermediate inputs and original weights came from the new baseline.
All runtime quantization, transpose, allocation, scatter and dequantization are timed.

| Matrix method | 40 ms pair time | 80 ms pair time | Exact checks |
|---|---:|---:|---|
| oneMKL current separate prepare | 1.706 ms | 1.374 ms | Passed |
| oneMKL shared prepare | 1.520 ms | 1.312 ms | Passed |
| oneDNN MatMul cached weights | 0.879 ms | 0.891 ms | Passed |
| oneDNN BRGeMM panel32 | 0.809 ms | 0.765 ms | Passed |
| oneDNN BRGeMM panel64 | 0.783 ms | 0.658 ms | Passed |

The matrix screen consumed 0.109 seconds of completed native calls and passed
60 output comparisons, including zero columns and rounding ties. The winning
BRGeMM64 path reduced pair time by 54.1% and 52.1%. Its smaller whole-decoder gain
is expected: this changes only the second projection pair. The first VNNI pair
and all other matrix operations remain unchanged.

[oneDNN 3.13.2](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.2)
was built with a sequential CPU runtime and no GPU runtime. Constant weights
are packed once and reused. The experimental bridge prepares 40/80 ms plans
and keeps the existing core for other lengths, adding about 12.6 MB of packed
weight storage. The public BRGeMM API remains experimental and is version-pinned.

For Snake, 42 captured activation/coefficient combinations represented the
intercepted calls in one 80 ms packet. High-accuracy oneMKL sine reduced weighted
Snake time by 20.4% with 256-value tiles and 24.6% over complete activation calls.
The largest isolated difference was 2.38e-7. Only dynamically intercepted Snake
calls change; symbol-bound calls elsewhere remain original.

## Correctness and decision

Matrix-only waveforms and checked states were bitwise identical to the existing
INT8 decoder. At 40 ms the checker compared 3,402 state arrays; at 80 ms it compared
2,322. This includes every packet during warm qualification and short mixed
partition cases, plus final states after every timed prefix. All timed output
packet counts were correct: 360 at 40 ms and 180 at 80 ms per variant. Empty calls,
reset, state preservation and reset replay passed.

The combined candidate had a largest waveform difference of 0.0001783, RMS error
6.54e-7 and cosine similarity above 0.9999999997. The largest checked state difference
was 1.91e-6. These are small differences, but this short screen does not establish
perceptual quality equivalence. Zero and tiny latent probes are execution edge
cases, not substitutes for recordings of natural silence.

Retain the matrix-only winner first. Keep Snake and the combined candidate as
separate experiments until their numerical change receives broader quality
qualification. Keep the cached MatMul, BRGeMM32 and shared-preparation alternatives
for reference; their isolated results did not beat BRGeMM64. No serving defaults,
wheels, production kernels, commits or pushes were changed in this experiment.

## Measurement corrections and evidence

The first full Snake timing harness retained all intermediate histories while
timing. Those reports are preserved but superseded by `snake-full*-r2.json`,
which retains only audio and final states during timing. Detailed intermediate
state checks run in the existing warm passes and mixed-packet gates. The original
Snake protocol-2 aggregate includes warm outputs; the matrix/combined aggregates
keep warm qualification separate. Per-prefix comparisons and timings are unaffected.

Two setup failures occurred before inference: a C++ name conflict and an external
weight symlink rejected by ONNX Runtime. The corrected build uses an unambiguous
name and a hardlink to the same immutable weight bytes inside the model directory.
Failed build/load receipts are preserved.

`results/summary.json` contains aggregate results and hashes. The `results/`
directory preserves raw aggregate reports, build receipts and graph manifests.
The executed source is saved in this experiment directory. Captured tensors and
model weights remain on Intel and are excluded from these results. Eight local
matrix/graph tests pass. The scripts target this saved Intel test installation;
they are not a production installer.
