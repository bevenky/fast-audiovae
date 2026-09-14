# Intel CPU streaming

The selected Intel recipe combines the accepted raw-history and phase-assembly
changes with the exact-output oneDNN BRGeMM64 matrix pair. It targets compatible
Linux Intel CPUs with usable AVX512-VNNI. CPU, streaming and one thread remain
the defaults. These kernels are included in the v0.4.1 native wheels.

## What changes

Six early convolution regions read their history directly, and the first
upsampling stage assembles its output without temporary concatenations. The
second projection pair packs its constant weights once and uses oneDNN for
40/80 ms calls. Other chunk sizes retain the existing oneMKL matrix path.

The first VNNI projection pair, quantization, weights, SLEEF Snake activation
and all 18 history tensors stay unchanged. The oneMKL sine experiment is not
included. Intel batch decoding and its existing worker schedules are unchanged;
older native wheels retain their previous Intel recipe.

## Accepted experiment

Intel Xeon Platinum 8280 identity exposed by a two-vCPU VM, one CPU thread,
ONNX Runtime 1.29.0 and causal streaming. Lower RTF is better.

| Output packet | Previous combined baseline RTF | Matrix candidate RTF | Decoder time reduction |
| --- | ---: | ---: | ---: |
| 40 ms | 0.42428 | **0.39365** | 7.22% |
| 80 ms | 0.28291 | **0.27231** | 3.75% |

The previous combined baseline already contains the history and phase changes.
These gains measure the additional matrix replacement. Each row uses fresh,
alternating reference and candidate measurements; timings from separate VM
sessions should not be combined into another claimed speedup.

Three Bengali, English and Spanish FLEURS prefixes, each 1.6 seconds, received
one warm pass and three measured repetitions at each packet size. The candidate
improved all nine paired samples at both sizes. RTF measures completed decoder
`Session.run` calls; loading, packing, encoding and verification are excluded.
AudioVAE2 produces 48 kHz audio from 64-channel latents at 25 Hz. No GPU was used.

Waveforms and checked histories matched the existing optimized INT8 decoder
bit for bit. The checks covered 3,402 state arrays at 40 ms and 2,322 at 80 ms,
including warm-pass packets, mixed chunk sizes and final timed states. Packet
counts, empty calls, reset and reset replay passed. This establishes no additional
numerical change on the checked inputs; it does not mean INT8 matches stock FP32
bit for bit or replace the earlier audio-quality evaluation.

[Matrix results and protocol](../experiments/intel-library-screen-v1/README.md)
and [earlier history/phase validation](../experiments/intel-streaming-transfer-v1/README.md).

## Serving integration validation

The installed development wheel passed public `load()` with CPU, streaming and
one-thread defaults. Waveforms matched exactly across 40 sequence pairs, with
4,266 history-array comparisons. Both variants passed all 16 behavior checks,
covering mixed/odd packets, zero/tiny inputs, empty calls, reset, independent
streams, partition equivalence and future-prefix invariance.

| Packet | Accepted matrix experiment RTF | Installed package RTF |
| --- | ---: | ---: |
| 40 ms | 0.39595 | 0.39408 |
| 80 ms | 0.26521 | 0.26655 |

The package stayed within about 0.5% of its paired reference. This confirms
integration preserved performance; it is not another claimed speedup. The
complete qualification used 27.66 seconds of native execution. Local validation
passed 528 tests and 1,963 subchecks.

Preparation used ONNX 1.22.0 and ORT 1.29.0. Original graph preparation completed
through the installed package. A subsequent runtime-only fix reused that
completed recipe with all 50 files verified unchanged, then passed new public
cache publication and cache reuse. No native compiler ran during public loading.

Repeated loading exposed an older LIBXSMM unload defect that closed stdin when
an unused session was destroyed. The runtime now retains verified operator
library handles for the process lifetime. Repeated-load checks pass; weights,
kernels and decoder calculations are unchanged. Preparation also releases four
large graph objects after their final use to reduce temporary memory usage.

[Integration evidence](../benchmarks/streaming/intel-serving-v1.json) and
[full aggregate report](../experiments/intel-serving-v1/result.json).

The package uses oneDNN 3.13.2 with a sequential CPU runtime and no GPU runtime.
Its library inventory, hashes, redistribution notices and relative dependency
paths are verified during packaging. The native wheel supplies these dependencies;
ordinary installation does not compile them. Version 0.4.1 packages this validated native implementation.
