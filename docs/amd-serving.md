# AMD CPU serving

The source checkout automatically selects the accepted AMD streaming kernels on
compatible Linux CPUs with usable AVX512-VNNI. CPU, streaming and one thread
remain the defaults. This update is not in the published v0.4.0 wheels yet.

The recipe combines the first two projections, processes six early convolution
histories directly, and assembles the first upsampling output without temporary
concatenations. It keeps the existing quantization, weights and 18 history states.
Batch and four-thread AMD use their existing recipes. Apple kernels are unchanged;
Intel has its own separately validated recipe. Older wheels retain their earlier AMD path.

## Short validation

AMD EPYC 9654, one CPU thread, 80 ms causal streaming. Three multilingual
1.6-second prefixes, two warmups and three measured repetitions. These RTFs
measure completed ONNX session calls; loading, Python validation and encoding
are excluded. Lower is better.

| ONNX Runtime | Saved accepted recipe RTF | Packaged recipe RTF |
| --- | ---: | ---: |
| 1.29.0 | 0.13516 | 0.13752 |
| 1.30.0 | 0.13750 | 0.14006 |

The rebuilt package took about 1.7–1.9% longer in these short comparisons. This
is a serving integration, not another speedup. The earlier matched comparison
measured the accepted recipe at 0.13754 and Pocket continuous Mimi at 0.11815;
Mimi was not rerun during packaging.

All 639 recorded waveform, history and stream-behavior checks passed under each
runtime. Checks cover mixed 40/80 ms calls, quiet inputs, reset, independent
streams, empty calls and future-prefix invariance. Every timed prefix also
matched the accepted output and final state exactly.

An installed development wheel then passed the public `load()` path under
1.29.0, including automatic CPU selection, fresh cache preparation, cache reuse,
and exact 40/80 ms output. All 11 native libraries, including OpenMP, loaded from
the package. An additional 171 waveform/state checks passed under 1.30.0 after
fixing AOCL's adjacent-library lookup. That lookup-only fix was not retimed.

This establishes parity with the existing optimized INT8 recipe on the checked
inputs. It does not imply lossless agreement with stock FP32 or replace the
earlier audio-quality panel. No GPU was used.

[Aggregate evidence](../benchmarks/streaming/amd-serving-v1.json) and
[accepted experiment](../experiments/amd-int8-streaming-transfer-v1/baseline.json).

## Packaging

`tools/package_x86_native.py` combines exported AMD and Intel native payloads
into a platform wheel. Supply both payloads for a combined release; the AMD-only
development wheel was used only for this validation. ELF dependencies, hashes,
glibc requirements, licenses and the CPU probe are checked before packaging.
No compiler is required when a user installs a native wheel.
