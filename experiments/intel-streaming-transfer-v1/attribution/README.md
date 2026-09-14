# Intel fused-stage attribution

This is an isolated, timing-only diagnostic of the existing one-thread Intel
INT8 streaming baseline. It does not rewrite the graph, replace model kernels,
change weights, or save audio, latents or intermediate values.

The preload shim is copied from
`experiments/amd-int8-next-v2/stage-attribution/extension/precision_probe_extended.cpp`,
with only the Linux guard message generalized. Its existing oneMKL branch is
selected by leaving `PROBE_AOCL` undefined. The native and pointwise declarations
come from that same extension; `precision.h` comes from the existing Intel core.

Build with the existing MKL header installation, then run:

```sh
python experiments/intel-streaming-transfer-v1/attribution/build.py \
  --mkl-include /var/tmp/fast-audiovae-20260907/mkl_candidate/dependency/include \
  --output /var/tmp/intel-streaming-attribution-build-v1
python experiments/intel-streaming-transfer-v1/attribution/run.py \
  --build /var/tmp/intel-streaming-attribution-build-v1/build.json \
  --output /var/tmp/intel-streaming-attribution-result-v1
```

Use the existing Python environment with ONNX Runtime 1.29.0. Both commands
require new output directories. The driver starts a child with the timing shim
preloaded and one CPU worker, then pins execution to CPU0. It verifies the
recorded baseline graph and library hashes before inference. The original Intel
graph is pinned to `59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516`.

One 80 ms Bengali packet is captured after two warm packets. Four extracted
stages receive two warmups and three off/on pairs each, totaling 35 model calls.
The cumulative completed-call budget is one second, checked after each call.
Each output and state must match both the captured whole decoder and its
uninstrumented counterpart byte for byte. Expected interception counts are
derived from each stage's actual tile size and packet length.

Results separate activation preparation, nested integer GEMM, remaining row
work, Snake, and FP32 pointwise work. Nested GEMM is counted only once. The
remaining duration includes depthwise convolution, copying, allocations,
epilogues and runtime dispatch. Timers and shim wrappers add overhead; these
figures locate broad bottlenecks and are not new RTF results or speedup claims.
The first packed VNNI projection bypasses the intercepted functions and is not
part of this four-stage diagnostic.
