# Apple first-pair INT8 projections

This library replaces only the current and previous projections in the first
decoder upsampling stage. It requires Apple ARM SME and SME2 and runs one worker.
Input, output, all surrounding operators and streaming state remain FP32.

Weights use symmetric per-output-channel INT8 quantization. Activations use
symmetric per-frame INT8 quantization across all 2,048 channels, with nearest-even
rounding. The full reduction accumulates into INT32. Dequantization computes
`float(dot) * float(weight_scale * activation_scale)` without FMA contraction.
No scale is shared between time frames.

The 40/80 ms path passes one or two frames directly to the qualified adapter.
Larger streaming packets gather pairs into fixed scratch buffers and scatter
their results back. They use the same arithmetic, so changing packet size cannot
switch precision or introduce dependence on future frames. The operator serializes
access to its reusable scratch buffers; separate sessions own separate buffers.

`upstream/` contains unchanged KleidiAI files at commit
`8730c61c5a176060c597cb7f54d5ba0580e44f09`. Their provenance and hashes are in
`upstream-provenance.json`. Apache 2.0 terms are included in
`licenses/KleidiAI-INT8-LICENSE.txt`.

Build offline with `python tools/build_apple_int8.py --offline`. The resulting
library links only system C++ and system runtime libraries. `validate.py` is an
optional development check against the original qualified experimental bridge;
that bridge is never part of the shipped runtime.
