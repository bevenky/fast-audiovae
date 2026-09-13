# Apple GPU work

This is the original development plan. The integrated implementation is documented in [Apple GPU decoding](apple-gpu.md).

The public selector is `device`, with `"cpu"` as the default. Only an explicit `load(device="gpu")` request selects the Apple GPU. Streaming remains the default mode. An unavailable GPU or unsupported operation raises an error instead of silently running the decoder on the CPU.

The first backend uses PyTorch 2.14 MPS and the existing checksum-verified ONNX tensors. It keeps FP32, the six sample-rate conditioning affine transforms, Snake operations, causal convolutions, upsampling and output tanh. There is no retraining or quantization. Every stream owns its bounded history on the GPU, while weights are shared and immutable. Public decode calls return ready NumPy audio on the CPU.

## Implementation sequence

1. Add optional GPU preparation/loading without changing the CPU build or dependencies. Provide streaming, reset, empty flush and independent batch calls with the existing NumPy input/output shapes.
2. Check the explicit model against stock CPU ONNX on the same saved latents. Cover 40/80 ms calls, startup, quiet input, expressive input, odd tails, two independent streams and exact sample accounting. Disable CPU fallback and fast math.
3. Run a bounded paired 40/80 ms timing check. GPU timing must include completion and the output transfer, not only host enqueue time. Keep the existing CPU baseline in the same session and state the small test scope.
4. Use the measured bottlenecks to choose the first optimization. Candidates are fixed-shape compilation to reduce small operation dispatch, fused Snake operations, and native MPS matrix paths for pointwise convolutions. Qualify each against the unfused GPU baseline before retaining it.

No GPU speedup is assumed. The specialized CPU path is already efficient at short packets, so GPU launch and synchronization costs may dominate. Multi-stream batching is a separate workload and cannot replace the requested single-stream latency measurement. [Initial measured results](apple-gpu-results.md) include original-weight AudioVAE2, Pocket Mimi GPU and the selected CPU decoder.

## Portable runtime versus native Metal

Keep PyTorch as the first GPU backend and numerical reference. It already calls Apple's native MPS/Metal operations. The model code can support other GPU backends later, but CUDA and other devices are not implemented or validated here. CPU remains the public default, and GPU selection stays explicit.

Try small compiled regions before committing to a second inference runtime. The isolated Snake compilation probe preserved its outputs exactly on three representative shapes, with median call reductions of approximately 21%, 6% and 10%. These measurements include a synchronization per isolated operation and do not establish a whole-decoder improvement. No compiled Snake change was promoted into the decoder.

A native Apple backend remains a candidate if compilation leaves a substantial measured gap. It could call MPSGraph/MPS matrix and convolution operations directly from a packaged native library, use Metal shaders for fused pointwise work, and retain per-stream buffers between calls. That would remove the PyTorch runtime dependency from that route, but require separate implementation and validation. Its speed advantage is unmeasured. The public `device` selector should not expose these internal implementation choices.

## Sources

- [PyTorch 2.14 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)
- [Apple MPS architecture](https://developer.apple.com/metal/pytorch/)
- [MPS environment controls](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)
- [GPU synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)
- [Apple's native GPU kernels](https://developer.apple.com/documentation/metalperformanceshaders)
- [Compiled native graph execution](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable)

CoreML remains an alternative to evaluate later. Its CPU-and-GPU setting allows CPU work, and conversion requires explicit FP32 precision to avoid the default FP16 transformation. These distinctions must be preserved in any comparison.

## Completed short experiments

The four candidates have now been tested. Full-step compilation gave the clearest reduction; standalone Snake compilation regressed, precise Metal Snake helped modestly, and projected-overlap reuse had no established extra benefit. The original-history compiled path is the preferred next candidate. See [results and qualification](apple-gpu-experiments.md). CPU defaults remain unchanged; the qualified hybrid matrix path is now integrated in the optional GPU loader.
