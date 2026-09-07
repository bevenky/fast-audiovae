# Closing the CPU gap to Mimi

The next useful change is a fused residual-stage implementation that keeps intermediate audio features in small reusable buffers. Changing another BLAS library or only widening the sine loop does not address all the remaining work.

This analysis used saved CPU profiles, source, assembly and static graph shapes. **No new benchmarks were run. All further speedups discussed here are unmeasured.** The [audit data](../benchmarks/cpu-optimization-audit.json) includes stage totals, matrix shapes, historical comparison extracts and source hashes.

Inference remains pinned to **ONNX Runtime 1.29.0**. The separate ONNX model-editing package is now 1.22.0; [static compatibility checks](../benchmarks/onnx-tooling-compatibility.json) reproduced the accepted native and fallback graphs byte-for-byte. This tooling update does not change the model operations or the recorded benchmark results.

## The target

| CPU | Fast AudioVAE2 RTF | Mimi RTF | Reduction in AudioVAE2 time needed to match Mimi |
|---|---:|---:|---:|
| Apple M5 Max, four threads | 0.02706 | 0.03285 | Already uses 17.62% less time |
| AMD EPYC 9654, four threads | 0.07629 | 0.05400 | 29.22% |
| Intel Xeon 8280 VM, two threads | 0.35355 | 0.17709 | 49.91% |

These are CPU-only, FP32, full-call decoder measurements on ten fixed multilingual clips, three repetitions each. They exclude encoding and profiling. AudioVAE2 produces 48 kHz; Mimi produces 24 kHz. The Intel MKL prototype reaches 0.32621, still requiring another 45.71% reduction to match Mimi. Its measured 7.73% gain costs about 211 MiB more resident memory. [Measurements and methodology](https://github.com/bevenky/fast-audiovae/blob/main/docs/multilingual.md)

The arithmetic difference is real: AudioVAE2 requires approximately **17.982 GFLOP per generated second**, versus Mimi's **7.678** for the saved 6.8-second shape, approximately 2.342 times as much. These are dense multiply-accumulate counts, with two FLOPs per MAC. They exclude nonlinearities, normalization, copying and other overhead. Mimi's exported dense attention makes its count duration-dependent; this is not a fixed streaming ratio or a predicted speed ratio. Convolution counts include conventional padded/boundary work, including transposed-convolution output later sliced away.

## Exactly where the time goes

Intel's saved two-thread profile covers three instrumented calls on a 6.8-second English recording:

| Operation | Share |
|---|---:|
| Upsampling matrix products | 32.94% |
| Residual pointwise matrices | 23.56% |
| Other matrix products | 0.02% |
| Standalone Snake | 15.42% |
| Depthwise convolution plus Snake | 14.13% |
| Bias and residual additions | 10.13% |
| Phase finishing | 1.84% |
| Other operations | 1.96% |

This is [instrumented operator time](https://github.com/bevenky/fast-audiovae/blob/main/benchmarks/intel-profile.json), not a decomposition of every clip in the RTF table. Matrix multiplication is the largest family, but approximately 40% is Snake, depthwise/Snake and additions.

The six upsampling stages shrink channels while increasing temporal resolution. From stage 3 onward, those changes exactly cancel in activation volume:

| Stage | Channels | Steps per audio second | Intel profile share | AMD profile share |
|---|---:|---:|---:|---:|
| 1 | 1,024 | 200 | 12.81% | 13.54% |
| 2 | 512 | 1,200 | 17.94% | 19.67% |
| 3 | 256 | 6,000 | 23.58% | 26.95% |
| 4 | 128 | 12,000 | 17.25% | 17.20% |
| 5 | 64 | 24,000 | 14.67% | 12.33% |
| 6 | 32 | 48,000 | 13.65% | 10.20% |

Each stage-3-through-6 feature tensor contains 1.536 million floats per audio second: **41.779 MB for 6.8 seconds**. Each residual block repeatedly visits that volume. These stages contain **88.14% of all Snake evaluations**, despite their smaller channel counts.

Stage means output temporal-rate domain; upsampling belongs to its destination stage. Under this grouping, stages 3-6 total 69.15% Intel, 66.69% AMD and 74.04% Apple, replacing the older 81% figure. AMD/Apple profiles use 5.64 seconds; Intel uses 6.8 seconds. The current AMD instrumented profile was substantially slower than its ordinary timing run, so its individual milliseconds should not be scaled onto the headline RTF. [Current profile records](https://github.com/bevenky/fast-audiovae/blob/main/benchmarks/multilingual/cpu-results.json)

A historical AMD acceptance campaign also profiled both decoders on the **same 6.8-second English clip, four threads**:

| Descriptive family | Fast AudioVAE2 | Mimi |
|---|---:|---:|
| Upsampling GEMMs / transposed convolution | 174.646 ms | 164.049 ms |
| Residual/other GEMMs / convolution plus transformer GEMMs | 132.049 ms | 102.377 ms |
| Snake plus depthwise/Snake / ELU | 110.468 ms | 26.054 ms |
| Add | 54.022 ms | 10.858 ms |
| Complete profile sum | 490.535 ms | 338.636 ms |

These families are not mathematically identical, but they reveal why the gap cannot be assigned only to upsampling or GEMM. Source: saved `remote_cpu/acceptance_results.json`, `phase_fused` and `mimi` profiles; accepted graph hashes match the current graphs, while the packaged native-library binary differs. This is historical attribution, separate from the [published multilingual comparison](https://github.com/bevenky/fast-audiovae/blob/main/docs/multilingual.md). No matched Intel Mimi operator profile was found.

## Implementation order

### 1. Fuse pre-Snake, depthwise convolution and post-Snake

Every residual unit currently executes:

```text
input -> Snake -> DW7 -> Snake -> pointwise matrix
   \---------------------------------> residual addition
```

There are 18 units. The first Snake output has one numerical consumer, the depthwise operation; the skip uses the original input. A combined kernel can transform new input values into a short local buffer, retain the transformed history required by dilations 1, 3 or 9, and run the existing ordered DW7 and post-Snake.

Maximum per-channel history is 54 floats. A 256-value tile plus history occupies 1,240 bytes per row worker. This removes 18 full pre-Snake tensors and their write/reread boundaries: **167.1168 MB per audio second of logical activation traffic**. That is not a measured DRAM-bandwidth saving or predicted RTF gain.

This is useful on both Apple and x86 while retaining their existing vForce/SLEEF sine routines. It is more substantial than deleting only the current 1 KiB sine scratch array. Keep history within each inference call and preserve global causal zero padding.

### 2. Fuse the two additions in each residual block

After every residual pointwise product, combine bias and skip additions in one SIMD pass, preserving:

```text
output = skip + (complete_matrix_product + bias)
```

This removes another **167.1168 MB/s of logical traffic** across the 18 blocks without replacing MLAS. It must not initialize a GEMM accumulator with bias or skip, because that changes the floating-point addition order.

A later, genuine GEMM epilogue can also remove the matrix-product intermediate, reaching approximately **334.2336 MB/s** of eliminated traffic across these boundaries. That larger figure includes the two-Add saving; do not add both together.

Some outputs feed both the next activation and the next skip connection. Graph matching must preserve both consumers. A single-output Add+Snake fusion cannot discard the residual that the next block still needs. [oneDNN ordered post-ops](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_post_ops.html) and [LIBXSMM TPP](https://libxsmm.readthedocs.io/en/latest/libxsmm_tpp/) provide implementation building blocks, not verified speedups for these shapes.

### 3. Keep an entire late residual stack in cache

Start with stage 3, then stages 4-6. Run all three residual units on a time slab with all channels, reusing buffers for skip, preactivation, depthwise output and matrix output. Publish the stack output instead of every intermediate.

The three units require 6, 18 and 54 positions of history. Independently recomputing overlap for every small slab adds work: a naive 256-position slab adds approximately 16.4% pointwise arithmetic. Prefer a long disjoint output segment per worker, initializing its boundary once and carrying histories between local slabs. Negative global time must remain correctly zero-padded at each layer.

A representative C=256, Q=128 layout needs roughly 462 KiB for three tiles and history, before weights and GEMM scratch. Tile choice therefore depends on all live arrays, not just one multiplication. Keep complete K reductions, one owner of parallelism, bounded scratch and safe concurrent calls.

The best matching research is [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755): schedule matrix and elementwise work together, using registers where possible and reusable tiles where GEMM or transcendental functions require more space. Its reported gains do not transfer directly to this decoder.

### 4. Optimize the actual matrix shapes inside that pipeline

The following targets cover the important Intel 6.8-second geometry. Each multiplication is `W[M,K] @ X[1,K,T]`:

| Target | Count | M | K | T | Intel share |
|---|---:|---:|---:|---:|---:|
| Stage 1 upsampling | 2 | 8,192 | 2,048 | 170 | 7.43% |
| Stage 1 residual | 3 | 1,024 | 1,024 | 1,360 | 4.14% |
| Stage 2 upsampling | 2 | 3,072 | 1,024 | 1,360 | 9.27% |
| Stage 2 residual | 3 | 512 | 512 | 8,160 | 4.92% |
| Stage 3 upsampling | 2 | 1,280 | 512 | 8,160 | 8.01% |
| Stage 3 residual plus stage 4 upsampling | 5 | 256 | 256 | 40,800 | 10.11% |
| Stage 4 residual plus stage 5 upsampling | 5 | 128 | 128 | 81,600 | 6.24% |
| Stage 5 residual plus stage 6 upsampling | 5 | 64 | 64 | 163,200 | 4.29% |
| Stage 6 residual | 3 | 32 | 32 | 326,400 | 2.10% |

The first two matrices contain 128 MiB of weights. They need a large-K policy; narrow K=32/64/128 matrices need efficient tiling and epilogues. A single universal replacement is unlikely to win everywhere.

An alternative is to retain `[time, channel]` layout through a complete stage, making immutable weights the right-hand GEMM operand and enabling persistent packing. Converting around every individual pointwise operation was already explored. The new proposal must carry the layout through Snake, depthwise, skip and upsampling consumers. Measure boundary conversion costs. [ORT MatMul packing implementation](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/cpu/math/matmul.cc)

### 5. Consume upsampling projections as tiles

Compute bounded current and previous projection tiles, preserve their separate complete reductions, then evaluate `(current + previous[t-1]) + bias`. Feed chronological output directly into the next stage's local buffer. Retain the previous projection at tile boundaries.

This removes full projection tensors; simply concatenating their weights does not. It changes scheduling and storage, not trained projection arithmetic.

## The Intel implementation path

The two-vCPU Xeon Platinum 8280 VM exposes AVX512F/DQ/BW/VL as well as AVX2. Intel documents two AVX512 FMA units per physical core. Our custom activations and depthwise kernels currently stop at AVX2; an AVX512 variant would handle sixteen FP32 values per vector instead of eight. The guest still has only two visible CPUs. [Intel processor specification](https://www.intel.com/content/www/us/en/products/sku/192478/intel-xeon-platinum-8280-processor-38-5m-cache-2-70-ghz/specifications.html)

| Work | Current implementation | Specific next candidate |
|---|---|---|
| Snake and DW7/Snake | AVX2 with external SLEEF u10 calls and local scratch passes | Guarded AVX512 u10 path; inline or LTO sine where assembly confirms fewer spills; preserve surrounding arithmetic order |
| Depthwise coefficients | Exact benchmark binary broadcasts seven weights plus bias for every eight outputs | Hoist constants outside the vector loop through an internal validated no-alias worker |
| Residual additions | Separate full-tensor bias and skip passes | One ordered SIMD pass, then a true matrix epilogue |
| Large matrix products | ORT MLAS; selective oneMKL prototype measured separately | Keep MKL as a control; evaluate oneDNN or LIBXSMM tile kernels with packed weights and ordered finishing |
| Late residual stacks | Full tensors between graph operators | Native stage pipeline with local histories and reusable tiles |

The matrix baseline already has advanced SIMD: [ORT 1.29's dispatch](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/mlas/lib/platform.cpp#L508) selects an AVX512 SGEMM kernel when CPU and OS state permit. This is a source capability, not an instruction trace proving every benchmarked call's selected kernel. Replacing MLAS is not a transition from scalar matrices to SIMD.

[SLEEF provides `Sleef_sinf16_u10avx512f`](https://github.com/shibatch/sleef/blob/3.9.0/docs/x86.xhtml), so a new approximation is unnecessary. Retain the same accuracy class and verify mixed-magnitude inputs because wider vectors can change argument-reduction decisions. Inlining can reduce external-call spills, but increased register pressure can reintroduce them. Keep AVX2 and SSE2 fallbacks and verify required CPUID features plus OS-enabled vector state before dispatch. Choose the winning implementation by validated CPU/shape policy, not by the Intel vendor string alone.

For matrix fusion, [oneDNN's FP32 BRGEMM API](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html) exposes packing requirements and post-operations. Use a CPU engine, FP32 inputs/accumulation/output, strict math, and one scheduler. This tests a different integration from ordinary oneMKL SGEMM. Strict FP32 prevents implicit lower-precision conversion but does not guarantee identical reduction ordering.

Intel SVML with explicit high accuracy and oneMKL vector math HA are secondary sine candidates if the SLEEF path remains costly. Their accuracy and whole-decoder speed must be compared; neither is an established improvement. AVX512 can also change frequency on this processor family, so sixteen lanes do not guarantee twice the throughput. [Intel vectorization guidance](https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-simd-vectorization-when-targeting-intel-xeon-processor-scalable-family.html)

As a scale check, halving the entire 29.55% Snake/DW share would reduce that instrumented profile's total time by only 14.78%, with all other costs unchanged. That is hypothetical arithmetic, not a predicted RTF. Closing the Intel gap requires matrix and dataflow improvements as well as SIMD.

## What is actually new

The existing decoder already has direct SIMD seven-tap depthwise convolution, vector sine and fused depthwise/Snake. It no longer performs the historical seven separate tensor multiplications.

Previously tested ordinary GEMM substitutions, combined phase matrices, per-layer layout conversions, larger sine tiles and isolated LIBXSMM/DLP matrices produced modest gains or regressions. The new work removes **whole producer-consumer boundaries** and carries layout/history through dependent operations. It should reuse existing matrix candidates as tile engines, not repeat their isolated screens as though they were new discoveries.

The audited Intel binary also repeatedly broadcasts seven depthwise weights and bias inside each vector iteration. Hoisting them in a validated no-alias worker is a concrete smaller opportunity. A naive register-Snake loop still spills around an external sine call; inlining/LTO and assembly inspection are required before claiming those accesses disappeared. Intel ISA selection needs its own guarded implementation review.

## Acceptance

Preserve weights, FP32 arithmetic, sine accuracy, input/output rates, causal direction and existing tolerances. Check changed intermediate boundaries, all fixed full clips, tiny/odd lengths, all dilations, future-prefix invariance, changing-length/repeated calls and concurrent sessions. Then compare paired complete-decoder timings with unchanged CPU threads, affinity and inputs, including new conversions and per-call scratch work.

Profile kernel arithmetic, packing, history, epilogues and scheduling separately; hardware counters are needed before declaring a stage memory-bound. Keep Apple's already-winning route unless a candidate passes its own checks. No existing paper or measured library substitution establishes the additional 29% AMD or 50% Intel reduction. This plan identifies concrete experiments with enough coverage to test that goal while preserving the trained network.
