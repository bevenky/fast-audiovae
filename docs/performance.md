# Performance and validation

These are historical 20-clip English development measurements on Apple M5 Max and AMD EPYC 9654. The [new multilingual benchmark](multilingual.md) includes Intel, Meta DACVAE and reconstruction-quality results. They are CPU-only results, not GPU, encoder-plus-decoder or whole-TTS timings.

## Decoder measurements

RTF is decoding wall time divided by generated audio duration; lower is better. Each campaign used ONNX Runtime 1.29.0, FP32, batch 1, the same 20 source speech clips, one warmup per shape and three timed repetitions. Loading, weight packing, encoding, file I/O and separate profiling were excluded. Thread spinning was disabled.

| CPU | Threads | Stock AudioVAE2 | Native AudioVAE2 | Optional AMD packing | Pocket continuous Mimi |
|---|---:|---:|---:|---:|---:|
| Apple M5 Max | 1 | 0.12523 | 0.05592 | N/A | 0.04642 |
| Apple M5 Max | 4 | 0.12102 | 0.03015 | N/A | 0.03577 |
| AMD EPYC 9654 | 1 | 0.40432 | 0.25580 | 0.24679 | 0.12195 |
| AMD EPYC 9654 | 4 | 0.25636 | 0.07246 | 0.07084 | 0.05125 |

Exact values and per-repetition results are in [benchmarks/results.json](../benchmarks/results.json).

Stock means the original ONNX export. It is not the prepared standard-ONNX fallback selected by `prefer_custom=False`, which also includes an upsampling rewrite.

AudioVAE2 produces **48 kHz** audio from 64-channel, 25 Hz continuous latents. Mimi here is Pocket's **continuous 24 kHz** variant with 32-channel, 12.5 Hz latents, not the original RVQ codec. Both decoder computations are FP32. Each uses its own encoded latents from the same source speech. Generated durations per repetition, including padding, are 163.88 and 164.24 seconds respectively. These different output bandwidths and latent contracts prevent interpreting the table as an equal-quality codec comparison.

At four threads, native Apple decoding took 15.72% less time than Mimi. AMD with optional packing took 38.22% more time than Mimi, although it reduced AudioVAE2 time by 72.37% versus stock. Packing saved a further 2.24% versus native AMD, or 3.52% at one thread. It costs roughly 164 MiB of additional weight storage; no precision reduction or compression is involved.

The AMD runs used one physical core or four sharing an L3 group, under a 16.15-core-equivalent quota. Cgroup counters showed zero additional throttled periods or throttled time during either timing block. CPU frequency was not measured. Results describe this environment, not universal CPU ratios. The subsequent [Intel campaign](multilingual.md) validates the native decoder at two threads on a two-vCPU VM. This older table does not include it.

## What passed

The AMD campaign completed 262 acceptance checks and 600 timed calls without failures. The selected packed decoder's maximum waveform difference was `1.07e-6` against stock ONNX and `2.20e-6` against saved upstream reconstructions, with unchanged `atol=1e-5, rtol=1e-4`. Same-shape future-prefix and repeated/length-change comparisons were exact. Separate operator fixtures cover dynamic shapes, coefficient ownership, batch offsets and concurrent sessions.

The corrected encoder policy was separately checked on Apple: all 108 latent/entrypoint and 94 reconstruction gates passed. Its 40 full-clip and 36 short-input latent comparisons were bitwise identical. Keeping upstream encoder Snake and depthwise accumulation was necessary; earlier replacements failed the latent gate despite passing waveform comparisons. The helper therefore retains those original operations. Small aggregate encoder timing gains were variable across clips and are not a promised per-input speedup.

The rebuilt sources passed all 40 Apple full-decoder comparisons at one/four threads, with maximum absolute difference `1.46e-6`, and all 40 AMD packed-decoder comparisons, with maximum difference `1.07e-6`. Both AMD graphs reproduce the previously accepted files byte-for-byte. The encoder helper passed all 76 full/short latent comparisons bitwise exactly. These are fresh numerical checks, separate from the timing campaigns above. See the [validation record](../benchmarks/validation.json).

These are numerical checks on speech derived from 16 kHz input. They do not establish universal perceptual equivalence, new listening scores, fullband reconstruction quality or identical results for every possible input. Fresh causal-call tests also do not establish cached-streaming latency or 40 ms chunk RTF.

## Check your inputs

Provide an NPZ containing `z` or `clip-name__z` FP32 arrays shaped `[1,64,L]`. From the repository root, validate against the pinned original export, then time fresh decoder calls:

```sh
python tests/validate_decoder.py --bundle artifacts --source .deps/models/audio_vae_decoder.onnx --cases user-latents.npz --output benchmark-results/parity.json
python benchmarks/decoder.py --bundle artifacts --cases user-latents.npz --threads 4 --output benchmark-results/decoder-4t.json
```

Validation defaults to one/four threads and exits nonzero on failed parity. Repeat timing with `--threads 1` and a separate output filename. Use `--amd-packed` for the optional AMD route. This timing command measures AudioVAE2 only; it does not reproduce the separate Mimi comparison automatically.

## Remaining CPU work

In a separate representative 6.8-second AMD profile, matrices consumed 61.37% of selected decoder time; Snake plus fused depthwise/Snake consumed 23.11%. These are instrumented wall-time shares from one shape, not a decomposition of every timed clip. Snake is already vectorized using SLEEF, so replacing scalar sine is not the remaining opportunity.

The 31 dense products total approximately **17.67 GFLOP per generated second**. Assuming conventional dense FP32 multiplication, two 256-bit FMA paths per Zen 4 core and sustained maximum 3.7 GHz gives optimistic matrix-only RTF floors of 0.149 at one core and 0.0373 at four. Combining the four-core floor with the current non-matrix share estimates full RTF near 0.065. Frequency was not measured, and this combines a representative profile with aggregate timing. It is a conditional arithmetic estimate, not an impossibility proof for other algorithms. [AMD architecture](https://www.amd.com/en/blogs/2025/leadership-hpc-performance-with-5th-generation-amd.html), [EPYC 9654 specifications](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9654.html).

Useful next experiments combine selective matrix kernels with ordered bias/residual or causal phase epilogues to reduce intermediate writes. Relevant starting points are [oneDNN CPU MatMul](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) and [BRGEMM](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html), with explicit [strict FP32 math](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html), [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755), and [GEMMFIP](https://arxiv.org/abs/2302.08417). Required reorders and packing belong inside the measured contract.

Completed AOCL-DLP and LIBXSMM screens found modest, shape-dependent matrix gains; neither is integrated. A separate Apple SGEMM prototype improved its aggregate three-clip screen but regressed one clip, so it is also excluded. None of these experiments proves a further full-decoder gain. New kernels must preserve the existing numerical gates, handle changing lengths and concurrent sessions, and win matched decoder measurements before becoming defaults.
