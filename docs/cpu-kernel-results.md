# CPU kernel experiment results

**Decoding time is now 16.9% lower on AMD and 26.4% lower on Intel than the previous fast decoder.** Each full campaign passes all 331 validation checks, and every timed clip mean improves. Matching Mimi still requires another **15.2% reduction on AMD** and **32.7% on Intel**.

## Decoder speed

Lower RTF is better. Compare models within each row; the machines have different CPU budgets.

| CPU and campaign | Previous fast | AVX512 + fusions | Stage candidate | Mimi |
|---|---:|---:|---:|---:|
| AMD EPYC 9654, 4 threads, latest full campaign | 0.07342 | 0.06501 | **0.06104** | 0.05173 |
| Intel Xeon Platinum 8280 VM, 2 threads, latest full campaign | 0.34654 | 0.29286 | **0.25510** | 0.17169 |

On AMD, stage reuse saves another **6.1%** over the fused AVX512 control and is **4.25× as fast as stock AudioVAE2** (0.25961 RTF). Clip improvements range from 15.7% to 18.2%. All repetitions are included, including the slower fifth. Counters showed no cgroup throttling or CPU steal.

Intel's stage/MKL candidate saves **12.9%** over its same-campaign fused control. Clip improvements versus previous fast range from 25.7% to 27.5%; the repeat RTF coefficient of variation is 0.32%. Historical screens remain separate in the evidence file.

Apple's full run passed correctness but showed unexplained timing drift across all models. Its timing is excluded from promotion, and its default remains unchanged. A separate Apple stage prototype passes 270 focused and 12 captured intermediate-output comparisons; this establishes neither full-decoder waveform quality nor speed for that prototype.

## What we implemented

- **Snake → depthwise → Snake fusion** removes full-tensor intermediates using bounded tiles and local causal history.
- **Ordered bias/residual fusion** preserves `skip + (product + bias)` and removes an intermediate tensor.
- **Guarded AVX512 kernels** process sixteen FP32 values per vector, with CPU and operating-system capability checks. Existing fallback paths remain available.
- **Three-unit stage reuse** carries a tile through three complete residual units, including pointwise products. Four reusable buffers and per-unit history reduce intermediate traffic. Internal time segments replay and discard the required 78-position causal prefix.

The stage pipeline targets memory movement and matrix products together, beyond changing a standalone BLAS call. It still computes the full channel dot products. This follows the approach described by [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755); [LIBXSMM TPP](https://libxsmm.readthedocs.io/en/latest/libxsmm_tpp/) supplies one implementation option. Published speedups are not predictions for this decoder.

AMD uses direct products in the final three stages, with four time segments. Intel uses two segments: 256 channels with 256-position tiles and LIBXSMM products; 128 channels with 128-position tiles and direct AVX512 products; 64/32 channels with 256-position tiles and direct AVX512 products. Intel's combined candidate adds twelve earlier MKL operations. These remain explicit experimental settings, not a universal hardware policy or silently promoted default.

## Remaining bottleneck

Separate profiles measure the same 6.8-second output, with three calls after two warmups. These are summed kernel wall times per call, not headline RTF. Each column describes that machine's latest candidate.

| Remaining work | AMD time / share | Intel time / share |
|---|---:|---:|
| Ordinary matrix products | **264.09 ms / 64.5%** | 198.35 ms / 11.7% |
| MKL matrix products | Not used | **593.21 ms / 34.9%** |
| Fused stages, including their matrix products | 81.21 ms / 19.8% | **680.01 ms / 40.0%** |
| Remaining Snake/depthwise operators | 30.07 ms / 7.3% | 101.50 ms / 6.0% |
| Remaining bias/residual additions | 14.17 ms / 3.5% | 38.95 ms / 2.3% |
| Upsampling phase merge | 11.09 ms / 2.7% | 43.44 ms / 2.6% |
| Other | 8.52 ms / 2.1% | 46.65 ms / 2.7% |

Compared with the fused controls, summed kernel time falls from 432.61 to 409.16 ms on AMD and 2,007.78 to 1,702.11 ms on Intel. Matrix work inside a stage is opaque to this profile; reduced ordinary MatMul time does not mean those dot products disappeared.

AMD's largest remaining matrix group is the 256-channel residual and upsampling products (**64.42 ms**), followed by earlier upsampling products (**49.02 ms** for the 512-input-channel group). On Intel, the 256-channel fused stage takes **255.59 ms** and the third upsampling projection pair takes **176.81 ms** inside MKL.

A bounded next Intel experiment would combine the fourth upsampling projection pair (**97.70 ms**), phase merge and 128-channel residual stage (**184.12 ms**) into a tiled region. This targets intermediate traffic while retaining the required matrix arithmetic. Its gain is unmeasured; neither this proposal nor isolated stage gains guarantee Mimi parity.

An isolated AMD 256-channel stage improved from 59.43 to 54.22 ms with LIBXSMM. This addition remains unadopted: its complete-decoder quality and RTF have not been measured. See the [next-step analysis](next-kernel-region.md).

## Quality and measurement limits

| Check set | Result |
|---|---|
| AMD latest full decoder campaign, 60 clips | 331 checks passed, none failed |
| Intel latest full decoder campaign, 60 clips | 331 checks passed, none failed |
| Apple full decoder campaign, 60 clips | 331 checks passed, none failed |
| Unchanged Apple encoder, 1 and 4 threads | 94 checks passed, none failed |

These are separate test sets. Full decoder checks cover output length, finite waveforms, dynamic lengths, repeat behavior and applicable causal prefixes against saved originals, with `atol=1e-5, rtol=1e-4`; AudioVAE2 also matches stock ONNX within the gate. Learned weights are unchanged, but matrix reduction rounding can differ. This demonstrates numerical agreement under the checks, not newly measured perceptual scores. UTMOS, DNSMOS and STOI were not rerun. No new encoder optimization is claimed.

All decoding used **ONNX Runtime 1.29.0, CPUExecutionProvider, FP32 and batch one**. Full campaigns validate 60 clips across ten languages and time ten clips with five repetitions after two warmups. RTF covers complete decoder calls divided by generated duration, excluding encoding, setup, warmup and profiling. It is not cached streaming RTF. AudioVAE2 retains 16 kHz input → 48 kHz output; the continuous Mimi decoder outputs 24 kHz and is also FP32. The additional memory cost of Intel's combined MKL candidate has not been measured.

The [evidence file](../benchmarks/kernel-experiments/results.json) contains hashes, per-clip changes, separate historical screens, native checks, stage measurements and CPU counters. Incomplete campaigns are excluded from performance comparisons.
