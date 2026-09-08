# Intel precision results

Selective INT8 reduced decoder time by **33.9%** against the matched optimized FP32 decoder. It achieved 4.01x stock throughput and an aggregate RTF 3.5% below Mimi in this run. It was faster than optimized FP32 on all ten timed clip means, and faster than Mimi on eight. It remains an experiment; the installed runtime defaults are unchanged.

| Decoder | RTF, lower is better | Listening pilot /100 | PESQ WB | STOI | UTMOS22 | DNSMOS overall |
|---|---:|---:|---:|---:|---:|---:|
| Stock FP32 | 0.657688 | 90.5 | 3.742 | 0.9360 | 2.257 | 2.765 |
| Optimized FP32 | 0.248212 | 90.5 | 3.742 | 0.9360 | 2.257 | 2.765 |
| Selective INT8 | **0.164109** | **90.5** | 3.717 | 0.9344 | 2.252 | 2.760 |
| Broad INT8 | 0.165647 | 85.375 | 3.694 | 0.9334 | 2.246 | 2.752 |
| Mimi | 0.170021 | Not rated | Not rescored | Not rescored | Not rescored | Not rescored |

Timing used an Intel Xeon Platinum 8280 VM, two vCPUs with affinity 0/1, CPU ONNX Runtime 1.29.0 and sequential oneMKL 2026.1.0. Ten fixed clips each received two warmups and five timed decoder calls. RTF excludes encoding, TTS generation, loading and warmup. No GPU was used. [Timing and per-clip results](../benchmarks/intel-precision/summary.json).

## What changes

Selected dense matrices use INT8 operands, complete-K INT32 accumulation and FP32 dequantization. Weight scales are per output row. Activation scales are calculated across channels for each time column, so future samples do not affect earlier scaling. Snake, depthwise convolution, residual arithmetic, normalization and waveform output remain FP32. There is no retraining and initializer bytes are unchanged, but inference is approximate.

Selective INT8 keeps small matrices in FP32. Broad INT8 quantizes those too, provided no speed advantage here and received a much lower Bengali listening rating. It is retained as an experimental control, not the recommended candidate. The separate FP16-operand control is excluded from the final campaign and makes no native FP16 acceleration claim.

The [source package](../experiments/intel-precision/README.md) includes native and fused sources, dependency pins and explicit build instructions. Its tested Intel path requires AVX512-VNNI. These Intel results do not establish performance on another CPU.

## Quality and validation

Automated quality covers 60 FLEURS clips across ten languages, scored in the common 16 kHz source bandwidth. The native output is 48 kHz. Selective INT8 changed PESQ by -0.02418, STOI by -0.001538 and UTMOS by -0.005295 on average; their paired bootstrap intervals stayed below zero. DNSMOS overall changed by -0.005009, with an interval including zero. Additional ESTOI, signal/background DNSMOS, P.808, spectral, SI-SDR and SNR results are in the [quality record](../benchmarks/intel-precision/quality-results.json).

The native 48 kHz listening pilot had one listener and eight rated clips. Selective INT8 and both FP32 versions averaged 90.5/100. This is not proof of equal quality or a completed standardized MUSHRA panel. English and Tamil were too quiet to rate. All eight completed clips were retained, including the high Telugu anchor rating. Every score below 20 belonged to the deliberately degraded low-pass control. Broad INT8's Bengali score of 50 was a separate candidate rating, versus 97 for optimized FP32. [All ratings and limitations](../benchmarks/intel-precision/human-ratings.json).

All 340 validation gate rows passed: 300 full-waveform rows and 40 boundary-fixture rows. Recovery revalidated 227 saved waveforms and generated 73 fresh full waveforms. All 250 timed calls were fresh and matched their validated waveform hashes. FP32 used the existing numerical tolerance. INT8 was checked for shape, finite values, unchanged input, exact repeated output and future-input invariance, with the existing short/long prefix checks. INT8 was not required to equal FP32; its quality changes are reported above.

## Later experiments

Seven subsequent complete-decoder screens tested reusable INT8 scratch, ordered residual fusion, larger tiles, direct VNNI, exact inline SLEEF sine and DW/post-Snake fusion. Improvements ranged from 0.68% to 4.54%; the original broad VNNI combination was 4.82% slower. None met the required 10% improvement. No candidate advanced to the full 60-clip validation and ten-clip timing gate.

Their [source archive](../experiments/intel-precision/archive/iteration3/README.md) and [paired results](../benchmarks/intel-precision/iteration3/results.md) preserve the rejected work separately. Each screen must be compared with its own baseline; their percentages cannot be added together.
