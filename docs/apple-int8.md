# Apple CPU streaming update

Version 0.5.0 automatically selects KleidiAI SME2 DOT INT8 for two large projections in the first upsampling stage. The rest of the decoder and its public inputs, outputs and streaming states remain float32. CPU, streaming and one thread remain the defaults.

The loader requires verified SME and SME2 support. Multiple workers and batch mode retain the previous FP32 kernels. Other Apple CPUs keep their compatible path. GPU selection is unchanged and remains explicit. No compiler or extra package is needed with the native wheel.

Packets of any length use the same one- or two-frame projection calculations internally. Packet length does not switch precision. Existing streaming state, reset and flush semantics are retained.

## Quality qualification

The longer check used 79 recordings totaling 860.16 decoded seconds: 60 FLEURS recordings across ten languages, expressive sounds, quiet speech, digital silence, transitions, long speech and one short music example. It covered 40 and 80 ms packets. Music coverage is limited; this is not a broad music benchmark or a human listening test.

All structural checks passed, including sample counts, finite values, state continuity and packet-size agreement. Across 839,124 state checks, no structural failures were found. The lowest waveform cosine against the previous FP32 decoder was 0.999895; the largest relative RMSE was 1.448%. No additional full-scale overshoots occurred. Across 7,882 quiet windows, mean residual RMS was 0.00000244 and maximum residual RMS was 0.0000380. Quiet windows here are a measurement cohort, not a claim of perfect silence.

Fresh paired scores against the original audio, using all 60 speech recordings and identical scorers:

| Metric | Previous FP32 | New Apple default |
| --- | ---: | ---: |
| PESQ-WB | 3.741549 | 3.741266 |
| STOI | 0.935986 | 0.935968 |
| ESTOI | 0.884635 | 0.884604 |
| UTMOS22 | 2.256901 | 2.256665 |
| DNSMOS P.835 overall | 2.765348 | 2.765783 |
| DNSMOS P.808 | 3.403932 | 3.403022 |
| SI-SDR, dB | 12.932260 | 12.928007 |

These tiny average changes do not establish zero quality loss on every recording. The largest individual PESQ reduction was 0.00836. Scores were computed without peak normalization or clipping, after trimming only known decoder padding. PESQ, STOI, UTMOS and DNSMOS use 16 kHz scoring audio and do not establish high-frequency fidelity.

## Speed

Apple M5 Max CPU, one thread, ONNX Runtime 1.30.0, same 79 recordings:

| Packet | Previous FP32 RTF | New Apple RTF | Decoding-time reduction |
| --- | ---: | ---: | ---: |
| 40 ms | 0.169313 | 0.126282 | 25.4% |
| 80 ms | 0.114592 | 0.099576 | 13.1% |

These are confirmation timings. State hashing between calls can affect cache behavior, so they should not be mixed with older benchmark sessions. A separate balanced short 80 ms check measured about 16% less decoding time on both English and Spanish. Differences between sessions remain possible. No new multithreaded, batch, GPU or Mimi speed claim is made by this release.

The shipping library contains the pinned Apache-2.0 KleidiAI implementation, with full-K signed INT8 accumulation and float32 dequantization. Provenance and source hashes are in [the native source record](../native/apple/streaming/first_pair_int8/upstream-provenance.json). The quantization and build recipe are retained with the sources for reproducibility.

The shipping decoder was replayed on all 79 recordings at both packet sizes. All 158 waveforms were byte-exact to the scored candidate. Fresh-wheel installation also passed 26 waveform checks without compiler access or PyTorch. [Release validation](releases/v0.4.2-validation.json) and [aggregate measurements](../benchmarks/streaming/apple-int8-v1.json).
