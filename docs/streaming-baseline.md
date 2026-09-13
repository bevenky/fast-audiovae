# Apple causal streaming baseline

Version 0.3.0 selects the measured Apple streaming recipe automatically when the OS reports SME and SME2 support. Its streaming graph is exactly `5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841`. This is the implementation previously measured at 0.07121946 RTF on a short 2.04-second prefix. That historical number is not a performance guarantee; the README uses the fresh matched measurements described below.

AudioVAE2 produces 48 kHz audio from 64-dimensional latents at 25 Hz. Pocket continuous Mimi produces 24 kHz audio from 32-dimensional latents at 12.5 Hz. Both retain decoder history and process only new latent frames. Encoding is performed beforehand and is outside the timing. This qualifies streaming decoding, not an end-to-end streaming encoder API.

## Timing

The current comparison uses Apple M5 Max, macOS 26.5.1 and ONNX Runtime 1.30.0 for all three decoders. The selected implementation is loaded from an installed platform wheel. Stock AudioVAE2 uses the pinned original ONNX export with explicit causal state and no native graph rewrites. Every AudioVAE2 variant receives the same frozen latents. Mimi uses its own encoder latents for the same recordings.

The three frozen FLEURS recordings are `bn_in_00151_1818`, `en_us_00103_1779` and `es_419_00060_1994`. Every codec and packet size gets two warmups and five measured repetitions. Stock and selected variants are paired, with order randomized inside each repetition. All numerical checks must pass before a timing enters the table.

RTF is the sum of decode calls and flush time divided by the duration actually returned, including terminal latent padding. The headline is the equal-weight mean of three per-recording median RTFs. Loading, stream creation, encoding, resampling, input preparation, output copies and external quality checks are excluded. Calls are warmed and unpaced. Whole-loop times are also retained in the detailed report.

The one-thread run showed substantial within-run timing drift across selected, stock and Mimi. For example, selected 80 ms repetitions on the Spanish recording ranged from about 0.0663 to 0.1304 RTF. All repetitions remain in the report; the headline does not select the fastest tail. These observations do not identify a thermal or scheduling cause. Because the four-thread run happened later, comparing the tables does not isolate the effect of thread count.

The thread settings are one and four ORT intra-op threads, with one inter-op thread and sequential graph execution. Nested BLAS workers are limited to one. Selected custom operators retain their declared one-worker implementation; their Accelerate calls enforce the threading policy on the calling thread. Four ORT threads do not mean four workers inside every custom operation. CPU affinity is not pinned on macOS. All inference uses CPUExecutionProvider, with no GPU execution.

AudioVAE2 is measured at 40 ms and 80 ms packets. Mimi's native latent frame is 80 ms, so no 40 ms Mimi measurement is reported. Splitting an already-produced 80 ms waveform into transport packets would not establish a 40 ms decoder.

The [current results](../benchmarks/streaming/apple-selected-20260913.json) include artifact identities, per-recording medians, state checks and an additional short comparison of the packaged implementation with the historical selected bundle. The [earlier ORT 1.29 baseline](../benchmarks/streaming/apple-one-thread-20260913.json) remains available separately.

## Preservation of the selected baseline

A separate contemporaneous check compared the saved selected bundle with the installed release on the same three recordings, at 80 ms and one thread, with two warmups and three measured pairs per recording. Both used the installed runtime and the identical selected graph. The saved bundle measured 0.06665 RTF and the packaged bundle 0.06631. Across all nine measured pairs, differences stayed between a 1.20% slowdown and a 1.58% speedup; all paired waveform hashes were identical. This check found no meaningful packaging regression. It does not replace the primary stock/Mimi comparison or prove a new kernel speedup.

## Numerical and state qualification

The broad corpus contains 78 inputs: 60 multilingual speech recordings, ten natural expressive or nonverbal recordings, two longer natural speech recordings, four encoded silence/quiet/transition controls, and constructed 60- and 120-second latent continuity sequences. Constructed sequences are state stress tests, not additional natural recordings. The silence controls were encoded from actual zero-valued audio rather than assuming zero latents represent silence.

At each thread setting, stock and selected decoders were checked at both packet sizes. All 624 streams, 128,292 chunks and 328,243,200 emitted samples passed. Maximum absolute streamed sample difference from stock continuous decoding was 4.0009618e-6. The elementwise gate was `atol=1e-5, rtol=1e-4`; no alignment, gain fitting, clipping or silence removal was applied.

Each thread setting also passed 72 state-behavior checks and six direct history checks. These cover reset, uneven packets, independent interleaving, future-latent perturbation, empty input, flush and invalid input. The 26 history tensors retain 683,264 bytes per stream. Startup and terminal samples are included in the waveform checks, and every complete latent frame must emit exactly 1,920 samples.

One additional natural instrumental recording passed all eight stock/selected, packet-size and thread combinations, with maximum error 2.0563602e-6. This is the complete 5.333-second [solo trumpet 06](https://freesound.org/people/sorohanro/sounds/77711/) by Mihai Sorohan, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), distributed through librosa's example-data repository. It was downmixed and resampled to the encoder's native 16 kHz input. This provides an instrumental numerical check, not a broad music listening evaluation.

The broad one-thread run used candidate r2. Candidate r3 added only a Python lock around selected-session calls to protect reused matrix scratch across simultaneous streams. Graphs, weights and native library bytes stayed identical. The final r3 package additionally passed a one-thread speech/expressive/silence subset, real simultaneous-stream checks at both thread settings, the full four-thread corpus, and both instrumental checks. Fresh timing uses r3. All accepted qualification processes exited cleanly, and before/after artifact hashes matched.

The historical base library was rebuilt from source with two additional operators unused by the selected graph. Existing arithmetic, tile size 256, compiler, SDK, headers and flags were unchanged. Its new binary identity is recorded rather than being presented as the original file.

The final source suite passed 277 tests and 1,959 subtests. Package initialization disables ORT telemetry before importing the runtime, avoiding an observed ORT 1.30 telemetry shutdown failure. An application that imports ORT first must set `ORT_DISABLE_TELEMETRY=1` before that import to prevent the uploader from starting.

## Quality

The README's perceptual quality table uses the earlier one-thread, 80 ms streaming speech panel on 60 FLEURS recordings across ten languages, totaling 534.58 seconds. PESQ and STOI compare each codec with the same original audio; UTMOS and DNSMOS are reference-free predictors. PESQ, STOI and UTMOS remain the earlier results. DNSMOS was subsequently rechecked on the saved waveforms as described below. No audio was re-encoded for this recheck, and it is not a fresh perceptual evaluation of the current kernels. The release is qualified separately for numerical agreement with stock. The trained values remain unchanged.

Only terminal padding was removed before scoring. There was no fitted time alignment, gain normalization, clipping or silence removal. The source bandwidth and metric sample rate were 16 kHz. These scores do not assess fidelity above 8 kHz or establish music quality. UTMOS22 and DNSMOS are predictions, not human ratings. [Quality scores and scorer provenance](../benchmarks/streaming/apple-quality-20260913.json).

The earlier panel had no missing scores or scorer warnings, and its 29 metric means were independently recomputed, including ESTOI and the additional DNSMOS components in the detailed report. UTMOS22 uses the pinned single strong learner. DNSMOS uses the non-personalized P.835 and P.808 models.

### DNSMOS revalidation

The earlier README's "DNSMOS overall" column meant P.835 overall. Those values were correct; the table now names P.835 explicitly and includes P.808, which ranks these codecs differently.

| Audio | DNSMOS P.835 overall | DNSMOS P.808 |
| --- | ---: | ---: |
| Original recordings | 2.775 | 3.428 |
| AudioVAE2 | 2.765 | 3.404 |
| Pocket Mimi | 2.894 | 3.339 |

The exact class from [Microsoft's pinned DNSMOS implementation](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py#L25-L112) was replayed on all 240 saved 16 kHz waveforms: original, stock AudioVAE2, optimized AudioVAE2 and Mimi for each recording. CPU-only ONNX Runtime 1.29.0 sessions used one thread. All 720 calibrated P.835 signal, background and overall per-recording values matched the saved scores exactly, with no warnings. The largest P.808 per-recording difference was 0.000000572, leaving every displayed mean unchanged. Source/output identities and pairing were checked before scoring.

The replay retained the shared float32, 16 kHz metric inputs and official windowing and calibration. It did not substitute the upstream file loader's float64/librosa resampling path. Among the two codecs, Mimi leads P.835 overall on 44 of 60 recordings, while AudioVAE2 leads P.808 on 39 of 60. This is a difference between predictors, not evidence of a scoring reversal or a human listening preference.

Meta DACVAE is excluded because the tested checkpoint requires future latent frames. Buffered decoding with lookahead does not qualify for this zero-lookahead decoder comparison. No new Intel or AMD speed measurement is claimed in this release; their existing native payload is preserved.
