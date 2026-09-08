# Streaming decoding

Streaming uses AudioVAE2's existing weights and causal history. Pass only new latent frames on each call. Each frame represents 40 ms of audio and produces 1,920 mono samples at 48 kHz.

## Prepare and load

The automatic loader prepares the right CPU recipe on first use. Streaming and one inference thread are the defaults:

```python
from fast_audiovae import load

decoder = load()
with decoder.stream() as stream:
    for latents in latent_chunks:
        audio = stream.decode_chunk(latents)
        play_or_send(audio)
```

No kernel flags or compiler are needed with a supported platform wheel. Use `load(mode="batch")` for full-sequence calls. `decoder.info` reports the selected recipe and any fallback. To prepare the cache ahead of time, run `fast-audiovae setup`; `--mode both` also prepares batch mode. Set `FAST_AUDIOVAE_CACHE` to choose a different cache location.

### Existing model bundles

The original explicit bundle API remains available. Prepare streaming alongside a manually prepared bundle:

```sh
fast-audiovae prepare --output artifacts --streaming
```

Or add it to an existing bundle:

```sh
fast-audiovae prepare-streaming artifacts
```

Preparation keeps the full-clip graphs and adds graphs with explicit history inputs and outputs. Prepare a fresh bundle to replace existing streaming artifacts.

```python
from fast_audiovae import load_streaming_decoder

# Use one inference thread.
decoder, selected = load_streaming_decoder("artifacts", threads=1)

with decoder.streaming_decode() as stream:
    for latents in latent_chunks:
        # NumPy float32 [1, 64, L], containing only new frames.
        audio = stream.decode_chunk(latents)
        # audio is NumPy float32 [1, 1, 1920 * L] at 48 kHz.
        play_or_send(audio)
```

`latent_chunks` and `play_or_send` are supplied by your application. One-frame chunks emit 40 ms of audio; five-frame chunks emit 200 ms. Chunk sizes can change within a stream, including a shorter final chunk.

The loader uses the same CPU selection policy as full-clip decoding. Inspect the selected backend with:

```sh
fast-audiovae inspect artifacts --streaming
```

## Existing INT8 recipes

The automatic loader handles the retained Intel and AMD recipes. When managing an existing INT8 bundle manually, use this preparation option:

```sh
fast-audiovae prepare-streaming artifacts --canonical-precision
```

First rebuild and package both the base native library and the fused precision libraries from the matching recipe. This option does not compile them. Preparation requires explicit backend 5 on every native Snake or fused stage, and rejects unconverted standard sine operations. The loader checks the required CPU backend and native math version before selecting the precision path.

The option retains the old graph files, but selects a new full-clip reference and its streaming graph. Two arithmetic details are made consistent across chunk sizes:

- A one-frame input to the initial 64-to-2048 projection is padded to two columns, avoiding a different matrix reduction. The dummy column is removed before activation or history updates. Inputs with two or more frames keep their original width.
- Partial AVX512 Snake blocks use the same vector sine calculation as complete blocks, avoiding chunk-dependent rounding in the tail.

These changes keep the trained weights. They can change rounding relative to the previously accepted INT8 decoder. The reconstruction comparison below checks the new reference against that accepted version.

## State and lifecycle

Each `streaming_decode()` call creates independent history. Streams can share a decoder and its immutable weights. Calls within one stream are serialized; send chunks in their intended audio order.

- `reset()` clears history for a new utterance without reloading weights.
- `flush()` returns an empty waveform. Complete latent frames already emit all their samples; there is no delayed tail to append.
- An empty chunk returns empty audio and leaves history unchanged.
- Leaving the context closes the stream. Create another stream for a new context.
- Failed input validation or inference leaves the previous history intact.

The base native and portable graphs use 26 history buffers, totaling 667.25 KiB per stream. This storage stays fixed as an utterance grows; it excludes returned audio and temporary inference buffers. Different fused graphs may combine histories into fewer tensors.

Streaming does not replay the complete utterance. Each causal layer retains its required context. The existing native depthwise and phase kernels process a bounded local context; fused streaming stages accept and return their history directly.

## Quality and validation

The base path retains the trained weights and operations. Chunk boundaries must agree with continuous decoding and the pinned upstream AudioVAE2 reference within `atol=1e-5`, `rtol=1e-4`, including samples around each boundary. Floating-point kernels can use different reduction orders for different chunk sizes, so bitwise identity is a stronger condition than this numerical gate.

The base floating-point path is checked directly against upstream AudioVAE2. INT8 already approximates that model. Its new canonical full-clip reference must be compared with the previously accepted INT8 output for quality, while streaming must independently pass the same strict numerical tolerance against that new reference. Passing streaming parity establishes no additional error beyond the tolerance; it does not remove the existing quantization approximation.

## Validation results

All three CPUs passed 540 streaming checks across the same 60 multilingual clips: one-, two- and five-frame chunks, uneven chunks, reset, final partial chunks and independent interleaved streams. Intel and AMD streaming outputs are bitwise identical to their respective canonical full decoders. Apple passed the strict numerical gate with maximum absolute difference 3.13e-6 and worst nonexact SNR 102.2 dB. Apple also passed 60 stored upstream comparisons and an additional 15-check run that included direct upstream streaming comparisons on one clip.

### Streaming projection update

The [streaming projection recipes](../experiments/streaming-matrix/README.md) improve the first upsampling pair while retaining the full-call graphs and causal history. They use explicitly prepared bundles and one inference thread.

| CPU | Chunk | Previous optimized RTF | New RTF | Less decoding time |
| --- | ---: | ---: | ---: | ---: |
| Apple M5 Max | 80 ms | 0.24715 | 0.14975 | 39.4% |
| Apple M5 Max | 160 ms | 0.13772 | 0.09077 | 34.1% |
| Intel Xeon Platinum 8280 VM | 80 ms | 0.37609 | 0.31152 | 17.2% |
| Intel Xeon Platinum 8280 VM | 160 ms | 0.27842 | 0.24781 | 11.0% |

These are new paired measurements on three frozen multilingual clips, with two warmups and five measured repetitions. Every clip improved at both chunk sizes. Each variant passed 126 timed waveform checks, 72 state checks and 180 complete streams across all 60 multilingual clips at 40, 80 and 160 ms. Intel remained bitwise identical to its accepted full decoder. Apple passed the existing strict tolerances against both accepted and stored upstream outputs, with maximum difference 3.93e-6 against the accepted decoder. [Results and provenance](../benchmarks/streaming/projection.json). Mimi was not rerun in this experiment, so use the earlier comparison below as historical context rather than a freshly matched speed ratio.

The full-call graph is byte-identical between the paired variants. Its control timings varied by 4.0% on Apple and 0.2% on Intel; no full-call improvement is claimed. The automatic loader selects these projection recipes for one-thread streaming on their respective CPUs.

### Earlier matched one-thread comparison

CPU-only ONNX Runtime 1.29.0, with one inference thread for all three decoders. Mimi's native frame is 80 ms, so the comparison uses 80 and 160 ms chunks. AudioVAE2 outputs 48 kHz; Mimi outputs 24 kHz. Lower RTF is better.

| CPU | Chunk | Stock AudioVAE2 | Optimized AudioVAE2 | Mimi |
| --- | ---: | ---: | ---: | ---: |
| Apple M5 Max | 80 ms | 0.30141 | 0.22213 | 0.05471 |
| Apple M5 Max | 160 ms | 0.18235 | 0.12176 | 0.03876 |
| AMD EPYC 9654 | 80 ms | 0.60055 | 0.18048 | 0.12297 |
| AMD EPYC 9654 | 160 ms | 0.47791 | 0.14214 | 0.11398 |
| Intel Xeon Platinum 8280 VM | 80 ms | 1.00410 | 0.37679 | 0.22228 |
| Intel Xeon Platinum 8280 VM | 160 ms | 0.80492 | 0.27850 | 0.18446 |

The table averages per-clip medians across three frozen Bengali, English and Spanish FLEURS clips, with two warmups and three randomized measured repetitions each. RTF includes every decode call and flush, divided by actual returned audio duration. Model loading, stream creation, input-chunk preparation and encoding are excluded. This is warmed, unpaced inference.

All 405 runs passed waveform and sample-count checks; independent audits checked 25,830 raw calls. Intel and AMD optimized streaming outputs are bitwise identical to their corresponding full outputs. Other paths passed `atol=1e-5`, `rtol=1e-4`, with maximum absolute error below 7.25e-7. These checks establish streaming parity, not equal quality between codecs.

The optimized native kernels were active. Intel and AMD used one-worker graphs with unchanged weights. Automatic preparation now reproduces those schedules. These results must not be mixed with the earlier two- and four-thread measurements below. [Results and provenance](../benchmarks/streaming/one-thread.json).

### Earlier multi-thread validation

CPU-only ONNX Runtime 1.29. The following RTFs are medians of three repetitions after two warmups on one preselected 9.96-second clip. They measure decoder calls and flush, excluding loading, encoding and TTS generation.

| CPU | Threads | 40 ms chunks | 200 ms chunks |
| --- | ---: | ---: | ---: |
| Apple M5 Max | 4 | 0.29517 | 0.08606 |
| AMD EPYC 9654 | 4 | 0.14336 | 0.05700 |
| Intel Xeon Platinum 8280 VM | 2 | 0.52166 | 0.27539 |

Full-clip regression screens used the same preselected clip and three randomized pairs. AMD median RTF changed from 0.03474 to 0.03394; Intel changed from 0.16860 to 0.17180, a 1.9% increase in time. That small Intel difference remains unresolved by this limited screen. Apple's existing full-call graph and native library are unchanged. These checks do not replace the broader full-clip README measurements or establish a streaming comparison against Mimi.

The new Intel and AMD full outputs are bitwise identical across all 60 clips. Comparing the new output with the accepted AMD output completed all 12 quality metrics with no errors. PESQ is unchanged at 3.71737024; STOI, UTMOS and DNSMOS overall remain 0.934448, 2.251605 and 2.760339 at the displayed precision. At native 48 kHz, only 91 of 25,683,840 samples changed, with maximum difference 5.96e-8. No meaningful degradation was detected on this corpus. This retains the earlier accepted quantization approximation; it is not a claim of exact agreement with the original FP32 decoder or a new human listening study.

[Validation summary and source hashes](../benchmarks/streaming/summary.json). The original streaming suite passed 183 tests and 1,944 subtests. Platform wheels now include the native dependencies; installation checks are separate from these recorded decoder measurements.

The optional encoder retains its existing 16 kHz input. This API streams decoder latents to 48 kHz audio; it does not add a stateful encoder API.

The interface follows the pinned upstream AudioVAE2 [`streaming_decode()` and `decode_chunk()`](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py). Our runtime keeps state in separate stream objects instead of patching shared model methods.
