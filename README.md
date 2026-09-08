# fast-audiovae

Fast CPU inference for VoxCPM2's AudioVAE2 decoder, using ONNX graph rewrites and native kernels. Latents `[1, 64, L]` at 25 Hz produce mono audio `[1, 1, 1920*L]` at 48 kHz. Supports full-clip and stateful streaming decoding.

## Run

Use Python 3.11 to 3.13. Install from the release wheels; pip picks the platform automatically:

```sh
python -m pip install fast-audiovae==0.2.0 --find-links https://github.com/bevenky/fast-audiovae/releases/expanded_assets/v0.2.0
```

With uv, use `uv pip install` with the same arguments. The first load downloads the pinned weights and prepares a local cache. Native wheels include the kernels and their CPU dependencies; no compiler or kernel flags are needed.

```python
from fast_audiovae import load

decoder = load()  # Streaming is the default.
with decoder.stream() as stream:
    # Pass only new latents: NumPy float32, shape [1, 64, L].
    audio = stream.decode_chunk(latents)
```

For a complete latent sequence, use `decoder = load(mode="batch")`, then `audio = decoder.decode(latents)`. Both return 48 kHz audio. [Streaming usage and validation](docs/streaming.md).

The loader selects the retained Apple, Intel or AMD recipe for the requested mode. Inference uses the CPU only and defaults to one thread. `decoder.info` shows the selection. Unsupported CPU or OS combinations use standard ONNX with an explicit fallback message.

Native wheels currently cover Apple ARM on macOS 26.2 or newer and compatible Intel/AMD Linux x86 systems with glibc 2.38 or newer. The loader also checks native library compatibility before using them.

The tables below report the previously accepted full-clip builds. [Streaming validation](docs/streaming.md#validation-results) covers the new API.

## Decoder speed

This table compares full-clip decoding. Lower RTF is better. RTF is total decoding time divided by generated audio duration; it excludes loading, encoding and TTS generation. Streaming chunk latency is measured separately.

| CPU | Threads | Base AudioVAE2 | Optimized AudioVAE2 | Mimi | Meta DAC-VAE, earlier run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple M5 Max | 4 | 0.09280 | **0.02470** | 0.02815 | 0.54146 |
| AMD EPYC 9654 | 4 | 0.25556 | **0.03435** | 0.05201 | 0.66065 |
| Intel Xeon Platinum 8280 VM | 2 | 0.65769 | **0.16411** | 0.17002 | 2.06885 |

CPU-only, ONNX Runtime 1.29. Base, optimized and Mimi values are matched within each row: ten clips on Apple and Intel, three on AMD, with five repetitions after warmup. DAC timings come from the earlier comparison and are shown for reference.

Measured results: [Apple](docs/apple-precision.md), [AMD](experiments/amd-precision/README.md), [Intel](experiments/intel-precision/README.md). These full-clip measurements use the thread counts shown; they are separate from the default one-thread streaming path.

## Reconstruction quality

Higher is better. These results use 60 FLEURS clips across ten languages, scored in the common 16 kHz source bandwidth.

| Codec | PESQ | STOI | UTMOS22 | DNSMOS overall |
| --- | ---: | ---: | ---: | ---: |
| Base AudioVAE2 | 3.742 | 0.9360 | 2.257 | 2.765 |
| Optimized AudioVAE2, Apple | 3.742 | 0.9360 | 2.257 | 2.765 |
| Optimized AudioVAE2, Intel and AMD | 3.717 | 0.9344 | 2.252 | 2.760 |
| Mimi | 2.130 | 0.8074 | 2.517 | 2.894 |
| Meta DAC-VAE | 4.284 | 0.9731 | 2.222 | 2.779 |

Intel and AMD were scored independently and agree at the displayed precision. UTMOS and DNSMOS are predictions, not listening-panel ratings. AudioVAE2 outputs 48 kHz and the tested continuous Mimi outputs 24 kHz; both decoders are causal. The tested 48 kHz Meta DAC-VAE is noncausal. [Methodology and detailed results](docs/multilingual.md).

## More

[Kernel experiments](docs/cpu-kernel-results.md) and [optional backends and encoder setup](docs/optional-backends.md) contain the detailed build and validation records. The encoder retains its existing 16 kHz input. The [streaming decoder](docs/streaming.md) carries independent history between chunks.

Architecture and weights originate from [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). Preparation uses the checksum-verified [pinned ONNX export](https://huggingface.co/ai4all8/VoxCPM2-ONNX/tree/ecb511b96675f041424b42f148bf72e301262586). Upstream model and dependency licenses apply.
