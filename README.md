# fast-audiovae

Fast inference for VoxCPM2's AudioVAE2 decoder, with optimized CPU kernels and optional Apple GPU execution. Latents `[1, 64, L]` at 25 Hz produce mono audio `[1, 1, 1920*L]` at 48 kHz. Supports full-clip and stateful streaming decoding.

## Run

Use Python 3.11 to 3.13. Install from the release wheels; pip picks the platform automatically:

```sh
python -m pip install fast-audiovae==0.4.0 --find-links https://github.com/bevenky/fast-audiovae/releases/expanded_assets/v0.4.0
```

This installs the CPU runtime and kernels without PyTorch or GPU dependencies. The wheel includes the small GPU integration code, but its dependencies are optional.

With uv, use `uv pip install` with the same arguments. The first load downloads the pinned weights and prepares a local cache. Native wheels include the kernels and their CPU dependencies; no compiler or kernel flags are needed.

```python
from fast_audiovae import load

decoder = load()  # Streaming is the default.
with decoder.stream() as stream:
    # Pass only new latents: NumPy float32, shape [1, 64, L].
    audio = stream.decode_chunk(latents)
```

For a complete latent sequence, use `decoder = load(mode="batch")`, then `audio = decoder.decode(latents)`. Both return 48 kHz audio. [Streaming usage and validation](docs/streaming.md).

The loader selects the CPU kernels automatically. Streaming and one inference thread are the defaults. Each latent frame produces 40 ms of audio; pass two frames for 80 ms packets. Use `load(threads=4)` to request four inference threads. `decoder.info` shows what was selected.

Apple CPUs with SME/SME2 use the new streaming kernels. Other Apple CPUs retain the compatible native path; unsupported systems use portable ONNX with a fallback message. CPU is always the default device.

The source checkout includes updated AMD and Intel kernels for one-thread streaming. These updates are not in the published v0.4.0 wheels yet. Older wheels keep their existing kernels. [AMD validation](docs/amd-serving.md) and [Intel validation](docs/intel-serving.md).

Native wheels currently cover Apple ARM on macOS 26.2 or newer and compatible Intel/AMD Linux x86 systems with glibc 2.38 or newer. The loader also checks native library compatibility before using them.

For Apple GPU support, add the optional `gpu` extra:

```sh
python -m pip install 'fast-audiovae[gpu]==0.4.0' --find-links https://github.com/bevenky/fast-audiovae/releases/expanded_assets/v0.4.0
```

The extra adds PyTorch 2.14.x alongside CPU support, so both CPU and Apple GPU execution are available. CPU remains the default; GPU runs only when you select `load(device="gpu")`. GPU loading prepares the optimized 40/80 ms paths automatically; first compilation takes extra time. [GPU usage and validation](docs/apple-gpu.md).

## Decoder speed

CPU-only causal streaming on Apple M5 Max, using ONNX Runtime 1.30.0. Lower RTF is better; 1.0 is real time. All decoders carry history between calls and need no future latent frames.

| Threads | Output chunk | Base AudioVAE2 RTF | Optimized AudioVAE2 RTF | Pocket Mimi RTF |
| ---: | --- | ---: | ---: | ---: |
| 1 | 40 ms | 0.7050 | **0.2316** | n/a |
| 1 | 80 ms | 0.4706 | **0.1390** | 0.1019 |
| 4 | 40 ms | 0.2279 | **0.0941** | n/a |
| 4 | 80 ms | 0.1705 | **0.0631** | 0.0357 |

Same three multilingual recordings, two warmups and five measured repetitions per case. The one-thread run had substantial timing drift across all codecs, so these observations do not establish a thread-count speedup or reproduce the historical 0.07122 short-clip result. RTF includes decode calls and flush; loading and encoding are excluded. AudioVAE2 outputs 48 kHz; Pocket continuous Mimi outputs 24 kHz and has an 80 ms minimum frame. [Protocol and results](docs/streaming-baseline.md).

Intel Xeon Platinum 8280 VM, **one CPU thread**, ONNX Runtime 1.29.0, causal streaming:

| Output chunk | Previous optimized AudioVAE2 RTF | New matrix baseline RTF |
| --- | ---: | ---: |
| 40 ms | 0.4243 | **0.3936** |
| 80 ms | 0.2829 | **0.2723** |

Short matched check on three 1.6-second multilingual prefixes, one warmup and three measured repetitions. Every measured pair improved. Outputs and all streaming states matched the previous optimized Intel baseline exactly; this is not a claim of bit-for-bit agreement with stock FP32. The Snake experiment is excluded. [Intel protocol and validation](docs/intel-serving.md).

Independent batch decoding on Apple M5 Max, **one CPU thread**, using `load(mode="batch")`:

| Complete input length | Previous AudioVAE2 batch RTF | Updated AudioVAE2 batch RTF | Pocket Mimi batch RTF |
| --- | ---: | ---: | ---: |
| 40 ms | 0.2341 | **0.1051** | n/a |
| 80 ms | 0.2198 | **0.0743** | 0.0486 |
| 960 ms | 0.0715 | **0.0535** | 0.0343 |

Short matched check on three recordings, with two warmups and three measured repetitions. Each batch call decodes its entire input with empty history; 40/80 ms rows are tiny independent inputs, not whole-recording throughput. In this same check, streaming the 960 ms inputs in 80 ms packets gave RTF 0.0763 for AudioVAE2 and 0.0560 for Mimi. These timings come from a different session than the streaming table above. Version 0.4.0 includes the updated batch path. Streaming remains the default. [Batch validation and earlier full-clip results](docs/apple-batch-validation.md).

Apple GPU causal streaming on the same M5 Max, using PyTorch 2.14.0, float32 and one host thread:

| Decoder | Output rate | 40 ms RTF | 80 ms RTF |
| --- | ---: | ---: | ---: |
| Original AudioVAE2 GPU | 48 kHz | 0.1696 | 0.0933 |
| Optimized AudioVAE2 GPU | 48 kHz | **0.0577** | **0.0321** |
| Pocket Mimi GPU | 24 kHz | n/a | 0.0634 |

Short matched check on three 960 ms multilingual segments, with one warmup and two measured repetitions. These are back-to-back stateful streaming calls returning completed CPU-ready audio, with load and compilation excluded. At 80 ms, optimized AudioVAE2 used about half Mimi's decoding time in this check. Paced live-stream latency can differ. Waveform and history checks passed; an additional eight-second continuation passed at both packet sizes. [GPU protocol and results](experiments/apple-gpu-v6/report.md).

## Reconstruction quality

Reference scores from the earlier one-thread, 80 ms streaming panel: the same 60 FLEURS recordings across ten languages. Higher is better. All metrics use 16 kHz audio; PESQ and STOI compare against the original, while UTMOS and DNSMOS score each recording without a reference.

| Audio | PESQ-WB | STOI | UTMOS22 | DNSMOS P.835 overall | DNSMOS P.808 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original recordings | Reference | Reference | 2.321 | 2.775 | 3.428 |
| AudioVAE2 | 3.742 | 0.9360 | 2.257 | 2.765 | 3.404 |
| Pocket Mimi | 2.130 | 0.8074 | 2.517 | 2.894 | 3.339 |

DNSMOS update: the earlier "overall" column was P.835, and its values were confirmed. The table now also shows P.808. Rechecking all saved recordings with Microsoft's pinned scorer confirmed that Mimi leads P.835 overall, while AudioVAE2 leads P.808. [Revalidation details](docs/streaming-baseline.md#dnsmos-revalidation).

The new kernels are checked separately for sample-level agreement with stock AudioVAE2. PESQ, STOI and UTMOS above remain the earlier panel results; the DNSMOS recheck used its saved audio. UTMOS and DNSMOS are predictions, not human ratings. These metrics do not assess frequencies above 8 kHz. Meta DACVAE is excluded because the tested checkpoint requires future frames. [Quality results and validation](docs/streaming-baseline.md#quality).

## More

[Kernel experiments](docs/cpu-kernel-results.md) and [optional backends and encoder setup](docs/optional-backends.md) contain the detailed build and validation records. The encoder retains its existing 16 kHz input. The [streaming decoder](docs/streaming.md) carries independent history between chunks.

Architecture and weights originate from [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). Preparation uses the checksum-verified [pinned ONNX export](https://huggingface.co/ai4all8/VoxCPM2-ONNX/tree/ecb511b96675f041424b42f148bf72e301262586). Upstream model and dependency licenses apply.
