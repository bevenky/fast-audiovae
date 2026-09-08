# fast-audiovae

Fast CPU inference for VoxCPM2's AudioVAE2 decoder, using ONNX graph rewrites and native kernels. Latents `[1, 64, L]` at 25 Hz produce mono audio `[1, 1, 1920*L]` at 48 kHz. Each call starts with fresh causal history.

## Run

Use Python 3.11 to 3.13. Native builds need Apple Command Line Tools on Apple ARM, or a C/C++ compiler, CMake and Make on Linux x86.

```sh
git clone https://github.com/bevenky/fast-audiovae.git
cd fast-audiovae
python -m pip install -e .
python tools/build.py
fast-audiovae prepare --output artifacts
```

Build and prepare on each target machine. Setup downloads pinned dependencies and model files. With uv, use `uv pip install -e .` inside an activated environment.

```python
from fast_audiovae import load_decoder

session, selected = load_decoder("artifacts")
# latents: contiguous NumPy float32 array, shape [1, 64, L]
audio = session.run(None, {session.get_inputs()[0].name: latents})[0]
```

Inference uses the CPU only. The loader automatically detects compatible CPU instructions and falls back to standard ONNX when needed. Set `threads` explicitly for a CPU quota; otherwise the loader uses up to four visible CPUs.

## Decoder speed

Lower RTF is better. RTF is total decoding time divided by generated audio duration; it excludes loading, encoding and TTS generation.

| CPU | Threads | Base AudioVAE2 | Optimized AudioVAE2 | Mimi | Meta DAC-VAE, earlier run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple M5 Max | 4 | 0.09280 | **0.02470** | 0.02815 | 0.54146 |
| AMD EPYC 9654 | 4 | 0.25556 | **0.03435** | 0.05201 | 0.66065 |
| Intel Xeon Platinum 8280 VM | 2 | 0.65769 | **0.16411** | 0.17002 | 2.06885 |

CPU-only, ONNX Runtime 1.29. Base, optimized and Mimi values are matched within each row: ten clips on Apple and Intel, three on AMD, with five repetitions after warmup. DAC timings come from the earlier comparison and are shown for reference.

Setup and measured results: [Apple](docs/apple-precision.md), [AMD](experiments/amd-precision/README.md), [Intel](experiments/intel-precision/README.md). The optimized Intel and AMD results use these setup recipes; they are not yet selected automatically.

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

[Kernel experiments](docs/cpu-kernel-results.md) and [optional backends and encoder setup](docs/optional-backends.md) contain the detailed build and validation records. The encoder retains its existing 16 kHz input. There is no cached streaming API.

Architecture and weights originate from [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). Preparation uses the checksum-verified [pinned ONNX export](https://huggingface.co/ai4all8/VoxCPM2-ONNX/tree/ecb511b96675f041424b42f148bf72e301262586). Upstream model and dependency licenses apply.
