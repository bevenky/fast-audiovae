# fast-audiovae

CPU inference optimizations for the AudioVAE2 decoder used by VoxCPM2. The trained weights and FP32 interface stay unchanged: input latents `[1, 64, L]` at 25 Hz produce mono audio `[1, 1, 1920*L]` at 48 kHz.

The package combines ONNX graph rewrites with native Snake, causal depthwise and phase-interleave operators. Every session uses ONNX Runtime's CPU execution provider. Backend selection checks platform and available CPU instructions; it does not benchmark or autotune your machine.

## Benchmarks

CPU-only decoder inference, FP32, ONNX Runtime 1.29.0. Ten fixed multilingual clips, three repetitions. Lower RTF is better.

| CPU | Threads | Stock RTF | Optimized RTF | Speedup |
|---|---:|---:|---:|---:|
| Apple M5 Max | 4 | 0.11351 | 0.02706 | 4.19× |
| AMD EPYC 9654 | 4 | 0.26078 | 0.07629 | 3.42× |
| Intel Xeon Platinum 8280 VM | 2 | 0.67968 | 0.35355 | 1.92× |

Stock is the original ONNX export. Optimized uses the default native backend. [Full results](docs/multilingual.md) include Mimi and Meta DACVAE, quality scores on 60 recordings, and numerical validation.

## Setup

Python 3.11 to 3.13 is recommended in a virtual environment. Decoder tests used Python 3.13 on macOS and 3.12 on Linux. Native builds require Apple Command Line Tools on Apple ARM, or a C/C++ toolchain, CMake and Make on Linux x86.

```sh
git clone https://github.com/bevenky/fast-audiovae.git
cd fast-audiovae
pip install -e .
python tools/build.py
fast-audiovae prepare --output artifacts
```

For standard ONNX without native operators, skip `tools/build.py`. Build dependencies and the pinned model are downloaded during setup, not package import. Generated files remain outside version control. Preparation produces a portable ONNX fallback and, when available, the locally built native backend.

```python
from fast_audiovae import load_decoder

session, selected = load_decoder("artifacts")
print(selected)
# latents: contiguous NumPy float32 array, shape [1, 64, L]
audio = session.run(None, {session.get_inputs()[0].name: latents})[0]
```

Calls start with fresh causal history. There is no cached streaming API. Model loading, encoding and TTS generation are separate from decoding.

The default uses up to four CPUs visible to the process, so a two-vCPU VM uses two workers. Set `threads` explicitly if needed, including when a container's CPU quota is smaller than its visible CPU count.

## Hardware

- **Apple ARM:** NEON/vForce; validated on M5 Max.
- **Linux x86:** guarded AVX2/SSE2 and SLEEF; validated on AMD EPYC 9654 and an Intel Xeon Platinum 8280 VM.
- **Other platforms, including generic ARM:** standard ONNX CPU fallback. Unavailable native dependencies also select the fallback.

Use `prefer_custom=False` to request the ONNX fallback. See [performance and validation](docs/performance.md) for measured results and limits.

## Optional AMD matrix packing

On AMD Zen 4 or newer running Linux, with AVX-512 enabled:

```sh
python tools/build.py --amd-packed
fast-audiovae prepare --output artifacts-amd --amd-build .build/amd/build.json
```

Then call `load_decoder("artifacts-amd", threads=4, prefer_packed=True)`. This option adds approximately 164 MiB of packed FP32 weights for a modest measured gain. Packing is a layout transformation, not compression. The loader checks AMD vendor, instructions, OS vector state and validated thread count before enabling it.

## Optional encoder preparation

```sh
pip install -e '.[encoder]'
```

```python
from fast_audiovae.encoder import prepare_encoder

handle = prepare_encoder(vae.encoder.eval())  # Existing upstream CPU FP32 model.
```

Use the upstream VAE's normal encoding path under `torch.inference_mode()`. Its existing 16 kHz encoder input and 48 kHz decoder output remain unchanged. The helper folds weight normalization and removes redundant pointwise padding copies while preserving original encoder Snake and depthwise operations. It does not export an encoder ONNX model. `handle.restore()` restores forwards; reload the checkpoint before training.

## Attribution

Model architecture and weights originate from [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). Preparation uses the checksum-verified [ai4all8/VoxCPM2-ONNX export](https://huggingface.co/ai4all8/VoxCPM2-ONNX/tree/ecb511b96675f041424b42f148bf72e301262586). Upstream model and dependency licenses apply.
