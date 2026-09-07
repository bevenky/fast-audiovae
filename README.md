# fast-audiovae

CPU inference optimizations for the AudioVAE2 decoder used by VoxCPM2. The trained weights and FP32 interface stay unchanged: input latents `[1, 64, L]` at 25 Hz produce mono audio `[1, 1, 1920*L]` at 48 kHz.

The package combines ONNX graph rewrites with native Snake, causal depthwise and phase-interleave operators. Every session uses ONNX Runtime's CPU execution provider. Backend selection checks platform and available CPU instructions; it does not benchmark or autotune your machine.

## Benchmarks

Original multilingual baseline using the previous native build. CPU-only decoder inference, FP32, ONNX Runtime 1.29.0. Ten fixed clips, three repetitions. Lower RTF is better.

| CPU | Threads | Stock AudioVAE2 | Fast AudioVAE2 | Mimi | Meta DAC-VAE |
|---|---:|---:|---:|---:|---:|
| Apple M5 Max | 4 | 0.11351 | 0.02706 | 0.03285 | 0.54146 |
| AMD EPYC 9654 | 4 | 0.26078 | 0.07629 | 0.05400 | 0.66065 |
| Intel Xeon Platinum 8280 VM | 2 | 0.67968 | 0.35355 | 0.17709 | 2.06885 |

Stock is the original ONNX export. Fast used the then-current default native backend: 4.19x faster on Apple, 3.42x on AMD and 1.92x on Intel. These are full-clip decoder calls; loading, encoding and TTS generation are excluded.

Latest kernel experiments use the same ten timed clips, five repetitions and full 60-clip validation:

| CPU | Previous fast | Fused AVX512 option | Stage experiment | Mimi |
|---|---:|---:|---:|---:|
| AMD EPYC 9654, 4 threads | 0.07342 | 0.06501 | **0.06104** | 0.05173 |
| Intel Xeon Platinum 8280 VM, 2 threads | 0.34654 | 0.29286 | **0.25510** | 0.17169 |

The stage experiments reduce time by 16.9% on AMD and 26.4% on Intel. All ten clip averages improve, with 331 validation checks passing on each host. The fused option is available through `prepare`; the larger stage experiments remain separate. Apple correctness passed, but unstable timing prevents a new performance claim. See [kernel results](docs/cpu-kernel-results.md).

A further Intel-only upsampling experiment reduced RTF from **0.25398 to 0.24424**, another **3.83% less decoding time**. Mimi measured 0.17327 in that same run. All ten clip averages improved and all 264 validation records passed. Runtime defaults remain unchanged. See [Intel results and bottlenecks](docs/intel-upsampling.md).

Reconstruction quality from the original 60-recording FLEURS comparison across ten languages, measured in the common 16 kHz source bandwidth. Higher scores are better. New kernels passed numerical checks; these quality metrics were not rerun.

| Codec | PESQ | STOI | UTMOS22 | DNSMOS P.835 overall | DNSMOS P.808 |
|---|---:|---:|---:|---:|---:|
| Fast AudioVAE2 | 3.742 | 0.936 | 2.257 | 2.765 | 3.404 |
| Pocket continuous Mimi | 2.130 | 0.807 | 2.517 | 2.894 | 3.339 |
| Meta DAC-VAE | 4.284 | 0.973 | 2.222 | 2.779 | 3.431 |
| Supertonic 3 | N/A | N/A | N/A | N/A | N/A |

Stock and fast AudioVAE2 agree at this precision. UTMOS and DNSMOS are learned predictions, not listening-panel ratings. AudioVAE2 and Pocket continuous Mimi are causal and output 48 kHz and 24 kHz respectively. The tested Meta DAC-VAE outputs 48 kHz, is noncausal and retains its full watermark.

The original comparison passed all 847 decoder validation checks. [Full results and methodology](docs/multilingual.md) include per-clip data and the MOS audit.

Supertonic 3 lacks a matching public audio encoder for this reconstruction test. MUSHRA listening scores are unmeasured for all models. See [comparison status](docs/codec-comparison-status.md).

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
- **Linux x86:** guarded AVX2/SSE2 and optional AVX512 with SLEEF; validated on AMD EPYC 9654 and an Intel Xeon Platinum 8280 VM.
- **Other platforms, including generic ARM:** standard ONNX CPU fallback. Unavailable native dependencies also select the fallback.

Use `prefer_custom=False` to request the ONNX fallback. See [performance and validation](docs/performance.md) for measured results and limits.

## Optional fused decoder

On compatible Linux x86 CPUs, enable AVX512 and fuse adjacent Snake, depthwise and residual operations:

```sh
fast-audiovae prepare --output artifacts-fused --native-backend avx512 --block-fusion both
```

Load this directory with `load_decoder("artifacts-fused")`. CPU and OS checks guard the native path; unsupported systems use the bundled standard ONNX fallback. The default preparation recipe stays unchanged. Larger stage and matrix experiments are kept separately in [experiments/cpu-stage](experiments/cpu-stage).

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
