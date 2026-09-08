# fast-audiovae

CPU inference for VoxCPM2's AudioVAE2 decoder, using ONNX graph rewrites and native kernels. FP32 latents `[1, 64, L]` at 25 Hz produce mono `[1, 1, 1920*L]` audio at 48 kHz. Calls start with fresh causal history; there is no cached streaming API.

## Run

Use Python 3.11 to 3.13. Native builds require Apple Command Line Tools on Apple ARM, or a C/C++ compiler, CMake and Make on Linux x86.

```sh
git clone https://github.com/bevenky/fast-audiovae.git
cd fast-audiovae
python -m pip install -e .
python tools/build.py
fast-audiovae prepare --output artifacts
fast-audiovae inspect artifacts
```

Build and prepare on each target machine. Setup downloads pinned dependencies and model files. To use standard ONNX on another platform, skip the native build. With uv, use `uv pip install -e .` inside an activated environment.

```python
from fast_audiovae import load_decoder

session, selected = load_decoder("artifacts")
print(selected)
# latents: contiguous NumPy float32 array, shape [1, 64, L]
audio = session.run(None, {session.get_inputs()[0].name: latents})[0]
```

Every session uses CPUExecutionProvider. The loader chooses a compatible native FP32 backend or standard ONNX CPU. Use `prefer_custom=False` to request the ONNX fallback. Omit `threads` for up to four visible CPUs; set it explicitly for CPU quotas or a two-vCPU VM.

- **Apple ARM:** use the public FP32 route above, with NEON/vForce. The latest SME2 INT8 experiment saved only 3.0% of decoder time, below the 10% adoption threshold.
- **Intel Linux:** the public route is FP32. [Selective INT8](experiments/intel-precision/README.md) is an explicit option for suitable AVX512-VNNI systems, using pinned sequential oneMKL.
- **AMD Linux:** the public route is FP32. [Selective INT8 with AOCL-DLP](experiments/amd-precision/README.md) is an explicit option for the tested EPYC configuration. Its 60-clip quality evaluation reproduces the accepted Intel INT8 result.

INT8 is not selected automatically. Experimental graphs and libraries require their separate instructions. CPU capability checks prevent unsupported execution; they do not predict speed on an untested machine.

## Decoder benchmarks

The matched FP32 codec comparison below used ONNX Runtime 1.29, ten fixed clips and three repetitions. Lower RTF is better. This historical comparison includes all three codecs; later experiments are linked below.

| CPU | Threads | Stock AudioVAE2 | Native AudioVAE2 FP32 | Mimi | Meta DAC-VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple M5 Max | 4 | 0.11351 | 0.02706 | 0.03285 | 0.54146 |
| AMD EPYC 9654 | 4 | 0.26078 | 0.07629 | 0.05400 | 0.66065 |
| Intel Xeon Platinum 8280 VM | 2 | 0.67968 | 0.35355 | 0.17709 | 2.06885 |

RTF measures full-clip decoder calls and excludes loading, encoding and TTS generation. Stock is the original export, not the rewritten ONNX fallback. Models have different latent and output-rate contracts.

Later matched experiments:

- **Intel:** [Selective INT8](docs/intel-precision.md) measured RTF **0.16411**, versus optimized FP32 0.24821 and Mimi 0.17002. That is 33.9% less time than FP32. Automated quality scores declined slightly; a single-listener pilot tied FP32.
- **AMD:** [AOCL selective INT8](docs/amd-precision.md) measured RTF **0.03435**, versus FP32 0.06056 and Mimi 0.05201. That is 43.3% less time in the final three-clip screen. Its fresh 60-clip quality scores reproduce the Intel INT8 result.
- **Apple:** [Retain FP32](docs/apple-precision.md). The latest ten-clip comparison measured FP32 RTF **0.02469**, INT8 0.02394 and Mimi 0.02811. INT8 saved only 3.0% and its fresh 60-clip quality scores were slightly below FP32.

Compare each experiment with its own control; do not combine speedups across campaigns.

## Reconstruction quality

The original 60-clip FLEURS comparison covers ten languages, scored in the common 16 kHz source bandwidth. These are FP32 results, not scores for the later INT8 variants. Higher is better.

| Codec | PESQ | STOI | UTMOS22 | DNSMOS overall | DNSMOS P.808 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AudioVAE2 FP32 | 3.742 | 0.936 | 2.257 | 2.765 | 3.404 |
| Pocket continuous Mimi | 2.130 | 0.807 | 2.517 | 2.894 | 3.339 |
| Meta DAC-VAE | 4.284 | 0.973 | 2.222 | 2.779 | 3.431 |

UTMOS and DNSMOS are predictions, not listening-panel ratings. AudioVAE2 outputs 48 kHz and Pocket continuous Mimi outputs 24 kHz; both tested decoders are causal. The tested 48 kHz Meta DAC-VAE is noncausal. No standardized MUSHRA panel is complete. Supertonic 3 has no matching public encoder for this reconstruction test. [Methodology and per-clip evidence](docs/multilingual.md).

## More

[FP32 fusion and stage experiments](docs/cpu-kernel-results.md), [optional FP32 backends and encoder setup](docs/optional-backends.md), and the explicit experiment packages contain the detailed build and validation records. The encoder retains its existing 16 kHz input; decoder optimization does not change that interface.

Architecture and weights originate from [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). Preparation uses the checksum-verified [pinned ONNX export](https://huggingface.co/ai4all8/VoxCPM2-ONNX/tree/ecb511b96675f041424b42f148bf72e301262586). Upstream model and dependency licenses apply.
