# Optional FP32 backends and encoder

These options extend the public FP32 setup. Intel and AMD INT8 have separate experiment instructions.

## Optional fused decoder

On compatible Linux x86 CPUs, enable AVX512 and fuse adjacent Snake, depthwise and residual operations:

```sh
fast-audiovae prepare --output artifacts-fused --native-backend avx512 --block-fusion both
```

Load this directory with `load_decoder("artifacts-fused")`. CPU and OS checks guard the native path; unsupported systems use the bundled standard ONNX fallback. The default preparation recipe stays unchanged. Larger stage and matrix experiments are kept separately in [experiments/cpu-stage](../experiments/cpu-stage).

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
