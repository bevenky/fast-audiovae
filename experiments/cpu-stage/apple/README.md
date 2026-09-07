# Apple stage experiment

This prototype keeps three residual units in small CPU tiles, reusing transformed
history between tiles. It uses NEON depthwise convolution, vForce Snake through
the existing Apple native library, and the shared FP32 matrix kernel. Weights,
causal padding and residual order are preserved.

Correctness passed on Apple Silicon with ONNX Runtime 1.29.0: all 270 focused
comparisons and 12 intermediate outputs captured from the actual decoder were
bitwise equal to the original native graph. The checks cover segmentation, short
inputs, tails, concurrent calls and future-input independence. See
[test-summary.json](test-summary.json) for the exact scope and artifact hashes.

No performance timing was run for this prototype. It is not enabled by default;
earlier Apple whole-decoder timings were unstable, so a controlled performance
screen and full waveform checks are still required.

From the repository root, using the CPU dependencies and header pins listed in
[dependency-pins.json](../dependency-pins.json):

```sh
python experiments/cpu-stage/apple/build.py \
  --ort-include "$ORT_INCLUDE" \
  --native-include native/apple \
  --native-library "$APPLE_NATIVE_LIBRARY" \
  --native-build-manifest "$APPLE_NATIVE_MANIFEST"
```

The builder checks the native library and headers against their manifests and
inherits the native library's minimum macOS version. It writes only to
`experiments/cpu-stage/.build/apple` and does not install dependencies.

```sh
python experiments/cpu-stage/apple/check_stage.py \
  --native-library "$APPLE_NATIVE_LIBRARY" \
  --stage-library experiments/cpu-stage/.build/apple/libstage_pipeline.dylib \
  --output experiments/cpu-stage/.build/apple/checks.json

PYTHONPATH=src python -m unittest discover \
  -s experiments/cpu-stage/apple/tests -v
```

`rewrite.py` creates a separate graph from an Apple native decoder, with explicit
stage selection. It rejects exposed intermediate outputs, incompatible residual
connections and incorrect dilations. Use backend `2`, matrix mode `0` and matrix
ISA `128`. `check_real.py` checks captured stage outputs from a supplied real case;
it does not measure RTF or establish whole-decoder audio quality.

Only source and validation summaries are included here. Models, audio, captured
arrays and compiled libraries are not included.
