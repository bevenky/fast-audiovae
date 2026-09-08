# Intel streaming projection pair

This recipe replaces the first two streaming upsampling projections with one CPU operator. It packs the fixed weights once, shares activation preparation, and uses AVX512-VNNI across output channels. It keeps the accepted integer values, scales and complete-K arithmetic.

The direct path supports one to four latent frames. Larger chunks use the accepted matrix core. No additional audio buffering or trained-weight changes are introduced. The full-call graph remains unchanged.

Validated on the Intel Xeon Platinum 8280 VM with one inference thread and ONNX Runtime 1.29.0. This recipe has not been validated on AMD.

Build from an existing validated Intel one-worker bundle. The source and output bundles must be on the same filesystem because the recipe hardlinks unchanged artifacts. Use a fresh output directory.

```sh
python experiments/streaming-matrix/intel/build_candidate.py \
  --baseline .build/intel-one-worker \
  --output .build/intel-stream-pair \
  --ort-include .deps/onnxruntime/include \
  --core .build/intel-core/libintel_precision_core.so

python experiments/streaming-matrix/intel/check_micro.py \
  --baseline .build/intel-one-worker \
  --candidate .build/intel-stream-pair \
  --output .build/intel-stream-pair-checks.json
```

Load the prepared bundle with `load_streaming_decoder(".build/intel-stream-pair", threads=1)`. The existing loader registers the new operator from the bundle manifest. This recipe does not change platform defaults.

The build requires the accepted sequential oneMKL core, its CPU dependencies, ORT API 29 headers, and a C++17 compiler. CPU and OS support for AVX512-VNNI are checked before the operator runs. The build disables fast math and floating-point contraction.

The arithmetic checker covers 35 cases, including empty and partial chunks, zero, extrema, subnormal inputs, rounding ties, and the larger-chunk fallback. The paired timing campaign passed 126 waveform checks and 72 state checks with bitwise equality.

Full-corpus qualification also passed all 180 complete streaming runs across 60 frozen multilingual clips, using one, two and four latent frames per call. Every output was bitwise identical to the accepted full decoder, with no missing samples.
