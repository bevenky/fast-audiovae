# Apple SME2 panels experiment

This opt-in CPU experiment prepares bounded INT8 time panels and runs complete-K matrix products through the pinned KleidiAI SME2 kernel. It also combines the late C256, C128, C64 and C32 decoder regions to reuse intermediate buffers. The measured 3.0% reduction in decoder time missed the 10% adoption threshold, and automated quality scores were slightly lower than FP32. The candidate remains experimental and FP32 remains the default.

| Decoder | Matched RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.092772 |
| Recommended Apple FP32 | 0.024685 |
| SME2 panels INT8 | 0.023938 |
| Mimi | 0.028114 |

These are equal-clip means over ten fixed clips, five repetitions and two warmups on Apple M5 Max, four ORT threads and ONNX Runtime 1.29 CPUExecutionProvider. The 95% clip-and-repeat bootstrap interval for time saved is 0.86% to 5.86%. All 1,620 execution checks passed; 60 full INT8 waveforms and 120 prefix comparisons were bitwise equal to the previous Apple INT8 implementation. That equality preserves the previous INT8 result, not FP32 fidelity. See [quality and profiling results](../../../docs/apple-precision.md) and [compact evidence](../../../benchmarks/apple-precision/sme2-panels.json). Earlier screens used different controls and should not be treated as a measured speedup into this result.

The 22 selected large products retain the previous experiment's per-row weight scales, per-time-column activation scales and full-K INT32 accumulation. Nine smaller products, Snake, depthwise convolution, inputs and outputs remain FP32. Learned initializer bytes and the 16 kHz input / 48 kHz output codec interface are unchanged. This is an explicit graph and library selection, not automatic INT8 dispatch.

Build from the repository root on native ARM macOS with an SME2-capable CPU, an Apple toolchain supporting SME2, Python 3.11+, ONNX Runtime 1.29.0, ONNX 1.22.0 and NumPy. Validation and WAV export also need SoundFile. Supply the pinned ORT headers and an existing accepted Apple FP32 native library. The builders download nothing.

```sh
python experiments/apple-precision/sme2-panels/build.py \
  --output-dir .build/apple-sme2-panels \
  --ort-include /path/to/onnxruntime/include \
  --ort-pins experiments/apple-precision/pins/ort.json

python experiments/apple-precision/sme2-panels/fused/build_fused.py \
  --output-dir .build/apple-sme2-panels-fused \
  --ort-include /path/to/onnxruntime/include \
  --native-include native/apple \
  --native-library /path/to/accepted/apple-native.dylib \
  --core-library /path/from/core-build.json/core_path

python experiments/apple-precision/sme2-panels/rewrite_fused.py \
  --source /path/to/accepted/apple/decoder.onnx \
  --output /path/to/new/decoder.onnx \
  --package-source src --tile-time 512 --segments 4 --shards 4
```

Use fresh build and graph destinations. `build.json` records the actual library names and hashes. The rewriter accepts only source graph SHA256 `fa7992825e807cac7ab1be405912ae3735031fb941a18dd81006005dc597bde3`. Register the accepted FP32 library, the new core operator bridge, and both fused libraries when opening the rewritten graph. Keep each library's recorded dependencies available. Rebuilding elsewhere produces new artifacts that need validation.

Run checks with CPU visibility and thread limits set before Python imports:

```sh
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python experiments/apple-precision/sme2-panels/check.py \
  --build .build/apple-sme2-panels/build.json --output /path/to/new/native-checks.json
python experiments/apple-precision/sme2-panels/check_parallel_edges.py \
  --build .build/apple-sme2-panels/build.json --output /path/to/new/parallel-checks.json
python experiments/apple-precision/sme2-panels/native/tests/check_panels.py \
  --library /path/from/core-build.json/core_path --library-sha256 CORE_SHA256 \
  --output /path/to/new/panel-checks.json
python experiments/apple-precision/sme2-panels/fused/check_fused.py \
  --native-library /path/to/accepted/apple-native.dylib \
  --core-ops /path/from/core-build.json/ops_path \
  --stage-library .build/apple-sme2-panels-fused/libapple_r4_stage.dylib \
  --upsample-library .build/apple-sme2-panels-fused/libapple_r4_upsample.dylib \
  --output /path/to/new/fused-checks.json
```

The baseline ARM wrapper checks SME2 support, streaming vector length and floating-point mode. Only upstream kernel objects receive SME2 compiler flags. Unsupported conditions fail explicitly; selecting this experiment does not replace the application's FP32 fallback policy.

For a fresh full comparison, supply the original four-model r3 reference config and its locally available model/corpus artifacts. Replace the capitalized hash values below with the SHA256 of each input file. The preparer verifies both configs, the build/library/graph identities and the unchanged cohort before writing relative paths to a fresh destination. The source and reference config can be the same r3 file.

```sh
python experiments/apple-precision/sme2-panels/validation/prepare_config_portable.py \
  --source-config /path/to/r3/config.json --source-config-sha256 REFERENCE_CONFIG_SHA256 \
  --reference-config /path/to/r3/config.json --reference-config-sha256 REFERENCE_CONFIG_SHA256 \
  --core-build .build/apple-sme2-panels/build.json --core-build-sha256 CORE_BUILD_SHA256 \
  --fused-build .build/apple-sme2-panels-fused/build.json --fused-build-sha256 FUSED_BUILD_SHA256 \
  --graph /path/to/new/decoder.onnx \
  --graph-manifest /path/to/new/decoder.json --graph-manifest-sha256 GRAPH_MANIFEST_SHA256 \
  --output /path/to/new/config.json

python experiments/apple-precision/sme2-panels/validation/campaign_portable.py \
  --config /path/to/new/config.json --config-sha256 PREPARED_CONFIG_SHA256 \
  --reference-config /path/to/r3/config.json --reference-config-sha256 REFERENCE_CONFIG_SHA256 \
  --harness experiments/apple-precision/tools/compare_decoders.py \
  --harness-sha256 593dc1df8b7ac7334add9211da869eac931aaf61d360656b15a3196467cecf01 \
  --mode full --threads 4 --output-dir /path/to/new/campaign
```

Keep the CPU environment above. `full` validates all 60 clips, exports FLOAT WAVs and times the fixed ten clips with two warmups and five paired repetitions. `--mode quality` runs validation and export without timing. The separately loaded r3 INT8 reference is compared with the original tolerances, with bitwise equality recorded separately. The primary timing aggregate gives each selected clip equal weight. Scoring the exported files is a separate step.

`validation/campaign.py`, `campaign_v2.py` and `prepare_config.py` preserve the executed workspace sources. The v2 correction bounds probes by real clip length after the original runner stopped on four short Mimi clips before timing. `campaign_portable.py` differs from v2 only in locating the pinned shared helper under the repository's `tools/` directory. The new preparer changes paths and provenance, not graph math. [Packaging provenance](packaging-provenance.json) records the hashes and exact path-only diff. Synthetic tests use `validation/check_protocol_portable.py` and `validation/check_config_portable.py`; neither loads a decoder.

No weights, corpus, audio, binaries or metric checkpoints are bundled. Existing `../quality/` and `../tools/` contain the shared scorer and benchmark helpers; supply `--scorer-dir` explicitly when using the scorer. Source and dependency hashes are in [source-manifest.json](source-manifest.json). See [third-party notices](THIRD_PARTY.md).
