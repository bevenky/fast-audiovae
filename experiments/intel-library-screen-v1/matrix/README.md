# Intel second-projection matrix screen

This isolated experiment compares the existing sequential oneMKL pair with one shared preparation, cached oneDNN MatMul weights, and the public BRGeMM API with 32/64 output-channel panels. It changes no production files.

Only the real second pair is enabled: two `3072 × 1024` projections with 8 or 16 input columns, from 40/80 ms streaming packets. It does not represent the first projection, which already uses a custom VNNI kernel.

The candidate uses the current symmetric INT8 quantization and FP32 scales. Quantization is still across channels per input column, including the current AVX512 conversion for 16 columns. It moves the unsigned offset from weights to the transposed input and compensates with precomputed weight sums. The corrected integer product and multiplication order are unchanged. Weight packing happens once. Runtime timings include quantization, transpose, temporary allocation, GEMM, compensation, dequantization and scattering back to channel-major output. The existing-core shared-preparation control separates that gain from the library comparison.

Requirements: Linux x86-64 with AVX512 VNNI, NumPy, the existing precision core and oneDNN **3.13.2**, built with CPU `SEQ`, GPU `NONE` and `ONEDNN_EXPERIMENTAL_UKERNEL=ON`. MatMul and Reorder must be built. No Torch, ORT inference, audio generation, network access or GPU is used by the runner.

```sh
python experiments/intel-library-screen-v1/matrix/build.py \
  --onednn-source /dev/shm/intel-library-screen-v1/oneDNN-3.13.2 \
  --onednn-build /dev/shm/intel-library-screen-v1/oneDNN-3.13.2/build \
  --output /dev/shm/intel-library-screen-v1/matrix-build

python experiments/intel-library-screen-v1/matrix/run.py \
  --build /dev/shm/intel-library-screen-v1/matrix-build/build.json \
  --capture /dev/shm/intel-library-screen-v1/matrix-inputs.npz \
  --capture-metadata /dev/shm/intel-library-screen-v1/capture.json \
  --core /var/tmp/fast-audiovae-intel-precision/native-large-build-r1/libintel_precision_core.so \
  --output /dev/shm/intel-library-screen-v1/matrix-result.json
```

The capture must contain FP32 `weight0`, `weight1`, `input_8`, `input_16`, and `expected0_8`, `expected1_8`, `expected0_16`, `expected1_16`. Expected outputs come from those exact nodes of the combined baseline graph. Array contents remain in the capture; the report contains only hashes, shapes, aggregate error and timings.

Before timing, the existing core must match the captured outputs bit for bit. Candidates must pass the same gate, plus zero and nearest-even rounding probes. Rejected candidates are not timed. Each shape uses two warmups and three alternating rounds, with a one-second completed-call budget including qualification calls. Setup and JIT compilation are reported separately. One additional call splits preparation from remaining work; its clock overhead is excluded from the main medians. This is a matrix screen, not a decoder RTF or audio-quality qualification.

The BRGeMM interface is public but experimental. Its packing and execution follow the [official example](https://uxlfoundation.github.io/oneDNN/page_cpu_brgemm_example_cpp.html); constant-weight MatMul uses the [documented descriptor/reorder flow](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html). The installed 3.13.2 headers, rather than the website's newer default documentation, are the compile-time authority.

## Optional decoder check

`bridge.py` builds a separate ORT operator and replaces only the two second-projection nodes. All other nodes, initializers and state interfaces are checked byte for byte. It caches both tested shapes; other packet sizes use the unchanged precision core. This experimental bridge retains roughly two extra copies of the pair's quantized weights for the two shape plans, in addition to the fallback weights. It is qualified only for serial, one-thread checks.

```sh
python experiments/intel-library-screen-v1/matrix/bridge.py \
  --source /dev/shm/intel-streaming-transfer-v1-graphs-r2/combined.onnx \
  --matrix-build /dev/shm/intel-library-screen-v1/matrix-build-r2/build.json \
  --ort-include /var/tmp/fast-audiovae-20260907/repo/.deps/onnxruntime/include \
  --core /var/tmp/fast-audiovae-intel-precision/native-large-build-r1/libintel_precision_core.so \
  --mode 4 --output /dev/shm/intel-library-screen-v1/matrix-bridge

python experiments/intel-library-screen-v1/matrix/full_check.py \
  --bridge /dev/shm/intel-library-screen-v1/matrix-bridge \
  --capture-metadata /dev/shm/intel-library-screen-v1/capture.json \
  --packet 80 --output /dev/shm/intel-library-screen-v1/matrix-full80.json
```

Use `--packet 40` for the second packet size. The driver follows the sibling Snake screen's bounded protocol: three 1.6-second prefixes, one warm pass, three alternating measured repetitions, mixed short partitions, and empty/reset checks. Matrix-only requires every output and state to match exactly. Optional `--snake-build /path/to/snake/build.json` runs the matrix plus Snake candidate in mode2 and leaves the baseline in mode0; that combined check reports numerical differences rather than applying the matrix-only equality gate. Results contain aggregates only, with a 25-second completed-inference budget per variant.
