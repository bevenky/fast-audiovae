# Intel CPU precision experiment

This is an opt-in decoder experiment for fast-audiovae. It changes selected matrix operands while retaining FP32 Snake, depthwise convolution, bias, residual operations and waveform output. It does not change the installed runtime or its defaults. On the tested Intel VM, selective INT8 reduced decoder time by 33.9% versus the matched optimized FP32 decoder. Automated quality scores fell slightly; an eight-clip, single-listener pilot tied the FP32 average. See the [complete results](../../docs/intel-precision.md).

The frozen INT8 core uses per-output-row weight scales and per-time-column activation scales across channels. It computes complete-K integer products with oneMKL, then returns FP32. Each call handles up to 2048 output rows and 512 time columns, with at most 4 MiB of integer scratch per active callback. There is no calibration across future samples.

The FP16 control rounds operands to binary16 but stores and accumulates them in FP32. Fixed, aligned 64x64 SGEMM panels keep the reduction geometry stable across lengths and worker partitions. This is **not native FP16 GEMM** or a claim of reduced model memory.

## Build and check

Use Linux x86-64 with AVX512-VNNI for the Intel INT8 path. The tested host was a Xeon Platinum 8280 VM. Dependencies are CPU ONNX Runtime **1.29.0**, ONNX **1.22.0**, oneMKL **2026.1.0** with sequential LP64 libraries, a C++17 compiler and Python 3.12 with NumPy. Recorded checks used NumPy 2.2.6 and GCC 13.3. Other host/toolchain combinations need their own validation.

Supply local dependency directories; hashes and source URLs are in `pins/`. Nothing is downloaded automatically. Run from this directory:

```sh
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset MKL_CBWR MKL_ENABLE_INSTRUCTIONS LIBXSMM_TARGET
python tools/verify_bundle.py
python tools/verify_publication.py

python native/build.py --output-dir build/native --cxx g++ \
  --mkl-include /path/to/mkl/include --mkl-library-dir /path/to/mkl/lib \
  --mkl-pins pins/mkl.json --ort-include /path/to/ort/include --ort-pins pins/ort.json

python native/check_micro.py --library build/native/libintel_precision_core.so \
  --ops build/native/libintel_precision_ops.so --backend 1 --output build/micro.json
```

`check_half_fixed.py` tests exact prefixes and row partitions; its scalar reference can come from the same core using backend zero. `check_candidate.py` additionally compares INT8 against the retained r2 core, which is not included. Its recorded reference-library hash is in `evidence/checks-large-r1-parity.json`.

## Integrate with a supplied decoder

`tools/rewrite.py --source /path/to/accepted-decoder.onnx --output-dir build/graphs` creates `int8_all`, `int8_large` and `fp16_all`. It requires the exact accepted source graph hash `1ddb6dcc2b0c3ccea90d309f6ebec10eb12e844fb6320cd2d525dbfd01d4cf29`. Learned initializer bytes stay unchanged. `int8_large` selects matrices with both dimensions at least 128, including the corresponding fused stages.

The supplied fused sources require the existing FP32 native library. Build that library using [fast-audiovae's](https://github.com/bevenky/fast-audiovae) `tools/build_x86.py` and its pinned SLEEF/ORT dependencies. Also supply a built LIBXSMM tree and source archive for commit `55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1`:

```sh
python tools/build_fused.py --precision-build build/native/build.json \
  --native-build /path/to/fast-audiovae/.build/x86/build.json \
  --ort-include /path/to/ort/include --libxsmm /path/to/built/libxsmm \
  --libxsmm-source-archive /path/to/libxsmm.tar.gz --output-dir build/fused
```

Load the rewritten graph with the source graph's existing custom libraries, plus the precision ORT bridge and the two new fused libraries, using only CPUExecutionProvider. This bundle does not add automatic runtime selection. Validate complete waveforms, short/long prefixes and repeated calls before measuring decoder RTF. Approximate variants require separate quality evaluation; numerical micro checks do not establish perceptual equality.

## Included evidence and limits

`native/` is byte-identical to the frozen candidate. Saved evidence covers 22 mode/shape cases, malformed-input checks, 12 ORT cases, 22 cross-core INT8 comparisons and 120 fixed-geometry half prefix checks. These native checks establish implementation behavior; the completed [campaign evidence](../../benchmarks/intel-precision/README.md) separately records 60-clip validation, ten-clip timing and quality assessment. The original evidence is preserved; public records identify any normalized host paths.

The [later source archive](archive/iteration3/README.md) retains seven rejected optimization screens. None met the required 10% whole-decoder improvement, and none changes the accepted implementation above.

Weights, ONNX models, audio, corpus archives, predictor assets and compiled libraries are not included. The machine-specific campaign/export launchers are omitted because they depend on private host paths and frozen local corpus archives. `source-map.json` records those boundaries and the parameterized adapters. The optional `tools/integrate_fused.py --repo /path/to/repo --output-dir build/generated` verifies pinned upstream sources and regenerates the supplied fused files exactly.

Source licensing is Apache-2.0. Dependency notices are in `THIRD_PARTY_NOTICES.md` and `licenses/`; their own terms apply.

The original release manifest is preserved at `pins/original-release-manifest.json`; `pins/original-release-bundle.json` records the prior source archive digest. The current `manifest.json` covers this accepted source package, updated documentation and provenance. The separate archive has its own source manifest.
