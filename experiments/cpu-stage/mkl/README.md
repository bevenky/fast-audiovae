# Optional Intel sequential MKL adapter

This retains the early-matrix adapter used with the stage experiment. Its C++ source and fixed graph rewriter are unchanged. The rewriter accepts only the recorded original native graph, replaces exactly 12 matrices across five shapes, and creates separate one-worker and two-worker graphs. It uses dynamic-time plain FP32 SGEMM with the complete reduction, original weights, alpha 1 and beta 0.

## Dependencies and build

Provide existing Linux x86-64 oneMKL **2026.1.0** headers and CPU libraries plus ONNX Runtime **1.29.0** headers. [dependency-pins.json](dependency-pins.json) records the exact official package, header and library hashes. Nothing is downloaded or installed by these scripts, and no Intel runtime binaries are included.

The direct link contains only `libmkl_intel_lp64.so.3`, `libmkl_sequential.so.3` and `libmkl_core.so.3`. The original VM dependency set also contained BLAS and VML dispatch libraries for `def`, `mc3`, `avx2` and `avx512`, eleven CPU libraries in total. The measured Xeon Platinum 8280 VM selected the AVX512 libraries. This subset omitted the newer AVX10 dispatch library and is not a general Intel redistribution bundle. Supply a complete supported Intel distribution when deploying elsewhere and revalidate that machine.

From the repository root, using the CPU-only environment described in [the experiment guide](../README.md):

```sh
python experiments/cpu-stage/mkl/build_decoder.py \
  --ort-include .deps/onnxruntime/include \
  --mkl-include "$CPU_MKL_INCLUDE" --mkl-library-dir "$CPU_MKL_LIBRARIES" \
  --output-dir experiments/cpu-stage/.build/mkl
```

Set the two `CPU_MKL_*` variables to your existing dependency directories. The builder checks the pinned CPU files, links the sequential LP64 layers, and records hashes of the actual dependencies and output. Its runtime search path points at the supplied library directory; keep that directory available or rebuild. ORT owns the workers, and the native constructor requires one BLAS thread. Do not substitute an OpenMP, TBB, SYCL or GPU threading/runtime layer.

## Rewrite and compose

```sh
python experiments/cpu-stage/mkl/prepare_decoder.py \
  --source "$CPU_STAGE_ORIGINAL" --threads 2 \
  --output experiments/cpu-stage/.build/mkl/decoder_2t.onnx
python experiments/cpu-stage/compose_candidates.py \
  --original "$CPU_STAGE_ORIGINAL" --base experiments/cpu-stage/.build/base.onnx \
  --candidate mkl experiments/cpu-stage/.build/mkl/decoder_2t.onnx experiments/cpu-stage/.build/mkl/decoder_2t.json \
  --candidate stage experiments/cpu-stage/.build/c128.onnx experiments/cpu-stage/.build/c128.stage.json \
  --output experiments/cpu-stage/.build/stage_mkl.onnx
```

Every candidate must derive from the same original. Add other disjoint stage derivatives through repeated `--candidate` arguments. Register the base native, stage and MKL libraries in the comparison harness's `custom_libraries` list, and match ORT's thread count to `--threads` in the MKL graph. Neither this adapter nor the composed graph enters the package's default loader.

[reference-hashes.json](reference-hashes.json) records the original and both expected derivative hashes. Static reproduction with your pinned graph runs no inference:

```sh
FAST_AUDIOVAE_NATIVE_SOURCE="$CPU_STAGE_ORIGINAL" \
  python -m unittest discover -s experiments/cpu-stage/tests -p test_mkl.py -v
```

These checks establish graph identity. Follow with the full decoder's numerical, causal and multilingual validation before timing. This optional adapter adds copied weights and MKL runtime memory; a stage microbenchmark cannot establish its full-decoder benefit.
