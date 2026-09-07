# Upsampling and residual-stage experiment

This fuses the fourth upsampling projection pair, phase merge and three C128 residual units into one CPU operator. The boundary is `[B,256,T]` to `[B,128,2T]`. It preserves the original FP32 weights and leaves runtime defaults unchanged.

The two projections retain separate complete-K reductions. Phase output remains `(current + previous) + bias`; residual additions retain their original order. Local tiles carry the previous projection and causal histories. Internal segments replay and discard the required prefix, seeding the preceding projection from real input. Global start uses positive zero. Matrix rounding can differ from ORT, so numerical validation remains required.

## Measured region result

On an Intel Xeon Platinum 8280 VM, two CPU threads and two segments:

| Implementation | Median region time |
|---|---:|
| Existing combined decoder's four-node region | 301.249 ms |
| LIBXSMM projections, 128-position output tiles | 267.188 ms |

That is **11.3% less region time**, measured over seven shuffled repetitions after two warmups on one captured speech activation representing 6.8 seconds. Only the production final output was timed. Direct AVX512 projection variants were slower. Intermediate and synthetic correctness checks passed; **whole-decoder RTF and full-corpus validation are pending**. This result does not establish an 11.3% decoder improvement or a memory saving.

## Build

Run from the repository root with ONNX 1.22.0, ONNX Runtime 1.29.0 and the normal native x86 build available. Linux x86 with the required AVX512 CPU/OS capabilities is required. Set `CPU_UP_XSMM` to an existing [pinned LIBXSMM checkout](../dependency-pins.json) containing `lib/libxsmm.a`. The builder downloads nothing, checks ORT header and native-library hashes, and records build dependencies.

```sh
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LIBXSMM_TARGET
export CPU_UP=experiments/cpu-stage/upsample
export CPU_UP_OUT=experiments/cpu-stage/.build/upsample-run
export CPU_UP_NATIVE="$(python -c 'import json; print(json.load(open(".build/x86/build.json"))["library"])')"
python "$CPU_UP/build.py" --output-dir "$CPU_UP_OUT" \
  --ort-include .deps/onnxruntime/include \
  --native-library "$CPU_UP_NATIVE" \
  --native-build-manifest .build/x86/build.json \
  --xsmm-source "$CPU_UP_XSMM"
```

Use a fresh output directory. Direct-only builds may omit `--xsmm-source`.

## Rewrite and check

Set `CPU_UP_SOURCE` to the existing combined stage/MKL graph described in the [stage experiment guide](../README.md). A fresh stock or native-only graph does not contain the required C128 `StageStackF32` region.

```sh
python "$CPU_UP/rewrite.py" --source "$CPU_UP_SOURCE" \
  --output "$CPU_UP_OUT/decoder.onnx" \
  --tile-time 128 --segments 2 --projection-mode 1 --projection-isa 0
python "$CPU_UP/check.py" --native-library "$CPU_UP_NATIVE" \
  --library "$CPU_UP_OUT/libupsample_stage.so" --mode 1 \
  --output "$CPU_UP_OUT/checks.json"
python "$CPU_UP/check_edges.py" --native-library "$CPU_UP_NATIVE" \
  --library "$CPU_UP_OUT/libupsample_stage.so" --mode 1 \
  --output "$CPU_UP_OUT/edges.json"
```

The checks require CPUs 0 and 1 and execute correctness only. They cover intermediate outputs, odd segment boundaries, tails, causal prefixes, repeated/concurrent calls and malformed inputs. Mode 0 tests direct projections; pair it with `--projection-mode 0 --projection-isa 512` when rewriting.

The rewrite preserves coefficients and external data, rejects ambiguous shapes or extra consumers of internal outputs, and writes a separate audit. For full-decoder validation, register the original graph's native, stage and MKL libraries **plus** `libupsample_stage.so` in the [comparison harness configuration](../../../benchmarks/example-comparison.json). Preserve artifact hashes and validate before timing. Existing libraries remain dependencies; this is not a standalone binary package.
