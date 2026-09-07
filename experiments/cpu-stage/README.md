# CPU stage experiments

These experiments fuse three causal residual units or a pointwise matrix multiply with its bias and residual additions. They keep the FP32 weights and interface. Matrix reduction uses explicit FMA, so validate numerical agreement with the original decoder before interpreting speed results.

This directory is separate from `fast_audiovae`, its runtime and its defaults. `source-hashes.json` records the unchanged native source revision. There are no model weights, binaries or captured recordings here.

## Build

The stage commands below target Linux x86. Run from the repository root after the normal package and native build setup. Use ONNX 1.22.0 and ONNX Runtime 1.29.0. The native x86 dependency uses SLEEF 3.9.0, commit `906ca7512ee483296780a81a21b9ca715d40dfe1`. Exact header and optional dependency pins are in [dependency-pins.json](dependency-pins.json).

```sh
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CPU_STAGE=experiments/cpu-stage
export CPU_STAGE_ORT=.deps/onnxruntime/include
export CPU_STAGE_NATIVE="$(python -c 'import json; print(json.load(open(".build/x86/build.json"))["library"])')"
python "$CPU_STAGE/matrix/build.py" --ort-include "$CPU_STAGE_ORT"
python "$CPU_STAGE/stage/build.py" --ort-include "$CPU_STAGE_ORT" \
  --native-library "$CPU_STAGE_NATIVE" --native-build-manifest .build/x86/build.json
```

Outputs and build manifests go under `experiments/cpu-stage/.build/`. Both builders accept `--output-dir`, compiler overrides and explicit dependency paths. Build manifests hash the actual linked libraries. Keep the native library at its linked location or rebuild after moving dependencies.

Direct matrix mode 0 needs no additional GEMM library. Optional modes 1 and 2 use [LIBXSMM commit 55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1](https://github.com/libxsmm/libxsmm/tree/55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1). Supply that source with `--xsmm-source` to both builders; add `--xsmm-static` to the matrix build. If its archive is missing, the matrix builder's explicit `--build-xsmm` option builds it with two jobs and installs nothing. The stage builder requires the existing static archive. Remove `LIBXSMM_TARGET`; LIBXSMM modes use ISA 0 and their own guarded dispatch.

## Rewrite and compose

Start with a fresh native graph prepared using default options. Every derivative must come from this same original. The example selects the C128 stage; it is an experiment configuration, not an automatic hardware policy.

```sh
fast-audiovae prepare --output "$CPU_STAGE/.build/original"
export CPU_STAGE_ORIGINAL="$CPU_STAGE/.build/original/decoder_native.onnx"
python -m fast_audiovae.graph.block_fusion --source "$CPU_STAGE_ORIGINAL" \
  --output "$CPU_STAGE/.build/base.onnx" --variant both --backend 5 \
  --expected-chains 18 --expected-adds 18
python "$CPU_STAGE/stage/rewrite.py" --source "$CPU_STAGE_ORIGINAL" \
  --output "$CPU_STAGE/.build/c128.onnx" --channels 128 --tile-time 128 \
  --segments 2 --backend 5 --matrix-mode 0 --matrix-isa 512
python "$CPU_STAGE/compose_candidates.py" --original "$CPU_STAGE_ORIGINAL" \
  --base "$CPU_STAGE/.build/base.onnx" --kind stage \
  --derivative "$CPU_STAGE/.build/c128.onnx" --output "$CPU_STAGE/.build/combined.onnx"
```

For matrix-only candidates, use `matrix/rewrite.py` with `--channels`, `--mode` and `--isa`. Compose with `--kind matrix`. Repeated `--candidate KIND MODEL AUDIT` arguments combine disjoint original regions, including different stage tile choices. The composer re-proves the base, checks hash-bound audits, preserves coefficients, and rejects overlaps or changed activation boundaries. Supply unfused original derivatives without `--native-chains`.

The older optional Intel MKL adapter remains separate. An existing original-derived MKL graph and its audit can be composed with `--kind mkl`; its library must also be registered for inference.

## Validate before timing

```sh
python -m unittest discover -s "$CPU_STAGE/tests" -v
python "$CPU_STAGE/matrix/bridge_probe.py" \
  --library "$CPU_STAGE/.build/matrix/libfused_pointwise.so" \
  --mode 0 --isa 512 --output "$CPU_STAGE/.build/matrix-check.json"
python "$CPU_STAGE/stage/check_stage.py" --native-library "$CPU_STAGE_NATIVE" \
  --stage-library "$CPU_STAGE/.build/stage/libstage_pipeline.so" \
  --backend 5 --matrix-isa 512 --output "$CPU_STAGE/.build/stage-check.json"
```

The unit tests are static. The two probes execute CPU correctness checks, including changing lengths and concurrent calls. LIBXSMM builds also need `matrix/check_xsmm_tiny.py`. `matrix/probe.py` and `stage/stage_probe.py` provide explicit isolated timing hooks. Stage capture currently requires a real 170-frame latent and uses two ORT workers. Compare against `--reference-fusion both` when assessing additional gains over the fused baseline.

For full decoder validation and RTF, use [the comparison harness](../../benchmarks/compare_decoders.py) and its [configuration template](../../benchmarks/example-comparison.json). Register the original native library and the stage or matrix library through `custom_libraries`, freeze hashes for all artifacts, and validate the full corpus before comparing timing. Isolated stage speedups cannot be added together or treated as decoder RTF.

The tested stage implementation requires Linux x86 with the selected AVX2 or AVX512 capabilities. Its segment count, tiles, compiler, dependencies and available CPU resources are part of the measurement. The standalone matrix core also has NEON and scalar paths, but packaging it here does not establish equivalent gains on Apple or other ARM CPUs. These experiments are not relocatable binary packages and do not change the production runtime fallback policy.
