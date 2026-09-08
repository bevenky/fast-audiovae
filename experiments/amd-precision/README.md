# AMD CPU precision experiment

The AOCL selective INT8 configuration reduced decoder time by **43.3%** against matched optimized FP32 in the completed AMD screen. Full runtime and quality scoring are complete, with small measured quality declines. This is a tested opt-in recipe. The public runtime defaults remain unchanged.

| Decoder | RTF |
| --- | ---: |
| Stock AudioVAE2 | 0.25556 |
| Optimized FP32 | 0.06056 |
| AOCL selective INT8, four workers | 0.03435 |
| Mimi | 0.05201 |

Three clips, five repetitions, two warmups, ORT 1.29.0 CPU-only, four outer workers and one AOCL inner worker. INT8 won all 15 matched pairs. The earlier oneMKL portability port was slower and is retained in `mkl/` only for reference. [Detailed results](../../docs/amd-precision.md) separate timing, runtime checks and quality.

`aocl/source/` and `common/` preserve the tested native source bytes. The new `build_aocl.py` adapter compiles all four intermediate objects directly from source, so no previous build objects or oneMKL installation are needed. The historical `aocl/build.py` remains unchanged for provenance.

From the repository root, with Python 3.11+ and supplied pinned dependencies:

```sh
python experiments/amd-precision/build_aocl.py \
  --aocl-root /path/to/aocl-build \
  --libxsmm-root /path/to/libxsmm \
  --ort-include /path/to/onnxruntime/include \
  --native-library /path/to/libfast_audiovae_x86.so \
  --output-dir .build/amd-int8
```

The AOCL root must contain `install/include` and `install/lib/libaocl-dlp.so`; LIBXSMM must contain `include` and `lib/libxsmm.a`. [Dependency pins](pins/dependencies.json) default to the measured library hashes. `--plan-only` verifies supplied inputs and prints commands without compiling. Explicit `--aocl-library-sha256`, `--libxsmm-library-sha256` and `--native-library-sha256` overrides bind rebuilt dependencies to their actual hashes. A replacement dependency or fresh output is always marked **unvalidated**; a matching source recipe does not imply the measured binary identity or speed.

Dependency sources: [pinned AOCL-DLP](https://github.com/amd/aocl-dlp/tree/c577191304a3db0029f2f12fcacbc8ad296a645d), [pinned LIBXSMM](https://github.com/libxsmm/libxsmm/tree/55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1), [ORT 1.29.0](https://github.com/microsoft/onnxruntime/tree/v1.29.0), and this repository's [native CPU sources](../../native/x86/). The [recorded dependency configuration](aocl/REBUILD.md) and [CMake cache](../../benchmarks/amd-precision/builds/aocl-dlp-CMakeCache.txt) retain the tested build settings; dependency archives and libraries are supplied separately.

The adapter has offline command/flag checks only. It has not been used for another native build or benchmark. Run `python experiments/amd-precision/tools/check_build_adapter.py` to verify its expansion of the recorded eight commands. The original executed commands and compiler identities are retained in [rebuild provenance](aocl/rebuild-provenance.json).

At runtime, this backend requires AMD Linux with CPU and OS AVX512 VNNI support. AOCL's calling-thread policy is set to one worker per GEMM callback; the experiment assumes an isolated process for that policy. All sessions must use CPUExecutionProvider, with GPU visibility disabled and library thread limits set before numerical imports. The C API keeps FP32 input/output and complete-K INT32 reductions with per-time-column activation scales.

`aocl/prepare_config.py` and `aocl/schedule.py` derive the pinned graph configuration from the recorded source. Only scheduling changes from two to four workers; no coefficients change. `mkl/check_int8.py`, `aocl/check_schedule.py` and the shared `tools/` scripts retain the measured tests. Historical checks use the original host's CPU IDs; choose valid CPU IDs on a different machine. Weights, model files, audio, prepared cases, dependency libraries and binary objects are not shipped. See [evidence provenance](../../benchmarks/amd-precision/publication.json).

Fresh scoring of 60 paired clips completed all 12 metrics without errors. PESQ changed from 3.74155 to 3.71737, STOI from 0.935986 to 0.934448 and UTMOS from 2.25690 to 2.25161. This approximation is not numerically identical to FP32; perceptual equivalence has not been established. [Quality details](../../docs/amd-precision.md#quality) describe the bandwidth and cohort limits.

To repeat the quality comparison with local source/export manifests and pinned metric assets, keep the explicit scorer location:

```sh
python experiments/amd-precision/quality/score_quality.py \
  --scorer-dir experiments/amd-precision/quality/scorer \
  --sources /path/to/source-manifest.json \
  --variant-manifest /path/to/local/quality-inputs.json \
  --baseline fast_fp32 --trim-audiovae2-right-padding \
  --assets /path/to/pinned-metric-assets \
  --output /path/to/new/comparison.json
```

The frozen wrapper verifies source and asset hashes and forces CPU scoring. Its old default scorer directory is not the packaged layout. The recorded scoring dependencies are required; predictor weights and audio are supplied separately.
