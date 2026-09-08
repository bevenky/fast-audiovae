# Small VNNI workspace experiment

This isolated candidate replaces eligible late INT8 workspace GEMMs with a direct AVX512 VNNI loop. It does not change quantization, model coefficients, the selected INT8 layers, or ordinary standalone GEMMs. No speed or waveform result has been measured for this candidate.

The accepted preparer still produces signed activation bytes and one activation scale per time column across K. The new preparer interleaves those existing bytes once into `[K/4,T,4]`. Each kernel call broadcasts four existing unsigned weight bytes, keeps four or eight output rows in INT32 registers, and emits 16 time positions per vector. The exact arithmetic order is:

```
integer = sum(U8_weight * S8_activation) - 128 * sum(S8_activation)
scale   = weight_scale * column_scale
product = float(integer) * scale
output  = skip + (product + bias)  // when both epilogue inputs are present
```

K is complete before conversion to FP32. The maximum supported K is 256, so even the uncompensated integer sum cannot overflow INT32. There is no saturating INT16 intermediate, split K, FMA, approximate sine, additional quantization, or internal thread pool. Source weight bytes remain row-major and immutable.

Dispatch requires backend 1, sequential oneMKL availability, independently checked CPU and OS AVX512F/BW/VNNI support, M and K between 4 and 256 in multiples of 4, T between 16 and 256 in multiples of 16, and output row boundaries in multiples of 4. Every other workspace geometry retains its existing oneMKL path, including previous-state seeds of T=1. Extra packed activation storage is at most 64 KiB per workspace. It is allocated once, guarded by the existing workspace busy flag, and invalidated before every preparation.

## Prepare and check

`apply_to_copy.py` requires explicit SHA256 pins and a fresh output directory. It checks seven unique source anchors, preserves the public header byte-for-byte, writes a separate core copy, and records source and output hashes. It never edits the input core or builds a library.

```sh
python3 apply_to_copy.py \
  --source ../core/precision.cpp --source-sha256 CORE_SHA256 \
  --header ../core/precision.h --header-sha256 HEADER_SHA256 \
  --output-dir source-r1
```

The standalone check needs GCC or Clang with AVX512 VNNI intrinsic support. Compile on the target CPU host using these semantics, without `-march=native` or fast math:

```sh
c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -pthread \
  check_direct.cpp -o check-direct
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./check-direct
```

This command only checks native arithmetic and dispatch. It does not run models or collect timings. The checker covers packed byte order, scalar INT64 reference dots, bitwise FP32 raw and residual outputs, both register block sizes, partial row intervals, signed zero, cancellation, unsupported geometry and ISA flags, overlap rejection, and independent concurrent calls. A machine without usable VNNI reports `skipped`, which is not a passing target check.

For the full candidate, use the existing iteration3 core build commands on the copied `precision.cpp`, with `-DIP_ITERATION3_NAMESPACE=1`, hidden symbols, `-Wl,-Bsymbolic-functions`, and the same pinned sequential CPU oneMKL 2026.1.0 headers and libraries. The copied header must be beside the copied source. Record `direct_vnni.h` in the new build manifest as an additional input. Re-link the iteration3 bridge and pipeline to the new core; changing only a filename or retaining a dependency on the first candidate core is insufficient. Register the existing distinct iteration3 ORT domains and check loaded library hashes. An optional `-DIP3_VNNI_ROWS=4` changes only the integer register blocking and must receive its own build provenance.

Run the existing native workspace checker against the new core, then `validate_candidate.py --screen`. Only the complete 60-clip parity and paired 10-clip campaign can establish acceptance. Candidate 1's result does not validate this kernel. Its generated config must use relative artifact paths, as required by the public harness.

## Source-derived exposure

For the unchanged two-segment schedule at latent length 255, C256 uses Q256 and the C128 upsample stage uses Q128. The counts below describe source execution geometry, not a measured speedup:

| Region | Workspace GEMM calls | Eligible direct calls | Fallback calls |
|---|---:|---:|---:|
| C256 three residual units | 720 | 714 | 6 |
| C128 two projections and three residual units | 4,791 | 4,783 | 8 |
| Total late workspaces | 5,511 | 5,497 | 14 |

C256 segments each contain 119 full tiles and one tail, of lengths 136 and 214. C128 segments each contain 478 full tiles and one tail, of lengths 16 and 94. Each C128 tile has five GEMMs. Its second segment also has one previous-projection seed at T=1. Only the three residual GEMMs of its 16-position tail meet the direct guard; the two projections have T=8.

There are 4,553 workspace preparations because each C128 projection pair shares one preparation. Of those, 4,541 meet the packing guard. Another 14 ordinary preparations and 1,248 standalone GEMM calls remain unchanged. These totals reproduce the existing instrumentation counts of 4,567 preparations, 5,539 row calls, and 6,759 GEMMs. The direct path covers 81.33% of the GEMM call count, which is not 81.33% of compute time. Changing Q or segments changes these counts.

The possible gain is avoiding repeated generic small GEMM dispatch, packing and INT32 output scratch in the late stages. Packing activations, finite checks, cache behavior, compiler register spills, and oneMKL's existing VNNI kernels may eliminate that advantage. Target native checks, whole-waveform parity, causality gates and paired wall timing remain required.
