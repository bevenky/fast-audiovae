# Inline accurate sine experiment

This isolated Intel CPU candidate removes the external call around the existing 16-lane SLEEF u10 sine. It uses an upstream-generated SLEEF 3.9.0 header, with no copied polynomial or replacement approximation. The original AVX2, SSE2 and padded tails, Snake operation order, histories and CPU guards stay unchanged. No speed or numerical-parity result has been established for this candidate.

Generate headers from the same verified source archive as the existing dependency, using the same GCC version in a fresh build directory. Preserve the original installation and archive. The source archive SHA256 is `af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908`, tag `3.9.0`, commit `906ca7512ee483296780a81a21b9ca715d40dfe1`.

```sh
cmake -S "$SLEEF_SOURCE" -B "$INLINE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSLEEF_BUILD_INLINE_HEADERS=ON -DSLEEF_ENABLE_LTO=OFF \
  -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_QUAD=OFF \
  -DSLEEF_BUILD_GNUABI_LIBS=OFF -DSLEEF_BUILD_TESTS=OFF \
  -DSLEEF_ENABLE_TESTER4=OFF -DSLEEF_ENABLE_TLFLOAT=OFF \
  -DSLEEF_DISABLE_OPENMP=ON -DSLEEF_ENABLE_CUDA=OFF \
  -DSLEEF_BUILD_BENCH=OFF
cmake --build "$INLINE_BUILD" --target inline_headers_util --parallel 2
```

Record the generation command, compiler, source-tree/archive hash and generated-header hash. `include/sleefinline_avx512f.h` must be the ordinary AVX512F version, whose explicit internal FMA operations match the existing symbol. The NOFMA and deterministic variants are different algorithms. The preparation script verifies pinned inputs and two exact source anchors; header text alone cannot prove its generation history.

```sh
python apply_to_copy.py \
  --row-source "$BASE_PIPELINE/row_nonlinear.h" \
  --row-sha256 "$ROW_SHA256" \
  --generated-header "$INLINE_BUILD/include/sleefinline_avx512f.h" \
  --generated-header-sha256 "$HEADER_SHA256" \
  --sleef-source-archive "$SLEEF_SOURCE_ARCHIVE" \
  --output-dir "$PREPARED"
```

Overlay the four prepared files into a fresh copy of the candidate pipeline. Keep all other sources byte-identical. The wrapper scopes generated definitions with GCC target push/pop, privately renames the sine, and requests full inlining through SLEEF's supported macro. Do not enable AVX512 for the whole generic translation unit. Keep linking the original pinned archive for unchanged tails and the comparison reference.

```sh
g++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -pthread \
  "$PREPARED/check_sine.cpp" "$PINNED_SLEEF_ARCHIVE" -lm \
  -o "$PREPARED/check-sine"
"$PREPARED/check-sine"
```

This test compares uint32 output bits for finite inputs, signed zero, subnormals, extremes, mixed vectors and both sides of the 125.0 range-reduction boundary. It enumerates 16 rounding/FTZ/DAZ combinations with masked exceptions and concurrent independent calls. A skipped CPU is not a pass. Repeat the existing `check_rows.cpp` against the modified helper, then full decoder waveform, repeat and causality checks. Compile provenance must include both new headers. Floating exception flags are outside the parity contract.

Inspect final disassembly before timing: the hot 16-lane Snake path should have no external u10 sine call, and the saved vector/register traffic should actually decrease. A compiler can trade the eliminated call for register pressure or instruction-cache growth. The unchanged arithmetic still costs time. Test one fixed whole-decoder comparison only after correctness passes; accept only the required measured gain.

The current four late residual stages perform approximately 36.864 million sine evaluations per output second, or 2.304 million 16-lane calls before segment warm work and tails. That establishes repeated-call opportunity, not a promised RTF saving. Their saved profile mixes sine, convolution, matrix work and memory operations; it does not isolate call overhead.

Upstream references: [header generation](https://github.com/shibatch/sleef/blob/3.9.0/src/libm/CMakeLists.txt#L500), [inline and LTO instructions](https://github.com/shibatch/sleef/blob/3.9.0/docs/compile.xhtml#L117), [u10 sine source](https://github.com/shibatch/sleef/blob/3.9.0/src/libm/sleefsimdsp.c#L975), [AVX512 FMA helpers](https://github.com/shibatch/sleef/blob/3.9.0/src/arch/helperavx512f.h#L383). Generated upstream material retains its Boost Software License notices.
