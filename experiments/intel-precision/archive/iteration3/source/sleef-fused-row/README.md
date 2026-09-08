# Depthwise output and post-Snake fusion

This separate Intel CPU experiment layers onto the pinned inline-SLEEF row helper. For backend5, each 16-value depthwise result stays in a vector through the accurate post-Snake. Only final output is stored. It removes the intermediate depthwise row store and reload for full vectors, while preserving seven ordered products, six additions, bias and the original Snake arithmetic.

The remaining fewer than 16 depthwise values use the existing scalar implementation, followed by the unchanged 8/4/padded4 sine grouping. The AVX2 branch is untouched. History copies the identical chronological `transformed[n:n+6*d]` interval; the fused branch copies it after post-Snake, which cannot read or modify that disjoint history buffer. The pre-Snake tile and its history storage remain necessary.

```sh
python apply_to_copy.py \
  --inline-source "$PREPARED_INLINE" \
  --source-manifest-sha256 "$PARENT_MANIFEST_SHA256" \
  --output-dir "$PREPARED_FUSED"
```

The script verifies every parent manifest artifact and copies the generated SLEEF header, wrapper and checker unchanged. Overlay its outputs into a fresh pipeline copy, preserving all other files. Retain the pinned SLEEF archive for the original tails and comparison reference. Build with `-fno-fast-math -ffp-contract=off` and the existing guarded SIMD targets.

Run the existing sine parity checker and `check_rows.cpp` with the modified row helper beside that test source. The latter covers every width 0 through 256, dilations 1/3/9, AVX2/AVX512, guard regions, repeated tiles and history cases shorter than the halo. Then require full decoder exact-waveform, repeat and causality gates before any timing.

No gain has been measured. This removes a pair of cache-local memory passes, not the depthwise products or sine calculation. Combining both computations may increase live registers and spills. Inspect the generated hot loop, then evaluate the same fixed whole-decoder workload. The original graph, runtime defaults and previous candidates remain unchanged.
