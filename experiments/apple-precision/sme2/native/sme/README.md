# SME2 wrapper

This wrapper adapts the pinned KleidiAI kernel to existing symmetric INT8 bytes and scales. It does not use the upstream asymmetric quantizer. See [upstream provenance](upstream-provenance.json) for the exact commit and six unchanged source/license files.

Weights are packed as the left operand and activations as the right operand. Each scale is computed over channels for one time column, so future samples cannot alter earlier scales. The output remains `float(dot) * (weight_scale * activation_scale)` with a complete INT32 K reduction. Packed bias is negative zero to preserve signed zero under the required floating-point mode.

The runtime checks SME2 support before invoking SME instructions, checks packed and active streaming vector lengths, and preserves the upstream lazy ZA-save helper. Only the upstream C and assembly units use SME2 build flags. All buffer sizes, tails, output ownership and alias restrictions are checked by the wrapper.

For correctness-only helper checks, run `python build_check.py --output-dir /path/to/fresh/check-build` from this directory. `--compile-only` builds without running the executable. This check is separate from the integrated ORT tests in `../../check.py`.

The supplementary `check_extremes.cpp` test links the same four objects produced by that command:

```sh
check_dir=/path/to/check-build
c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off \
  check_extremes.cpp "$check_dir/wrapper.o" "$check_dir/kernel_c.o" \
  "$check_dir/kernel_asm.o" "$check_dir/common_sme.o" \
  -o "$check_dir/check_extremes"
"$check_dir/check_extremes"
```

It covers K=16384 sign extremes, concurrent immutable packed inputs, and errors. This is a documented reproduction command; its saved original result was produced separately from `build_check.py`. Neither helper collects timings.
