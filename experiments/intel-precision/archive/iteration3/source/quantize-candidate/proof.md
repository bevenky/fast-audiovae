# Guarded exact-byte INT8 quantization

This candidate changes only the quotient used before the existing clamp to [-127,127] and nearest-even integer conversion. Scale calculation, the per-time-column reduction over K, quantized sums, scalar tails, GEMMs and FP16 stay unchanged. It does not preserve quotient bits. It must preserve every resulting signed INT8 byte and sum. No performance result has been measured here.

For each 16-column block, compute an ordinary FP32 division `r = RN(1/scale)` once. Eligible values use `qhat = RN(value*r)`. Every uncertain lane uses the original masked FP32 `value/scale` division instead. A zero fast mask performs the original full division.

Fast lanes require all of the following:

- MXCSR rounding is nearest when the local plan is created and remains unchanged while the plan is used. The current K loop makes no intervening external calls.
- Scale is positive and in [FLT_MIN, 2^126], and the computed reciprocal is normal and finite.
- The original value is normal or an exact signed zero, identified by integer bits.
- `qhat` is normal with magnitude at most 128, or both the value and `qhat` are exact zeros.
- The distance from `qhat` to its nearest integer is strictly less than `0.5 - 2^-12`. Ties and the entire conservative exclusion band use division.

Let `u=2^-24` and `q=value/scale` in real arithmetic. For normal results, reciprocal and multiplication give `qhat=q(1+d1)(1+d2)` with `|d1|,|d2|<=u`. Direct rounded division gives `qref=q(1+d3)`, so `|qhat-qref| <= |q|(3u+u^2)`. The guard `|qhat|<=128` bounds `|q| <= 128/(1-u)^2`, giving a discrepancy below 0.000023. This is much smaller than the excluded 0.000244140625 half-integer margin. The two rounded quotients therefore cannot land on opposite integer-rounding regions. Clamping both to [-127,127] cannot introduce a disagreement.

The distance calculation is exact: subtraction from the nearest integer satisfies Sterbenz's condition when the rounded integer is nonzero, and subtracting zero is exact. If the reference quotient is subnormal near the smallest normal boundary, both quotients are far below 0.5 and convert to integer zero. Overflow, subnormal input, subnormal approximate product, invalid scale, and non-nearest modes take exact division. Integer-bit classification prevents DAZ from mistaking subnormal inputs for true zero. FTZ/DAZ do not alter any accepted nonzero normal operands or products. The contract covers values and quantized bytes, not floating-point sticky exception flags.

`apply_to_copy.py` checks explicit source/header hashes and three unique anchors, then creates a separate core and provenance file. It can be applied to a hash-pinned combined VNNI derivative without modifying the original. Any other private headers referenced by that derivative must also accompany its build.

```sh
c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -pthread check_guarded.cpp -o check-guarded
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./check-guarded
```

The checker compares original hardware division against the candidate through the identical clamp and integer conversion. It covers half-integers, nextafter neighbors, both exclusion-band edges, random finite FP32 inputs and scales, signed zero, subnormals, extremes, all four rounding modes, all four FTZ/DAZ settings, and concurrent independent calls. A skipped CPU is not a passing native validation. Build provenance must include this header. The full native preparation and whole-decoder exact-waveform checks remain mandatory before timing or acceptance.
