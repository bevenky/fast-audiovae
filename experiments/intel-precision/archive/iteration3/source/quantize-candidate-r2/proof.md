# Guarded exact-byte quantization, reduced-predicate candidate

R1 passed native exact-byte checks but was slower than division. R2 removes the per-vector normal-input and normal-product predicates. It retains the existing reciprocal plan, nearest-rounding requirement, the `|qhat| <= 128` bound, the `2^-12` exclusion band around half-integers, and masked original division on every uncertain lane. This is an unmeasured separate source candidate, not an accepted optimization.

The caller already validates finite input values while finding each column maximum. No scale calculation, clamp, integer rounding, quantized sum, scalar tail, GEMM, FP16 path or persistent state changes. The plan still requires a positive scale in [FLT_MIN, 2^126] and a normal finite reciprocal computed with actual RN division. The floating-point environment uses masked exceptions, and its rounding mode must stay unchanged within the local K loop. A non-nearest plan always uses original division. Floating-point exception flags are outside the numerical-byte contract.

## Why the removed predicates are unnecessary

Let `x` be the actual FP32 operand, or signed zero if DAZ treats a subnormal input as zero. The same operand enters both multiplication and the original division. Scale and reciprocal are normal, so DAZ cannot modify those operands.

When the exact quotient and product round normally, the usual bound remains valid even if the original input is subnormal: representable input values are exact operands, not newly rounded real values. With `u=2^-24`, `r=RN(1/s)`, `qhat=RN(x*r)` and `qref=RN(x/s)`,

```
|qhat-qref| <= |x/s| * (3u + u^2)
|qhat| <= 128 implies |x/s| <= 128/(1-u)^2
```

The difference is below 0.000023, while the excluded half-integer margin is 0.000244140625. For nonzero rounded integers the distance subtraction is exact by Sterbenz's lemma. Subtracting zero is exact unless FTZ or DAZ makes a tiny subnormal zero, which does not change the conclusion.

If either quotient or product underflows, including an FTZ flush, the positive normal reciprocal is still accurate within relative `u`. The other result is therefore also tiny, near or below the smallest normal value, and far below 0.5. Both convert to integer zero. If DAZ zeroes the input, both operations produce signed zero and again quantize to zero. No input-normality check is needed. This also covers the narrow case in which one rounded result is normal and the other subnormal.

Overflow, infinities and NaNs in the approximate result fail the ordered bound comparison and use the original division. A quotient that is normal and within 128 cannot cross an integer-rounding boundary because the whole conservative half-integer band uses original division. Identical clamping to [-127,127] preserves equality.

## Validation and build

The checker and hash-copy integrator are byte-identical to R1. The same 16 rounding/FTZ/DAZ combinations, half-integers, nextafter neighbors, guard-band edges, random finite FP32 patterns, signed-zero/subnormal/extreme cases and concurrent independent calls must pass again. R1 results do not validate R2.

```sh
c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -pthread check_guarded.cpp -o check-guarded
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./check-guarded
```

Apply `apply_to_copy.py` to an explicitly hash-pinned source with a fresh output directory, retain any other private headers, and record the new header hash in the build manifest. The prepared core retains the original public header bytes. Native preparation, workspace and whole-decoder checks remain required. Guard instructions, mask handling and reciprocal work may still outweigh the division savings; only target measurements can establish a gain.
