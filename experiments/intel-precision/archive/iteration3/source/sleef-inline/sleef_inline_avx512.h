/* Include only the upstream-generated, hash-pinned SLEEF 3.9.0 AVX512F header.
 * The caller keeps the existing CPU/OS guard and the external 8/4-lane tails.
 * Compile with -fno-fast-math -ffp-contract=off. */
#pragma once
#if !defined(__GNUC__) || defined(__clang__) || !defined(__x86_64__)
#error "This isolated inline experiment requires GCC on x86_64"
#endif
#if defined(__FAST_MATH__)
#error "Inline SLEEF requires fast math disabled"
#endif
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

/* Public generated functions default to static inline. SLEEF supports this
 * override so that the complete sine body can inline into the Snake loop.
 * Rename its definition to retain the pinned external symbol as a reference.
 * The upstream polynomial and its explicit internal FMA intrinsics are intact. */
#pragma push_macro("SLEEF_INLINE")
#pragma push_macro("SLEEF_ALWAYS_INLINE")
#pragma push_macro("Sleef_sinf16_u10avx512f")
#undef SLEEF_INLINE
#undef SLEEF_ALWAYS_INLINE
#define SLEEF_INLINE static inline __attribute__((always_inline))
#define SLEEF_ALWAYS_INLINE inline __attribute__((always_inline))
#define Sleef_sinf16_u10avx512f ip3_sleef_inline_sinf16_u10avx512f
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma")
#pragma GCC diagnostic push
/* GCC C++ does not implement the generated STDC FP_CONTRACT pragma. The
 * mandatory command-line -ffp-contract=off supplies its intended behavior. */
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
/* The generated all-functions header contains helpers unused by this sine-only
 * translation unit. Keep these exceptions scoped to upstream definitions. */
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "sleefinline_avx512f.h"
#pragma GCC diagnostic pop
#pragma GCC pop_options
#pragma pop_macro("Sleef_sinf16_u10avx512f")
#pragma pop_macro("SLEEF_ALWAYS_INLINE")
#pragma pop_macro("SLEEF_INLINE")
