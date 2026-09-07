#ifndef NCC_NATIVE_KERNELS_H
#define NCC_NATIVE_KERNELS_H

#include <stdint.h>

#if defined(_WIN32)
#define NCC_API __declspec(dllexport)
#else
#define NCC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ncc_backend {
    NCC_AUTO = 0, NCC_SCALAR = 1, NCC_NEON = 2, NCC_SSE2 = 3, NCC_AVX2 = 4
};
enum ncc_status {
    NCC_OK = 0, NCC_INVALID_ARGUMENT = 1, NCC_UNSUPPORTED_BACKEND = 2,
    NCC_OVERLAPPING_BUFFER = 3, NCC_SIZE_OVERFLOW = 4,
    NCC_THREADS_UNAVAILABLE = 5
};
enum ncc_capability {
    NCC_CAP_SCALAR = 1, NCC_CAP_NEON = 2, NCC_CAP_SSE2 = 4,
    NCC_CAP_AVX2 = 8, NCC_CAP_OPENMP = 16, NCC_CAP_VFORCE = 32
};

NCC_API uint32_t ncc_abi_version(void); /* Currently 1. */
NCC_API uint64_t ncc_capabilities(void); /* Runtime-usable capabilities. */
NCC_API int32_t ncc_backend_available(int32_t backend);
NCC_API int32_t ncc_selected_backend(void);
NCC_API const char *ncc_backend_name(int32_t backend);
NCC_API const char *ncc_status_string(int32_t status);
NCC_API const char *ncc_snake_math_name(int32_t backend);

/* All arrays are contiguous FP32. x/y: [B,C,T], weights: [C,7], bias: [C].
 * Zero causal history. Output shape equals input shape. Any positive dilation
 * is accepted. Arithmetic order: tap 0 product, additions of taps 1..6, bias.
 * No read buffer may overlap y. Float pointers must have 4-byte alignment.
 * B/C/T may be zero (no-op; pointers may be NULL), but cannot be negative.
 * backend and threads are still validated for a no-op. threads must be >0;
 * requesting >1 requires a build with OpenMP. No mutable global stream state.
 */
NCC_API int32_t ncc_dw7_f32(
    const float *x, const float *weights, const float *bias, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads);

/* Optional history: [B,C,6*dilation], chronological oldest to newest.
 * NULL history means zero history. History is read-only, never updated here.
 * alpha/reciprocal: both NULL for DW only, or both [C] for fused DW -> Snake.
 * Snake order: a*x, sin, square, reciprocal*square, x+correction.
 * reciprocal must be supplied already computed; the kernel does not change
 * denominator, alpha, precision, or normalization. This is inference only.
 * Fused execution uses bounded scratch, without a full DW output tensor.
 */
NCC_API int32_t ncc_dw7_f32_ex(
    const float *x, const float *weights, const float *bias,
    const float *history, const float *alpha, const float *reciprocal, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads);

NCC_API int32_t ncc_dw7_snake_f32(
    const float *x, const float *weights, const float *bias,
    const float *alpha, const float *reciprocal, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads);

/* Forced SCALAR uses scalar sinf. Other available backends use vForce on
 * Accelerate-enabled Apple builds, otherwise scalar sinf with vectorized
 * surrounding arithmetic. Sine library changes can change last-bit rounding.
 * SIMD names describe the arithmetic ISA; they do not promise SIMD sine.
 */
NCC_API int32_t ncc_snake_f32(
    const float *x, const float *alpha, const float *reciprocal, float *y,
    int64_t B, int64_t C, int64_t T, int32_t backend, int32_t threads);

/* Local-history pre-Snake -> DW7 -> post-Snake, dilations 1, 3 or 9.
 * Coefficients are mandatory [C]; weights [C,7], x/y [B,C,T]. No read
 * buffer may overlap y. Zero-sized calls allow NULL pointers, but still
 * validate dilation/backend/threads. History is private to this call.
 */
NCC_API int32_t ncc_snake_dw7_snake_f32(
    const float *x, const float *weights, const float *bias,
    const float *alpha_pre, const float *reciprocal_pre,
    const float *alpha_post, const float *reciprocal_post, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads);

/* Exactly skip + (product + bias), with no reassociation. Same pointer,
 * shape and threading requirements as ncc_snake_f32; read inputs may alias.
 */
NCC_API int32_t ncc_bias_residual_f32(
    const float *product, const float *bias, const float *skip, float *y,
    int64_t B, int64_t C, int64_t T, int32_t backend, int32_t threads);

#ifdef __cplusplus
}
#endif
#endif
