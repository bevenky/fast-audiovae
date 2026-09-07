/* Portable FP32 inference kernels. Build with FP contraction and fast math off. */
#include "native_kernels.h"
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "These kernels require fast math to be disabled."
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define NCC_HAVE_NEON 1
#include <arm_neon.h>
#else
#define NCC_HAVE_NEON 0
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NCC_HAVE_X86 1
#include <immintrin.h>
#include <cpuid.h>
#define NCC_SSE_TARGET __attribute__((target("sse2")))
#define NCC_AVX_TARGET __attribute__((target("avx2,no-fma")))
#else
#define NCC_HAVE_X86 0
#endif
#if defined(__APPLE__) && defined(NCC_USE_ACCELERATE)
#define NCC_HAVE_VFORCE 1
#include <Accelerate/Accelerate.h>
#else
#define NCC_HAVE_VFORCE 0
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define NCC_TILE 256

uint32_t ncc_abi_version(void) { return 1; }

uint64_t ncc_capabilities(void) {
#if ATOMIC_INT_LOCK_FREE == 2
    /* Immutable ISA metadata only, never per-stream state. Concurrent first
     * callers may both detect features; publishing the same value is safe. */
    static atomic_uint cached = 0;
    unsigned int prior = atomic_load_explicit(&cached, memory_order_relaxed);
    if (prior) return prior;
#endif
    uint64_t flags = NCC_CAP_SCALAR;
#if NCC_HAVE_NEON
    /* Advanced SIMD is part of the AArch64 user-space ABI. */
    flags |= NCC_CAP_NEON;
#endif
#if NCC_HAVE_X86
    unsigned int a, b, c, d;
    unsigned int max_leaf = __get_cpuid_max(0, NULL);
    if (max_leaf >= 1) {
        __cpuid_count(1, 0, a, b, c, d);
        if (d & (1u << 26)) flags |= NCC_CAP_SSE2;
        /* AVX requires CPU AVX/XSAVE, OSXSAVE, and OS-enabled XMM+YMM state. */
        const unsigned int required = (1u << 26) | (1u << 27) | (1u << 28);
        if ((c & required) == required && max_leaf >= 7) {
            uint32_t xcr_low, xcr_high;
            __asm__ volatile("xgetbv" : "=a"(xcr_low), "=d"(xcr_high) : "c"(0));
            (void)xcr_high;
            if ((xcr_low & 6u) == 6u) {
                __cpuid_count(7, 0, a, b, c, d);
                if (b & (1u << 5)) flags |= NCC_CAP_AVX2;
            }
        }
    }
#endif
#ifdef _OPENMP
    flags |= NCC_CAP_OPENMP;
#endif
#if NCC_HAVE_VFORCE
    flags |= NCC_CAP_VFORCE;
#endif
#if ATOMIC_INT_LOCK_FREE == 2
    atomic_store_explicit(&cached, (unsigned int)flags, memory_order_relaxed);
#endif
    return flags;
}

int32_t ncc_selected_backend(void) {
    uint64_t c = ncc_capabilities();
    if (c & NCC_CAP_AVX2) return NCC_AVX2;
    if (c & NCC_CAP_SSE2) return NCC_SSE2;
    if (c & NCC_CAP_NEON) return NCC_NEON;
    return NCC_SCALAR;
}
int32_t ncc_backend_available(int32_t backend) {
    uint64_t c = ncc_capabilities();
    switch (backend) {
        case NCC_AUTO: case NCC_SCALAR: return 1;
        case NCC_NEON: return !!(c & NCC_CAP_NEON);
        case NCC_SSE2: return !!(c & NCC_CAP_SSE2);
        case NCC_AVX2: return !!(c & NCC_CAP_AVX2);
        default: return 0;
    }
}
const char *ncc_backend_name(int32_t backend) {
    switch (backend) {
        case NCC_AUTO: return "auto";
        case NCC_SCALAR: return "scalar";
        case NCC_NEON: return "neon";
        case NCC_SSE2: return "sse2";
        case NCC_AVX2: return "avx2";
        default: return "invalid";
    }
}
const char *ncc_status_string(int32_t s) {
    switch (s) {
        case NCC_OK: return "ok";
        case NCC_INVALID_ARGUMENT: return "invalid argument";
        case NCC_UNSUPPORTED_BACKEND: return "unsupported runtime backend";
        case NCC_OVERLAPPING_BUFFER: return "output overlaps a read buffer";
        case NCC_SIZE_OVERFLOW: return "array size or address range overflow";
        case NCC_THREADS_UNAVAILABLE: return "multiple threads require OpenMP";
        default: return "unknown status";
    }
}
const char *ncc_snake_math_name(int32_t backend) {
    if (!ncc_backend_available(backend)) return "unavailable";
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
#if NCC_HAVE_VFORCE
    if (backend != NCC_SCALAR) return "apple-vforce";
#endif
    return "scalar-libm-sinf";
}

static int checked_bytes(uint64_t count, size_t *bytes) {
    if (count > SIZE_MAX / sizeof(float)) return NCC_SIZE_OVERFLOW;
    *bytes = (size_t)count * sizeof(float);
    return NCC_OK;
}
static int multiply(uint64_t a, uint64_t b, uint64_t *result) {
    if (b && a > UINT64_MAX / b) return NCC_SIZE_OVERFLOW;
    *result = a * b;
    return NCC_OK;
}
static int valid_pointer(const void *p, size_t bytes) {
    if (!p || ((uintptr_t)p % _Alignof(float))) return NCC_INVALID_ARGUMENT;
    if (bytes > UINTPTR_MAX - (uintptr_t)p) return NCC_SIZE_OVERFLOW;
    return NCC_OK;
}
static int check_read(const float *p, size_t bytes, const float *y, size_t ybytes) {
    int status = valid_pointer(p, bytes);
    if (status) return status;
    uintptr_t a = (uintptr_t)p, b = (uintptr_t)y;
    if ((a <= b && b - a < bytes) || (b < a && a - b < ybytes))
        return NCC_OVERLAPPING_BUFFER;
    return NCC_OK;
}
static int validate_dimensions(int64_t B, int64_t C, int64_t T,
                               int32_t backend, int32_t threads,
                               uint64_t *rows, size_t *data_bytes,
                               size_t *channel_bytes) {
    if (B < 0 || C < 0 || T < 0 || threads <= 0)
        return NCC_INVALID_ARGUMENT;
    if (!ncc_backend_available(backend)) return NCC_UNSUPPORTED_BACKEND;
#ifndef _OPENMP
    if (threads > 1) return NCC_THREADS_UNAVAILABLE;
#endif
    if (!B || !C || !T) {
        *rows = 0; *data_bytes = 0; *channel_bytes = 0;
        return NCC_OK;
    }
    uint64_t count;
    int s = multiply((uint64_t)B, (uint64_t)C, rows);
    if (s) return s;
    s = multiply(*rows, (uint64_t)T, &count);
    if (s) return s;
    s = checked_bytes(count, data_bytes);
    if (s) return s;
    return checked_bytes((uint64_t)C, channel_bytes);
}

static float history_sample(const float *x, const float *history,
                            int64_t index, int64_t halo) {
    if (index >= 0) return x[index];
    return history ? history[halo + index] : 0.0f;
}
static float dw_one(const float *x, const float *history, const float *w,
                    const float *bias, int64_t t, int64_t dilation) {
    int64_t halo = 6 * dilation;
    float sum = history_sample(x, history, t - halo, halo) * w[0];
    for (int k = 1; k < 7; ++k) {
        float product = history_sample(x, history, t - (6 - k) * dilation, halo) * w[k];
        sum = sum + product;
    }
    if (bias) sum = sum + *bias;
    return sum;
}
typedef void (*dw_range_fn)(const float *, const float *, const float *,
                            const float *, float *, int64_t, int64_t, int64_t);

static void dw_scalar(const float *x, const float *history, const float *w,
                      const float *bias, float *y, int64_t start,
                      int64_t count, int64_t dilation) {
    for (int64_t i = 0; i < count; ++i)
        y[i] = dw_one(x, history, w, bias, start + i, dilation);
}

/* Independent SIMD lanes are consecutive time positions. Taps remain ordered. */
#if NCC_HAVE_NEON
static void dw_neon(const float *x, const float *history, const float *w,
                    const float *bias, float *y, int64_t start,
                    int64_t count, int64_t dilation) {
    int64_t i = 0, halo = 6 * dilation;
    for (; i < count && start + i < halo; ++i)
        y[i] = dw_one(x, history, w, bias, start + i, dilation);
    for (; i <= count - 4; i += 4) {
        const float *p = x + start + i - halo;
        float32x4_t acc = vmulq_f32(vld1q_f32(p), vdupq_n_f32(w[0]));
        for (int k = 1; k < 7; ++k) {
            float32x4_t product = vmulq_f32(vld1q_f32(p + k * dilation), vdupq_n_f32(w[k]));
            acc = vaddq_f32(acc, product);
        }
        if (bias) acc = vaddq_f32(acc, vdupq_n_f32(*bias));
        vst1q_f32(y + i, acc);
    }
    for (; i < count; ++i) y[i] = dw_one(x, history, w, bias, start + i, dilation);
}
#endif

#if NCC_HAVE_X86
NCC_SSE_TARGET
static void dw_sse2(const float *x, const float *history, const float *w,
                    const float *bias, float *y, int64_t start,
                    int64_t count, int64_t dilation) {
    int64_t i = 0, halo = 6 * dilation;
    for (; i < count && start + i < halo; ++i)
        y[i] = dw_one(x, history, w, bias, start + i, dilation);
    for (; i <= count - 4; i += 4) {
        const float *p = x + start + i - halo;
        __m128 acc = _mm_mul_ps(_mm_loadu_ps(p), _mm_set1_ps(w[0]));
        for (int k = 1; k < 7; ++k) {
            __m128 product = _mm_mul_ps(_mm_loadu_ps(p + k * dilation), _mm_set1_ps(w[k]));
            acc = _mm_add_ps(acc, product);
        }
        if (bias) acc = _mm_add_ps(acc, _mm_set1_ps(*bias));
        _mm_storeu_ps(y + i, acc);
    }
    for (; i < count; ++i) y[i] = dw_one(x, history, w, bias, start + i, dilation);
}
NCC_AVX_TARGET
static void dw_avx2(const float *x, const float *history, const float *w,
                    const float *bias, float *y, int64_t start,
                    int64_t count, int64_t dilation) {
    int64_t i = 0, halo = 6 * dilation;
    for (; i < count && start + i < halo; ++i)
        y[i] = dw_one(x, history, w, bias, start + i, dilation);
    for (; i <= count - 8; i += 8) {
        const float *p = x + start + i - halo;
        __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(p), _mm256_set1_ps(w[0]));
        for (int k = 1; k < 7; ++k) {
            __m256 product = _mm256_mul_ps(_mm256_loadu_ps(p + k * dilation), _mm256_set1_ps(w[k]));
            acc = _mm256_add_ps(acc, product);
        }
        if (bias) acc = _mm256_add_ps(acc, _mm256_set1_ps(*bias));
        _mm256_storeu_ps(y + i, acc);
    }
    for (; i < count; ++i) y[i] = dw_one(x, history, w, bias, start + i, dilation);
}
#endif

static dw_range_fn choose_dw(int32_t backend) {
#if NCC_HAVE_NEON
    if (backend == NCC_NEON) return dw_neon;
#endif
#if NCC_HAVE_X86
    if (backend == NCC_AVX2) return dw_avx2;
    if (backend == NCC_SSE2) return dw_sse2;
#endif
    return dw_scalar;
}

/* The portable fallback deliberately uses libm sine rather than an unchecked
 * short polynomial. Apple builds may use documented in-place vForce sine.
 * Scalar arithmetic and all explicit SIMD operations preserve the same order. */
typedef void (*scale_fn)(const float *, float *, float, int);
typedef void (*finish_fn)(const float *, const float *, float *, float, int);
static void scale_scalar(const float *x, float *s, float a, int n) {
    for (int i = 0; i < n; ++i) s[i] = a * x[i];
}
static void finish_scalar(const float *x, const float *s, float *y, float r, int n) {
    for (int i = 0; i < n; ++i) {
        float q = s[i] * s[i];
        float u = r * q;
        y[i] = x[i] + u;
    }
}
#if NCC_HAVE_NEON
static void scale_neon(const float *x, float *s, float a, int n) {
    int i = 0; float32x4_t av = vdupq_n_f32(a);
    for (; i <= n - 4; i += 4) vst1q_f32(s+i, vmulq_f32(av, vld1q_f32(x+i)));
    for (; i < n; ++i) s[i] = a * x[i];
}
static void finish_neon(const float *x, const float *s, float *y, float r, int n) {
    int i = 0; float32x4_t rv = vdupq_n_f32(r);
    for (; i <= n - 4; i += 4) {
        float32x4_t sv = vld1q_f32(s+i);
        float32x4_t q = vmulq_f32(sv, sv);
        float32x4_t u = vmulq_f32(rv, q);
        vst1q_f32(y+i, vaddq_f32(vld1q_f32(x+i), u));
    }
    finish_scalar(x+i, s+i, y+i, r, n-i);
}
#endif
#if NCC_HAVE_X86
NCC_SSE_TARGET
static void scale_sse2(const float *x, float *s, float a, int n) {
    int i = 0; __m128 av = _mm_set1_ps(a);
    for (; i <= n-4; i+=4) _mm_storeu_ps(s+i, _mm_mul_ps(av, _mm_loadu_ps(x+i)));
    for (; i<n; ++i) s[i] = a*x[i];
}
NCC_SSE_TARGET
static void finish_sse2(const float *x, const float *s, float *y, float r, int n) {
    int i = 0; __m128 rv = _mm_set1_ps(r);
    for (; i<=n-4; i+=4) {
        __m128 sv = _mm_loadu_ps(s+i);
        __m128 q = _mm_mul_ps(sv,sv);
        __m128 u = _mm_mul_ps(rv,q);
        _mm_storeu_ps(y+i, _mm_add_ps(_mm_loadu_ps(x+i),u));
    }
    finish_scalar(x+i,s+i,y+i,r,n-i);
}
NCC_AVX_TARGET
static void scale_avx2(const float *x, float *s, float a, int n) {
    int i = 0; __m256 av = _mm256_set1_ps(a);
    for (; i<=n-8; i+=8) _mm256_storeu_ps(s+i, _mm256_mul_ps(av,_mm256_loadu_ps(x+i)));
    for (; i<n; ++i) s[i] = a*x[i];
}
NCC_AVX_TARGET
static void finish_avx2(const float *x, const float *s, float *y, float r, int n) {
    int i = 0; __m256 rv = _mm256_set1_ps(r);
    for (; i<=n-8; i+=8) {
        __m256 sv = _mm256_loadu_ps(s+i);
        __m256 q = _mm256_mul_ps(sv,sv);
        __m256 u = _mm256_mul_ps(rv,q);
        _mm256_storeu_ps(y+i,_mm256_add_ps(_mm256_loadu_ps(x+i),u));
    }
    finish_scalar(x+i,s+i,y+i,r,n-i);
}
#endif
static void choose_snake(int backend, scale_fn *scale, finish_fn *finish) {
    *scale = scale_scalar; *finish = finish_scalar;
#if NCC_HAVE_NEON
    if (backend == NCC_NEON) { *scale = scale_neon; *finish = finish_neon; }
#endif
#if NCC_HAVE_X86
    if (backend == NCC_SSE2) { *scale = scale_sse2; *finish = finish_sse2; }
    if (backend == NCC_AVX2) { *scale = scale_avx2; *finish = finish_avx2; }
#endif
}
static void snake_tile(const float *x, float *y, float a, float r, int n,
                       int backend, scale_fn scale, finish_fn finish) {
    float sine[NCC_TILE];
    scale(x, sine, a, n);
#if NCC_HAVE_VFORCE
    if (backend != NCC_SCALAR) {
        vvsinf(sine, sine, &n);
    } else
#else
    (void)backend;
#endif
    {
        for (int i = 0; i < n; ++i) sine[i] = sinf(sine[i]);
    }
    finish(x, sine, y, r, n);
}

int32_t ncc_dw7_f32_ex(
    const float *x, const float *weights, const float *bias,
    const float *history, const float *alpha, const float *reciprocal, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads) {
    if (dilation <= 0 || (!!alpha != !!reciprocal)) return NCC_INVALID_ARGUMENT;
    uint64_t rows, weight_count, history_count;
    size_t data_bytes, channel_bytes, weight_bytes, history_bytes;
    int status = validate_dimensions(B,C,T,backend,threads,&rows,&data_bytes,&channel_bytes);
    if (status || !rows) return status;
    status = valid_pointer(y,data_bytes);
    if (status) return status;
    status = multiply((uint64_t)C,7,&weight_count);
    if (status) return status;
    status = checked_bytes(weight_count,&weight_bytes);
    if (status) return status;
    status = check_read(x,data_bytes,y,data_bytes);
    if (status) return status;
    status = check_read(weights,weight_bytes,y,data_bytes);
    if (status) return status;
    if (bias && (status=check_read(bias,channel_bytes,y,data_bytes))) return status;
    if (alpha && (status=check_read(alpha,channel_bytes,y,data_bytes))) return status;
    if (reciprocal && (status=check_read(reciprocal,channel_bytes,y,data_bytes))) return status;
    int64_t halo = 6 * (int64_t)dilation;
    if (history) {
        status = multiply(rows,(uint64_t)halo,&history_count);
        if (status) return status;
        status = checked_bytes(history_count,&history_bytes);
        if (status) return status;
        status = check_read(history,history_bytes,y,data_bytes);
        if (status) return status;
    }
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
    dw_range_fn dw = choose_dw(backend);
    scale_fn scale; finish_fn finish;
    choose_snake(backend,&scale,&finish);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t row = 0; row < (int64_t)rows; ++row) {
        int64_t c = row % C;
        const float *xr = x + row*T;
        const float *hr = history ? history + row*halo : NULL;
        const float *wr = weights + c*7;
        const float *br = bias ? bias+c : NULL;
        float *yr = y + row*T;
        if (!alpha) {
            dw(xr,hr,wr,br,yr,0,T,dilation);
        } else {
            float dw_values[NCC_TILE];
            for (int64_t t=0; t<T; t+=NCC_TILE) {
                int n = (T-t < NCC_TILE) ? (int)(T-t) : NCC_TILE;
                dw(xr,hr,wr,br,dw_values,t,n,dilation);
                snake_tile(dw_values,yr+t,alpha[c],reciprocal[c],n,backend,scale,finish);
            }
        }
    }
    return NCC_OK;
}
int32_t ncc_dw7_f32(const float *x, const float *weights, const float *bias,
                     float *y, int64_t B, int64_t C, int64_t T,
                     int32_t dilation, int32_t backend, int32_t threads) {
    return ncc_dw7_f32_ex(x,weights,bias,NULL,NULL,NULL,y,B,C,T,dilation,backend,threads);
}
int32_t ncc_dw7_snake_f32(const float *x, const float *weights, const float *bias,
                           const float *alpha, const float *reciprocal, float *y,
                           int64_t B, int64_t C, int64_t T,
                           int32_t dilation, int32_t backend, int32_t threads) {
    if (!alpha || !reciprocal) return NCC_INVALID_ARGUMENT;
    return ncc_dw7_f32_ex(x,weights,bias,NULL,alpha,reciprocal,y,B,C,T,dilation,backend,threads);
}
int32_t ncc_snake_f32(const float *x, const float *alpha, const float *reciprocal,
                       float *y, int64_t B, int64_t C, int64_t T,
                       int32_t backend, int32_t threads) {
    uint64_t rows;
    size_t data_bytes, channel_bytes;
    int status = validate_dimensions(B,C,T,backend,threads,&rows,&data_bytes,&channel_bytes);
    if (status || !rows) return status;
    status = valid_pointer(y,data_bytes);
    if (status) return status;
    status = check_read(x,data_bytes,y,data_bytes);
    if (status) return status;
    status = check_read(alpha,channel_bytes,y,data_bytes);
    if (status) return status;
    status = check_read(reciprocal,channel_bytes,y,data_bytes);
    if (status) return status;
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
    scale_fn scale; finish_fn finish;
    choose_snake(backend,&scale,&finish);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t row = 0; row < (int64_t)rows; ++row) {
        int64_t c = row%C;
        for (int64_t t=0; t<T; t+=NCC_TILE) {
            int n = (T-t < NCC_TILE) ? (int)(T-t) : NCC_TILE;
            snake_tile(x+row*T+t,y+row*T+t,alpha[c],reciprocal[c],n,backend,scale,finish);
        }
    }
    return NCC_OK;
}


int32_t ncc_snake_dw7_snake_f32(
    const float *x, const float *weights, const float *bias,
    const float *alpha_pre, const float *reciprocal_pre,
    const float *alpha_post, const float *reciprocal_post, float *y,
    int64_t B, int64_t C, int64_t T, int32_t dilation,
    int32_t backend, int32_t threads) {
    if (dilation != 1 && dilation != 3 && dilation != 9) return NCC_INVALID_ARGUMENT;
    uint64_t rows, weight_count;
    size_t data_bytes, channel_bytes, weight_bytes;
    int status = validate_dimensions(B,C,T,backend,threads,&rows,&data_bytes,&channel_bytes);
    if (status || !rows) return status;
    if ((status=valid_pointer(y,data_bytes))) return status;
    if ((status=multiply((uint64_t)C,7,&weight_count))) return status;
    if ((status=checked_bytes(weight_count,&weight_bytes))) return status;
    if ((status=check_read(x,data_bytes,y,data_bytes))) return status;
    if ((status=check_read(weights,weight_bytes,y,data_bytes))) return status;
    const float *coefficients[] = {bias, alpha_pre, reciprocal_pre, alpha_post, reciprocal_post};
    for (size_t i=0; i<5; ++i)
        if ((status=check_read(coefficients[i],channel_bytes,y,data_bytes))) return status;
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
    dw_range_fn dw = choose_dw(backend);
    scale_fn scale; finish_fn finish;
    choose_snake(backend,&scale,&finish);
    const int halo = 6*dilation;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t row=0; row<(int64_t)rows; ++row) {
        const int64_t c = row%C;
        float transformed[54+NCC_TILE], dw_values[NCC_TILE];
        for (int i=0; i<halo; ++i) transformed[i] = 0.0f;
        for (int64_t t=0; t<T; t+=NCC_TILE) {
            const int n = T-t < NCC_TILE ? (int)(T-t) : NCC_TILE;
            snake_tile(x+row*T+t,transformed+halo,alpha_pre[c],reciprocal_pre[c],n,backend,scale,finish);
            // Contiguous positive-zero/history halo lets the interior SIMD
            // path start immediately, including at every later tile boundary.
            dw(transformed,NULL,weights+7*c,bias+c,dw_values,halo,n,dilation);
            snake_tile(dw_values,y+row*T+t,alpha_post[c],reciprocal_post[c],n,backend,scale,finish);
            memmove(transformed,transformed+n,(size_t)halo*sizeof(float));
        }
    }
    return NCC_OK;
}

int32_t ncc_bias_residual_f32(
    const float *product, const float *bias, const float *skip, float *y,
    int64_t B, int64_t C, int64_t T, int32_t backend, int32_t threads) {
    uint64_t rows;
    size_t data_bytes, channel_bytes;
    int status = validate_dimensions(B,C,T,backend,threads,&rows,&data_bytes,&channel_bytes);
    if (status || !rows) return status;
    if ((status=valid_pointer(y,data_bytes))) return status;
    if ((status=check_read(product,data_bytes,y,data_bytes))) return status;
    if ((status=check_read(skip,data_bytes,y,data_bytes))) return status;
    if ((status=check_read(bias,channel_bytes,y,data_bytes))) return status;
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int64_t row=0; row<(int64_t)rows; ++row) {
        const float b = bias[row%C];
        const float *p = product+row*T, *s = skip+row*T;
        float *out = y+row*T;
        int64_t t=0;
#if NCC_HAVE_NEON
        if (backend == NCC_NEON) {
            const float32x4_t bv = vdupq_n_f32(b);
            for (; t<=T-4; t+=4) {
                const float32x4_t biased = vaddq_f32(vld1q_f32(p+t),bv);
                vst1q_f32(out+t,vaddq_f32(vld1q_f32(s+t),biased));
            }
        }
#endif
        for (; t<T; ++t) {
            const float biased = p[t]+b;
            out[t] = s[t]+biased;
        }
    }
    return NCC_OK;
}
