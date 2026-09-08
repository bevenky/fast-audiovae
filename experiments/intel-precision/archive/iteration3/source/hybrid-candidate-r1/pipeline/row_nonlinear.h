/* Row-local ordered FP32 Snake -> DW7 -> Snake for the Intel experiment.
 * Callers validate shape, immutable finite coefficients and native backend.
 * Link the exact pinned SLEEF 3.9.0 archive, never replacement approximations.
 * Compile with -fno-fast-math -ffp-contract=off. */
#pragma once
#include "stage_dw.h"
#include <cstring>
#if defined(__FAST_MATH__)
#error "Row nonlinear operations require fast math disabled"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
#if !SP_X86
#error "This isolated row pipeline requires x86 and a guarded AVX2/AVX512 backend"
#endif

/* Public vector ABI declarations match the accepted native implementation.
 * Symbols are provided by the pinned archive recorded in source-provenance. */
#define ROW_SSE __attribute__((target("sse2")))
#define ROW_AVX2 __attribute__((target("avx2,no-fma")))
#define ROW_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma")))
extern "C" {
ROW_SSE extern __m128 Sleef_sinf4_u10sse2(__m128);
ROW_AVX2 extern __m256 Sleef_sinf8_u10avx2(__m256);
ROW_AVX512 extern __m512 Sleef_sinf16_u10avx512f(__m512);
}

/* Every real element preserves alpha*x -> sin -> sin*sin -> reciprocal*square
 * -> x+correction. In particular, do not contract the last two operations.
 * Sine tails are 16 -> 8 -> 4 -> padded4, identical to the old native policy. */
ROW_SSE static inline void row_snake_sse_tail(const float* x,float* y,float alpha,float reciprocal,int n){
    const __m128 av=_mm_set1_ps(alpha),rv=_mm_set1_ps(reciprocal);
    int i=0;
    for(;i+4<=n;i+=4){
        const __m128 v=_mm_loadu_ps(x+i);
        const __m128 s=Sleef_sinf4_u10sse2(_mm_mul_ps(av,v));
        const __m128 square=_mm_mul_ps(s,s);
        const __m128 correction=_mm_mul_ps(rv,square);
        _mm_storeu_ps(y+i,_mm_add_ps(v,correction));
    }
    if(i<n){
        alignas(16) float scaled[4]={0.f,0.f,0.f,0.f};
        alignas(16) float sine[4];
        for(int j=0;i+j<n;++j)scaled[j]=alpha*x[i+j];
        _mm_store_ps(sine,Sleef_sinf4_u10sse2(_mm_load_ps(scaled)));
        for(int j=0;i+j<n;++j){
            const float square=sine[j]*sine[j];
            const float correction=reciprocal*square;
            y[i+j]=x[i+j]+correction;
        }
    }
}
ROW_AVX2 static inline void row_snake_avx2(const float* x,float* y,float alpha,float reciprocal,int n){
    const __m256 av=_mm256_set1_ps(alpha),rv=_mm256_set1_ps(reciprocal);
    int i=0;
    for(;i+8<=n;i+=8){
        const __m256 v=_mm256_loadu_ps(x+i);
        const __m256 s=Sleef_sinf8_u10avx2(_mm256_mul_ps(av,v));
        const __m256 square=_mm256_mul_ps(s,s);
        const __m256 correction=_mm256_mul_ps(rv,square);
        _mm256_storeu_ps(y+i,_mm256_add_ps(v,correction));
    }
    row_snake_sse_tail(x+i,y+i,alpha,reciprocal,n-i);
}
ROW_AVX512 static inline void row_snake_avx512(const float* x,float* y,float alpha,float reciprocal,int n){
    const __m512 av=_mm512_set1_ps(alpha),rv=_mm512_set1_ps(reciprocal);
    int i=0;
    for(;i+16<=n;i+=16){
        const __m512 v=_mm512_loadu_ps(x+i);
        const __m512 s=Sleef_sinf16_u10avx512f(_mm512_mul_ps(av,v));
        const __m512 square=_mm512_mul_ps(s,s);
        const __m512 correction=_mm512_mul_ps(rv,square);
        _mm512_storeu_ps(y+i,_mm512_add_ps(v,correction));
    }
    row_snake_avx2(x+i,y+i,alpha,reciprocal,n-i);
}
static inline void row_snake(const float* x,float* y,float alpha,float reciprocal,int n,int backend){
    if(backend==5)row_snake_avx512(x,y,alpha,reciprocal,n);
    else row_snake_avx2(x,y,alpha,reciprocal,n);
}

/* n is in [0,256], dilation is 1/3/9, backend is 4/5. History contains exactly
 * 6*d chronological pre-Snake samples and is owned by this segment/channel.
 * x/y/history/weights are disjoint. This internal helper does not revalidate
 * immutable parent metadata on each row. No read or write occurs for n==0. */
static inline void row_snake_dw_snake(const float* x,const float* weights,float bias,
                                    float ap,float rp,float aq,float rq,
                                    float* history,float* y,int n,int d,int backend){
    if(!n)return;
    const int halo=6*d;
    alignas(64) float transformed[54+256];
    alignas(64) float depthwise[256];
    std::memcpy(transformed,history,static_cast<size_t>(halo)*sizeof(float));
    row_snake(x,transformed+halo,ap,rp,n,backend);
    sp_dw(transformed,weights,bias,depthwise,n,d,backend);
    // This copy also preserves the chronological halo when n < 6*d.
    std::memcpy(history,transformed+n,static_cast<size_t>(halo)*sizeof(float));
    row_snake(depthwise,y,aq,rq,n,backend);
}
