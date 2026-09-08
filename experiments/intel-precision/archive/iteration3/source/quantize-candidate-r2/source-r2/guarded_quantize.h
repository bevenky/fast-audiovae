#ifndef FAST_AUDIO_GUARDED_QUANTIZE_H
#define FAST_AUDIO_GUARDED_QUANTIZE_H

// Experimental CPU-only quotient for the existing [-127,127] RN-even
// quantizer. The quotient need not match; the quantized integer must match.
// Compile with -fno-fast-math -ffp-contract=off. No approximate reciprocal.
// R2 removes redundant normal input/product predicates. Subnormal outcomes
// stay far below half an integer; DAZ applies equally to both input paths.
#include <cstdint>
#if defined(__FAST_MATH__)
#error "Guarded quantization requires ordinary FP32 division and multiplication"
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#define IP3_GQ_X86 1
#define IP3_GQ_TARGET __attribute__((target("avx512f,avx512bw")))
#else
#define IP3_GQ_X86 0
#endif

namespace ip3_guarded_quantize {

static inline bool host_available() {
#if IP3_GQ_X86
    unsigned a,b,c,d;uint32_t lo,hi;
    if(!__get_cpuid(1,&a,&b,&c,&d)||!(c&bit_AVX)||!(c&bit_OSXSAVE))return false;
    __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));(void)hi;
    if((lo&0xe6)!=0xe6||!__get_cpuid_count(7,0,&a,&b,&c,&d))return false;
    return (b&(1u<<16))&&(b&(1u<<30));
#else
    return false;
#endif
}

#if IP3_GQ_X86
struct Plan {
    __m512 scale;
    __m512 reciprocal;
    __mmask16 safe_scale;
};

IP3_GQ_TARGET
static inline Plan make_plan(__m512 scale) {
    Plan result{scale,_mm512_setzero_ps(),0};
    // Directed rounding takes the unchanged divide path. The normal reciprocal
    // and unchanged input operand keep FTZ/DAZ integer-rounding behavior safe.
    if((_mm_getcsr()&_MM_ROUND_MASK)!=_MM_ROUND_NEAREST)return result;
    const __m512i bits=_mm512_castps_si512(scale);
    // [FLT_MIN, 2^126] ensures positive normal scales and a normal finite
    // reciprocal. Integer classification cannot misread subnormals under DAZ.
    result.safe_scale=_mm512_cmp_epu32_mask(bits,_mm512_set1_epi32(0x00800000),_MM_CMPINT_GE)
        &_mm512_cmp_epu32_mask(bits,_mm512_set1_epi32(0x7e800000),_MM_CMPINT_LE);
    if(!result.safe_scale)return result;
    result.reciprocal=_mm512_mask_div_ps(result.reciprocal,result.safe_scale,
        _mm512_set1_ps(1.f),scale);
    const __m512i reciprocal_bits=_mm512_castps_si512(result.reciprocal);
    result.safe_scale&=_mm512_cmp_epu32_mask(reciprocal_bits,_mm512_set1_epi32(0x00800000),_MM_CMPINT_GE)
        &_mm512_cmp_epu32_mask(reciprocal_bits,_mm512_set1_epi32(0x7f800000),_MM_CMPINT_LT);
    return result;
}

// fast_lanes is optional diagnostics for the standalone checker. Production
// omits it. No mutable state, allocation, quantization scale, or history here.
IP3_GQ_TARGET
__attribute__((always_inline)) static inline __m512 quotient(
        __m512 value,const Plan& plan,uint16_t* fast_lanes=nullptr) {
    if(!plan.safe_scale){
        if(fast_lanes)*fast_lanes=0;
        return _mm512_div_ps(value,plan.scale);
    }
    const __m512 approximate=_mm512_mul_ps(value,plan.reciprocal);
    const __m512i abs_mask=_mm512_set1_epi32(0x7fffffff);
    const __m512i quotient_bits=_mm512_and_si512(_mm512_castps_si512(approximate),abs_mask);
    const __m512 absolute=_mm512_castsi512_ps(quotient_bits);
    __mmask16 fast=plan.safe_scale
        &_mm512_cmp_ps_mask(absolute,_mm512_set1_ps(128.f),_CMP_LE_OQ);
    // For |q|<=128, this difference from the nearest integer is exact in FP32.
    // The exclusion band is over ten times the reciprocal-product error bound.
    const __m512 integer=_mm512_roundscale_ps(absolute,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    const __m512 difference=_mm512_castsi512_ps(_mm512_and_si512(
        _mm512_castps_si512(_mm512_sub_ps(absolute,integer)),abs_mask));
    fast&=_mm512_cmp_ps_mask(difference,_mm512_set1_ps(0.5f-0x1p-12f),_CMP_LT_OQ);
    if(fast_lanes)*fast_lanes=fast;
    const __mmask16 fallback=static_cast<__mmask16>(~fast);
    if(!fallback)return approximate;
    return _mm512_mask_div_ps(approximate,fallback,value,plan.scale);
}
#endif
}  // namespace ip3_guarded_quantize
#endif
