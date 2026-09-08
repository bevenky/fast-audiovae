/* Include after the unchanged row_snake_avx2 definition and pinned inline sine.
 * Validated internal contract: n in [0,256], d in {1,3,9}; x has 6*d history
 * plus n transformed samples, and x/weights/y are disjoint. Backend5 only.
 * Compile with -fno-fast-math -ffp-contract=off. */
#pragma once
#if defined(__FAST_MATH__)
#error "Ordered depthwise and Snake require fast math disabled"
#endif

ROW_AVX512 static inline void row_dw_post_snake_avx512(
        const float* x,const float* w,float bias,float alpha,float reciprocal,
        float* y,int n,int d){
    if(!n)return;
    const __m512 av=_mm512_set1_ps(alpha),rv=_mm512_set1_ps(reciprocal);
    int i=0;
    for(;i+16<=n;i+=16){
        // Identical first multiply, six ordered multiply/add pairs and bias.
        // Do not contract these operations. SLEEF's own FMA is intentional.
        __m512 a=_mm512_mul_ps(_mm512_loadu_ps(x+i),_mm512_set1_ps(w[0]));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+d),_mm512_set1_ps(w[1])));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+2*d),_mm512_set1_ps(w[2])));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+3*d),_mm512_set1_ps(w[3])));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+4*d),_mm512_set1_ps(w[4])));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+5*d),_mm512_set1_ps(w[5])));
        a=_mm512_add_ps(a,_mm512_mul_ps(_mm512_loadu_ps(x+i+6*d),_mm512_set1_ps(w[6])));
        const __m512 value=_mm512_add_ps(a,_mm512_set1_ps(bias));
        const __m512 sine=ip3_sleef_inline_sinf16_u10avx512f(_mm512_mul_ps(av,value));
        const __m512 square=_mm512_mul_ps(sine,sine);
        const __m512 correction=_mm512_mul_ps(rv,square);
        _mm512_storeu_ps(y+i,_mm512_add_ps(value,correction));
    }
    if(i<n){
        // The original AVX512 DW uses scalar arithmetic for this remainder.
        // Translating its start index changes no operand or operation order.
        alignas(64) float tail[16];
        sp_dw_scalar(x+i,w,bias,tail,n-i,d);
        row_snake_avx2(tail,y+i,alpha,reciprocal,n-i);
    }
}
