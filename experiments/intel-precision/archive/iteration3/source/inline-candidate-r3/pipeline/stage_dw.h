/* Isolated interior-only adaptation of fast-audiovae native/x86 DW7.
 * Input includes the chronological 6*d transformed halo. No scalar causal
 * boundary is re-entered at a tile boundary; only the final SIMD tail is scalar.
 * Caller proves disjoint buffers and gates backend through native CPUID/XCR0.
 */
#pragma once
#include <cstddef>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SP_X86 1
#endif

static inline void sp_dw_scalar(const float* x,const float* w,float b,float* y,int n,int d,int first=0) {
  for(int i=first;i<n;++i) {
    float a=x[i]*w[0];
    for(int tap=1;tap<7;++tap) a=a+x[i+tap*d]*w[tap];
    y[i]=a+b;
  }
}
#if SP_X86
#define SP_DW(NAME,ATTR,V,N,LOAD,STORE,BROAD,MUL,ADD) \
ATTR static void NAME(const float* x,const float* w,float b,float* __restrict y,int n,int d) { \
  const V w0=BROAD(w[0]),w1=BROAD(w[1]),w2=BROAD(w[2]),w3=BROAD(w[3]); \
  const V w4=BROAD(w[4]),w5=BROAD(w[5]),w6=BROAD(w[6]),bb=BROAD(b); \
  int i=0; \
  for(;i+N<=n;i+=N) { \
    V a=MUL(LOAD(x+i),w0); \
    a=ADD(a,MUL(LOAD(x+i+d),w1)); \
    a=ADD(a,MUL(LOAD(x+i+2*d),w2)); \
    a=ADD(a,MUL(LOAD(x+i+3*d),w3)); \
    a=ADD(a,MUL(LOAD(x+i+4*d),w4)); \
    a=ADD(a,MUL(LOAD(x+i+5*d),w5)); \
    a=ADD(a,MUL(LOAD(x+i+6*d),w6)); \
    STORE(y+i,ADD(a,bb)); \
  } \
  sp_dw_scalar(x,w,b,y,n,d,i); \
}
SP_DW(sp_dw_avx2,__attribute__((target("avx2,no-fma"))),__m256,8,
      _mm256_loadu_ps,_mm256_storeu_ps,_mm256_set1_ps,_mm256_mul_ps,_mm256_add_ps)
SP_DW(sp_dw_avx512,__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma"))),__m512,16,
      _mm512_loadu_ps,_mm512_storeu_ps,_mm512_set1_ps,_mm512_mul_ps,_mm512_add_ps)
#endif
static inline void sp_dw(const float* x,const float* w,float b,float* y,int n,int d,int backend) {
#if SP_X86
  if(backend==5) return sp_dw_avx512(x,w,b,y,n,d);
  if(backend==4) return sp_dw_avx2(x,w,b,y,n,d);
#else
  (void)backend;
#endif
  sp_dw_scalar(x,w,b,y,n,d);
}
