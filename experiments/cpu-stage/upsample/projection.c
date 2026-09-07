/* Raw full-K projection for the isolated stride2 stage experiment. */
#include "projection.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__FAST_MATH__)
#error "Raw projections require fast math disabled"
#endif
#if defined(UP_WITH_LIBXSMM)
#include <libxsmm.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <cpuid.h>
#define UP_X86 1
#define UP_AVX512 __attribute__((target("avx512f,avx2,fma")))
#else
#define UP_X86 0
#endif

typedef struct {
    int time, mode;
#ifdef UP_WITH_LIBXSMM
    libxsmm_gemmfunction gemm;
#endif
} up_projection_plan;

static int usable_avx512(void) {
#if UP_X86
    unsigned a,b,c,d;
    uint32_t xl,xh;
    if (!__get_cpuid(1,&a,&b,&c,&d) || !(c&bit_AVX) ||
        !(c&bit_OSXSAVE) || !(c&bit_FMA)) return 0;
    __asm__ volatile("xgetbv" : "=a"(xl),"=d"(xh) : "c"(0));
    (void)xh;
    if ((xl&0xe6)!=0xe6 || !__get_cpuid_count(7,0,&a,&b,&c,&d)) return 0;
    return (b&bit_AVX512F) && (b&bit_AVX2);
#else
    return 0;
#endif
}

#if UP_X86
/* Four output rows by 32 consecutive times. Qinput32 takes this vector path.
 * Masked 16-time tails also cover the single real prior-column seed. */
UP_AVX512 static void raw_avx512(const float *w,const float *x,float *restrict y,int t) {
    int n=0;
    for (; n+32<=t; n+=32) {
        for (int oc=0; oc<256; oc+=4) {
            __m512 a00=_mm512_setzero_ps(), a01=_mm512_setzero_ps();
            __m512 a10=_mm512_setzero_ps(), a11=_mm512_setzero_ps();
            __m512 a20=_mm512_setzero_ps(), a21=_mm512_setzero_ps();
            __m512 a30=_mm512_setzero_ps(), a31=_mm512_setzero_ps();
            for (int k=0;k<256;++k) {
                const __m512 x0=_mm512_loadu_ps(x+(size_t)k*t+n);
                const __m512 x1=_mm512_loadu_ps(x+(size_t)k*t+n+16);
                const __m512 w0=_mm512_set1_ps(w[(size_t)oc*256+k]);
                const __m512 w1=_mm512_set1_ps(w[(size_t)(oc+1)*256+k]);
                const __m512 w2=_mm512_set1_ps(w[(size_t)(oc+2)*256+k]);
                const __m512 w3=_mm512_set1_ps(w[(size_t)(oc+3)*256+k]);
                a00=_mm512_fmadd_ps(x0,w0,a00);a01=_mm512_fmadd_ps(x1,w0,a01);
                a10=_mm512_fmadd_ps(x0,w1,a10);a11=_mm512_fmadd_ps(x1,w1,a11);
                a20=_mm512_fmadd_ps(x0,w2,a20);a21=_mm512_fmadd_ps(x1,w2,a21);
                a30=_mm512_fmadd_ps(x0,w3,a30);a31=_mm512_fmadd_ps(x1,w3,a31);
            }
            _mm512_storeu_ps(y+(size_t)oc*t+n,a00);_mm512_storeu_ps(y+(size_t)oc*t+n+16,a01);
            _mm512_storeu_ps(y+(size_t)(oc+1)*t+n,a10);_mm512_storeu_ps(y+(size_t)(oc+1)*t+n+16,a11);
            _mm512_storeu_ps(y+(size_t)(oc+2)*t+n,a20);_mm512_storeu_ps(y+(size_t)(oc+2)*t+n+16,a21);
            _mm512_storeu_ps(y+(size_t)(oc+3)*t+n,a30);_mm512_storeu_ps(y+(size_t)(oc+3)*t+n+16,a31);
        }
    }
    for (;n<t;n+=16) {
        const int count=t-n<16?t-n:16;
        const __mmask16 mask=(__mmask16)((1u<<count)-1u);
        for (int oc=0;oc<256;oc+=4) {
            __m512 acc[4]={_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps()};
            for (int k=0;k<256;++k) {
                const __m512 xx=_mm512_maskz_loadu_ps(mask,x+(size_t)k*t+n);
                for (int j=0;j<4;++j)
                    acc[j]=_mm512_fmadd_ps(xx,_mm512_set1_ps(w[(size_t)(oc+j)*256+k]),acc[j]);
            }
            for (int j=0;j<4;++j)_mm512_mask_storeu_ps(y+(size_t)(oc+j)*t+n,mask,acc[j]);
        }
    }
}
#endif

int up_projection_init(void) {
    if (!usable_avx512()) return 0;
#ifdef UP_WITH_LIBXSMM
    if (getenv("LIBXSMM_TARGET")) return -1;
    libxsmm_init();
#endif
    return 512;
}

int up_projection_has_xsmm(void) {
#ifdef UP_WITH_LIBXSMM
    return 1;
#else
    return 0;
#endif
}

void *up_projection_create(int time,int mode,int isa) {
    if (time<1||time>128||(mode!=0&&mode!=1)||
        (mode==0?isa!=512:isa!=0)||!usable_avx512()) return NULL;
    up_projection_plan *p=(up_projection_plan*)calloc(1,sizeof(*p));
    if (!p)return NULL;
    p->time=time;p->mode=mode;
    if (mode==1) {
#ifdef UP_WITH_LIBXSMM
        if (getenv("LIBXSMM_TARGET")){free(p);return NULL;}
        /* Row-major Y[64,time] is column-major [time,64]. The same raw
         * full-K kernel serves four disjoint output-channel groups. */
        const libxsmm_gemm_shape shape=libxsmm_create_gemm_shape(time,64,256,time,256,time,
            LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
        p->gemm=libxsmm_dispatch_gemm(shape,LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
        if (!p->gemm){free(p);return NULL;}
#else
        free(p);return NULL;
#endif
    }
    return p;
}

static int valid_pointer(const void *p,size_t bytes) {
    const uintptr_t address=(uintptr_t)p;
    return p && !(address%sizeof(float)) && address<=UINTPTR_MAX-bytes;
}
static int overlap(const void *a,size_t an,const void *b,size_t bn) {
    const uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b;
    return aa<=bb?bb-aa<an:aa-bb<bn;
}

int up_projection_run(const void *plan,const float *w,const float *x,float *y) {
    const up_projection_plan *p=(const up_projection_plan*)plan;
    if (!p||p->time<1||p->time>128||(p->mode!=0&&p->mode!=1)) return -1;
    const size_t wb=256u*256u*sizeof(float), bytes=256u*(size_t)p->time*sizeof(float);
    if (!valid_pointer(w,wb)||!valid_pointer(x,bytes)||!valid_pointer(y,bytes)||
        overlap(y,bytes,w,wb)||overlap(y,bytes,x,bytes))return -1;
    if (p->mode==0) {
#if UP_X86
        raw_avx512(w,x,y,p->time);return 0;
#else
        return -1;
#endif
    }
#ifdef UP_WITH_LIBXSMM
    libxsmm_gemm_param args;memset(&args,0,sizeof(args));
    args.a.primary=(void*)x;
    for (int oc=0;oc<256;oc+=64) {
        args.b.primary=(void*)(w+(size_t)oc*256);
        args.c.primary=y+(size_t)oc*p->time;
        p->gemm(&args);
    }
    return 0;
#else
    return -1;
#endif
}

void up_projection_destroy(void *plan){free(plan);}
