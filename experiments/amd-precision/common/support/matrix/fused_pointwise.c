/* Isolated CPU experiment. W[C,K] X[K,T] -> (dot + bias[C]) + skip[C,T].
 * Every dot completes its full K reduction before either addition. FP32 only.
 * Explicit FMA inside the dot can differ from MLAS in rounding. No split K,
 * quantization, activation transpose, copied weights or internal threadpool.
 * Callers partition disjoint [first_block,last_block) ranges on one shared plan.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#if defined(FX_WITH_LIBXSMM)
#include <libxsmm.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#include <cpuid.h>
#define FX_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef void (*fx_direct)(int,int,int,int,int,const float*,const float*,const float*,const float*,float*,int);
typedef struct {
  int c,k,t,tt,tc,mode,isa,skip_first;
  fx_direct direct;
#ifdef FX_WITH_LIBXSMM
  libxsmm_gemmfunction gemm[2][2];
#endif
} fx_plan;

static void scalar(int c,int k,int t,int begin,int end,const float*w,const float*x,const float*b,const float*s,float*y,int skip_first) {
  for(int oc=0;oc<c;++oc) for(int n=begin;n<end;++n) {
    float v=0.0f;
    for(int ic=0;ic<k;++ic) v=fmaf(w[(size_t)oc*k+ic],x[(size_t)ic*t+n],v);
    v=v+b[oc];
    if(s) v=skip_first?s[(size_t)oc*t+n]+v:v+s[(size_t)oc*t+n];
    y[(size_t)oc*t+n]=v;
  }
}

/* Fixed four-output-channel microtiles reuse each contiguous input vector.
 * The full dot and both additions stay in the same source-level accumulator.
 * Inspect compiler output to check actual register allocation on each target.
 */
#define FX_IMPL(NAME,ATTR,V,LANES,NV,NC,ZERO,LOAD,STORE,BROAD,FMA,ADD) \
ATTR static void NAME(int c,int k,int t,int begin,int end,const float*w,const float*x,const float*b,const float*s,float*y,int skip_first) { \
  int n=begin; \
  for(;n<=end-(LANES)*(NV);n+=(LANES)*(NV)) { \
    for(int oc=0;oc+NC<=c;oc+=NC) { \
      V acc[NC][NV]; \
      for(int i=0;i<NC;++i) for(int j=0;j<NV;++j) acc[i][j]=ZERO(); \
      for(int ic=0;ic<k;++ic) { \
        V xx[NV]; \
        for(int j=0;j<NV;++j) xx[j]=LOAD(x+(size_t)ic*t+n+j*LANES); \
        for(int i=0;i<NC;++i) { \
          V ww=BROAD(w[(size_t)(oc+i)*k+ic]); \
          for(int j=0;j<NV;++j) acc[i][j]=FMA(xx[j],ww,acc[i][j]); \
        } \
      } \
      for(int i=0;i<NC;++i) { \
        V bb=BROAD(b[oc+i]); \
        for(int j=0;j<NV;++j) { \
          V value=ADD(acc[i][j],bb); \
          if(s) { V ss=LOAD(s+(size_t)(oc+i)*t+n+j*LANES); value=skip_first?ADD(ss,value):ADD(value,ss); } \
          STORE(y+(size_t)(oc+i)*t+n+j*LANES,value); \
        } \
      } \
    } \
  } \
  if(n<end) scalar(c,k,t,n,end,w,x,b,s,y,skip_first); \
}

#if FX_X86
#define AVX2_ATTR __attribute__((target("avx2,fma")))
#define AVX512_ATTR __attribute__((target("avx512f,fma")))
FX_IMPL(avx2,AVX2_ATTR,__m256,8,2,4,_mm256_setzero_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_set1_ps,_mm256_fmadd_ps,_mm256_add_ps)
FX_IMPL(avx512,AVX512_ATTR,__m512,16,4,4,_mm512_setzero_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_set1_ps,_mm512_fmadd_ps,_mm512_add_ps)
FX_IMPL(avx512_c8,AVX512_ATTR,__m512,16,2,8,_mm512_setzero_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_set1_ps,_mm512_fmadd_ps,_mm512_add_ps)
static int available_isa(void) {
  unsigned a,b,c,d; uint32_t xl,xh;
  if(!__get_cpuid(1,&a,&b,&c,&d) || !(c&bit_AVX) || !(c&bit_OSXSAVE) || !(c&bit_FMA)) return 0;
  __asm__ volatile("xgetbv" : "=a"(xl),"=d"(xh) : "c"(0));
  (void)xh;
  if((xl&6)!=6 || !__get_cpuid_count(7,0,&a,&b,&c,&d)) return 0;
  if((xl&0xe6)==0xe6 && (b&bit_AVX512F) && (b&bit_AVX2)) return 512;
  return (b&bit_AVX2) ? 256 : 0;
}
#elif defined(__aarch64__)
static inline float32x4_t neon_zero(void){ return vdupq_n_f32(0.0f); }
static inline float32x4_t neon_fma(float32x4_t a,float32x4_t b,float32x4_t c){return vfmaq_f32(c,a,b);}
FX_IMPL(neon,,float32x4_t,4,4,4,neon_zero,vld1q_f32,vst1q_f32,vdupq_n_f32,neon_fma,vaddq_f32)
static int available_isa(void){return 128;}
#else
static int available_isa(void){return 0;}
#endif

int fx_init(void) {
#ifdef FX_WITH_LIBXSMM
  if(getenv("LIBXSMM_TARGET")) return -1;
  libxsmm_init();
  libxsmm_set_target_archid(libxsmm_cpuid(NULL));
#endif
  return available_isa();
}
int fx_has_xsmm(void) {
#ifdef FX_WITH_LIBXSMM
  return 1;
#else
  return 0;
#endif
}

/* mode 0: direct register epilogue. mode 1: XSMM scratch-tile epilogue.
 * mode 2: XSMM writes output then separate global finishing passes (control).
 * isa 0 auto, 1 scalar, 256 AVX2, 512 AVX512, 128 NEON; unsupported fails.
 * tt <=256 and tc<=64 bound per-worker scratch to <=64KiB (no heap in run).
 */
void *fx_create_ordered(int c,int k,int t,int tt,int tc,int mode,int isa,int skip_first) {
  if(!(c==32||c==64||c==128||c==256)||k!=c||t<=0||tt<=0||tt>256||tc<=0||tc>64||mode<0||mode>2||(mode&&isa)) return NULL;
  if((size_t)c > SIZE_MAX/(size_t)t/sizeof(float)) return NULL;
  fx_plan*p=(fx_plan*)calloc(1,sizeof(*p)); if(!p)return NULL;
  p->c=c;p->k=k;p->t=t;p->tt=tt;p->tc=tc;p->mode=mode;p->isa=available_isa();p->direct=scalar;
  p->skip_first=!!skip_first;
  if(isa==1)p->isa=0;
  else if(isa && isa!=p->isa && !(isa==256&&p->isa==512)){free(p);return NULL;}
  else if(isa)p->isa=isa;
#if FX_X86
  if(p->isa==512)p->direct=tc==8?avx512_c8:avx512;
  if(p->isa==256)p->direct=avx2;
#elif defined(__aarch64__)
  if(p->isa==128)p->direct=neon;
#endif
  if(mode) {
#ifdef FX_WITH_LIBXSMM
    for(int ti=0;ti<2;++ti)for(int ci=0;ci<2;++ci) {
      /* Short dimensions have no full tile. Dispatching the unused full-time
       * shape would set m=tt>lda=t, which LIBXSMM correctly rejects. Only make
       * kernels whose full/tail category can be selected by run_xsmm. */
      if((!ti&&t<tt)||(!ci&&c<tc))continue;
      int nt=ti?t%tt:tt, nc=ci?c%tc:tc;
      if(nt&&nc){
        int ldout=mode==1?tt:t;
        libxsmm_gemm_shape sh=libxsmm_create_gemm_shape(nt,nc,k,t,k,ldout,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
        p->gemm[ti][ci]=libxsmm_dispatch_gemm(sh,LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
        if(!p->gemm[ti][ci]){free(p);return NULL;}
      }
    }
#else
    free(p);return NULL;
#endif
  }
  return p;
}
void *fx_create(int c,int k,int t,int tt,int tc,int mode,int isa){return fx_create_ordered(c,k,t,tt,tc,mode,isa,0);}
int fx_blocks(const void*plan){const fx_plan*p=plan;return p?(p->t-1)/p->tt+1:-1;}
int fx_isa(const void*plan){const fx_plan*p=plan;return p?p->isa:-1;}
static int overlaps(const void*a,size_t an,const void*b,size_t bn){
  uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b;
  return an&&bn&&(aa<=bb?bb-aa<an:aa-bb<bn);
}
#ifdef FX_WITH_LIBXSMM
/* Isolate the scratch stack frame from the direct SIMD path. */
__attribute__((noinline)) static int run_xsmm(const fx_plan*p,const float*w,const float*x,const float*b,const float*s,float*y,int first,int last) {
  _Alignas(64) float scratch[64*256];
  for(int block=first;block<last;++block){
    int t0=block*p->tt,nt=p->t-t0<p->tt?p->t-t0:p->tt,ti=nt<p->tt;
    for(int c0=0;c0<p->c;c0+=p->tc){
      int nc=p->c-c0<p->tc?p->c-c0:p->tc,ci=nc<p->tc;
      libxsmm_gemm_param q;memset(&q,0,sizeof(q));
      q.a.primary=(void*)(x+t0);q.b.primary=(void*)(w+(size_t)c0*p->k);
      q.c.primary=p->mode==1?scratch:y+(size_t)c0*p->t+t0;
      p->gemm[ti][ci](&q);
      if(p->mode==1)for(int oc=0;oc<nc;++oc)for(int n=0;n<nt;++n){
        size_t dst=(size_t)(c0+oc)*p->t+t0+n;
        float v=scratch[(size_t)oc*p->tt+n]+b[c0+oc];
        if(s)v=p->skip_first?s[dst]+v:v+s[dst];
        y[dst]=v;
      }
    }
  }
  if(p->mode==2){
    int begin=first*p->tt,end=last==fx_blocks(p)?p->t:last*p->tt;
    for(int oc=0;oc<p->c;++oc)for(int n=begin;n<end;++n)y[(size_t)oc*p->t+n]+=b[oc];
    if(s)for(int oc=0;oc<p->c;++oc)for(int n=begin;n<end;++n){size_t i=(size_t)oc*p->t+n;y[i]=p->skip_first?s[i]+y[i]:y[i]+s[i];}
  }
  return 0;
}
#endif
int fx_run_range(const void*plan,const float*w,const float*x,const float*b,const float*s,float*y,int first,int last) {
  const fx_plan*p=plan;
  if(!p||!w||!x||!b||!y||first<0||last<first||last>fx_blocks(p))return -1;
  size_t bytes=(size_t)p->c*p->t*sizeof(float);
  if(overlaps(y,bytes,w,(size_t)p->c*p->k*sizeof(float))||overlaps(y,bytes,x,bytes)||
     overlaps(y,bytes,b,(size_t)p->c*sizeof(float))||(s&&overlaps(y,bytes,s,bytes)))return -1;
  if(first==last)return 0;
  if(p->mode==0){
    int begin=first*p->tt,end=last==fx_blocks(p)?p->t:last*p->tt;
    p->direct(p->c,p->k,p->t,begin,end,w,x,b,s,y,p->skip_first);return 0;
  }
#ifdef FX_WITH_LIBXSMM
  return run_xsmm(p,w,x,b,s,y,first,last);
#else
  return -1;
#endif
}
void fx_destroy(void*plan){free(plan);}
