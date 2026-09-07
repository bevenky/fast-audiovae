/* Pin-specific FP32 packed-A adapter. No CBLAS global state or thread setters. */
#include "packed_a.h"
#include "blis.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <pthread.h>
#include <cpuid.h>
#if defined(__FAST_MATH__)
#error "Adapter arithmetic must not use fast math"
#endif
#if defined(BLIS_ENABLE_MULTITHREADING) || defined(BLIS_ENABLE_OPENMP)
#error "This adapter requires the separate serial AOCL build"
#endif
_Static_assert(sizeof(f77_int)==4,"Expected LP64 BLAS integers");
_Static_assert(sizeof(dim_t)==8,"Expected64-bit BLIS dimensions");
struct ncc_aocl_pack { float* data; size_t bytes; dim_t m,k; cntx_t* context; };
static pthread_once_t init_once=PTHREAD_ONCE_INIT;
static void initialize_blis(void) { bli_init(); }
uint32_t ncc_aocl_adapter_abi(void) { return 1; }
int ncc_aocl_cpu_supported(void) {
    unsigned a,b,c,d;
    if (__get_cpuid_max(0,NULL)<7) return 0;
    __cpuid_count(0,0,a,b,c,d);
    /* AMD vendor plus Zen4 ISA prerequisites and OS-enabled vector state. */
    if (b!=0x68747541u || d!=0x69746e65u || c!=0x444d4163u) return 0;
    __cpuid_count(1,0,a,b,c,d);
    unsigned need=(1u<<12)|(1u<<26)|(1u<<27)|(1u<<28);
    if ((c&need)!=need) return 0;
    uint32_t lo,hi;__asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));(void)hi;
    if ((lo&0xe6u)!=0xe6u) return 0;
    __cpuid_count(7,0,a,b,c,d);
    need=(1u<<5)|(1u<<16)|(1u<<17)|(1u<<30)|(1u<<31);
    return (b&need)==need;
}
struct ncc_aocl_pack* ncc_aocl_create(const float* weights,int64_t m,int64_t k) {
    if (!ncc_aocl_cpu_supported() || !weights || m<=0 || k<=0 || m>INT_MAX || k>INT_MAX) return NULL;
    pthread_once(&init_once,initialize_blis);
    cntx_t* context=bli_gks_query_cntx();
    const dim_t mr=bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT,BLIS_MR,context);
    if (mr<=0 || mr>INT_MAX) return NULL;
    uint64_t mp=((uint64_t)m+mr-1)/mr*mr,kp=((uint64_t)k+mr-1)/mr*mr;
    if (mp>SIZE_MAX/(uint64_t)k || kp>SIZE_MAX/(uint64_t)m) return NULL;
    uint64_t count=mp*k > kp*m ? mp*k : kp*m;
    /* Match the pinned CBLAS conservative A allocation, avoiding LP64 overflow. */
    if (count>INT_MAX/sizeof(float)) return NULL;
    struct ncc_aocl_pack* p=calloc(1,sizeof(*p));if (!p) return NULL;
    p->bytes=(size_t)count*sizeof(float);p->m=m;p->k=k;p->context=context;
    if (posix_memalign((void**)&p->data,64,p->bytes)) { free(p);return NULL; }
    /* Padding is initialized for reproducible ownership probes, never read as data. */
    memset(p->data,0,p->bytes);
    float one=1.0f;char identifier='A';
    obj_t src=BLIS_OBJECT_INITIALIZER,dst=BLIS_OBJECT_INITIALIZER,alpha=BLIS_OBJECT_INITIALIZER_1X1;
    /* Mirror RowMajor/NoTrans CBLAS: view W as column-major KxM, transposed. */
    bli_obj_init_finish(BLIS_FLOAT,k,m,(float*)weights,1,k,&src);
    bli_obj_init_finish(BLIS_FLOAT,k,m,p->data,1,k,&dst);
    bli_obj_set_conjtrans(BLIS_TRANSPOSE,&src);
    bli_obj_init_finish_1x1(BLIS_FLOAT,&one,&alpha);
    rntm_t runtime=BLIS_RNTM_INITIALIZER;bli_rntm_set_num_threads(1,&runtime);
    bli_pack_full_init(&identifier,&alpha,&src,&dst,context,&runtime);
    return p;
}
void ncc_aocl_destroy(struct ncc_aocl_pack* p) { if(p) {free(p->data);free(p);} }
size_t ncc_aocl_packed_bytes(const struct ncc_aocl_pack* p) { return p ? p->bytes:0; }
int ncc_aocl_compute(const struct ncc_aocl_pack* p,const float* x,float* y,int64_t n,int64_t full_t) {
    if (!p || !x || !y || n<=0 || full_t<n || full_t>INT_MAX || n>INT_MAX) return -1;
    float zero=0.0f;
    obj_t a=BLIS_OBJECT_INITIALIZER,b=BLIS_OBJECT_INITIALIZER,c=BLIS_OBJECT_INITIALIZER,beta=BLIS_OBJECT_INITIALIZER_1X1;
    bli_obj_init_finish(BLIS_FLOAT,p->m,p->k,p->data,1,p->k,&a);
    bli_obj_set_conjtrans(BLIS_PACKED,&a);
    bli_obj_init_finish(BLIS_FLOAT,n,p->k,(float*)x,1,full_t,&b);
    bli_obj_set_conjtrans(BLIS_TRANSPOSE,&b);
    bli_obj_init_finish(BLIS_FLOAT,p->m,n,y,full_t,1,&c);
    bli_obj_init_finish_1x1(BLIS_FLOAT,&zero,&beta);
    rntm_t runtime=BLIS_RNTM_INITIALIZER;bli_rntm_set_num_threads(1,&runtime);
    bli_gemm_compute_init(&a,&b,&beta,&c,p->context,&runtime);
    return 0;
}
