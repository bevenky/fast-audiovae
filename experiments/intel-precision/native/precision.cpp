// Experimental CPU matrix approximation. FP32 nonlinear paths are external.
#include "precision.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>
#if defined(IP_WITH_MKL)
#include "mkl_cblas.h"
#include "mkl_service.h"
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#define IP_X86 1
#define IP_AVX512 __attribute__((target("avx512f,avx512bw,avx512vnni")))
#define IP_F16C __attribute__((target("avx,f16c")))
#else
#define IP_X86 0
#endif
#if defined(__FAST_MATH__)
#error "Precision experiments must retain ordinary FP32 semantics"
#endif
namespace {
thread_local char error_text[512]={0};
void require(bool v,const char* s){if(!v)throw std::invalid_argument(s);}
void failed() noexcept {
    try{throw;}catch(const std::exception& e){std::snprintf(error_text,sizeof(error_text),"%s",e.what());}
    catch(...){std::snprintf(error_text,sizeof(error_text),"Unknown precision exception");}
}
size_t count(int a,int b,size_t bytes){
    require(a>=0&&b>=0,"Negative matrix dimension");
    const size_t x=static_cast<size_t>(a),y=static_cast<size_t>(b);
    require(!y||x<=std::numeric_limits<size_t>::max()/y/bytes,"Matrix byte size overflow");
    return x*y;
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    if(!na||!nb)return false;
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    require(x<=UINTPTR_MAX-na&&y<=UINTPTR_MAX-nb,"Pointer range overflow");
    return x<y+nb&&y<x+na;
}
int cpu_vnni(){
#if IP_X86
    unsigned a,b,c,d;uint32_t lo,hi;
    if(!__get_cpuid(1,&a,&b,&c,&d)||!(c&bit_AVX)||!(c&bit_OSXSAVE))return 0;
    __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));(void)hi;
    if((lo&0xe6)!=0xe6||!__get_cpuid_count(7,0,&a,&b,&c,&d))return 0;
    return (b&(1u<<16))&&(b&(1u<<30))&&(c&(1u<<11));
#else
    return 0;
#endif
}
int cpu_f16c(){
#if IP_X86
    unsigned a,b,c,d;uint32_t lo,hi;
    if(!__get_cpuid(1,&a,&b,&c,&d)||!(c&bit_AVX)||!(c&bit_OSXSAVE)||!(c&(1u<<29)))return 0;
    __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));(void)hi;
    return (lo&6)==6;
#else
    return 0;
#endif
}
int capabilities(){
    static const int cpu=(cpu_vnni()?2:0)|(cpu_f16c()?4:0);
#if defined(IP_WITH_MKL)
    return cpu|(mkl_get_max_threads()==1?1:0);
#else
    return cpu;
#endif
}
// Explicit nearest-even integer choice. Ordinary FP32 divisions use the
// process floating-point environment, which the experiment keeps at nearest.
int quantize(float v,float scale){
    float q=std::max(-127.f,std::min(127.f,v/scale));
    int n=static_cast<int>(std::floor(q));const float fraction=q-static_cast<float>(n);
    if(fraction>.5f||(fraction==.5f&&(n&1)))++n;
    return n;
}
float scale_of(float mx){
    if(mx==0.f)return 1.f;
    const float s=mx/127.f;
    return s>0.f?s:mx; // avoid division by zero for subnormal magnitudes
}
uint32_t round_shift(uint32_t v,int shift){
    const uint32_t q=v>>shift,mask=(uint32_t(1)<<shift)-1,r=v&mask,half=uint32_t(1)<<(shift-1);
    return q+(r>half||(r==half&&(q&1)));
}
float half_round(float f){
    require(std::isfinite(f),"Nonfinite FP16 operand");
    uint32_t u;std::memcpy(&u,&f,4);
    const uint32_t sign=(u>>16)&0x8000,man=u&0x7fffff;
    const int exponent=int((u>>23)&255)-127+15;
    uint32_t h;
    if(exponent<=0){
        h=sign|(exponent < -10?0:round_shift(man|0x800000,14-exponent));
    }else{
        require(exponent<31,"FP16 operand overflow");
        const uint32_t rounded=round_shift(man,13);
        h=sign+(uint32_t(exponent)<<10)+rounded;
        require((h&0x7c00)!=0x7c00,"FP16 operand overflow");
    }
    uint32_t bits=(h&0x8000)<<16,e=(h>>10)&31,m=h&1023;
    if(e){bits|=(e+112)<<23;bits|=m<<13;}
    else if(m){int unbiased=-14;while(!(m&1024)){m<<=1;--unbiased;}bits|=uint32_t(unbiased+127)<<23;bits|=(m&1023)<<13;}
    float out;std::memcpy(&out,&bits,4);return out;
}
#if defined(IP_WITH_MKL)
struct Packed{
    void* data=nullptr;size_t bytes;
    explicit Packed(size_t n):bytes(n){
        require(n>0,"oneMKL returned an empty packed buffer");
        data=mkl_malloc(n,64);if(!data)throw std::bad_alloc();
    }
    ~Packed(){mkl_free(data);}
    Packed(const Packed&)=delete;Packed& operator=(const Packed&)=delete;
};
// The key always contains exact geometry. A cache's owning Plan/Input fixes
// K, source data, precision, row-major layout and sequential packing threads.
// An evicted buffer stays alive through shared ownership while a worker uses it.
class PackCache{
    using Key=std::tuple<int,int,int>;
    struct Entry{std::shared_ptr<Packed> buffer;uint64_t stamp;};
    mutable std::mutex mutex_;
    mutable std::map<Key,Entry> entries_;
    mutable size_t bytes_=0;
    mutable uint64_t clock_=0;
public:
    size_t limit=1u<<20;
    template<class Factory>std::shared_ptr<Packed> get(Key key,Factory factory)const{
        std::lock_guard<std::mutex> lock(mutex_);
        auto found=entries_.find(key);
        if(found!=entries_.end()){found->second.stamp=++clock_;return found->second.buffer;}
        auto buffer=factory();
        if(buffer->bytes>limit)return buffer;
        while(!entries_.empty()&&(bytes_>limit-buffer->bytes||entries_.size()>=256)){
            auto oldest=std::min_element(entries_.begin(),entries_.end(),
                    [](const auto& a,const auto& b){return a.second.stamp<b.second.stamp;});
            bytes_-=oldest->second.buffer->bytes;entries_.erase(oldest);
        }
        entries_.emplace(key,Entry{buffer,++clock_});bytes_+=buffer->bytes;
        return buffer;
    }
    size_t bytes()const{std::lock_guard<std::mutex> lock(mutex_);return bytes_;}
};
size_t pack_budget(size_t payload){
    // Clamp before doubling so the accounting remains safe at accepted sizes.
    return payload>=(32u<<20)?size_t(64u<<20):std::max<size_t>(1u<<20,2*payload);
}
#endif
struct Plan{
    int m,k,mode,backend;
    const float* original=nullptr;
    std::vector<uint8_t> w8;
    std::vector<float> w16,scales;
#if defined(IP_WITH_MKL)
    PackCache packed_weights;
#endif
};
struct Input{
    int k,t,mode;
    const float* original=nullptr;
    std::vector<int8_t> x8;
    std::vector<float> x16,scales;
    std::vector<int32_t> sums;
#if defined(IP_WITH_MKL)
    PackCache packed_activations;
#endif
};
#if IP_X86
IP_F16C void half_round_vector(const float* source,float* output,size_t count){
    size_t i=0;const __m256 sign=_mm256_set1_ps(-0.f),largest=_mm256_set1_ps(std::numeric_limits<float>::max());
    for(;count-i>=8;i+=8){
        const __m256 x=_mm256_loadu_ps(source+i);
        const __m128i half=_mm256_cvtps_ph(x,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        const __m256 y=_mm256_cvtph_ps(half);
        require(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign,y),largest,_CMP_LE_OQ))==255,
                "Nonfinite or overflowing FP16 operand");
        _mm256_storeu_ps(output+i,y);
    }
    for(;i<count;++i)output[i]=half_round(source[i]);
}
IP_AVX512 int prepare_columns_vector(const float* values,Input& x){
    const __m512 zero=_mm512_setzero_ps(),one=_mm512_set1_ps(1.f),limit=_mm512_set1_ps(127.f);
    const __m512 negative=_mm512_set1_ps(-127.f),largest=_mm512_set1_ps(std::numeric_limits<float>::max());
    const __m512i abs_mask=_mm512_set1_epi32(0x7fffffff);
    int begin=0;
    for(;x.t-begin>=16;begin+=16){
        __m512 mx=zero;
        for(int k=0;k<x.k;++k){
            const __m512 v=_mm512_loadu_ps(values+size_t(k)*x.t+begin);
            const __m512 ab=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v),abs_mask));
            require(_mm512_cmp_ps_mask(ab,largest,_CMP_LE_OQ)==0xffff,"Nonfinite INT8 activation");
            mx=_mm512_max_ps(mx,ab);
        }
        __m512 scale=_mm512_div_ps(mx,limit);
        scale=_mm512_mask_mov_ps(scale,_mm512_cmp_ps_mask(scale,zero,_CMP_EQ_OQ),mx);
        scale=_mm512_mask_mov_ps(scale,_mm512_cmp_ps_mask(mx,zero,_CMP_EQ_OQ),one);
        _mm512_storeu_ps(x.scales.data()+begin,scale);__m512i sums=_mm512_setzero_si512();
        for(int k=0;k<x.k;++k){
            const __m512 v=_mm512_loadu_ps(values+size_t(k)*x.t+begin);
            const __m512 q=_mm512_max_ps(negative,_mm512_min_ps(limit,_mm512_div_ps(v,scale)));
            const __m512i qi=_mm512_cvt_roundps_epi32(q,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
            sums=_mm512_add_epi32(sums,qi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(x.x8.data()+size_t(k)*x.t+begin),_mm512_cvtsepi32_epi8(qi));
        }
        _mm512_storeu_si512(x.sums.data()+begin,sums);
    }
    return begin;
}
IP_AVX512 void dequant_vector(const int32_t* product,float* output,int n,float sw,const float* sx,const int32_t* sums){
    const __m512 weight_scale=_mm512_set1_ps(sw),largest=_mm512_set1_ps(std::numeric_limits<float>::max());
    const __m512i abs_mask=_mm512_set1_epi32(0x7fffffff);int j=0;
    for(;n-j>=16;j+=16){
        const __m512i raw=_mm512_loadu_si512(product+j);
        const __m512i correction=_mm512_slli_epi32(_mm512_loadu_si512(sums+j),7);
        const __m512 scale=_mm512_mul_ps(weight_scale,_mm512_loadu_ps(sx+j));
        const __m512 y=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(raw,correction)),scale);
        const __m512 ab=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(y),abs_mask));
        require(_mm512_cmp_ps_mask(ab,largest,_CMP_LE_OQ)==0xffff,"Nonfinite precision output");
        _mm512_storeu_ps(output+j,y);
    }
    for(;j<n;++j){output[j]=float(product[j]-128*sums[j])*(sw*sx[j]);require(std::isfinite(output[j]),"Nonfinite precision output");}
}
#endif
void half_round_array(const float* source,float* output,size_t n,bool vector){
#if IP_X86
    if(vector&&(capabilities()&4)){half_round_vector(source,output,n);return;}
#else
    (void)vector;
#endif
    for(size_t i=0;i<n;++i)output[i]=half_round(source[i]);
}
void scalar(const Plan& p,const Input& x,float* y,int first,int last){
    for(int m=first;m<last;++m)for(int t=0;t<x.t;++t){
        if(p.mode==8){
            int32_t acc=0;
            for(int k=0;k<p.k;++k)acc+=(int(p.w8[size_t(m)*p.k+k])-128)*int(x.x8[size_t(k)*x.t+t]);
            y[size_t(m)*x.t+t]=static_cast<float>(acc)*(p.scales[m]*x.scales[t]);
        }else{
            float acc=0.f;
            for(int k=0;k<p.k;++k)acc=std::fma(p.w16[size_t(m)*p.k+k],x.x16[size_t(k)*x.t+t],acc);
            y[size_t(m)*x.t+t]=acc;
        }
        require(std::isfinite(y[size_t(m)*x.t+t]),"Nonfinite precision output");
    }
}
#if defined(IP_WITH_MKL)
constexpr int HALF_TILE=64;
std::shared_ptr<Packed> half_a(const Plan& p,int first,int rows){
    return p.packed_weights.get({first,rows,HALF_TILE},[&](){
        auto buffer=std::make_shared<Packed>(count(HALF_TILE,p.k,sizeof(float))*sizeof(float));
        std::memset(buffer->data,0,buffer->bytes);
        std::memcpy(buffer->data,p.w16.data()+size_t(first)*p.k,size_t(rows)*p.k*sizeof(float));
        return buffer;
    });
}
std::shared_ptr<Packed> half_b(const Input& x,int time,int cols){
    return x.packed_activations.get({time,HALF_TILE,cols},[&](){
        auto buffer=std::make_shared<Packed>(count(x.k,HALF_TILE,sizeof(float))*sizeof(float));
        std::memset(buffer->data,0,buffer->bytes);
        auto out=static_cast<float*>(buffer->data);
        for(int k=0;k<x.k;++k)
            std::memcpy(out+size_t(k)*HALF_TILE,x.x16.data()+size_t(k)*x.t+time,size_t(cols)*sizeof(float));
        return buffer;
    });
}
void mkl(const Plan& p,const Input& x,float* y,int first,int last){
    static_assert(sizeof(MKL_INT)==4&&sizeof(MKL_INT32)==4,"LP64 oneMKL required");
    require(mkl_get_max_threads()==1,"Sequential oneMKL required");
    if(p.mode==16){
        // Half rounding can amplify tiny shape-dependent SGEMM differences at
        // later layers. Keep M/N/leading dimensions and 64-byte alignment fixed
        // for every tile, including tails and arbitrary output-row partitions.
        // Padding adds independent output rows/columns, never terms to K or
        // values to a real causal history. Actual operands remain half-rounded
        // FP32; the complete-K product accumulates and returns FP32.
        Packed product(size_t(HALF_TILE)*HALF_TILE*sizeof(float));
        auto out=static_cast<float*>(product.data);
        for(int t=0;t<x.t;){
            const int cols=std::min(HALF_TILE,x.t-t);const auto b=half_b(x,t,cols);
            for(int m=first;m<last;){
                const int rows=std::min(HALF_TILE,last-m);const auto a=half_a(p,m,rows);
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,HALF_TILE,HALF_TILE,p.k,
                    1.f,static_cast<const float*>(a->data),p.k,
                    static_cast<const float*>(b->data),HALF_TILE,0.f,out,HALF_TILE);
                for(int r=0;r<rows;++r)for(int j=0;j<cols;++j){
                    const float v=out[size_t(r)*HALF_TILE+j];
                    require(std::isfinite(v),"Nonfinite precision output");y[size_t(m+r)*x.t+t+j]=v;
                }
                m+=rows;
            }
            t+=cols;
        }
        return;
    }
    // oneMKL's ROW-MAJOR s8u8 routine requires UNSIGNED A and SIGNED B.
    // Store qw+128 in A; remove its offset after the complete-K integer dot.
    // BCT layout is retained. Larger complete-K row panels reduce GEMM calls.
    // INT32 scratch is bounded to 2048x512 elements (4MiB) per callback.
    // Opaque INT8 packing was rejected because its minimum buffer was >12MiB.
    constexpr int MR=2048,NT=512;
    std::vector<MKL_INT32> product(size_t(std::min(MR,last-first))*std::min(NT,x.t));
    const MKL_INT32 zero=0;
    for(int m=first;m<last;){
        const int rows=std::min(MR,last-m);
        for(int t=0;t<x.t;){
            const int cols=std::min(NT,x.t-t);
            cblas_gemm_s8u8s32(CblasRowMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,
                rows,cols,p.k,1.f,p.w8.data()+size_t(m)*p.k,p.k,0,
                x.x8.data()+t,x.t,0,0.f,product.data(),cols,&zero);
            for(int r=0;r<rows;++r){
#if IP_X86
                dequant_vector(product.data()+size_t(r)*cols,y+size_t(m+r)*x.t+t,cols,
                               p.scales[m+r],x.scales.data()+t,x.sums.data()+t);
#else
                for(int j=0;j<cols;++j){
                    const int32_t corrected=product[size_t(r)*cols+j]-128*x.sums[t+j];
                    const float v=static_cast<float>(corrected)*(p.scales[m+r]*x.scales[t+j]);
                    require(std::isfinite(v),"Nonfinite precision output");y[size_t(m+r)*x.t+t+j]=v;
                }
#endif
            }
            t+=cols;
        }
        m+=rows;
    }
}
#endif
}
extern "C" {
int ip_abi(void){return 1;}
int ip_capabilities(void){return capabilities();}
const char* ip_last_error(void){return error_text;}
void* ip_create(int m,int k,const float* w,int mode,int backend){
    error_text[0]=0;
    try{
        require(m>0&&k>0&&k<=16384,"Require M>0 and 0<K<=16384");
        require(w,"Missing weights");require(mode==8||mode==16,"Precision mode must be 8 or16");
        require(backend==0||backend==1,"Backend must be scalar0 or oneMKL1");
        require(backend==0||(capabilities()&1),"Sequential oneMKL backend unavailable");
        require(backend==0||mode!=8||(capabilities()&2),"W8A8 oneMKL requires CPU and OS AVX512 VNNI");
#if defined(IP_WITH_MKL)
        if(backend==1&&mode==8){
            const char* instruction=std::getenv("MKL_ENABLE_INSTRUCTIONS");
            require(!instruction||!*instruction||std::strcmp(instruction,"AVX512_E1")==0,
                    "W8A8 requires unset MKL_ENABLE_INSTRUCTIONS or AVX512_E1");
            const char* branch=std::getenv("MKL_CBWR");
            require(!branch||!*branch||std::strcmp(branch,"AUTO")==0||std::strcmp(branch,"AVX512_E1")==0,
                    "W8A8 requires unset MKL_CBWR, AUTO or AVX512_E1");
        }
#endif
        const auto n=count(m,k,sizeof(float));auto p=std::make_unique<Plan>();
        p->m=m;p->k=k;p->mode=mode;p->backend=backend;p->original=w;
        if(mode==16){
            p->w16.resize(n);half_round_array(w,p->w16.data(),n,backend==1);
#if defined(IP_WITH_MKL)
            p->packed_weights.limit=pack_budget(n*sizeof(float));
#endif
        }
        else{
            p->w8.resize(n);p->scales.resize(m);
            for(int r=0;r<m;++r){
                float mx=0.f;
                for(int c=0;c<k;++c){const float v=w[size_t(r)*k+c];require(std::isfinite(v),"Nonfinite INT8 weight");mx=std::max(mx,std::fabs(v));}
                const float s=p->scales[r]=scale_of(mx);
                for(int c=0;c<k;++c)p->w8[size_t(r)*k+c]=static_cast<uint8_t>(quantize(w[size_t(r)*k+c],s)+128);
            }
        }
        return p.release();
    }catch(...){failed();return nullptr;}
}
void ip_destroy_plan(void* p){delete static_cast<Plan*>(p);}
void* ip_prepare(const void* raw,const float* values,int time){
    error_text[0]=0;
    try{
        require(raw,"Missing weight plan");const auto& p=*static_cast<const Plan*>(raw);
        require(time>=0,"Negative time");require(values||!time,"Missing input");
        const auto n=count(p.k,time,sizeof(float));auto x=std::make_unique<Input>();
        x->k=p.k;x->t=time;x->mode=p.mode;x->original=values;
        if(p.mode==16){
            x->x16.resize(n);half_round_array(values,x->x16.data(),n,p.backend==1);
#if defined(IP_WITH_MKL)
            x->packed_activations.limit=pack_budget(n*sizeof(float));
#endif
        }
        else{
            x->x8.resize(n);x->scales.assign(time,0.f);x->sums.assign(time,0);
            // Each column owns its range. Time blocking changes access locality
            // only, never the reduction domain or arithmetic for a column.
            int begin=0;
#if IP_X86
            if(p.backend==1&&(capabilities()&2))begin=prepare_columns_vector(values,*x);
#endif
            for(;begin<time;){
                const int end=begin+std::min(64,time-begin);
                for(int k=0;k<p.k;++k)for(int t=begin;t<end;++t){
                    const float v=values[size_t(k)*time+t];require(std::isfinite(v),"Nonfinite INT8 activation");
                    x->scales[t]=std::max(x->scales[t],std::fabs(v));
                }
                for(int t=begin;t<end;++t)x->scales[t]=scale_of(x->scales[t]);
                for(int k=0;k<p.k;++k)for(int t=begin;t<end;++t){
                    const int q=quantize(values[size_t(k)*time+t],x->scales[t]);
                    x->x8[size_t(k)*time+t]=static_cast<int8_t>(q);x->sums[t]+=q;
                }
                begin=end;
            }
        }
        return x.release();
    }catch(...){failed();return nullptr;}
}
void ip_destroy_input(void* p){delete static_cast<Input*>(p);}
int ip_run_rows(const void* pr,const void* xr,float* y,int first,int last){
    error_text[0]=0;
    try{
        require(pr&&xr,"Missing plan or input");const auto& p=*static_cast<const Plan*>(pr);const auto& x=*static_cast<const Input*>(xr);
        require(p.k==x.k&&p.mode==x.mode,"Prepared input K/mode mismatch");
        require(first>=0&&last>=first&&last<=p.m,"Invalid output row interval");
        const auto bytes=count(p.m,x.t,sizeof(float))*sizeof(float);
        require(y||!bytes,"Missing output");
        require(!overlap(y,bytes,x.original,count(x.k,x.t,4)*4),"Output overlaps original input");
        require(!overlap(y,bytes,p.original,count(p.m,p.k,4)*4),"Output overlaps original weights");
        if(first==last||!x.t)return 0;
        if(p.backend==0)scalar(p,x,y,first,last);
#if defined(IP_WITH_MKL)
        else mkl(p,x,y,first,last);
#else
        else throw std::invalid_argument("oneMKL backend unavailable");
#endif
        return 0;
    }catch(...){failed();return -1;}
}
size_t ip_plan_bytes(const void* raw){
    try{
        if(!raw)return 0;const auto& p=*static_cast<const Plan*>(raw);
        size_t n=sizeof(p)+p.w8.size()+(p.w16.size()+p.scales.size())*sizeof(float);
#if defined(IP_WITH_MKL)
        n+=p.packed_weights.bytes();
#endif
        return n;
    }catch(...){failed();return 0;}
}
size_t ip_input_bytes(const void* raw){
    try{
        if(!raw)return 0;const auto& x=*static_cast<const Input*>(raw);
        size_t n=sizeof(x)+x.x8.size()+(x.x16.size()+x.scales.size())*sizeof(float)+x.sums.size()*sizeof(int32_t);
#if defined(IP_WITH_MKL)
        n+=x.packed_activations.bytes();
#endif
        return n;
    }catch(...){failed();return 0;}
}
}
