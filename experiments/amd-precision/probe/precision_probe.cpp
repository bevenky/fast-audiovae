// Optional CPU attribution shim. No tensors, weights or outputs are modified.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "precision.h"
#include <limits>
#if defined(PROBE_AOCL)
extern "C" {
#include "classic/aocl_gemm_interface_apis.h"
}
#else
#include "mkl_cblas.h"
#endif
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <sys/stat.h>
#include <time.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <cstdlib>

#if !defined(__linux__) || !defined(__x86_64__)
#error "This optional preload probe targets the controlled AMD Linux process"
#endif
#define PROBE_API __attribute__((visibility("default")))

namespace {
#if !defined(PROBE_AOCL)
static_assert(sizeof(MKL_INT)==4&&sizeof(MKL_INT32)==4,"LP64 oneMKL required");
#endif
using Prepare=decltype(&ip_prepare);
using Rows=decltype(&ip_run_rows);
#if defined(PROBE_AOCL)
using Gemm=decltype(&aocl_gemm_u8s8s32os32);
#define PROBE_GEMM aocl_gemm_u8s8s32os32
#define PROBE_GEMM_NAME "aocl_gemm_u8s8s32os32"
constexpr size_t GEMM_ARGUMENTS=17;
#else
using Gemm=decltype(&cblas_gemm_s8u8s32);
#define PROBE_GEMM cblas_gemm_s8u8s32
#define PROBE_GEMM_NAME "cblas_gemm_s8u8s32"
constexpr size_t GEMM_ARGUMENTS=18;
#endif
template<class>struct Signature;
template<class R,class... A>struct Signature<R(*)(A...)>{
    using Return=R;using Args=std::tuple<A...>;
};
template<class R,class... A>struct Signature<R(*)(A...) noexcept>:Signature<R(*)(A...)>{};
static_assert(std::is_same<Signature<Gemm>::Return,void>::value,"Expected void GEMM ABI");
static_assert(std::tuple_size<Signature<Gemm>::Args>::value==GEMM_ARGUMENTS,"Unexpected pinned GEMM ABI");
template<size_t I>using GemmArg=typename std::tuple_element<I,Signature<Gemm>::Args>::type;

// Snapshot: [ABI, fields, enabled, active, prepare calls/ns/failures,
// row calls/ns/failures, GEMM calls/ns, nested GEMM calls/ns,
// outside-row GEMM calls/ns, clock errors, resolution errors, control errors,
// resolved]. All counters are uint64_t and all durations are nanoseconds.
enum Field:size_t {ABI,FIELDS,ENABLED,ACTIVE,PREP_CALLS,PREP_NS,PREP_FAILURES,
    ROW_CALLS,ROW_NS,ROW_FAILURES,GEMM_CALLS,GEMM_NS,NESTED_CALLS,NESTED_NS,
    OUTSIDE_CALLS,OUTSIDE_NS,CLOCK_ERRORS,RESOLUTION_ERRORS,CONTROL_ERRORS,
    RESOLVED,FIELD_COUNT};
constexpr uint64_t EnabledBit=uint64_t(1)<<63;
constexpr uint64_t ActiveMask=EnabledBit-1;
std::atomic<uint64_t> gate{0}; // Starts off. Low bits count tracked wrappers.
std::atomic<uint64_t> counters[FIELD_COUNT]{};
std::mutex control_mutex;
std::once_flag resolve_once;
std::atomic<bool> resolve_done{false};
bool resolve_ok=false;
Prepare real_prepare=nullptr;
Rows real_rows=nullptr;
Gemm real_gemm=nullptr;
void* core_handle=nullptr;
void* mkl_handle=nullptr;
thread_local bool resolving=false;
thread_local unsigned row_depth=0;

[[noreturn]] void fatal_resolution() noexcept {
    constexpr char message[]="precision_probe: cannot safely resolve original CPU functions\n";
    (void)!write(STDERR_FILENO,message,sizeof(message)-1);
    _exit(127);
}

bool same_file(void* symbol,const char* expected) noexcept {
    Dl_info owner{};struct stat actual{},wanted{};
    return dladdr(symbol,&owner)&&owner.dli_fname&&
        stat(owner.dli_fname,&actual)==0&&stat(expected,&wanted)==0&&
        actual.st_dev==wanted.st_dev&&actual.st_ino==wanted.st_ino;
}
void* symbol(const char* name,const char* env,void*& retained,void* self) noexcept {
    const char* path=std::getenv(env);
    if(!path||path[0]!='/'){
        counters[RESOLUTION_ERRORS].fetch_add(1,std::memory_order_relaxed);return nullptr;
    }
    dlerror();void* found=dlsym(RTLD_NEXT,name);const char* problem=dlerror();
    if(!problem&&found&&found!=self&&same_file(found,path))return found;
    if(!retained)retained=dlopen(path,RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL);
    if(retained){
        dlerror();found=dlsym(retained,name);problem=dlerror();
        if(!problem&&found&&found!=self&&same_file(found,path))return found;
    }
    counters[RESOLUTION_ERRORS].fetch_add(1,std::memory_order_relaxed);
    return nullptr;
}

bool initialize() noexcept {
    if(resolving)fatal_resolution(); // Never reenter call_once through dlsym.
    const int saved_errno=errno;
    if(!resolve_done.load(std::memory_order_acquire)){
        try{
            std::call_once(resolve_once,[]{
                resolving=true;
                real_prepare=reinterpret_cast<Prepare>(symbol("ip_prepare","IP_PROBE_CORE_LIBRARY",core_handle,
                    reinterpret_cast<void*>(&ip_prepare)));
                real_rows=reinterpret_cast<Rows>(symbol("ip_run_rows","IP_PROBE_CORE_LIBRARY",core_handle,
                    reinterpret_cast<void*>(&ip_run_rows)));
                real_gemm=reinterpret_cast<Gemm>(symbol(PROBE_GEMM_NAME,"IP_PROBE_GEMM_LIBRARY",mkl_handle,
                    reinterpret_cast<void*>(&PROBE_GEMM)));
                resolve_ok=real_prepare&&real_rows&&real_gemm;
                resolving=false;resolve_done.store(true,std::memory_order_release);
            });
        }catch(...){
            resolving=false;counters[RESOLUTION_ERRORS].fetch_add(1,std::memory_order_relaxed);
            errno=saved_errno;return false;
        }
    }
    errno=saved_errno;return resolve_ok;
}

bool enter() noexcept {
    uint64_t state=gate.load(std::memory_order_acquire);
    while(state&EnabledBit){
        if((state&ActiveMask)==ActiveMask){
            counters[CONTROL_ERRORS].fetch_add(1,std::memory_order_relaxed);return false;
        }
        if(gate.compare_exchange_weak(state,state+1,std::memory_order_acq_rel,std::memory_order_acquire))return true;
    }
    return false;
}
bool now(uint64_t& value) noexcept {
    const int saved_errno=errno;timespec time{};
    const int result=clock_gettime(CLOCK_MONOTONIC,&time);errno=saved_errno;
    if(result||time.tv_sec<0||time.tv_nsec<0||time.tv_nsec>=1000000000L){
        counters[CLOCK_ERRORS].fetch_add(1,std::memory_order_relaxed);return false;
    }
    value=uint64_t(time.tv_sec)*1000000000ull+uint64_t(time.tv_nsec);return true;
}
struct Span {
    Field calls,ns;bool tracked,start_valid,nested;uint64_t start=0;
    Span(Field c,Field n):calls(c),ns(n),tracked(enter()),start_valid(false),nested(row_depth!=0){
        if(tracked){
            counters[calls].fetch_add(1,std::memory_order_relaxed);
            start_valid=now(start);
            if(calls==ROW_CALLS)++row_depth;
        }
    }
    void failure(Field field)const noexcept {
        if(tracked)counters[field].fetch_add(1,std::memory_order_relaxed);
    }
    ~Span(){
        if(!tracked)return;
        uint64_t stop=0,elapsed=0;const bool valid=now(stop);
        if(start_valid&&valid){
            if(stop>=start)elapsed=stop-start;
            else counters[CLOCK_ERRORS].fetch_add(1,std::memory_order_relaxed);
        }
        counters[ns].fetch_add(elapsed,std::memory_order_relaxed);
        if(calls==GEMM_CALLS){
            counters[nested?NESTED_CALLS:OUTSIDE_CALLS].fetch_add(1,std::memory_order_relaxed);
            counters[nested?NESTED_NS:OUTSIDE_NS].fetch_add(elapsed,std::memory_order_relaxed);
        }
        if(calls==ROW_CALLS)--row_depth;
        gate.fetch_sub(1,std::memory_order_release);
    }
};
int control_error() noexcept {
    counters[CONTROL_ERRORS].fetch_add(1,std::memory_order_relaxed);return -1;
}
}

extern "C" {
PROBE_API int ip_probe_initialize(){return initialize()?0:-1;}
PROBE_API int ip_probe_set_enabled(int enabled){
    if(enabled!=0&&enabled!=1)return control_error();
    if(enabled&&!initialize())return -1;
    std::lock_guard<std::mutex> lock(control_mutex);
    uint64_t expected=enabled?0:EnabledBit;
    const uint64_t desired=enabled?EnabledBit:0;
    if(gate.compare_exchange_strong(expected,desired,std::memory_order_acq_rel))return 0;
    if(expected==desired)return 0; // Idempotent only at a quiescent boundary.
    return control_error();
}
PROBE_API int ip_probe_reset(){
    std::lock_guard<std::mutex> lock(control_mutex);
    if(gate.load(std::memory_order_acquire)!=0)return control_error();
    // Resolution errors describe process setup and cannot be erased by a phase reset.
    for(size_t i=0;i<FIELD_COUNT;++i)if(i!=RESOLUTION_ERRORS)counters[i].store(0,std::memory_order_relaxed);
    return 0;
}
PROBE_API int ip_probe_snapshot(uint64_t* output,size_t fields){
    if(!output||fields<FIELD_COUNT)return control_error();
    std::lock_guard<std::mutex> lock(control_mutex);
    const uint64_t state=gate.load(std::memory_order_acquire);
    if(state)return control_error();
    for(size_t i=0;i<FIELD_COUNT;++i)output[i]=counters[i].load(std::memory_order_relaxed);
    output[ABI]=1;output[FIELDS]=FIELD_COUNT;output[ENABLED]=(state&EnabledBit)?1:0;
    output[ACTIVE]=state&ActiveMask;output[RESOLVED]=resolve_done.load(std::memory_order_acquire)&&resolve_ok?1:0;
    return 0;
}

PROBE_API void* ip_prepare(const void* plan,const float* input,int time){
    if(!initialize())fatal_resolution();
    Span span(PREP_CALLS,PREP_NS);void* result=real_prepare(plan,input,time);
    if(!result)span.failure(PREP_FAILURES);return result;
}
PROBE_API int ip_run_rows(const void* plan,const void* input,float* output,int first,int last){
    if(!initialize())fatal_resolution();
    Span span(ROW_CALLS,ROW_NS);const int result=real_rows(plan,input,output,first,last);
    if(result)span.failure(ROW_FAILURES);return result;
}
// Types are taken from the selected library's actual pinned C declaration.
#if defined(PROBE_AOCL)
PROBE_API void aocl_gemm_u8s8s32os32(
    GemmArg<0> order,GemmArg<1> ta,GemmArg<2> tb,GemmArg<3> m,GemmArg<4> n,
    GemmArg<5> k,GemmArg<6> alpha,GemmArg<7> a,GemmArg<8> lda,GemmArg<9> af,
    GemmArg<10> b,GemmArg<11> ldb,GemmArg<12> bf,GemmArg<13> beta,
    GemmArg<14> c,GemmArg<15> ldc,GemmArg<16> metadata) {
    if(!initialize())fatal_resolution();
    Span span(GEMM_CALLS,GEMM_NS);
    real_gemm(order,ta,tb,m,n,k,alpha,a,lda,af,b,ldb,bf,beta,c,ldc,metadata);
}
#else
// Types are taken from the actual pinned oneMKL declaration. This preserves
// pointer signedness, offset widths and LP64 scalar arguments exactly.
PROBE_API void cblas_gemm_s8u8s32(
    GemmArg<0> layout,GemmArg<1> trans_a,GemmArg<2> trans_b,GemmArg<3> offset,
    GemmArg<4> m,GemmArg<5> n,GemmArg<6> k,GemmArg<7> alpha,
    GemmArg<8> a,GemmArg<9> lda,GemmArg<10> ao,GemmArg<11> b,
    GemmArg<12> ldb,GemmArg<13> bo,GemmArg<14> beta,GemmArg<15> c,
    GemmArg<16> ldc,GemmArg<17> co) noexcept {
    if(!initialize())fatal_resolution();
    Span span(GEMM_CALLS,GEMM_NS);
    real_gemm(layout,trans_a,trans_b,offset,m,n,k,alpha,a,lda,ao,b,ldb,bo,beta,c,ldc,co);
}
#endif
}
