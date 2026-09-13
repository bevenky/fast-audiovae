// Explicit FP32 SME JIT; constant transpose once, direct BCT-compatible output.
#include "onnxruntime_c_api.h"
#include "libxsmm.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <vector>
#include <sys/sysctl.h>
#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__FAST_MATH__)
#error Apple ARM64 without fast-math required
#endif
#define EXPORT __attribute__((visibility("default")))
namespace {
thread_local char error_text[256]={};
std::mutex jit_mutex;
std::atomic<uint64_t> jit_invocations{0};
std::atomic<uint64_t> parallel_invocations{0},parallel_tasks{0},parallel_fpcr_checks{0};
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
void failed()noexcept{try{throw;}catch(const std::exception& e){std::snprintf(error_text,sizeof(error_text),"%s",e.what());}catch(...){std::snprintf(error_text,sizeof(error_text),"LIBXSMM error");}}
bool flag(const char* key){int value=0;size_t size=sizeof(value);return sysctlbyname(key,&value,&size,nullptr,0)==0 && value==1;}
bool supported(){return flag("hw.optional.arm.FEAT_SME") && flag("hw.optional.arm.FEAT_SME2");}
void fpcr_check(){uint64_t f;__asm__ volatile("mrs %0, fpcr":"=r"(f));require((f&((uint64_t(3)<<22)|(uint64_t(1)<<24)|3))==0,"Nearest FP32 without FZ/FIZ/AH required");}
struct Code {
  libxsmm_gemmfunction fn=nullptr;
  size_t bytes=0,starts=0,stops=0,mopas=0;
  uint64_t hash=14695981039346656037ULL;
};
struct Region {
  size_t k,n,max_m;
  std::vector<float> transposed_weight;
  Code code[2];
  int detected=0,target=0;
  uint64_t runs=0;
  std::atomic_flag busy=ATOMIC_FLAG_INIT;
  Region(size_t K,size_t N,size_t M,const float* w,size_t count):k(K),n(N),max_m(M){
    require(supported(),"Runtime SME and SME2 required");fpcr_check();
    require(k==2048 && n==8192 && max_m==2 && w && count==n*k,"Only first-pair FP32 W[N,K] is supported");
    transposed_weight.resize(count);
    for(size_t depth=0;depth<k;++depth)for(size_t channel=0;channel<n;++channel){
      float v=w[channel*k+depth];require(std::isfinite(v),"Nonfinite original weight");
      transposed_weight[(channel/64)*k*64+depth*64+channel%64]=v;
    }
    std::lock_guard<std::mutex> lock(jit_mutex);
    libxsmm_init();detected=libxsmm_cpuid(nullptr);
    int previous=libxsmm_get_target_archid();
    struct Restore{int value;~Restore(){libxsmm_set_target_archid(value);}} restore{previous};
    libxsmm_set_target_archid(LIBXSMM_AARCH64_APPL_M4);
    target=libxsmm_get_target_archid();require(target==2501,"Explicit Apple SME target not selected");
    for(size_t m=1;m<=2;++m){
      // Column-major A[T,K] aliases BCT X[K,T]. B is once-prepared N64-panel W
      // supplied with TRANS_B and ldb64. Column-major C[T,N] aliases BCT Y[N,T].
      const auto shape=libxsmm_create_gemm_shape(m,64,k,m,64,m,
        LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
      auto& c=code[m-1];
      c.fn=libxsmm_dispatch_gemm(shape,LIBXSMM_GEMM_FLAG_TRANS_B|LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
      require(c.fn!=nullptr,"Explicit FP32 SME dispatch returned no kernel");
      libxsmm_xmmfunction f{};f.gemm=c.fn;
      libxsmm_mmkernel_info info{};libxsmm_kernel_info ki{};
      require(libxsmm_get_mmkernel_info(f,&info)==0 && libxsmm_get_kernel_info(f.ptr_const,&ki)==0,"JIT descriptor unavailable");
      require(info.m==m && info.n==64 && info.k==k && info.lda==m && info.ldb==64 && info.ldc==m &&
        info.iprecision==LIBXSMM_DATATYPE_F32 && info.oprecision==LIBXSMM_DATATYPE_F32 &&
        info.flags==(LIBXSMM_GEMM_FLAG_TRANS_B|LIBXSMM_GEMM_FLAG_BETA_0|LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) &&
        info.prefetch==LIBXSMM_GEMM_PREFETCH_NONE && !ki.is_reference_kernel && ki.code_size>0 && ki.code_size<=1024*1024,
        "Returned JIT descriptor/precision differs from explicit request");
      c.bytes=ki.code_size;
      const auto* bytes=static_cast<const uint8_t*>(f.ptr_const);
      for(size_t i=0;i<c.bytes;++i){c.hash^=bytes[i];c.hash*=1099511628211ULL;}
      constexpr uint32_t variables=0x3u|(0x1fu<<5)|(0x1fu<<16)|(7u<<10)|(7u<<13);
      for(size_t i=0;i+4<=c.bytes;i+=4){
        uint32_t op;std::memcpy(&op,bytes+i,4);
        c.starts+=op==0xd503477f;c.stops+=op==0xd503467f;
        c.mopas+=(op&~variables)==0x80800000;
      }
      require(c.starts>0 && c.stops>0 && c.mopas>0,"Generated code is not the requested FP32 SME MOPA path");
    }
  }
  void run(size_t m,const float* x,size_t xc,float* y,size_t yc){
    require(m>0 && m<=2 && x && y && xc==k*m && yc==n*m,"Invalid BCT buffers");fpcr_check();
    auto xa=reinterpret_cast<uintptr_t>(x),ya=reinterpret_cast<uintptr_t>(y);
    require(xa<=UINTPTR_MAX-xc*4 && ya<=UINTPTR_MAX-yc*4 && !(xa<ya+yc*4 && ya<xa+xc*4),"Buffers overlap or overflow");
    require(!busy.test_and_set(std::memory_order_acquire),"Concurrent region reuse prohibited");
    struct Release{std::atomic_flag& f;~Release(){f.clear(std::memory_order_release);}} release{busy};
    libxsmm_gemm_param p{};p.a.primary=const_cast<float*>(x);p.b.primary=transposed_weight.data();p.c.primary=y;
    for(size_t column=0;column<n;column+=64){
      p.b.primary=transposed_weight.data()+column*k;p.c.primary=y+column*m;
      code[m-1].fn(&p);jit_invocations.fetch_add(1,std::memory_order_relaxed);
    }
    ++runs;
  }
  // A task owns a contiguous group of complete N64 panels. ORT supplies the
  // workers; neither packed weights nor JIT code is changed during execution.
  struct TaskResult {
    size_t completed=0;
    bool fpcr_checked=false,done=false;
    unsigned error=0;
  };
  struct ParallelWork {
    Region* region;
    size_t m,panels_per_task;
    const float* x;
    float* y;
    std::array<TaskResult,128> results{};
  };
  static void panel_task(void* raw,size_t task)noexcept {
    auto& work=*static_cast<ParallelWork*>(raw);
    auto& result=work.results[task];
    size_t completed=0;
    try {
      // FPCR is thread-local, so checking only the initiating thread is not
      // sufficient when an ORT worker executes the existing SME kernel.
      fpcr_check();result.fpcr_checked=true;
      auto& r=*work.region;
      const size_t begin=task*work.panels_per_task;
      const size_t end=begin+work.panels_per_task<128?begin+work.panels_per_task:128;
      libxsmm_gemm_param p{};p.a.primary=const_cast<float*>(work.x);
      for(size_t panel=begin;panel<end;++panel){
        p.b.primary=r.transposed_weight.data()+panel*64*r.k;
        p.c.primary=work.y+panel*64*work.m;
        r.code[work.m-1].fn(&p);++completed;
      }
    }catch(const std::exception&){result.error=1;}
    catch(...){result.error=2;}
    result.completed=completed;result.done=true;
  }
  void run_parallel(size_t m,const float* x,size_t xc,float* y,size_t yc,
                    const OrtApi* api,const OrtKernelContext* context,size_t panels_per_task){
    require(m>0 && m<=2 && x && y && xc==k*m && yc==n*m,"Invalid BCT buffers");fpcr_check();
    require(api && context && api->KernelContext_ParallelFor,"ORT parallel context required");
    require(panels_per_task>0 && panels_per_task<=128,"Panel group size must be in [1,128]");
    auto xa=reinterpret_cast<uintptr_t>(x),ya=reinterpret_cast<uintptr_t>(y);
    require(xa<=UINTPTR_MAX-xc*4 && ya<=UINTPTR_MAX-yc*4 && !(xa<ya+yc*4 && ya<xa+xc*4),"Buffers overlap or overflow");
    require(!busy.test_and_set(std::memory_order_acquire),"Concurrent region reuse prohibited");
    struct Release{std::atomic_flag& f;~Release(){f.clear(std::memory_order_release);}} release{busy};
    ParallelWork work{this,m,panels_per_task,x,y,{}};
    const size_t tasks=(128+panels_per_task-1)/panels_per_task;
    // ParallelFor joins before returning. Each callback writes only its own
    // result slot and output panels; the Region guard remains held throughout.
    OrtStatus* status=api->KernelContext_ParallelFor(context,panel_task,tasks,0,&work);
    struct StatusRelease{const OrtApi* api;OrtStatus* status;~StatusRelease(){if(status)api->ReleaseStatus(status);}} status_release{api,status};
    size_t completed=0,finished=0,checked=0;unsigned callback_error=0;
    for(size_t task=0;task<tasks;++task){
      const auto& result=work.results[task];
      completed+=result.completed;finished+=result.done;checked+=result.fpcr_checked;
      callback_error|=result.error;
    }
    // Publish actual work only after all callbacks have joined, including any
    // work completed before an error. Failed runs do not increment runs.
    jit_invocations.fetch_add(completed,std::memory_order_relaxed);
    parallel_tasks.fetch_add(finished,std::memory_order_relaxed);
    parallel_fpcr_checks.fetch_add(checked,std::memory_order_relaxed);
    if(status)throw std::runtime_error(api->GetErrorMessage(status));
    require(!callback_error,"Parallel panel worker FPCR check or execution failed");
    require(finished==tasks && checked==tasks && completed==128,"Incomplete parallel panel execution");
    ++runs;parallel_invocations.fetch_add(1,std::memory_order_relaxed);
  }
};
// External paired-region scheduler owns this lease until every task joins.
// Region storage is immutable during the lease. Each tile owns whole N64 outputs.
struct PanelWork {
  Region* region;
  size_t m;
  const float* x;
  float* y;
  std::array<std::atomic<unsigned char>,128> panels{};
  bool acquired=false;
  PanelWork(Region* r,size_t M,const float* X,size_t xc,float* Y,size_t yc)
      :region(r),m(M),x(X),y(Y) {
    require(r && m>0 && m<=2 && x && y && xc==r->k*m && yc==r->n*m,"Invalid panel lease buffers");
    fpcr_check();
    auto xa=reinterpret_cast<uintptr_t>(x),ya=reinterpret_cast<uintptr_t>(y);
    require(xa<=UINTPTR_MAX-xc*4 && ya<=UINTPTR_MAX-yc*4 && !(xa<ya+yc*4 && ya<xa+xc*4),"Panel lease buffers overlap");
    require(!r->busy.test_and_set(std::memory_order_acquire),"Concurrent region reuse prohibited");
    acquired=true;
    for(auto& v:panels)v.store(0,std::memory_order_relaxed);
  }
  ~PanelWork(){if(acquired)region->busy.clear(std::memory_order_release);}
  void tile(size_t first,size_t count) {
    require(count && first<128 && count<=128-first,"Panel tile out of bounds");
    fpcr_check();
    for(size_t p=first;p<first+count;++p){
      unsigned char zero=0;
      require(panels[p].compare_exchange_strong(zero,1,std::memory_order_relaxed),"Panel scheduled twice");
    }
    libxsmm_gemm_param param{};param.a.primary=const_cast<float*>(x);
    for(size_t p=first;p<first+count;++p){
      param.b.primary=region->transposed_weight.data()+p*64*region->k;
      param.c.primary=y+p*64*m;
      region->code[m-1].fn(&param);
      panels[p].store(2,std::memory_order_release);
    }
  }
  void finish() {
    for(auto& p:panels)require(p.load(std::memory_order_acquire)==2,"Incomplete panel lease");
    jit_invocations.fetch_add(128,std::memory_order_relaxed);++region->runs;
  }
};
}
extern "C" {
EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions*,const OrtApiBase*){return nullptr;}
EXPORT uint64_t av_libxsmm_panel_api_calls(){return jit_invocations.load(std::memory_order_relaxed);}
EXPORT uint64_t av_libxsmm_panel_parallel_calls(){return parallel_invocations.load(std::memory_order_relaxed);}
EXPORT uint64_t av_libxsmm_panel_parallel_tasks(){return parallel_tasks.load(std::memory_order_relaxed);}
EXPORT uint64_t av_libxsmm_panel_parallel_fpcr_checks(){return parallel_fpcr_checks.load(std::memory_order_relaxed);}
EXPORT int av_libxsmm_panel_supported(){return supported()?1:0;}
EXPORT const char* av_libxsmm_panel_error(){return error_text;}
EXPORT void* av_libxsmm_panel_create(size_t k,size_t n,size_t max_m,const float* w,size_t count){error_text[0]=0;try{return new Region(k,n,max_m,w,count);}catch(...){failed();return nullptr;}}
EXPORT void av_libxsmm_panel_destroy(void* h){delete static_cast<Region*>(h);}
EXPORT int av_libxsmm_panel_run(void* h,size_t m,const float* x,size_t xc,float* y,size_t yc){error_text[0]=0;try{require(h,"Missing handle");static_cast<Region*>(h)->run(m,x,xc,y,yc);return 0;}catch(...){failed();return -1;}}
EXPORT int av_libxsmm_panel_run_parallel(void* h,size_t m,const float* x,size_t xc,float* y,size_t yc,const OrtApi* api,const OrtKernelContext* context,size_t panels_per_task){error_text[0]=0;try{require(h,"Missing handle");static_cast<Region*>(h)->run_parallel(m,x,xc,y,yc,api,context,panels_per_task);return 0;}catch(...){failed();return -1;}}
EXPORT void* av_libxsmm_panel_begin(void* h,size_t m,const float* x,size_t xc,float* y,size_t yc){error_text[0]=0;try{return new PanelWork(static_cast<Region*>(h),m,x,xc,y,yc);}catch(...){failed();return nullptr;}}
EXPORT int av_libxsmm_panel_tile(void* work,size_t first,size_t count){error_text[0]=0;try{require(work,"Missing panel lease");static_cast<PanelWork*>(work)->tile(first,count);return 0;}catch(...){failed();return -1;}}
EXPORT int av_libxsmm_panel_finish(void* work){error_text[0]=0;std::unique_ptr<PanelWork> lease(static_cast<PanelWork*>(work));try{require(work,"Missing panel lease");lease->finish();return 0;}catch(...){failed();return -1;}}
EXPORT void av_libxsmm_panel_abort(void* work){delete static_cast<PanelWork*>(work);}
EXPORT uint64_t av_libxsmm_panel_stat(void* h,int field){
  if(!h)return 0;auto& p=*static_cast<Region*>(h);
  switch(field){case 0:return p.detected;case 1:return p.target;case 2:return p.transposed_weight.size()*4;case 3:return p.runs;}
  if(field>=10 && field<24){auto& c=p.code[(field-10)/7];switch((field-10)%7){case 0:return c.bytes;case 1:return c.starts;case 2:return c.stops;case 3:return c.mopas;case 4:return c.hash;case 5:return 1;case 6:return 0;}}
  return 0;
}
}
