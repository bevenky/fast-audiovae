// Experiment only: one ORT task queue for the two first-projection branches.
// Inputs: X[B,2048,M], constant Wcurrent[8192,2048], Wprevious[8192,2048].
// Outputs: current[B,8192,M], previous[B,8192,M]. No bias/phase/state rewrite.
// task_grain is N64 panels per task, NOT a worker count. ORT owns the pool;
// threads=1 still describes each unchanged inner native matrix kernel.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "multitile_tasks.h"

#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__FAST_MATH__)
#error Apple ARM64 without fast-math required
#endif
#define EXPORT __attribute__((visibility("default")))
extern "C" {
int av_multitile_supported();
int av_libxsmm_panel_supported();
const char* av_libxsmm_panel_error();
void* av_libxsmm_panel_create(size_t,size_t,size_t,const float*,size_t);
void av_libxsmm_panel_destroy(void*);
int av_libxsmm_panel_run(void*,size_t,const float*,size_t,float*,size_t);
void* av_libxsmm_panel_begin(void*,size_t,const float*,size_t,float*,size_t);
int av_libxsmm_panel_tile(void*,size_t,size_t);
int av_libxsmm_panel_finish(void*);
void av_libxsmm_panel_abort(void*);
}
namespace {
constexpr const char* kDomain="fast.audiovae.apple.paired.first.v1";
constexpr size_t kPanels=128,kMaxTasks=2*kPanels;
std::atomic<uint64_t> calls{0},packs{0},serial_calls{0},parallel_calls{0},
                      fallback_calls{0},fallback_thread_checks{0},task_calls{0};
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
OrtStatus* error(const OrtApi& api) noexcept {
  try{throw;}
  catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"Paired projection bridge failure");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
  require(value!=nullptr && value.IsTensor(),"Dense tensor required");
  auto info=value.GetTensorTypeAndShapeInfo();
  require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
  require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
  return info.GetShape();
}
size_t count(int64_t a,int64_t b,int64_t c=1){
  size_t result=1;
  for(int64_t v:{a,b,c}){
    require(v>=0,"Negative dimension");
    require(v==0 || result<=std::numeric_limits<size_t>::max()/static_cast<size_t>(v),"Count overflow");
    result*=static_cast<size_t>(v);
  }
  require(result<=std::numeric_limits<size_t>::max()/sizeof(float),"Byte count overflow");
  return result;
}
bool disjoint(const float* a,size_t ac,const float* b,size_t bc){
  if(!ac || !bc)return true;
  if(!a || !b || ac>UINTPTR_MAX/4 || bc>UINTPTR_MAX/4)return false;
  auto aa=reinterpret_cast<uintptr_t>(a),bb=reinterpret_cast<uintptr_t>(b);
  if(aa>UINTPTR_MAX-ac*4 || bb>UINTPTR_MAX-bc*4)return false;
  return !(aa<bb+bc*4 && bb<aa+ac*4);
}
int64_t optional_trace(Ort::ConstKernelInfo info){
  try{return info.GetAttribute<int64_t>("trace");}
  catch(const Ort::Exception& e){
    if(e.GetOrtErrorCode()!=ORT_FAIL ||
       std::strcmp(e.what(),"No attribute with name:'trace'is defined.")!=0)throw;
    return 0;
  }
}
struct CurrentDestroy{void operator()(void* p)const noexcept{av_libxsmm_panel_destroy(p);}};
struct PreviousDestroy{void operator()(void* p)const noexcept{av_multitile_destroy(p);}};
struct CurrentLease{
  void* work=nullptr;
  ~CurrentLease(){if(work)av_libxsmm_panel_abort(work);}
  int finish(){void* p=work;work=nullptr;return av_libxsmm_panel_finish(p);}
};
struct PreviousLease{
  void* region=nullptr;
  ~PreviousLease(){if(region)(void)av_multitile_finish(region,0);}
  int finish(){void* p=region;region=nullptr;return av_multitile_finish(p,1);}
};

// Trace is opt-in and belongs to an isolated diagnostic call. No clocks,
// thread-ID queries or overlap atomics execute in callbacks when trace=0.
// Snapshot is the last traced batch (normally B=1); read only after Session.run.
// Times are mach_absolute_time ticks, including the instrumentation overhead.
struct TraceRecord{
  uint64_t worker=0,start=0,end=0,branch=0,first_panel=0,panel_count=0,status=0;
};
struct Trace{
  uint64_t records=0,mode=0,m=0,grain=0,prepare_start=0,dispatch_start=0,
           dispatch_end=0,finish_end=0,status=1,batch=0,timebase_num=0,timebase_den=0;
  std::atomic<uint64_t> active{0},peak{0};
  std::array<TraceRecord,kMaxTasks> task{};
  void start(size_t i,size_t branch,size_t first,size_t length)noexcept{
    auto& r=task[i];r.branch=branch;r.first_panel=first;r.panel_count=length;
    (void)pthread_threadid_np(nullptr,&r.worker);
    r.start=mach_absolute_time();
    const auto current=active.fetch_add(1,std::memory_order_relaxed)+1;
    auto previous=peak.load(std::memory_order_relaxed);
    while(current>previous && !peak.compare_exchange_weak(previous,current,std::memory_order_relaxed)){}
  }
  void end(size_t i,int result)noexcept{
    task[i].end=mach_absolute_time();
    task[i].status=static_cast<uint64_t>(static_cast<int64_t>(result));
    active.fetch_sub(1,std::memory_order_relaxed);
  }
};
struct Snapshot{
  std::array<uint64_t,16> summary{};
  std::array<TraceRecord,kMaxTasks> task{};
};
std::mutex trace_mutex;
Snapshot last_trace;
void publish(const Trace& t){
  std::lock_guard<std::mutex> lock(trace_mutex);
  last_trace.summary={1,t.records,t.mode,t.m,t.grain,t.prepare_start,t.dispatch_start,
    t.dispatch_end,t.finish_end,t.peak.load(std::memory_order_relaxed),t.status,t.batch,
    t.timebase_num,t.timebase_den,0,0};
  last_trace.task=t.task;
}
struct PublishTrace{
  Trace* trace;
  ~PublishTrace()noexcept{if(trace){if(!trace->finish_end)trace->finish_end=mach_absolute_time();
    try{publish(*trace);}catch(...){/* Diagnostics cannot throw during cleanup. */}}}
};
struct TaskResult{int status=-1;bool done=false;};
struct Work{
  void* current;
  void* previous;
  size_t grain;
  Trace* trace;
  std::array<TaskResult,kMaxTasks> result{};
};
void task(void* raw,size_t index)noexcept{
  auto& work=*static_cast<Work*>(raw);
  // ORT emits each index once. Each pair addresses the same N range on
  // different branches, so consecutive tasks interleave current/previous.
  const size_t group=index/2,branch=index%2,first=group*work.grain;
  const size_t width=std::min(work.grain,kPanels-first);
  if(work.trace)work.trace->start(index,branch,first,width);
  int result=-1;
  try{result=branch ? av_multitile_task(work.previous,group)
                    : av_libxsmm_panel_tile(work.current,first,width);}
  catch(...){result=-2;}
  work.result[index].status=result;
  work.result[index].done=true;
  if(work.trace)work.trace->end(index,result);
}
struct Kernel{
  const OrtApi& api;
  int64_t k,n,max_m,grain,serial,trace_enabled;
  std::unique_ptr<void,CurrentDestroy> current;
  std::unique_ptr<void,PreviousDestroy> previous;
  std::atomic_flag busy=ATOMIC_FLAG_INIT;
  Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
    Ort::ConstKernelInfo info(raw);
    require(info.GetAttribute<int64_t>("matrix_abi")==1,"Matrix ABI mismatch");
    require(info.GetAttribute<int64_t>("threads")==1,"Inner kernels require threads=1");
    k=info.GetAttribute<int64_t>("k");n=info.GetAttribute<int64_t>("n");
    max_m=info.GetAttribute<int64_t>("max_m");
    grain=info.GetAttribute<int64_t>("task_grain");serial=info.GetAttribute<int64_t>("serial_mode");
    trace_enabled=optional_trace(info);
    require(k==2048 && n==8192 && max_m==2,"Only first-pair geometry is supported");
    require(grain>0 && grain<=128,"task_grain must be in [1,128] N64 panels");
    require(serial==0 || serial==1,"serial_mode must be 0 or 1");
    require(trace_enabled==0 || trace_enabled==1,"trace must be 0 or 1");
    require(av_libxsmm_panel_supported()==1 && av_multitile_supported()==1,"Runtime SME and SME2 required");
    int is_current_constant=0,is_previous_constant=0;
    auto wc=info.GetTensorConstantInput(1,&is_current_constant);
    auto wp=info.GetTensorConstantInput(2,&is_previous_constant);
    require(is_current_constant && is_previous_constant && wc!=nullptr && wp!=nullptr,
            "Both weights must be non-overridable constant inputs");
    require(shape(wc)==std::vector<int64_t>({n,k}) && shape(wp)==std::vector<int64_t>({n,k}),
            "Both constant weights must have shape [N,K]");
    current.reset(av_libxsmm_panel_create(k,n,max_m,wc.GetTensorData<float>(),count(n,k)));
    if(!current)throw std::runtime_error(av_libxsmm_panel_error());
    previous.reset(av_multitile_create(k,n,max_m,wp.GetTensorData<float>(),count(n,k)));
    if(!previous)throw std::runtime_error(av_multitile_error());
    // Current N64 panel granularity must also preserve the original KAI tile.
    const auto alignment=av_multitile_stat(previous.get(),11);
    require(alignment && 64%alignment==0,"Multitile alignment is incompatible with N64 panels");
    packs.fetch_add(2,std::memory_order_relaxed);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw)noexcept{
    try{
      Ort::KernelContext ctx(raw);
      auto x=ctx.GetInput(0),wc=ctx.GetInput(1),wp=ctx.GetInput(2);
      auto dims=shape(x);
      require(dims.size()==3 && dims[0]>=0 && dims[1]==k && dims[2]>=0 && dims[2]<=INT32_MAX,
              "Expected [B,2048,M] within BLAS column limits");
      require(shape(wc)==std::vector<int64_t>({n,k}) && shape(wp)==std::vector<int64_t>({n,k}),
              "Constant weight shape changed");
      const auto total_x=count(dims[0],k,dims[2]),total_y=count(dims[0],n,dims[2]),weight_count=count(n,k);
      auto outdims=dims;outdims[1]=n;
      auto yc=ctx.GetOutput(0,outdims),yp=ctx.GetOutput(1,outdims);
      if(!dims[0] || !dims[2])return nullptr;
      const auto xc=count(k,dims[2]),out_count=count(n,dims[2]);
      const float* input=x.GetTensorData<float>();
      const float* current_weight=wc.GetTensorData<float>();const float* previous_weight=wp.GetTensorData<float>();
      float* current_output=yc.GetTensorMutableData<float>();float* previous_output=yp.GetTensorMutableData<float>();
      require(disjoint(current_output,total_y,previous_output,total_y),"Projection outputs must not alias");
      for(float* out:{current_output,previous_output}){
        require(disjoint(out,total_y,input,total_x) && disjoint(out,total_y,current_weight,weight_count) &&
                disjoint(out,total_y,previous_weight,weight_count),"Output overlaps input or immutable weights");
      }
      // All scratch lives in the two existing handles; concurrent reuse of one
      // Kernel is rejected, while ORT workers within this call share a single lease.
      require(!busy.test_and_set(std::memory_order_acquire),"Concurrent paired-kernel reuse prohibited");
      struct Release{std::atomic_flag& flag;~Release(){flag.clear(std::memory_order_release);}} release{busy};
      const bool fallback=dims[2]>max_m;
      if(fallback){
        require(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)==0,"Single-thread BLAS policy failed");
        require(BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED,"BLAS policy was not applied");
        fallback_thread_checks.fetch_add(1,std::memory_order_relaxed);
      }
      for(int64_t b=0;b<dims[0];++b){
        const float* in=input+b*xc;float* out_current=current_output+b*out_count;
        float* out_previous=previous_output+b*out_count;
        std::unique_ptr<Trace> trace;
        if(trace_enabled){
          trace=std::make_unique<Trace>();trace->mode=fallback?2:static_cast<uint64_t>(serial);
          trace->m=dims[2];trace->grain=grain;trace->batch=b;
          mach_timebase_info_data_t timebase{};mach_timebase_info(&timebase);
          trace->timebase_num=timebase.numer;trace->timebase_den=timebase.denom;
          trace->prepare_start=mach_absolute_time();
        }
        PublishTrace publish_trace{trace.get()};
        if(serial || fallback){
          if(trace){trace->records=2;trace->dispatch_start=mach_absolute_time();}
          for(size_t branch=0;branch<2;++branch){
            if(trace)trace->start(branch,branch,0,kPanels);
            int result=0;
            if(fallback){
              cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,static_cast<int>(n),static_cast<int>(dims[2]),
                static_cast<int>(k),1.0f,branch?previous_weight:current_weight,static_cast<int>(k),in,
                static_cast<int>(dims[2]),0.0f,branch?out_previous:out_current,static_cast<int>(dims[2]));
              fallback_calls.fetch_add(1,std::memory_order_relaxed);
            }else{
              result=branch?av_multitile_run(previous.get(),dims[2],in,xc,out_previous,out_count)
                           :av_libxsmm_panel_run(current.get(),dims[2],in,xc,out_current,out_count);
            }
            if(trace)trace->end(branch,result);
            if(result)throw std::runtime_error(branch?av_multitile_error():av_libxsmm_panel_error());
          }
          if(trace)trace->dispatch_end=mach_absolute_time();
          if(!fallback)serial_calls.fetch_add(1,std::memory_order_relaxed);
        }else{
          CurrentLease current_lease;
          current_lease.work=av_libxsmm_panel_begin(current.get(),dims[2],in,xc,out_current,out_count);
          if(!current_lease.work)throw std::runtime_error(av_libxsmm_panel_error());
          PreviousLease previous_lease;
          if(av_multitile_begin(previous.get(),dims[2],in,xc,out_previous,out_count,64*static_cast<size_t>(grain)))
            throw std::runtime_error(av_multitile_error());
          previous_lease.region=previous.get();
          const size_t groups=(kPanels+static_cast<size_t>(grain)-1)/static_cast<size_t>(grain);
          require(av_multitile_task_count(previous.get())==groups,"Backend task count differs from common panel groups");
          Work work{current_lease.work,previous.get(),static_cast<size_t>(grain),trace.get(),{}};
          if(trace){trace->records=2*groups;trace->dispatch_start=mach_absolute_time();}
          // Both branches are prepared before this single synchronous join.
          // On failure the leases below abort only after ParallelFor returns.
          OrtStatus* status=api.KernelContext_ParallelFor(raw,task,2*groups,0,&work);
          if(trace)trace->dispatch_end=mach_absolute_time();
          struct StatusRelease{const OrtApi& api;OrtStatus* status;~StatusRelease(){if(status)api.ReleaseStatus(status);}}
            status_release{api,status};
          size_t done=0;bool success=true;
          for(size_t i=0;i<2*groups;++i){done+=work.result[i].done;success&=work.result[i].done && work.result[i].status==0;}
          task_calls.fetch_add(done,std::memory_order_relaxed);
          if(status)throw std::runtime_error(api.GetErrorMessage(status));
          require(success && done==2*groups,"Paired task failed or did not complete; see optional task trace");
          // finish validates complete coverage. It consumes/releases its lease
          // even on error. Any still-owned peer is aborted by RAII after join.
          if(current_lease.finish())throw std::runtime_error(av_libxsmm_panel_error());
          if(previous_lease.finish())throw std::runtime_error(av_multitile_error());
          parallel_calls.fetch_add(1,std::memory_order_relaxed);
        }
        calls.fetch_add(1,std::memory_order_relaxed);
        if(trace){trace->finish_end=mach_absolute_time();trace->status=0;}
      }
      return nullptr;
    }catch(...){return error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
  Op(){start_ver_=1;end_ver_=1;}
  const char* GetName()const{return "PairedFirstProjectionsF32";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 3;}
  size_t GetOutputTypeCount()const{return 2;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept{
    *out=nullptr;try{*out=new Kernel(api,info);return nullptr;}catch(...){return error(api);}
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& ctx)noexcept{
    try{
      auto dims=ctx.GetInputShape(0);require(dims.size()==3,"Rank-three input required");
      require(ctx.GetAttrInt("k")==2048 && ctx.GetAttrInt("n")==8192 && ctx.GetAttrInt("max_m")==2,
              "Only first-pair geometry is supported");
      dims[1]=8192;
      auto first=ctx.SetOutputShape(0,dims);if(first)return first.release();
      return ctx.SetOutputShape(1,dims).release();
    }catch(...){return error(Ort::GetApi());}
  }
};
std::mutex registration_mutex;const OrtApi* registered=nullptr;Op op;
std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" EXPORT uint64_t av_paired_first_calls(){return calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_weight_packs(){return packs.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_serial_calls(){return serial_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_parallel_calls(){return parallel_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_task_calls(){return task_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_fallback_calls(){return fallback_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_paired_first_fallback_thread_checks(){return fallback_thread_checks.load(std::memory_order_relaxed);}
// Summary fields (index ignored): 0version,1record_count,2mode(0pool/1serial/2BLAS),
// 3M,4grain,5prepare_start,6dispatch_start,7dispatch_end,8finish_end,9peak_callbacks,
// 10status(0success),11batch_index,12timebase_numer,13timebase_denom.
// Task fields (index < record_count):32worker_id,33start,34end,35branch(0current),
// 36first_N64_panel,37panel_count,38signed_return_code cast to uint64.
// The dispatch envelope includes execution AND join; it is not pure wait time.
// Read after one traced Session.run in a serialized diagnostic harness.
extern "C" EXPORT uint64_t av_paired_first_trace_get(size_t field,size_t index)noexcept{
  try{
    std::lock_guard<std::mutex> lock(trace_mutex);
    if(field<last_trace.summary.size())return last_trace.summary[field];
    if(index>=last_trace.summary[1] || index>=kMaxTasks)return 0;
    const auto& t=last_trace.task[index];
    switch(field){case 32:return t.worker;case 33:return t.start;case 34:return t.end;
      case 35:return t.branch;case 36:return t.first_panel;case 37:return t.panel_count;case 38:return t.status;}
  }catch(...){}
  return 0;
}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
  const OrtApi* api=base->GetApi(ORT_API_VERSION);
  if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API 29 required");
  try{
    std::lock_guard<std::mutex> lock(registration_mutex);
    require(!registered || registered==api,"Multiple runtimes unsupported");
    if(!registered){Ort::InitApi(api);auto fresh=std::make_unique<Ort::CustomOpDomain>(kDomain);fresh->Add(&op);
      domain=std::move(fresh);registered=api;}
    return api->AddCustomOpDomain(options,*domain);
  }catch(...){return error(*api);}
}
