// ORT allocation/dispatch bridge to the unchanged, separately pinned SME2 core.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "geometry.h"
#include <Accelerate/Accelerate.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if !defined(__APPLE__) || !defined(__aarch64__)
#error Apple ARM64 required
#endif
#define EXPORT __attribute__((visibility("default")))
extern "C" {
int av_sweep_supported();
const char* av_sweep_error();
void* av_sweep_create(size_t,size_t,size_t,const float*,size_t);
void av_sweep_destroy(void*);
int av_sweep_run(void*,size_t,const float*,size_t,float*,size_t);
}
namespace {
constexpr const char* kDomain="fast.audiovae.apple.matrix.sweep.v1";
std::atomic<uint64_t> calls{0},packs{0},fallback_calls{0},fallback_thread_checks{0};
void require(bool ok,const char* message) {if(!ok)throw std::invalid_argument(message);}
OrtStatus* error(const OrtApi& api) noexcept {
  try {throw;}
  catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"SME2 bridge failure");}
}
std::vector<int64_t> shape(Ort::ConstValue value) {
  require(value!=nullptr && value.IsTensor(),"Dense tensor required");
  auto info=value.GetTensorTypeAndShapeInfo();
  require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
  require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensors required");
  return info.GetShape();
}
size_t count(int64_t a,int64_t b,int64_t c=1) {
  size_t result=1;
  for(int64_t v:{a,b,c}) {
    require(v>=0,"Negative dimension");
    require(v==0 || result<=std::numeric_limits<size_t>::max()/static_cast<size_t>(v),"Count overflow");
    result*=static_cast<size_t>(v);
  }
  require(result<=std::numeric_limits<size_t>::max()/4,"Byte count overflow");return result;
}
struct Destroy {void operator()(void* p)const{av_sweep_destroy(p);}};
struct Kernel {
  const OrtApi& api;int64_t k,n,max_m;std::unique_ptr<void,Destroy> region;
  Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a) {
    Ort::ConstKernelInfo info(raw);
    require(info.GetAttribute<int64_t>("matrix_abi")==1,"Matrix ABI mismatch");
    require(info.GetAttribute<int64_t>("threads")==1,"One thread required");
    k=info.GetAttribute<int64_t>("k");n=info.GetAttribute<int64_t>("n");
    max_m=info.GetAttribute<int64_t>("max_m");
    require(approved_max_m(k,n)>0 && max_m==static_cast<int64_t>(approved_max_m(k,n)),"Unapproved matrix geometry");
    require(av_sweep_supported()==1,"Runtime SME and SME2 required");
    int constant=0;auto weight=info.GetTensorConstantInput(1,&constant);
    require(constant && weight!=nullptr,"Non-overridable constant weight required");
    require(shape(weight)==std::vector<int64_t>({n,k}),"Weight [N,K] mismatch");
    region.reset(av_sweep_create(k,n,max_m,weight.GetTensorData<float>(),count(n,k)));
    if(!region)throw std::runtime_error(av_sweep_error());
    packs.fetch_add(1,std::memory_order_relaxed);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw)noexcept {
    try {
      Ort::KernelContext ctx(raw);auto x=ctx.GetInput(0),w=ctx.GetInput(1);auto dims=shape(x);
      require(dims.size()==3 && dims[0]>=0 && dims[1]==k && dims[2]>=0 && dims[2]<=INT32_MAX,
              "Expected [B,K,M] with a supported BLAS column count");
      require(shape(w)==std::vector<int64_t>({n,k}),"Constant weight shape changed");
      count(dims[0],k,dims[2]);count(dims[0],n,dims[2]);
      auto outdims=dims;outdims[1]=n;auto out=ctx.GetOutput(0,outdims);
      if(!dims[0] || !dims[2])return nullptr;
      const size_t xc=count(k,dims[2]),yc=count(n,dims[2]);
      const float* xp=x.GetTensorData<float>();float* yp=out.GetTensorMutableData<float>();
      if(dims[2]>max_m) {
        // Preserve the existing streaming API for arbitrary longer packets.
        // Only the measured <=80 ms region uses the fixed SME scratch buffers.
        // This is the same corrected weight-left SGEMM orientation already
        // qualified independently, with the live immutable ORT weight.
        require(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)==0,"Single-thread BLAS policy failed");
        require(BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED,"BLAS policy was not applied");
        fallback_thread_checks.fetch_add(1,std::memory_order_relaxed);
        const float* wp=w.GetTensorData<float>();
        for(int64_t b=0;b<dims[0];++b) {
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,static_cast<int>(n),static_cast<int>(dims[2]),
            static_cast<int>(k),1.0f,wp,static_cast<int>(k),xp+b*xc,static_cast<int>(dims[2]),
            0.0f,yp+b*yc,static_cast<int>(dims[2]));
          fallback_calls.fetch_add(1,std::memory_order_relaxed);
        }
        return nullptr;
      }
      for(int64_t b=0;b<dims[0];++b) {
        if(av_sweep_run(region.get(),dims[2],xp+b*xc,xc,yp+b*yc,yc))
          throw std::runtime_error(av_sweep_error());
        calls.fetch_add(1,std::memory_order_relaxed);
      }
      return nullptr;
    }catch(...){return error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true> {
  Op(){start_ver_=1;end_ver_=1;}
  const char* GetName()const{return "Sme2WeightLeftF32";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 2;}
  size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** result)const noexcept {
    *result=nullptr;try{*result=new Kernel(api,info);return nullptr;}catch(...){return error(api);}
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& ctx)noexcept {
    try{auto dims=ctx.GetInputShape(0);require(dims.size()==3,"Rank-three input required");
      auto n=ctx.GetAttrInt("n"),k=ctx.GetAttrInt("k"),m=ctx.GetAttrInt("max_m");
      require(approved_max_m(k,n)>0 && m==static_cast<int64_t>(approved_max_m(k,n)),"Unapproved shape");dims[1]=n;
      return ctx.SetOutputShape(0,dims).release();
    }catch(...){return error(Ort::GetApi());}
  }
};
std::mutex mutex;const OrtApi* registered=nullptr;Op op;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" EXPORT uint64_t av_sweep_ort_calls(){return calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_sweep_ort_rhs_packs(){return packs.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_sweep_ort_fallback_calls(){return fallback_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av_sweep_ort_fallback_thread_checks(){return fallback_thread_checks.load(std::memory_order_relaxed);}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
  const OrtApi* api=base->GetApi(ORT_API_VERSION);
  if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API 29 required");
  try{std::lock_guard<std::mutex> lock(mutex);require(!registered || registered==api,"Multiple runtimes unsupported");
    if(!registered){Ort::InitApi(api);auto fresh=std::make_unique<Ort::CustomOpDomain>(kDomain);fresh->Add(&op);
      domain=std::move(fresh);registered=api;}
    return api->AddCustomOpDomain(options,*domain);
  }catch(...){return error(*api);}
}
