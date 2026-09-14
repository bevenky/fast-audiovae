#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "adapter.h"
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__FAST_MATH__)
#error "Apple ARM CPU without fast math required"
#endif
#define EXPORT __attribute__((visibility("default")))
extern "C" {
int av8_quant(size_t,size_t,const float*,int8_t*,float*);
int av8_quant_scalar(size_t,size_t,const float*,int8_t*,float*);
}
namespace {
constexpr const char* kDomain="fast.audiovae.apple.firstpair.int8.v1";
constexpr size_t K=2048, N=8192, Tile=2;
std::atomic<uint64_t> calls{},packs{},quant_calls{},empty_calls{};
void require(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
OrtStatus* error(const OrtApi& api) noexcept {
  try{throw;}
  catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"Apple INT8 projection failure");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
  require(value && value.IsTensor(),"Dense tensor required");
  require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
  auto info=value.GetTensorTypeAndShapeInfo();
  require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 tensor required");
  return info.GetShape();
}
size_t count(int64_t a,int64_t b,int64_t c=1){
  size_t n=1;
  for(int64_t v:{a,b,c}){
    require(v>=0,"Negative dimension");
    require(!v || n<=SIZE_MAX/static_cast<size_t>(v),"Tensor count overflow");n*=static_cast<size_t>(v);
  }
  require(n<=SIZE_MAX/sizeof(float),"Tensor byte count overflow");return n;
}
using Plan=std::unique_ptr<void,void(*)(void*)>;
struct Kernel {
  const OrtApi& api;
  Plan plan{nullptr,av8_destroy};
  std::vector<int8_t> q;
  std::vector<float> scales,input_tile,output_tile;
  std::mutex mutex;
  Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
    Ort::ConstKernelInfo info(raw);
    require(info.GetAttribute<int64_t>("matrix_abi")==1 && info.GetAttribute<int64_t>("threads")==1,"ABI1 and one worker required");
    require(info.GetAttribute<int64_t>("k")==K && info.GetAttribute<int64_t>("n")==N &&
            info.GetAttribute<int64_t>("max_m")==Tile,"Expected K2048/N8192 with a two-frame tile");
    int constant=0;auto weight=info.GetTensorConstantInput(1,&constant);
    require(constant && weight,"Non-overridable constant weights required");
    require(shape(weight)==std::vector<int64_t>({N,K}),"Expected constant W[8192,2048]");
    const float* w=weight.GetTensorData<float>();
    std::vector<int8_t> qw(N*K);std::vector<float> sw(N);
    for(size_t n=0;n<N;++n)
      require(av8_quant_scalar(K,1,w+n*K,qw.data()+n*K,sw.data()+n)==0,"Weight quantization failed");
    plan.reset(av8_create(K,N,qw.data(),sw.data()));
    require(bool(plan),av8_error());
    q.resize(K*Tile);scales.resize(Tile);input_tile.resize(K*Tile);output_tile.resize(N*Tile);
    packs.fetch_add(1,std::memory_order_relaxed);
  }
  void run_tile(size_t t,const float* x,float* y){
    require(av8_quant(K,t,x,q.data(),scales.data())==0,"Activation quantization failed");
    quant_calls.fetch_add(1,std::memory_order_relaxed);
    require(av8_run_q(plan.get(),t,q.data(),scales.data(),y,0)==0,av8_error());
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try{
      std::lock_guard<std::mutex> guard(mutex);
      Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);auto dims=shape(input);
      require(dims.size()==3 && dims[0]>=0 && dims[1]==K && dims[2]>=0,"Expected [B,2048,T]");
      require(shape(ctx.GetInput(1))==std::vector<int64_t>({N,K}),"Weight shape changed");
      count(dims[0],K,dims[2]);count(dims[0],N,dims[2]);
      auto output=ctx.GetOutput(0,{dims[0],N,dims[2]});
      if(!dims[0] || !dims[2]){empty_calls.fetch_add(1,std::memory_order_relaxed);return nullptr;}
      const size_t t=static_cast<size_t>(dims[2]);const size_t xc=K*t,yc=N*t;
      const float* x=input.GetTensorData<float>();float* y=output.GetTensorMutableData<float>();
      for(int64_t b=0;b<dims[0];++b){
        const float* xb=x+static_cast<size_t>(b)*xc;float* yb=y+static_cast<size_t>(b)*yc;
        if(t<=Tile){
          // Keep the qualified 40/80 ms arithmetic and layout path unchanged.
          run_tile(t,xb,yb);
        }else{
          // Quantization is per frame, so packet size cannot change precision or
          // depend on a future frame. Large packets reuse the same bounded tile.
          for(size_t start=0;start<t;){
            const size_t m=std::min(Tile,t-start);
            for(size_t k=0;k<K;++k)
              std::copy_n(xb+k*t+start,m,input_tile.data()+k*m);
            run_tile(m,input_tile.data(),output_tile.data());
            for(size_t n=0;n<N;++n)
              std::copy_n(output_tile.data()+n*m,m,yb+n*t+start);
            start+=m;
          }
        }
        calls.fetch_add(1,std::memory_order_relaxed);
      }
      return nullptr;
    }catch(...){return error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
  Op(){start_ver_=1;end_ver_=1;}
  const char* GetName()const{return "FirstPairInt8";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 2;}size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** result)const noexcept{
    *result=nullptr;try{*result=new Kernel(api,info);return nullptr;}catch(...){return error(api);}
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& ctx)noexcept{
    try{auto dims=ctx.GetInputShape(0);require(dims.size()==3,"Rank-three input required");
      require(ctx.GetAttrInt("k")==K && ctx.GetAttrInt("n")==N && ctx.GetAttrInt("max_m")==Tile,"Shape attributes changed");
      dims[1]=N;return ctx.SetOutputShape(0,dims).release();
    }catch(...){return error(Ort::GetApi());}
  }
};
std::mutex registration_mutex;const OrtApi* registered=nullptr;Op op;std::unique_ptr<Ort::CustomOpDomain> domain;
}
// The unused integer preserves the validation counter ABI; there is one kernel.
extern "C" EXPORT uint64_t av8_ort_calls(int){return calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av8_ort_packs(int){return packs.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av8_ort_quant_calls(int){return quant_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT uint64_t av8_ort_empty_calls(int){return empty_calls.load(std::memory_order_relaxed);}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
  const OrtApi* api=base->GetApi(ORT_API_VERSION);
  if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
  try{std::lock_guard<std::mutex> guard(registration_mutex);require(!registered||registered==api,"Multiple runtimes unsupported");
    if(!registered){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>(kDomain);d->Add(&op);domain=std::move(d);registered=api;}
    return api->AddCustomOpDomain(options,*domain);
  }catch(...){return error(*api);}
}
