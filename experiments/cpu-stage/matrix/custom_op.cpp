#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "fused_pointwise.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool ok,const char*msg){if(!ok)throw std::invalid_argument(msg);}
OrtStatus* error(const OrtApi&api) noexcept {
  try{throw;}catch(const Ort::Exception&e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument&e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception&e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"Pointwise fusion failed");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
  require(value!=nullptr&&value.IsTensor(),"Expected tensor");
  auto info=value.GetTensorTypeAndShapeInfo();
  require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"Only FP32 is supported");
  require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensors required");
  return info.GetShape();
}
struct Kernel {
  const OrtApi&api;
  int channels,tt,tc,mode,isa,skip_first,blocks_per_task;
  const float *weight,*bias;
  std::mutex cache_mutex;
  using Plan=std::shared_ptr<void>;
  // Shape-only immutable plans; no input, output or streaming history is retained.
  std::vector<std::pair<int,Plan>> cache;
  Kernel(const OrtApi&a,const OrtKernelInfo*raw):api(a){
    Ort::ConstKernelInfo info(raw);
    require(info.GetAttribute<int64_t>("native_abi")==1,"Pointwise ABI mismatch");
    auto integer=[&](const char*name){auto v=info.GetAttribute<int64_t>(name);require(v>=0&&v<=INT32_MAX,"Invalid integer attribute");return static_cast<int>(v);};
    channels=integer("channels");tt=integer("tile_time");tc=integer("tile_channels");
    mode=integer("mode");isa=integer("isa");skip_first=integer("skip_first");
    blocks_per_task=integer("blocks_per_task");
    require(channels==32||channels==64||channels==128||channels==256,"Supported C/K are 32,64,128,256");
    require(tt>0&&tt<=256&&tc>0&&tc<=64&&mode<=2&&skip_first<=1&&blocks_per_task>0&&blocks_per_task<=1024,"Invalid tiling or epilogue policy");
    require(isa==0||isa==1||isa==128||isa==256||isa==512,"Unknown ISA override");
    require(!mode||isa==0,"LIBXSMM modes use automatic dispatch, not a per-operator ISA override");
    require(!mode||fx_has_xsmm(),"This build has no LIBXSMM backend");
    int fixed=0;auto w=info.GetTensorConstantInput(0,&fixed);
    require(fixed&&shape(w)==std::vector<int64_t>{channels,channels},"W must be immutable FP32 [C,C]");
    weight=w.GetTensorData<float>();
    fixed=0;auto b=info.GetTensorConstantInput(2,&fixed);
    require(fixed&&shape(b)==std::vector<int64_t>{channels},"Bias must be immutable FP32 [C]");
    bias=b.GetTensorData<float>();
    for(int i=0;i<channels*channels;++i)require(std::isfinite(weight[i]),"Nonfinite W");
    for(int i=0;i<channels;++i)require(std::isfinite(bias[i]),"Nonfinite bias");
  }
  Plan plan(int time){
    std::lock_guard<std::mutex> lock(cache_mutex);
    for(const auto&item:cache)if(item.first==time)return item.second;
    Plan p(fx_create_ordered(channels,channels,time,tt,tc,mode,isa,skip_first),fx_destroy);
    require(bool(p),"Pointwise plan rejected by shape/ISA/JIT guard");
    if(cache.size()==4)cache.erase(cache.begin());
    cache.emplace_back(time,p);return p;
  }
  struct Work{const Kernel*self;const void*plan;const float*x,*skip;float*y;int blocks;std::atomic<int>failure{0};};
  static void task(void*opaque,size_t index)noexcept{
    auto&w=*static_cast<Work*>(opaque);
    int first=static_cast<int>(index)*w.self->blocks_per_task;
    int last=first+std::min(w.blocks-first,w.self->blocks_per_task);
    int status=fx_run_range(w.plan,w.self->weight,w.x,w.self->bias,w.skip,w.y,first,last);
    if(status)w.failure.store(status,std::memory_order_relaxed);
  }
  OrtStatus*ComputeV2(OrtKernelContext*raw)noexcept{
    try{
      Ort::KernelContext context(raw);auto x=context.GetInput(1);auto skip=context.GetInput(3);
      auto dims=shape(x);
      require(dims.size()==3&&dims[0]==1&&dims[1]==channels&&dims[2]>=0&&dims[2]<=INT32_MAX,"Expected FP32 [1,C,T]");
      require(shape(skip)==dims,"Skip must have identical shape with no broadcasting");
      require(static_cast<uint64_t>(dims[2])<=std::numeric_limits<size_t>::max()/sizeof(float)/channels,"Output size overflow");
      auto y=context.GetOutput(0,dims);if(dims[2]==0)return nullptr;
      auto p=plan(static_cast<int>(dims[2]));int blocks=fx_blocks(p.get());
      Work work{this,p.get(),x.GetTensorData<float>(),skip.GetTensorData<float>(),y.GetTensorMutableData<float>(),blocks};
      size_t tasks=1+static_cast<size_t>(blocks-1)/blocks_per_task;
      if(tasks==1)task(&work,0);else context.ParallelFor(task,tasks,0,&work);
      require(!work.failure.load(std::memory_order_relaxed),"Pointwise computation failed");return nullptr;
    }catch(...){return error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
  Op(){start_ver_=1;end_ver_=1;}
  const char*GetName()const{return "PointwiseBiasResidualF32";}
  const char*GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 4;}
  size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus*CreateKernelV2(const OrtApi&a,const OrtKernelInfo*i,void**out)const noexcept{
    *out=nullptr;try{*out=new Kernel(a,i);return nullptr;}catch(...){return error(a);}
  }
  static OrtStatus*InferOutputShape(Ort::ShapeInferContext&c)noexcept{
    try{auto s=c.GetInputShape(1);require(s.size()==3,"Expected rank-three pointwise input");s[1]=c.GetAttrInt("channels");return c.SetOutputShape(0,s).release();}
    catch(...){return error(Ort::GetApi());}
  }
};
Op op;std::mutex registration_mutex;const OrtApi*loaded=nullptr;std::unique_ptr<Ort::CustomOpDomain>domain;
}
extern "C" __attribute__((visibility("default"))) OrtStatus*ORT_API_CALL RegisterCustomOps(OrtSessionOptions*options,const OrtApiBase*base){
  auto*api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"Requires ORT API29");
  try{
    std::lock_guard<std::mutex>lock(registration_mutex);
    require(!loaded||loaded==api,"Multiple ORT runtimes unsupported");
    if(!loaded){
      require(fx_init()>=0,"Forced LIBXSMM_TARGET is prohibited");Ort::InitApi(api);
      auto d=std::make_unique<Ort::CustomOpDomain>("audio.cpu.pointwise.experimental");d->Add(&op);domain=std::move(d);loaded=api;
    }
    return api->AddCustomOpDomain(options,*domain);
  }catch(...){return error(*api);}
}
