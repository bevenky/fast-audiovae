// Experiment-only portable adaptation of native/apple/streaming/phase/native_ops.cpp.
// Same phase indexing and two ordered FP32 additions; no projection is recomputed.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__)
#error Fast math is forbidden
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
#define EXPORT __attribute__((visibility("default")))
namespace {
constexpr const char* DOMAIN="fast.audiovae.amd.phase.state.v1";
void require(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
OrtStatus* error(const OrtApi& api) noexcept {
 try{throw;}catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
 catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
 catch(...){return api.CreateStatus(ORT_FAIL,"Phase state failure");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
 require(value&&value.IsTensor(),"Tensor required");
 require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
 auto info=value.GetTensorTypeAndShapeInfo();
 require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
 return info.GetShape();
}
bool overlap(const float* a,size_t an,const float* b,size_t bn) noexcept {
 if(!an||!bn)return false;
 const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
 return x<y ? y-x<an : x-y<bn;
}
struct Kernel {
 const OrtApi& api;int64_t channels;int stride;std::vector<float> bias;
 Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
  Ort::ConstKernelInfo info(raw);
  require(info.GetAttribute<int64_t>("phase_state_abi")==1,"Phase ABI1 required");
  require(info.GetAttribute<int64_t>("threads")==1,"One native worker required");
  channels=info.GetAttribute<int64_t>("channels");const auto s=info.GetAttribute<int64_t>("stride");
  require(channels>0&&channels<=1024,"Channels outside audited range");
  require(s==2||s==5||s==6||s==8,"Unsupported stride");stride=int(s);
  int fixed=0;auto value=info.GetTensorConstantInput(3,&fixed);
  require(fixed&&value&&shape(value)==std::vector<int64_t>{channels},"Fixed FP32 bias[C] required");
  const auto* p=value.GetTensorData<float>();bias.assign(p,p+channels);
  for(float x:bias)require(std::isfinite(x),"Finite bias required");
 }
 template<int S> static void row(const float* cur,const float* prev,const float* hist,
                                 float* out,float* next,int64_t t,float bias) noexcept {
  // Phase-major input to chronological output. Fixed S lets the compiler unroll
  // the phase loop. Keep this neutral implementation distinct from Apple NEON.
  for(int64_t j=0;j<t;++j)for(int phase=0;phase<S;++phase){
   const auto i=static_cast<int64_t>(phase)*t+j;
   const float p=j ? prev[i-1] : hist[phase];
   const float sum=cur[i]+p;
   out[j*S+phase]=sum+bias;
  }
  for(int phase=0;phase<S;++phase)next[phase]=prev[static_cast<int64_t>(phase)*t+t-1];
 }
 OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
  try{
   Ort::KernelContext ctx(raw);auto cv=ctx.GetInput(0),pv=ctx.GetInput(1),hv=ctx.GetInput(2);
   auto d=shape(cv);const int64_t width=channels*stride;
   require(d.size()==3&&d[0]==1&&d[1]==width&&d[2]>=0,"Expected current[1,C*S,T]");
   require(shape(pv)==d&&shape(hv)==std::vector<int64_t>{1,width,1},"Projection/history shape mismatch");
   const int64_t t=d[2];require(t<=INT64_MAX/stride,"Output shape overflow");
   require(static_cast<uint64_t>(t)<=std::numeric_limits<size_t>::max()/(sizeof(float)*width),"Tensor size overflow");
   auto ov=ctx.GetOutput(0,{1,channels,t*stride}),nv=ctx.GetOutput(1,{1,width,1});
   const auto* c=cv.GetTensorData<float>();const auto* p=pv.GetTensorData<float>();const auto* h=hv.GetTensorData<float>();
   auto* o=ov.GetTensorMutableData<float>();auto* n=nv.GetTensorMutableData<float>();
   const size_t bytes=static_cast<size_t>(width)*t*sizeof(float),hb=static_cast<size_t>(width)*sizeof(float);
   require(!overlap(o,bytes,c,bytes)&&!overlap(o,bytes,p,bytes)&&!overlap(o,bytes,h,hb)
           &&!overlap(n,hb,c,bytes)&&!overlap(n,hb,p,bytes)&&!overlap(n,hb,h,hb)
           &&!overlap(o,bytes,n,hb),"Outputs may not overlap inputs or each other");
   if(!t){std::memcpy(n,h,hb);return nullptr;}
   for(int64_t ch=0;ch<channels;++ch){
    const auto offset=ch*stride*t,ho=ch*stride;
    switch(stride){
     case 2:row<2>(c+offset,p+offset,h+ho,o+offset,n+ho,t,bias[ch]);break;
     case 5:row<5>(c+offset,p+offset,h+ho,o+offset,n+ho,t,bias[ch]);break;
     case 6:row<6>(c+offset,p+offset,h+ho,o+offset,n+ho,t,bias[ch]);break;
     case 8:row<8>(c+offset,p+offset,h+ho,o+offset,n+ho,t,bias[ch]);break;
    }
   }
   return nullptr;
  }catch(...){return error(api);}
 }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
 Op(){start_ver_=1;end_ver_=1;}
 const char* GetName()const{return "StatefulPhaseFinishF32";}
 const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
 size_t GetInputTypeCount()const{return 4;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 2;}
 ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatus* CreateKernelV2(const OrtApi& a,const OrtKernelInfo* i,void** out)const noexcept{
  *out=nullptr;try{*out=new Kernel(a,i);return nullptr;}catch(...){return error(a);}
 }
};
Op op;std::mutex mutex;const OrtApi* initialized=nullptr;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
 const auto* api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API headers incompatible");
 try{std::lock_guard<std::mutex> lock(mutex);require(!initialized||initialized==api,"Mixed ORT runtimes");
  if(!initialized){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>(DOMAIN);d->Add(&op);domain=std::move(d);initialized=api;}
  return api->AddCustomOpDomain(options,*domain);
 }catch(...){return error(*api);}
}
