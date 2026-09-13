// Isolated state-aware phase assembly. Based on the existing ordered phase loop.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__)
#error FP32 arithmetic requires fast math disabled
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#define EXPORT __attribute__((visibility("default")))
namespace {
const char* kDomain="fast.audiovae.apple.phase.state.v1";
void Require(bool v,const char* s){if(!v)throw std::invalid_argument(s);}
OrtStatus* Error(const OrtApi& api) noexcept {
 try{throw;}catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
 catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
 catch(...){return api.CreateStatus(ORT_FAIL,"Phase state error");}
}
std::vector<int64_t> Shape(Ort::ConstValue v){
 Require(v!=nullptr&&v.IsTensor(),"Tensor required");
 auto i=v.GetTensorTypeAndShapeInfo();
 Require(i.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
 Require(v.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU required");
 return i.GetShape();
}
struct Kernel{
 const OrtApi& api;int64_t channels;int stride;std::vector<float> bias;
 Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
  Ort::ConstKernelInfo info(raw);
  Require(info.GetAttribute<int64_t>("phase_state_abi")==1,"ABI mismatch");
  Require(info.GetAttribute<int64_t>("threads")==1,"One thread required");
  channels=info.GetAttribute<int64_t>("channels");auto s=info.GetAttribute<int64_t>("stride");
  Require(channels>0&&channels<=1024,"Invalid channels");
  Require(s==2||s==5||s==6||s==8,"Invalid phase count");stride=(int)s;
  int fixed=0;auto v=info.GetTensorConstantInput(3,&fixed);
  Require(fixed&&v!=nullptr&&Shape(v)==std::vector<int64_t>{channels},"Fixed bias required");
  const float* p=v.GetTensorData<float>();bias.assign(p,p+channels);
  for(float x:bias)Require(std::isfinite(x),"Finite bias required");
 }
#if 1
  static void TransposeFour(const float32x4_t* phase, float32x4_t* time) noexcept {
    const auto ab = vtrnq_f32(phase[0], phase[1]);
    const auto cd = vtrnq_f32(phase[2], phase[3]);
    time[0] = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
    time[1] = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
    time[2] = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    time[3] = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
  }
#endif

  template <int phases>
  static void FinishRow(const float* current, const float* previous, float* output,
                        int64_t time, float bias_value, const float* history) noexcept {
    int64_t t = 0;
#if 1
    const auto bias_vec = vdupq_n_f32(bias_value);
    for (; t <= time - 4; t += 4) {
      float32x4_t values[phases];
      for (int p = 0; p < phases; ++p) {
        const auto offset = static_cast<int64_t>(p) * time;
        const auto cur = vld1q_f32(current + offset + t);
        // t=0 needs [history, previous[0], previous[1], previous[2]]. All
        // four loaded previous values exist because time >= t+4.
        const auto prev = t == 0
            ? vextq_f32(vdupq_n_f32(history[p]), vld1q_f32(previous + offset), 3)
            : vld1q_f32(previous + offset + t - 1);
        const auto sum = vaddq_f32(cur, prev);
        values[p] = vaddq_f32(sum, bias_vec);
      }
      if constexpr (phases == 2) {
        const float32x4x2_t pair = {{values[0], values[1]}};
        vst2q_f32(output + t * phases, pair);
      } else {
        float32x4_t front[4];
        TransposeFour(values, front);
        if constexpr (phases == 8) {
          float32x4_t back[4];
          TransposeFour(values + 4, back);
          for (int k = 0; k < 4; ++k) {
            vst1q_f32(output + (t + k) * phases, front[k]);
            vst1q_f32(output + (t + k) * phases + 4, back[k]);
          }
        } else if constexpr (phases == 6) {
          const auto tail = vzipq_f32(values[4], values[5]);
          for (int k = 0; k < 4; ++k)
            vst1q_f32(output + (t + k) * phases, front[k]);
          vst1_f32(output + t * phases + 4, vget_low_f32(tail.val[0]));
          vst1_f32(output + (t + 1) * phases + 4, vget_high_f32(tail.val[0]));
          vst1_f32(output + (t + 2) * phases + 4, vget_low_f32(tail.val[1]));
          vst1_f32(output + (t + 3) * phases + 4, vget_high_f32(tail.val[1]));
        } else {  // phases == 5
          float tail[4];
          vst1q_f32(tail, values[4]);
          for (int k = 0; k < 4; ++k) {
            vst1q_f32(output + (t + k) * phases, front[k]);
            output[(t + k) * phases + 4] = tail[k];
          }
        }
      }
    }
#endif
    // The compiler sees a fixed phase count and can unroll it. Keep two
    // separate additions, including the retained projection at t=0.
    for (; t < time; ++t) {
      for (int p = 0; p < phases; ++p) {
        const auto offset = static_cast<int64_t>(p) * time + t;
        const float prev = t ? previous[offset - 1] : history[p];
        const float sum = current[offset] + prev;
        output[t * phases + p] = sum + bias_value;
      }
    }
  }

 OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
  try{
   Ort::KernelContext c(raw);auto x=c.GetInput(0),p=c.GetInput(1),h=c.GetInput(2);auto d=Shape(x);
   Require(d.size()==3&&d[0]==1&&d[1]==channels*stride&&d[2]>=0,"Invalid projection shape");
   Require(Shape(p)==d&&Shape(h)==std::vector<int64_t>{1,channels*stride,1},"Invalid projection/history");
   Require((uint64_t)d[2]<=std::numeric_limits<size_t>::max()/(sizeof(float)*channels*stride),"Size overflow");
   auto y=c.GetOutput(0,std::vector<int64_t>{1,channels,d[2]*stride});
   auto hn=c.GetOutput(1,std::vector<int64_t>{1,channels*stride,1});
   const auto* xp=x.GetTensorData<float>();const auto* pp=p.GetTensorData<float>();const auto* hp=h.GetTensorData<float>();
   auto* yp=y.GetTensorMutableData<float>();auto* np=hn.GetTensorMutableData<float>();
   const int64_t t=d[2];
   if(!t){std::memcpy(np,hp,(size_t)(channels*stride)*sizeof(float));return nullptr;}
   for(int64_t row=0;row<channels;++row){
    auto offset=row*stride*t;auto ho=row*stride;
    switch(stride){
     case 2:FinishRow<2>(xp+offset,pp+offset,yp+offset,t,bias[row],hp+ho);break;
     case 5:FinishRow<5>(xp+offset,pp+offset,yp+offset,t,bias[row],hp+ho);break;
     case 6:FinishRow<6>(xp+offset,pp+offset,yp+offset,t,bias[row],hp+ho);break;
     case 8:FinishRow<8>(xp+offset,pp+offset,yp+offset,t,bias[row],hp+ho);break;
    }
    for(int phase=0;phase<stride;++phase)np[ho+phase]=pp[offset+phase*t+t-1];
   }
   return nullptr;
  }catch(...){return Error(api);}
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
 OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept{
  *out=nullptr;try{*out=new Kernel(api,info);return nullptr;}catch(...){return Error(api);}
 }
};
Op op;std::mutex registration_mutex;const OrtApi* registered_api=nullptr;
std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
 const OrtApi* api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
 try{std::lock_guard<std::mutex> lock(registration_mutex);Require(!registered_api||registered_api==api,"Mixed runtimes");
  if(!registered_api){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>(kDomain);d->Add(&op);domain=std::move(d);registered_api=api;}
  return api->AddCustomOpDomain(options,*domain);
 }catch(...){return Error(*api);}
}
