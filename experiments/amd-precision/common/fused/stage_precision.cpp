// Experimental mixed-precision matrices in the accepted fused stage. FP32 IO and nonlinearities.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "native_kernels.h"
#include "fused_pointwise.h"
#include "stage_dw.h"
#include "precision.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__)
#error "Stage pipeline requires exact ordered network operations"
#endif
namespace {
void require(bool v,const char* msg){if(!v)throw std::invalid_argument(msg);}
OrtStatus* error(const OrtApi& api) noexcept {
  try{throw;}catch(const Ort::Exception&e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument&e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception&e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"Stage pipeline failed");}
}
std::vector<int64_t> shape(Ort::ConstValue v){
  require(v!=nullptr&&v.IsTensor(),"Expected dense tensor");
  auto i=v.GetTensorTypeAndShapeInfo();
  require(i.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
  require(v.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
  return i.GetShape();
}
std::vector<float> constant(Ort::ConstKernelInfo info,size_t index,const std::vector<int64_t>& dims){
  int fixed=0;auto v=info.GetTensorConstantInput(index,&fixed);
  require(fixed&&shape(v)==dims,"Immutable constant has wrong shape");
  const float* p=v.GetTensorData<float>();size_t n=v.GetTensorTypeAndShapeInfo().GetElementCount();
  for(size_t i=0;i<n;++i)require(std::isfinite(p[i]),"Nonfinite stage constant");
  return {p,p+n};
}
struct Unit { std::vector<float> w,b,ap,rp,aq,rq,pw,pb; };
struct Kernel {
  const OrtApi& api;
  int c,q,segments,backend,matrix_mode,matrix_isa;
  bool debug;
  std::array<Unit,3> units;
  int precision_mode;
  std::array<std::shared_ptr<void>,3> precision;
  using Plan=std::shared_ptr<void>;
  std::mutex plan_mutex;
  std::vector<Plan> plans; // At most Q immutable shape plans, no audio/history.
  Kernel(const OrtApi&a,const OrtKernelInfo* raw,bool dbg):api(a),debug(dbg){
    Ort::ConstKernelInfo info(raw);
    auto integer=[&](const char* name){auto x=info.GetAttribute<int64_t>(name);require(x>=0&&x<=INT32_MAX,"Invalid integer attribute");return static_cast<int>(x);};
    require(ncc_abi_version()==1&&integer("native_abi")==1,"Native ABI mismatch");
    c=integer("channels");q=integer("tile_time");segments=integer("segments");
    backend=integer("backend");matrix_mode=integer("matrix_mode");matrix_isa=integer("matrix_isa");
    require(c==32||c==64||c==128||c==256,"C must be 32,64,128,256");
    require(q==64||q==128||q==256,"Q must be 64,128,256");
    require(segments>=1&&segments<=64,"Invalid contiguous segment count");
    require((backend==4||backend==5)&&ncc_backend_available(backend)&&ncc_vector_sine_available(backend),"Requested accurate x86 vector backend unavailable");
    require(matrix_mode<=2&&(!matrix_mode||fx_has_xsmm()),"Unavailable matrix backend");
    require(matrix_mode ? matrix_isa==0 : (matrix_isa==1||matrix_isa==256||matrix_isa==512),
            "Direct requires explicit ISA; LIBXSMM requires ISA0 native dispatch");
    require(fx_init()>=0,"Matrix runtime initialization failed");
    for(int u=0;u<3;++u){
      auto& z=units[u];size_t i=1+8*u;
      z.w=constant(info,i++,{c,1,7});z.b=constant(info,i++,{c});
      z.ap=constant(info,i++,{c});z.rp=constant(info,i++,{c});
      z.aq=constant(info,i++,{c});z.rq=constant(info,i++,{c});
      z.pw=constant(info,i++,{c,c});z.pb=constant(info,i++,{c});
    }
    precision_mode=integer("precision_mode");
    require(precision_mode==8||precision_mode==16,"Precision mode must be 8 or 16");
    for(int u=0;u<3;++u){
      precision[u]=std::shared_ptr<void>(ip_create(c,c,units[u].pw.data(),precision_mode,1),ip_destroy_plan);
      require(bool(precision[u]),ip_last_error());
    }
    plans.resize(q+1);
  }
  Plan plan(int n){
    std::lock_guard<std::mutex> lock(plan_mutex);
    if(!plans[n]){
      // Each call is already one local tile. In a short final tile, tt=n
      // also avoids asking XSMM to dispatch an unused full-Q shape with lda<Q.
      Plan p(fx_create_ordered(c,c,n,n,std::min(c,64),matrix_mode,matrix_isa,1),fx_destroy);
      require(bool(p),"Matrix plan rejected shape/ISA/JIT policy");plans[n]=p;
    }
    return plans[n];
  }
  struct Work {
    const Kernel* self;const float* x;std::array<float*,3> out;
    int64_t b,t;int parts;std::vector<Plan> plans;
    std::atomic<int> failure{0};
  };
  static void segment(void* opaque,size_t task) noexcept {
    auto& work=*static_cast<Work*>(opaque);const auto& k=*work.self;
    try{
      const int part=static_cast<int>(task%work.parts);
      const size_t batch=task/work.parts;
      const int64_t first=work.t*part/work.parts,last=work.t*(part+1)/work.parts;
      const int64_t warm=std::max<int64_t>(0,first-78);
      const size_t tile=static_cast<size_t>(k.c)*k.q;
      std::vector<float> scratch(4*tile);
      std::vector<float> history(static_cast<size_t>(k.c)*78,0.0f);
      float* a=scratch.data();float* b=a+tile;float* pre=b+tile;float* post=pre+tile;
      const float* input=work.x+batch*k.c*work.t;
      alignas(64) float row[54+256];
      for(int64_t t=warm;t<last;t+=k.q){
        const int n=static_cast<int>(std::min<int64_t>(k.q,last-t));
        for(int c=0;c<k.c;++c)std::memcpy(a+static_cast<size_t>(c)*n,input+static_cast<size_t>(c)*work.t+t,n*sizeof(float));
        int history_offset=0;
        for(int u=0;u<3;++u){
          const Unit& z=k.units[u];const int d=u==0?1:u==1?3:9,halo=6*d;
          int status=ncc_snake_f32(a,z.ap.data(),z.rp.data(),pre,1,k.c,n,k.backend,1);
          require(status==0,"Pre-Snake failed");
          for(int c=0;c<k.c;++c){
            float* h=history.data()+history_offset+static_cast<size_t>(c)*halo;
            std::memcpy(row,h,halo*sizeof(float));
            std::memcpy(row+halo,pre+static_cast<size_t>(c)*n,n*sizeof(float));
            sp_dw(row,z.w.data()+7*c,z.b[c],post+static_cast<size_t>(c)*n,n,d,k.backend);
            std::memcpy(h,row+n,halo*sizeof(float));
          }
          history_offset+=k.c*halo;
          status=ncc_snake_f32(post,z.aq.data(),z.rq.data(),pre,1,k.c,n,k.backend,1);
          require(status==0,"Post-Snake failed");
          std::unique_ptr<void,decltype(&ip_destroy_input)> prepared(
              ip_prepare(k.precision[u].get(),pre,n),ip_destroy_input);
          require(bool(prepared),ip_last_error());
          require(ip_run_rows(k.precision[u].get(),prepared.get(),b,0,k.c)==0,ip_last_error());
          for(int oc=0;oc<k.c;++oc)for(int it=0;it<n;++it){
            const size_t offset=static_cast<size_t>(oc)*n+it;
            const float dot_bias=b[offset]+z.pb[oc];
            b[offset]=a[offset]+dot_bias;
          }
          std::swap(a,b);
          if(work.out[u]){
            const int skip=static_cast<int>(std::max<int64_t>(0,first-t));
            if(skip<n)for(int c=0;c<k.c;++c)
              std::memcpy(work.out[u]+(batch*k.c+c)*work.t+t+skip,
                          a+static_cast<size_t>(c)*n+skip,(n-skip)*sizeof(float));
          }
        }
      }
    }catch(...){work.failure.store(1,std::memory_order_relaxed);}
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try{
      Ort::KernelContext context(raw);auto input=context.GetInput(0);auto dims=shape(input);
      require(dims.size()==3&&dims[0]>=0&&dims[1]==c&&dims[2]>=0&&dims[2]<=INT32_MAX,"Expected FP32 [B,C,T]");
      require(static_cast<uint64_t>(dims[0])<=std::numeric_limits<size_t>::max()/sizeof(float)/c/std::max<int64_t>(1,dims[2]),"Input size overflow");
      std::array<float*,3> outputs{};
      if(debug)for(int u=0;u<3;++u)outputs[u]=context.GetOutput(u,dims).GetTensorMutableData<float>();
      else outputs[2]=context.GetOutput(0,dims).GetTensorMutableData<float>();
      if(!dims[0]||!dims[2])return nullptr;
      const int parts=static_cast<int>(std::min<int64_t>(segments,dims[2]));
      Work work{this,input.GetTensorData<float>(),outputs,dims[0],dims[2],parts,{}};
      work.plans.resize(q+1);work.plans[q]=plan(q);
      for(int p=0;p<parts;++p){
        int64_t first=dims[2]*p/parts,last=dims[2]*(p+1)/parts,warm=std::max<int64_t>(0,first-78);
        int tail=static_cast<int>((last-warm)%q);if(tail)work.plans[tail]=plan(tail);
      }
      const size_t tasks=static_cast<size_t>(dims[0])*parts;
      if(tasks==1)segment(&work,0);else context.ParallelFor(segment,tasks,0,&work);
      require(work.failure.load(std::memory_order_relaxed)==0,"Stage worker failed");return nullptr;
    }catch(...){return error(api);}
  }
};
template<bool debug>struct Op:Ort::CustomOpBase<Op<debug>,Kernel,true>{
  Op(){this->start_ver_=1;this->end_ver_=1;}
  const char* GetName()const{return debug?"StageStackDebugF32":"StageStackF32";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 25;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  size_t GetOutputTypeCount()const{return debug?3:1;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus* CreateKernelV2(const OrtApi&a,const OrtKernelInfo*i,void**out)const noexcept{
    *out=nullptr;try{*out=new Kernel(a,i,debug);return nullptr;}catch(...){return error(a);}
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& ctx)noexcept{
    try{const auto& s=ctx.GetInputShape(0);require(s.size()==3,"Stage requires rank3");
      for(size_t i=0;i<(debug?3:1);++i){auto status=ctx.SetOutputShape(i,s);if(status)return status.release();}return nullptr;
    }catch(...){return error(Ort::GetApi());}
  }
};
std::mutex registration;const OrtApi* registered=nullptr;
std::unique_ptr<Ort::CustomOpDomain> domain;Op<false> op;Op<true> debug_op;
}
extern "C" NCC_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
  const OrtApi* api=base->GetApi(ORT_API_VERSION);
  if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
  try{std::lock_guard<std::mutex> lock(registration);
    require(!registered||registered==api,"Multiple ORT runtimes unsupported");
    if(!registered){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.precision.stage.experimental");d->Add(&op);d->Add(&debug_op);domain=std::move(d);registered=api;}
    return api->AddCustomOpDomain(options,*domain);
  }catch(...){return error(*api);}
}
