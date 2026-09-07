#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "packed_a.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace {
void require(bool v,const char* m) { if(!v)throw std::invalid_argument(m); }
OrtStatus* error(const OrtApi& api) noexcept {
  try {throw;}catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
  catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
  catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  catch(...){return api.CreateStatus(ORT_FAIL,"AOCL adapter failure");}
}
std::vector<int64_t> shape(Ort::ConstValue v) {
  require(v!=nullptr && v.IsTensor(),"Expected tensor");auto t=v.GetTensorTypeAndShapeInfo();
  require(t.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"Only FP32 is supported");
  require(v.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"AOCL inputs must be CPU tensors");
  return t.GetShape();
}
struct Kernel {
  const OrtApi& api; int64_t m,k,tiles;
  using OwnedPack=std::unique_ptr<ncc_aocl_pack,decltype(&ncc_aocl_destroy)>;
  std::vector<OwnedPack> packs;
  std::vector<int64_t> row_starts;
  Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a) {
    require(ncc_aocl_cpu_supported(),"AOCL adapter requires compatible AMD AVX512 CPU and OS state");
    Ort::ConstKernelInfo info(raw);
    require(info.GetAttribute<int64_t>("native_abi")==1,"AOCL adapter ABI mismatch");
    require(info.GetAttribute<std::string>("aocl_pin")=="25cad99a6840855ade0a49871197f48ee0e1d317","AOCL pin mismatch");
    m=info.GetAttribute<int64_t>("rows");k=info.GetAttribute<int64_t>("inner");tiles=info.GetAttribute<int64_t>("row_tiles");
    require(m>0&&k>0&&m<=INT32_MAX&&k<=INT32_MAX&&tiles>=1&&tiles<=64,"Invalid AOCL shape or tile count");
    require(static_cast<uint64_t>(m)<=std::numeric_limits<size_t>::max()/sizeof(float)/k,"Weight byte overflow");
    int constant=0;auto w=info.GetTensorConstantInput(0,&constant);
    require(constant&&shape(w)==std::vector<int64_t>{m,k},"W must be immutable FP32 [rows,inner] initializer");
    const auto* p=w.GetTensorData<float>();
    for(int64_t i=0;i<m*k;++i)require(std::isfinite(p[i]),"Nonfinite W initializer");
    /* Packing copies coefficients into an owned aligned immutable buffer. */
    const int64_t count=tiles<m?tiles:m;
    for(int64_t tile=0;tile<count;++tile) {
      const int64_t begin=m*tile/count,end=m*(tile+1)/count;
      OwnedPack pack(ncc_aocl_create(p+begin*k,end-begin,k),ncc_aocl_destroy);
      require(bool(pack),"AOCL row packing failed");
      row_starts.push_back(begin);packs.push_back(std::move(pack));
    }
  }
  struct Work {const Kernel* self;const float* x;float* y;int64_t time,tasks_per_batch;std::atomic<int> failure{0};};
  static void Task(void* opaque,size_t index) noexcept {
    auto& w=*static_cast<Work*>(opaque);const auto& s=*w.self;
    const int64_t batch=index/w.tasks_per_batch,tile=index%w.tasks_per_batch;
    /* Each immutable pack owns a disjoint M block; every task computes
     * the full N/time range and complete K dot products. */
    const int64_t row_begin=s.row_starts[tile];
    int status=ncc_aocl_compute(s.packs[tile].get(),w.x+batch*s.k*w.time,
                               w.y+(batch*s.m+row_begin)*w.time,w.time,w.time);
    if(status)w.failure.store(status,std::memory_order_relaxed);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);auto x=context.GetInput(1);auto dims=shape(x);
      require(dims.size()==3&&dims[0]>=0&&dims[1]==k&&dims[2]>=0&&dims[2]<=INT32_MAX,"Expected FP32 [B,inner,T] input");
      uint64_t b=dims[0],t=dims[2],maxc=static_cast<uint64_t>(m>k?m:k);
      require(!b||t<=std::numeric_limits<size_t>::max()/sizeof(float)/b/maxc,"AOCL input/output byte overflow");
      auto out=context.GetOutput(0,std::vector<int64_t>{dims[0],m,dims[2]});if(!b||!t)return nullptr;
      int64_t tasks=static_cast<int64_t>(packs.size());
      require(b<=std::numeric_limits<size_t>::max()/tasks,"AOCL task count overflow");
      Work work{this,x.GetTensorData<float>(),out.GetTensorMutableData<float>(),dims[2],tasks};
      if(b*tasks==1)Task(&work,0);else context.ParallelFor(Task,b*tasks,0,&work);
      require(!work.failure.load(std::memory_order_relaxed),"AOCL packed computation failed");return nullptr;
    }catch(...){return error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true> {
 Op(){start_ver_=1;end_ver_=1;}
 const char* GetName()const{return "PackedRowsMatMulF32";}
 const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
 size_t GetInputTypeCount()const{return 2;} size_t GetOutputTypeCount()const{return 1;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatus* CreateKernelV2(const OrtApi& a,const OrtKernelInfo* i,void** out)const noexcept { *out=nullptr;try{*out=new Kernel(a,i);return nullptr;}catch(...){return error(a);} }
 static OrtStatus* InferOutputShape(Ort::ShapeInferContext& c)noexcept {try{auto s=c.GetInputShape(1);require(s.size()==3,"AOCL X must have rank3");s[1]=c.GetAttrInt("rows");return c.SetOutputShape(0,s).release();}catch(...){return error(Ort::GetApi());}}
};
Op op;std::mutex mutex;const OrtApi* loaded=nullptr;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" AOCL_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
 auto* api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"Requires ORT API29");
 try {require(ncc_aocl_cpu_supported(),"Unsupported CPU for optional AOCL adapter");std::lock_guard<std::mutex> guard(mutex);
   require(!loaded||loaded==api,"Multiple ORT runtimes unsupported");
   if(!loaded){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>("venky.audio.cpu.aocl.rows");d->Add(&op);domain=std::move(d);loaded=api;}
   return api->AddCustomOpDomain(options,*domain);
 }catch(...){return error(*api);}
}
