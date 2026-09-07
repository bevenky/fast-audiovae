// Isolated CPU-only FP32 sequential MKL. ORT owns all workers. No model rewrite.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "mkl_cblas.h"
#include "mkl_service.h"
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool c,const char* s) { if(!c) throw std::invalid_argument(s); }
OrtStatus* Error(const OrtApi& api) noexcept {
  try { throw; }
  catch(const Ort::Exception& e) { return api.CreateStatus(e.GetOrtErrorCode(),e.what()); }
  catch(const std::exception& e) { return api.CreateStatus(ORT_FAIL,e.what()); }
  catch(...) { return api.CreateStatus(ORT_FAIL,"Unknown Intel MKL MatMul exception"); }
}
std::vector<int64_t> Shape(Ort::ConstValue v) {
  Require(v && v.IsTensor(),"Expected dense tensor");
  Require(v.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensors only");
  const auto i=v.GetTensorTypeAndShapeInfo();
  Require(i.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 only");
  return i.GetShape();
}
struct Kernel {
  const OrtApi& api;
  int m,k,nfixed,mode,shards;
  struct Free { void operator()(float* p) const { mkl_free(p); } };
  std::vector<std::unique_ptr<float,Free>> packed;
  std::vector<float> weights;
  Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a) {
    static_assert(sizeof(MKL_INT)==4,"LP64 MKL required");
    Require(mkl_get_max_threads()==1,"Sequential MKL is required");
    Ort::ConstKernelInfo info(raw);
    const int64_t mm=info.GetAttribute<int64_t>("M"),kk=info.GetAttribute<int64_t>("K");
    Require(mm>0 && kk>0 && mm<=INT32_MAX && kk<=INT32_MAX,"Invalid matrix dimensions");
    Require(static_cast<uint64_t>(mm)<=SIZE_MAX/static_cast<uint64_t>(kk)/sizeof(float),"Weight size overflow");
    m=static_cast<int>(mm);k=static_cast<int>(kk);
    const auto pp=info.GetAttribute<int64_t>("mode"),ss=info.GetAttribute<int64_t>("shards"),nn=info.GetAttribute<int64_t>("N");
    Require(pp==0 || pp==1,"Mode must be plain=0 or packed-A=1");
    Require(ss==1 || ss==2,"Exactly1 or2 shards supported");
    Require(nn>=0 && nn<=INT32_MAX && (pp==0 || nn>0),"N must fit LP64; zero is reserved for dynamic plain SGEMM");
    mode=static_cast<int>(pp);shards=static_cast<int>(ss);nfixed=static_cast<int>(nn);
    int constant=0;const auto w=info.GetTensorConstantInput(0,&constant);
    Require(constant && w,"Weights must be an immutable constant initializer");
    Require(Shape(w)==std::vector<int64_t>({m,k}),"Constant matrix shape mismatch");
    const auto* data=w.GetTensorData<float>();
    weights.assign(data,data+static_cast<size_t>(m)*k);
    if(mode==1) {
      const int jobs=std::min(shards,m);
      for(int job=0;job<jobs;++job) {
        const int begin=static_cast<int>((static_cast<int64_t>(m)*job)/jobs);
        const int end=static_cast<int>((static_cast<int64_t>(m)*(job+1))/jobs);
        const size_t bytes=cblas_sgemm_pack_get_size(CblasAMatrix,end-begin,nfixed,k);
        Require(bytes>0 && bytes<=(static_cast<size_t>(1)<<34),"Invalid packed buffer size");
        std::unique_ptr<float,Free> p(static_cast<float*>(mkl_malloc(bytes,64)));
        if(!p) throw std::bad_alloc();
        cblas_sgemm_pack(CblasRowMajor,CblasAMatrix,CblasNoTrans,end-begin,nfixed,k,
                        1.f,weights.data()+static_cast<size_t>(begin)*k,k,p.get());
        packed.emplace_back(std::move(p));
      }
    }
  }
  struct Work { const Kernel* self; const float* x;float* y;int n;int jobs;std::atomic<int> error{0}; };
  static void Run(void* opaque,size_t job) noexcept {
    auto& w=*static_cast<Work*>(opaque);const auto& q=*w.self;
    const int begin=static_cast<int>((static_cast<int64_t>(q.m)*job)/w.jobs);
    const int end=static_cast<int>((static_cast<int64_t>(q.m)*(job+1))/w.jobs);
    if(q.mode==0) {
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,end-begin,w.n,q.k,
                  1.f,q.weights.data()+static_cast<size_t>(begin)*q.k,q.k,
                  w.x,w.n,0.f,w.y+static_cast<size_t>(begin)*w.n,w.n);
    } else {
      cblas_sgemm_compute(CblasRowMajor,CblasPacked,CblasNoTrans,end-begin,w.n,q.k,
                         q.packed[job].get(),q.k,w.x,w.n,0.f,
                         w.y+static_cast<size_t>(begin)*w.n,w.n);
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext ctx(raw);auto x=ctx.GetInput(1);const auto shape=Shape(x);
      Require(shape.size()==3 && shape[0]==1 && shape[1]==k && shape[2]>0 && shape[2]<=INT32_MAX,
              "Expected contiguous B1 FP32[K,T] with positive LP64 T");
      Require(static_cast<uint64_t>(m)<=SIZE_MAX/static_cast<uint64_t>(shape[2])/sizeof(float),"Output size overflow");
      const auto n=static_cast<int>(shape[2]);
      Require(mode==0 || n==nfixed,"Packed mode requires original N, including packing cache key");
      auto y=ctx.GetOutput(0,{1,m,n});
      const int jobs=std::min(shards,m);
      Work w{this,x.GetTensorData<float>(),y.GetTensorMutableData<float>(),n,jobs};
      if(jobs==1)Run(&w,0);else ctx.ParallelFor(Run,jobs,jobs,&w);
      Require(w.error.load()==0,"MKL worker error");
      return nullptr;
    }catch(...) {return Error(api);}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true> {
  Op(){start_ver_=1;end_ver_=1;}
  const char* GetName()const{return "IntelPlainMatMulF32";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 2;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatus* CreateKernelV2(const OrtApi& a,const OrtKernelInfo* i,void** r)const noexcept {
    *r=nullptr;try{*r=new Kernel(a,i);return nullptr;}catch(...){return Error(a);}
  }
};
std::mutex mu;const OrtApi* active=nullptr;Op op;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* o,const OrtApiBase* b) {
  const auto* a=b->GetApi(ORT_API_VERSION);
  if(!a)return b->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
  try {
    std::lock_guard<std::mutex> lock(mu);
    if(active && active!=a)return a->CreateStatus(ORT_FAIL,"Multiple ORT runtimes unsupported");
    if(!active){Ort::InitApi(a);domain=std::make_unique<Ort::CustomOpDomain>("venky.audio.intel.decoder.experimental");domain->Add(&op);active=a;}
    return a->AddCustomOpDomain(o,*domain);
  }catch(...){return Error(*a);}
}

extern "C" __attribute__((visibility("default"))) int IntelSequentialThreads() { return mkl_get_max_threads(); }
extern "C" __attribute__((visibility("default"))) void IntelVersion(char* dst,int n) { mkl_get_version_string(dst,n); }

extern "C" __attribute__((visibility("default"))) size_t IntelPackBytes(int m,int n,int k,int shards) {
  if(m<=0 || n<=0 || k<=0 || (shards!=1 && shards!=2)) return 0;
  size_t total=0;const int jobs=std::min(m,shards);
  for(int j=0;j<jobs;++j) {
    const int begin=static_cast<int>((static_cast<int64_t>(m)*j)/jobs);
    const int end=static_cast<int>((static_cast<int64_t>(m)*(j+1))/jobs);
    total+=cblas_sgemm_pack_get_size(CblasAMatrix,end-begin,n,k);
  }
  return total;
}
