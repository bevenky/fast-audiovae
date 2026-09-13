// Experimental CPU-only FP32 SGEMM. No model/state changes or hidden workers.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <Accelerate/Accelerate.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error This candidate requires Apple ARM64
#endif
#define EXPORT __attribute__((visibility("default")))
namespace {
constexpr const char* kDomain = "fast.audiovae.apple.layout.v3";
std::atomic<uint64_t> calls{0}, thread_checks{0};
void Require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
OrtStatus* Error(const OrtApi& api) noexcept {
  try { throw; }
  catch (const Ort::Exception& e) { return api.CreateStatus(e.GetOrtErrorCode(), e.what()); }
  catch (const std::invalid_argument& e) { return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what()); }
  catch (const std::exception& e) { return api.CreateStatus(ORT_FAIL, e.what()); }
  catch (...) { return api.CreateStatus(ORT_FAIL, "Apple matrix kernel failure"); }
}
std::vector<int64_t> Shape(Ort::ConstValue value) {
  Require(value != nullptr && value.IsTensor(), "Expected dense FP32 tensor");
  auto info = value.GetTensorTypeAndShapeInfo();
  Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FP32 only");
  Require(value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU,
          "CPU tensors only");
  return info.GetShape();
}
size_t Count(int64_t a, int64_t b, int64_t c=1) {
  Require(a >= 0 && b >= 0 && c >= 0, "Negative tensor dimension");
  size_t value = static_cast<size_t>(a);
  for (int64_t v : {b,c}) {
    Require(!v || value <= std::numeric_limits<size_t>::max()/static_cast<size_t>(v), "Tensor size overflow");
    value *= static_cast<size_t>(v);
  }
  Require(value <= std::numeric_limits<size_t>::max()/sizeof(float), "Tensor byte size overflow");
  return value;
}
struct Kernel {
  const OrtApi& api;
  int64_t k, n;
  Kernel(const OrtApi& a, const OrtKernelInfo* raw) : api(a) {
    Ort::ConstKernelInfo info(raw);
    Require(info.GetAttribute<int64_t>("matrix_abi")==1, "Matrix ABI mismatch");
    Require(info.GetAttribute<int64_t>("threads")==1, "One thread required");
    k=info.GetAttribute<int64_t>("k"); n=info.GetAttribute<int64_t>("n");
    Require(k>0 && n>0 && k<=INT32_MAX && n<=INT32_MAX, "BLAS dimensions out of range");
    int constant=0;
    auto weight=info.GetTensorConstantInput(1,&constant);
    Require(constant && weight!=nullptr, "Weight must be a non-overridable constant initializer");
    Require(Shape(weight)==std::vector<int64_t>({n,k}), "Constant Weight shape mismatch");
    Count(k,n);
    // Do not copy 64 MiB per kernel, retain a borrowed initializer pointer, or
    // allocate a second packing cache. Compute obtains ORT's live input value.
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      auto x=context.GetInput(0), w=context.GetInput(1);
      auto dims=Shape(x);
      Require(dims.size()==3 && dims[0]>=0 && dims[1]>=0 && dims[2]>=0 && dims[2]<=INT32_MAX && dims[1]==k,
              "Expected FP32 [batch, k, rows]");
      Require(Shape(w)==std::vector<int64_t>({n,k}), "Weight shape changed");
      Count(dims[0],dims[2],k); Count(dims[0],dims[2],n);
      auto outdims=dims; outdims[1]=n;
      auto out=context.GetOutput(0,outdims);
      if (!dims[0] || !dims[2]) return nullptr;
      // This policy is thread-local, so construction-time setup is insufficient.
      // Keep the caller single-threaded after the call as well. No ORT ParallelFor.
      Require(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)==0,
              "Accelerate does not support the requested single-thread policy");
      Require(BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED,
              "Accelerate single-thread policy was not applied");
      thread_checks.fetch_add(1,std::memory_order_relaxed);
      const float* xp=x.GetTensorData<float>();
      const float* wp=w.GetTensorData<float>();
      float* yp=out.GetTensorMutableData<float>();
      const size_t xs=Count(dims[2],k), ys=Count(dims[2],n);
      for (int64_t b=0;b<dims[0];++b) {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    static_cast<int>(n),static_cast<int>(dims[2]),static_cast<int>(k),
                    1.0f,wp,static_cast<int>(k),xp+b*xs,static_cast<int>(dims[2]),
                    0.0f,yp+b*ys,static_cast<int>(dims[2]));
        calls.fetch_add(1,std::memory_order_relaxed);
      }
      return nullptr;
    } catch (...) { return Error(api); }
  }
};
struct Op : Ort::CustomOpBase<Op,Kernel,true> {
  Op() { start_ver_=1; end_ver_=1; }
  const char* GetName() const { return "WeightLeftSgemmF32"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** result) const noexcept {
    *result=nullptr;
    try { *result=new Kernel(api,info); return nullptr; } catch (...) { return Error(api); }
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& context) noexcept {
    try {
      auto shape=context.GetInputShape(0);
      Require(shape.size()==3,"Matrix input rank must be three");
      const auto n=context.GetAttrInt("n");
      Require(n>0 && n<=INT32_MAX,"Invalid output width");
      shape[1]=n;
      return context.SetOutputShape(0,shape).release();
    } catch (...) { return Error(Ort::GetApi()); }
  }
};
std::mutex registration_mutex;
const OrtApi* registered_api=nullptr;
Op op;
std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" EXPORT uint64_t av2_layout_call_count() { return calls.load(std::memory_order_relaxed); }
extern "C" EXPORT uint64_t av2_layout_thread_check_count() { return thread_checks.load(std::memory_order_relaxed); }
extern "C" EXPORT int av2_layout_caller_is_single_threaded() {
  return BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED;
}
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
  const OrtApi* api=base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL,"Requires ORT API 29");
  try {
    std::lock_guard<std::mutex> lock(registration_mutex);
    Require(!registered_api || registered_api==api,"Multiple ORT runtimes unsupported");
    if (!registered_api) {
      Ort::InitApi(api);
      auto fresh=std::make_unique<Ort::CustomOpDomain>(kDomain);
      fresh->Add(&op); domain=std::move(fresh); registered_api=api;
    }
    return api->AddCustomOpDomain(options,*domain);
  } catch (...) { return Error(*api); }
}
