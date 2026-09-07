// Native FP32 CPU operators for the explicitly rewritten AudioVAE2 graph.
// No Torch, Python, ctypes, OpenMP, hidden stream state, or input mutation.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "native_kernels.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#define NCC_PHASE_NEON 1
#include <arm_neon.h>
#else
#define NCC_PHASE_NEON 0
#endif

namespace {
constexpr const char* kDomain = "venky.audio.cpu";
enum class Mode { Snake, DW7, DW7Snake, SnakeDW7Snake };

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

OrtStatus* Error(const OrtApi& api) noexcept {
  try { throw; }
  catch (const Ort::Exception& e) { return api.CreateStatus(e.GetOrtErrorCode(), e.what()); }
  catch (const std::invalid_argument& e) { return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what()); }
  catch (const std::exception& e) { return api.CreateStatus(ORT_FAIL, e.what()); }
  catch (...) { return api.CreateStatus(ORT_FAIL, "Unknown native codec custom-op exception"); }
}

std::vector<int64_t> FloatShape(Ort::ConstValue value) {
  Require(value != nullptr && value.IsTensor(), "Expected a dense tensor");
  auto info = value.GetTensorTypeAndShapeInfo();
  Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Only FP32 is supported");
  Require(value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU,
          "Native codec inputs must reside in CPU memory");
  return info.GetShape();
}

std::vector<float> Constant(Ort::ConstKernelInfo info, size_t index,
                            const std::vector<int64_t>& expected) {
  int is_constant = 0;
  auto value = info.GetTensorConstantInput(index, &is_constant);
  Require(is_constant && value != nullptr, "Coefficient/weight must be a non-overridable constant initializer");
  const auto shape = FloatShape(value);
  Require(shape == expected, "Unexpected constant initializer shape");
  const auto count = value.GetTensorTypeAndShapeInfo().GetElementCount();
  const auto* data = value.GetTensorData<float>();
  // This is a borrowed OrtValue. Copy only the small coefficients/DW weights;
  // never ReleaseValue it, retain an unowned pointer, or share mutable packing.
  return {data, data + count};
}

struct Kernel {
  const OrtApi& api;
  Mode mode;
  int64_t channels;
  int32_t dilation;
  int32_t backend;
  size_t row_batches;
  std::vector<float> weights, bias, alpha, reciprocal, alpha_pre, reciprocal_pre;

  Kernel(const OrtApi& api_, const OrtKernelInfo* raw_info, Mode mode_) : api(api_), mode(mode_) {
    Ort::ConstKernelInfo info(raw_info);
    Require(ncc_abi_version() == 1, "Native codec C ABI mismatch");
    Require(info.GetAttribute<int64_t>("native_abi") == 1, "Graph/native ABI mismatch");
    channels = info.GetAttribute<int64_t>("channels");
    Require(channels > 0 && channels <= std::numeric_limits<int32_t>::max(), "Invalid channel count");
    auto backend_value = info.GetAttribute<int64_t>("backend");
    Require(backend_value >= NCC_AUTO && backend_value <= NCC_AVX2, "Invalid backend");
    backend = static_cast<int32_t>(backend_value);
    Require(backend == NCC_AUTO || ncc_backend_available(backend), "Requested CPU ISA unavailable");
    if (backend == NCC_AUTO) backend = ncc_selected_backend();
    auto batches = info.GetAttribute<int64_t>("row_batches");
    Require(batches >= 0, "row_batches must be nonnegative");
    row_batches = static_cast<size_t>(batches);
    dilation = 1;
    if (mode != Mode::Snake) {
      const auto d = info.GetAttribute<int64_t>("dilation");
      Require(d > 0 && d <= std::numeric_limits<int32_t>::max() / 6, "Invalid causal DW7 dilation");
      dilation = static_cast<int32_t>(d);
      if (mode == Mode::SnakeDW7Snake) Require(d == 1 || d == 3 || d == 9, "Triple fusion dilation must be 1, 3 or 9");
      weights = Constant(info, 1, {channels, 1, 7});
      bias = Constant(info, 2, {channels});
    }
    if (mode != Mode::DW7) {
      // Deliberately fail instead of silently substituting scalar sine on a
      // different platform. The rewriter's auto policy leaves Snake unchanged.
      Require(info.GetAttribute<int64_t>("require_vforce") == 1,
              "This custom Snake implementation requires explicit vForce policy");
      Require((ncc_capabilities() & NCC_CAP_VFORCE) && backend != NCC_SCALAR,
              "Fast vForce sine unavailable: use the original graph or DW-only variant");
      if (mode == Mode::SnakeDW7Snake) {
        alpha_pre = Constant(info, 3, {channels});
        reciprocal_pre = Constant(info, 4, {channels});
      }
      const size_t start = mode == Mode::Snake ? 1 : mode == Mode::SnakeDW7Snake ? 5 : 3;
      alpha = Constant(info, start, {channels});
      reciprocal = Constant(info, start + 1, {channels});
    }
  }

  struct Work {
    const Kernel* self;
    const float* x;
    float* y;
    int64_t time;
    std::atomic<int32_t> error{NCC_OK};
  };

  static void Row(void* opaque, size_t row) noexcept {
    auto& work = *static_cast<Work*>(opaque);
    const auto& k = *work.self;
    const auto c = row % static_cast<size_t>(k.channels);
    const auto offset = row * static_cast<size_t>(work.time);
    int32_t result;
    if (k.mode == Mode::Snake) {
      result = ncc_snake_f32(work.x + offset, k.alpha.data() + c,
                             k.reciprocal.data() + c, work.y + offset,
                             1, 1, work.time, k.backend, 1);
    } else if (k.mode == Mode::SnakeDW7Snake) {
      result = ncc_snake_dw7_snake_f32(work.x + offset, k.weights.data() + 7*c,
          k.bias.data()+c, k.alpha_pre.data()+c, k.reciprocal_pre.data()+c,
          k.alpha.data()+c, k.reciprocal.data()+c, work.y+offset,
          1, 1, work.time, k.dilation, k.backend, 1);
    } else {
      result = ncc_dw7_f32_ex(work.x + offset, k.weights.data() + 7 * c,
                              k.bias.data() + c, nullptr,
                              k.mode == Mode::DW7 ? nullptr : k.alpha.data() + c,
                              k.mode == Mode::DW7 ? nullptr : k.reciprocal.data() + c,
                              work.y + offset, 1, 1, work.time, k.dilation, k.backend, 1);
    }
    if (result != NCC_OK) work.error.store(result, std::memory_order_relaxed);
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context(raw_context);
      auto input = context.GetInput(0);
      const auto shape = FloatShape(input);
      Require(shape.size() == 3 && shape[0] >= 0 && shape[1] == channels && shape[2] >= 0,
              "Expected contiguous FP32 [batch, channels, time] input");
      Require(static_cast<uint64_t>(shape[0]) <= std::numeric_limits<size_t>::max() / channels,
              "Row count overflow");
      auto output = context.GetOutput(0, shape);
      if (!shape[0] || !shape[2]) return nullptr;
      Work work{this, input.GetTensorData<float>(), output.GetTensorMutableData<float>(), shape[2]};
      const auto rows = static_cast<size_t>(shape[0]) * static_cast<size_t>(channels);
      // ORT's own intra-op pool. Each native C invocation is single threaded;
      // the native C object is also compiled without OpenMP.
      if (rows == 1) Row(&work, 0);
      else context.ParallelFor(Row, rows, row_batches, &work);
      const auto status = work.error.load(std::memory_order_relaxed);
      if (status != NCC_OK) return api.CreateStatus(ORT_FAIL, ncc_status_string(status));
      return nullptr;
    } catch (...) { return Error(api); }
  }
};

template <Mode mode>
struct Op : Ort::CustomOpBase<Op<mode>, Kernel, true> {
  Op() { this->start_ver_ = 1; this->end_ver_ = 1; }
  const char* GetName() const {
    return mode == Mode::Snake ? "SnakeF32" : mode == Mode::DW7 ? "CausalDW7F32" :
           mode == Mode::SnakeDW7Snake ? "SnakeDW7SnakeF32" : "CausalDW7SnakeF32";
  }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return mode == Mode::SnakeDW7Snake ? 7 : mode == Mode::DW7Snake ? 5 : 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** result) const noexcept {
    *result = nullptr;
    try { *result = new Kernel(api, info, mode); return nullptr; }
    catch (...) { return Error(api); }
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& context) noexcept {
    try {
      const auto& shape = context.GetInputShape(0);
      Require(shape.size() == 3, "Native codec custom op requires rank three");
      return context.SetOutputShape(0, shape).release();
    } catch (...) { return Error(Ort::GetApi()); }
  }
};

struct BiasResidualKernel {
  const OrtApi& api;
  int64_t channels;
  int32_t backend;
  size_t row_batches;
  std::vector<float> bias;

  BiasResidualKernel(const OrtApi& a, const OrtKernelInfo* raw_info) : api(a) {
    Ort::ConstKernelInfo info(raw_info);
    Require(ncc_abi_version() == 1 && info.GetAttribute<int64_t>("native_abi") == 1,
            "Bias/residual graph/native ABI mismatch");
    channels = info.GetAttribute<int64_t>("channels");
    Require(channels > 0 && channels <= std::numeric_limits<int32_t>::max(), "Invalid bias channel count");
    auto b = info.GetAttribute<int64_t>("backend");
    Require(b >= NCC_AUTO && b <= NCC_AVX2 && ncc_backend_available(static_cast<int32_t>(b)),
            "Requested bias/residual CPU ISA unavailable");
    backend = b == NCC_AUTO ? ncc_selected_backend() : static_cast<int32_t>(b);
    auto batches = info.GetAttribute<int64_t>("row_batches");
    Require(batches >= 0, "row_batches must be nonnegative");
    row_batches = static_cast<size_t>(batches);
    bias = Constant(info, 1, {channels});
  }
  struct Work {
    const BiasResidualKernel* self;
    const float* product;
    const float* skip;
    float* output;
    int64_t time;
    std::atomic<int32_t> error{NCC_OK};
  };
  static void Row(void* opaque, size_t row) noexcept {
    auto& w = *static_cast<Work*>(opaque);
    const auto& k = *w.self;
    const auto offset = row * static_cast<size_t>(w.time);
    const auto status = ncc_bias_residual_f32(w.product+offset,
        k.bias.data()+row%static_cast<size_t>(k.channels), w.skip+offset,
        w.output+offset, 1, 1, w.time, k.backend, 1);
    if (status != NCC_OK) w.error.store(status, std::memory_order_relaxed);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context(raw_context);
      auto product = context.GetInput(0), skip = context.GetInput(2);
      auto shape = FloatShape(product);
      Require(shape.size() == 3 && shape[0] >= 0 && shape[1] == channels && shape[2] >= 0,
              "Expected contiguous FP32 [batch, channels, time] product");
      Require(FloatShape(skip) == shape, "Bias/residual skip shape must exactly match product");
      Require(static_cast<uint64_t>(shape[0]) <= std::numeric_limits<size_t>::max()/channels,
              "Bias/residual row count overflow");
      auto output = context.GetOutput(0, shape);
      if (!shape[0] || !shape[2]) return nullptr;
      Work work{this, product.GetTensorData<float>(), skip.GetTensorData<float>(),
                output.GetTensorMutableData<float>(), shape[2]};
      const auto rows = static_cast<size_t>(shape[0])*static_cast<size_t>(channels);
      if (rows == 1) Row(&work, 0);
      else context.ParallelFor(Row, rows, row_batches, &work);
      const auto status = work.error.load(std::memory_order_relaxed);
      if (status != NCC_OK) return api.CreateStatus(ORT_FAIL, ncc_status_string(status));
      return nullptr;
    } catch (...) { return Error(api); }
  }
};

struct BiasResidualOp : Ort::CustomOpBase<BiasResidualOp, BiasResidualKernel, true> {
  BiasResidualOp() { this->start_ver_ = 1; this->end_ver_ = 1; }
  const char* GetName() const { return "BiasResidualF32"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** result) const noexcept {
    *result = nullptr;
    try { *result = new BiasResidualKernel(api, info); return nullptr; }
    catch (...) { return Error(api); }
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& context) noexcept {
    try {
      const auto& shape = context.GetInputShape(0);
      Require(shape.size() == 3 && context.GetInputShape(2).size() == 3,
              "Bias/residual inputs require rank three");
      return context.SetOutputShape(0, shape).release();
    } catch (...) { return Error(Ort::GetApi()); }
  }
};

// Phase order remains sum=current+previous(t-1), then output=sum+bias.
struct PhaseKernel {
  const OrtApi& api;
  int64_t channels;
  int stride;
  size_t row_batches;
  std::vector<float> bias;

  PhaseKernel(const OrtApi& a, const OrtKernelInfo* raw_info) : api(a) {
    Ort::ConstKernelInfo info(raw_info);
    Require(ncc_abi_version() == 1 && info.GetAttribute<int64_t>("native_abi") == 1,
            "Phase graph/native ABI mismatch");
    channels = info.GetAttribute<int64_t>("channels");
    Require(channels > 0 && channels <= std::numeric_limits<int32_t>::max(), "Invalid phase output channels");
    const auto s = info.GetAttribute<int64_t>("stride");
    Require(s == 2 || s == 5 || s == 6 || s == 8, "Phase stride must be 2, 5, 6, or 8");
    stride = static_cast<int>(s);
    Require(info.GetAttribute<int64_t>("previous_shift") == 1,
            "Phase previous input must be unshifted with previous_shift=1");
    const auto batches = info.GetAttribute<int64_t>("row_batches");
    Require(batches >= 0, "row_batches must be nonnegative");
    row_batches = static_cast<size_t>(batches);
    bias = Constant(info, 2, {channels});
  }

  struct Work {
    const PhaseKernel* self;
    const float* current;
    const float* previous;
    float* output;
    int64_t time;
  };

#if NCC_PHASE_NEON
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
                        int64_t time, float bias_value) noexcept {
    int64_t t = 0;
#if NCC_PHASE_NEON
    const auto bias_vec = vdupq_n_f32(bias_value);
    for (; t <= time - 4; t += 4) {
      float32x4_t values[phases];
      for (int p = 0; p < phases; ++p) {
        const auto offset = static_cast<int64_t>(p) * time;
        const auto cur = vld1q_f32(current + offset + t);
        // t=0 needs [0, previous[0], previous[1], previous[2]]. All
        // four loaded previous values exist because time >= t+4.
        const auto prev = t == 0
            ? vextq_f32(vdupq_n_f32(0.0f), vld1q_f32(previous + offset), 3)
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
    // separate additions, including addition of positive zero at t=0.
    for (; t < time; ++t) {
      for (int p = 0; p < phases; ++p) {
        const auto offset = static_cast<int64_t>(p) * time + t;
        const float prev = t ? previous[offset - 1] : 0.0f;
        const float sum = current[offset] + prev;
        output[t * phases + p] = sum + bias_value;
      }
    }
  }

  static void Row(void* opaque, size_t row) noexcept {
    const auto& work = *static_cast<Work*>(opaque);
    const auto& k = *work.self;
    const auto offset = row * static_cast<size_t>(work.time) * static_cast<size_t>(k.stride);
    const auto c = row % static_cast<size_t>(k.channels);
    const auto* cur = work.current + offset;
    const auto* prev = work.previous + offset;
    auto* out = work.output + offset;
    switch (k.stride) {
      case 2: FinishRow<2>(cur, prev, out, work.time, k.bias[c]); break;
      case 5: FinishRow<5>(cur, prev, out, work.time, k.bias[c]); break;
      case 6: FinishRow<6>(cur, prev, out, work.time, k.bias[c]); break;
      case 8: FinishRow<8>(cur, prev, out, work.time, k.bias[c]); break;
    }
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context(raw_context);
      auto cur = context.GetInput(0);
      auto prev = context.GetInput(1);
      const auto shape = FloatShape(cur);
      Require(shape.size() == 3 && shape[0] >= 0 && shape[1] == channels * stride && shape[2] >= 0,
              "Phase inputs must have shape [B, channels*stride, T]");
      Require(FloatShape(prev) == shape, "Current/previous projection shapes differ");
      Require(shape[2] <= std::numeric_limits<int64_t>::max() / stride, "Phase output length overflow");
      Require(static_cast<uint64_t>(shape[0]) <= std::numeric_limits<size_t>::max() / channels,
              "Phase row count overflow");
      auto out = context.GetOutput(0, std::vector<int64_t>{shape[0], channels, shape[2] * stride});
      if (!shape[0] || !shape[2]) return nullptr;
      Work work{this, cur.GetTensorData<float>(), prev.GetTensorData<float>(),
                out.GetTensorMutableData<float>(), shape[2]};
      const auto rows = static_cast<size_t>(shape[0]) * static_cast<size_t>(channels);
      if (rows == 1) Row(&work, 0);
      else context.ParallelFor(Row, rows, row_batches, &work);
      return nullptr;
    } catch (...) { return Error(api); }
  }
};

struct PhaseOp : Ort::CustomOpBase<PhaseOp, PhaseKernel, true> {
  PhaseOp() { this->start_ver_ = 1; this->end_ver_ = 1; }
  const char* GetName() const { return "PhaseSumBiasInterleaveF32"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** result) const noexcept {
    *result = nullptr;
    try { *result = new PhaseKernel(api, info); return nullptr; }
    catch (...) { return Error(api); }
  }
  static OrtStatus* InferOutputShape(Ort::ShapeInferContext& context) noexcept {
    try {
      auto shape = context.GetInputShape(0);
      Require(shape.size() == 3, "Phase inputs must have rank three");
      const int64_t channels = context.GetAttrInt("channels");
      const int64_t stride = context.GetAttrInt("stride");
      Require(channels > 0 && (stride == 2 || stride == 5 || stride == 6 || stride == 8),
              "Invalid phase channels or stride");
      shape[1] = channels;
      std::string symbol;
      if (shape[2].IsInt() && shape[2].AsInt() >= 0) {
        Require(shape[2].AsInt() <= std::numeric_limits<int64_t>::max() / stride,
                "Phase shape inference overflow");
        shape[2] = shape[2].AsInt() * stride;
      } else {
        symbol = (shape[2].IsInt() ? "unknown_phase_T" : std::string(shape[2].AsSym()))
                 + "_times_" + std::to_string(stride);
        shape[2] = symbol.c_str();
      }
      return context.SetOutputShape(0, shape).release();
    } catch (...) { return Error(Ort::GetApi()); }
  }
};

// Domain and operator objects must outlive every session using them. Only the
// registration path touches this state; kernels themselves are reentrant.
std::mutex registration_mutex;
const OrtApi* registered_api = nullptr;
Op<Mode::Snake> snake_op;
Op<Mode::DW7> dw_op;
Op<Mode::DW7Snake> fused_op;
Op<Mode::SnakeDW7Snake> chain_op;
BiasResidualOp bias_residual_op;
PhaseOp phase_op;
std::unique_ptr<Ort::CustomOpDomain> domain;
}  // namespace

extern "C" NCC_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                            const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (!api) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "This library requires ONNX Runtime API 29 (1.29+)");
  try {
    std::lock_guard<std::mutex> lock(registration_mutex);
    if (registered_api && registered_api != api)
      return api->CreateStatus(ORT_FAIL, "Loading multiple ORT runtimes into one custom-op library is unsupported");
    if (!registered_api) {
      Ort::InitApi(api);
      auto fresh = std::make_unique<Ort::CustomOpDomain>(kDomain);
      fresh->Add(&snake_op); fresh->Add(&dw_op); fresh->Add(&fused_op);
      fresh->Add(&phase_op); fresh->Add(&chain_op); fresh->Add(&bias_residual_op);
      domain = std::move(fresh);
      registered_api = api;
    }
    return api->AddCustomOpDomain(options, *domain);
  } catch (...) { return Error(*api); }
}
