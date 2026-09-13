// Experimental FP32 Apple streaming operators. Original DW arithmetic is linked
// unchanged from native/apple/native_kernels.c; no mutable stream state is hidden.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "native_kernels.h"
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#if defined(__FAST_MATH__)
#error "FP32 candidate requires fast math disabled"
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr const char* kDomain = "fast.audiovae.apple.state.v2";
constexpr size_t kScratch = 4096;
enum class Mode { StateDW, StateDWSnake, PackedSnake, PackedDWSnake };
void Require(bool ok, const char* text) { if (!ok) throw std::invalid_argument(text); }
OrtStatus* Error(const OrtApi& api) noexcept {
  try { throw; }
  catch (const Ort::Exception& e) { return api.CreateStatus(e.GetOrtErrorCode(), e.what()); }
  catch (const std::invalid_argument& e) { return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what()); }
  catch (const std::exception& e) { return api.CreateStatus(ORT_FAIL, e.what()); }
  catch (...) { return api.CreateStatus(ORT_FAIL, "Unknown Apple state candidate failure"); }
}
std::vector<int64_t> Shape(Ort::ConstValue value) {
  Require(value != nullptr && value.IsTensor(), "Dense tensor required");
  auto info = value.GetTensorTypeAndShapeInfo();
  Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FP32 required");
  Require(value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU, "CPU tensor required");
  return info.GetShape();
}
std::vector<float> Constant(Ort::ConstKernelInfo info, size_t index, std::vector<int64_t> shape) {
  int fixed = 0;
  auto value = info.GetTensorConstantInput(index, &fixed);
  Require(fixed && value != nullptr && Shape(value) == shape, "Wrong or overridable constant");
  size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
  const float* p = value.GetTensorData<float>();
  for (size_t i = 0; i < count; ++i) Require(std::isfinite(p[i]), "Nonfinite constant");
  return {p, p + count};
}

struct Kernel {
  const OrtApi& api;
  Mode mode;
  int64_t channels;
  int32_t dilation;
  bool stateful, packed;
  size_t tile;
  std::vector<float> weights, bias, alpha, reciprocal;
  // Preallocated bounded scratch is private to this operator instance, guarded
  // across simultaneous sessions/streams. It is not an audio-history cache.
  std::array<float, kScratch> sine;
  std::mutex scratch_mutex;

  Kernel(const OrtApi& a, const OrtKernelInfo* raw, Mode m) : api(a), mode(m) {
    Ort::ConstKernelInfo info(raw);
    Require(ncc_abi_version() == 1 && ncc_backend_available(NCC_NEON), "Original NEON DW ABI required");
    Require(ncc_capabilities() & NCC_CAP_VFORCE, "Apple vForce required");
    Require(info.GetAttribute<int64_t>("candidate_abi") == 1, "Candidate ABI mismatch");
    channels = info.GetAttribute<int64_t>("channels");
    Require(channels > 0 && channels <= std::numeric_limits<int32_t>::max(), "Invalid channels");
    stateful = mode == Mode::StateDW || mode == Mode::StateDWSnake;
    const bool snake = mode != Mode::StateDW;
    packed = mode == Mode::PackedSnake || mode == Mode::PackedDWSnake;
    if (mode == Mode::StateDWSnake) {
      const int64_t value = info.GetAttribute<int64_t>("packed_snake");
      Require(value == 0 || value == 1, "packed_snake must be 0 or 1");
      packed = value == 1;
    }
    tile = kScratch;
    if (packed) {
      auto value = info.GetAttribute<int64_t>("scratch_elements");
      Require(value >= 256 && value <= static_cast<int64_t>(kScratch) && value % 256 == 0,
              "Scratch must be a multiple of 256, at most 4096");
      tile = static_cast<size_t>(value);
    }
    size_t coefficient = stateful ? 2 : 1;
    dilation = 1;
    if (mode != Mode::PackedSnake) {
      auto d = info.GetAttribute<int64_t>("dilation");
      Require(d == 1 || d == 3 || d == 9, "Dilation must be 1, 3 or 9");
      dilation = static_cast<int32_t>(d);
      weights = Constant(info, coefficient++, {channels, 1, 7});
      bias = Constant(info, coefficient++, {channels});
    }
    if (snake) {
      alpha = Constant(info, coefficient++, {channels});
      reciprocal = Constant(info, coefficient, {channels});
    }
  }

  // Flattening BCT changes only sine call granularity. Channel coefficients,
  // multiplication/addition order and original x in the Snake finish are kept.
  // input==output is permitted: a whole scratch tile is read before its finish.
  void Packed(const float* input, float* output, int64_t time) {
    const size_t tsize = static_cast<size_t>(time);
    const size_t count = static_cast<size_t>(channels) * tsize;
    std::lock_guard<std::mutex> lock(scratch_mutex);
    for (size_t base = 0; base < count; base += tile) {
      const size_t n = std::min(tile, count - base);
      for (size_t offset = 0; offset < n;) {
        const size_t index = base + offset, channel = index / tsize;
        const size_t span = std::min(n - offset, tsize - index % tsize);
        const auto av = vdupq_n_f32(alpha[channel]);
        size_t j = 0;
        for (; j + 4 <= span; j += 4)
          vst1q_f32(sine.data() + offset + j, vmulq_f32(av, vld1q_f32(input + index + j)));
        for (; j < span; ++j) sine[offset + j] = alpha[channel] * input[index + j];
        offset += span;
      }
      int length = static_cast<int>(n);
      vvsinf(sine.data(), sine.data(), &length);
      for (size_t offset = 0; offset < n;) {
        const size_t index = base + offset, channel = index / tsize;
        const size_t span = std::min(n - offset, tsize - index % tsize);
        const auto rv = vdupq_n_f32(reciprocal[channel]);
        size_t j = 0;
        for (; j + 4 <= span; j += 4) {
          const auto s = vld1q_f32(sine.data() + offset + j);
          const auto square = vmulq_f32(s, s);
          const auto correction = vmulq_f32(rv, square);
          vst1q_f32(output + index + j, vaddq_f32(vld1q_f32(input + index + j), correction));
        }
        for (; j < span; ++j) {
          const float square = sine[offset + j] * sine[offset + j];
          const float correction = reciprocal[channel] * square;
          output[index + j] = input[index + j] + correction;
        }
        offset += span;
      }
    }
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      auto xvalue = context.GetInput(0);
      auto shape = Shape(xvalue);
      Require(shape.size() == 3 && shape[0] == 1 && shape[1] == channels && shape[2] >= 0,
              "Input must be FP32 [1,C,T]");
      const int64_t time = shape[2], halo = 6 * dilation;
      Require(static_cast<uint64_t>(time) <= std::numeric_limits<size_t>::max() / (4 * static_cast<uint64_t>(channels)),
              "Input size overflow");
      const float* x = xvalue.GetTensorData<float>();
      const float* history = nullptr;
      if (stateful) {
        auto hvalue = context.GetInput(1);
        Require(Shape(hvalue) == std::vector<int64_t>({1, channels, halo}), "History shape mismatch");
        history = hvalue.GetTensorData<float>();
      }
      auto yvalue = context.GetOutput(0, shape);
      float* y = yvalue.GetTensorMutableData<float>();
      if (time) {
        if (mode == Mode::PackedSnake) Packed(x, y, time);
        else {
          const bool fused = mode != Mode::StateDW && !packed;
          const auto status = ncc_dw7_f32_ex(x, weights.data(), bias.data(), history,
              fused ? alpha.data() : nullptr, fused ? reciprocal.data() : nullptr,
              y, 1, channels, time, dilation, NCC_NEON, 1);
          Require(status == NCC_OK, ncc_status_string(status));
          if (packed) Packed(y, y, time);
        }
      }
      if (stateful) {
        auto nextvalue = context.GetOutput(1, std::vector<int64_t>{1, channels, halo});
        float* next = nextvalue.GetTensorMutableData<float>();
        for (int64_t c = 0; c < channels; ++c) {
          float* target = next + c * halo;
          if (time >= halo) std::memcpy(target, x + c * time + time - halo, static_cast<size_t>(halo) * sizeof(float));
          else {
            std::memcpy(target, history + c * halo + time, static_cast<size_t>(halo - time) * sizeof(float));
            if (time) std::memcpy(target + halo - time, x + c * time, static_cast<size_t>(time) * sizeof(float));
          }
        }
      }
      return nullptr;
    } catch (...) { return Error(api); }
  }
};

template <Mode mode> struct Op : Ort::CustomOpBase<Op<mode>, Kernel, true> {
  Op() { this->start_ver_ = 1; this->end_ver_ = 1; }
  const char* GetName() const {
    if constexpr (mode == Mode::StateDW) return "StatefulDW7F32";
    if constexpr (mode == Mode::StateDWSnake) return "StatefulDW7SnakeF32";
    if constexpr (mode == Mode::PackedDWSnake) return "PackedDW7SnakeF32";
    return "PackedSnakeF32";
  }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const {
    if constexpr (mode == Mode::StateDW) return 4;
    if constexpr (mode == Mode::StateDWSnake) return 6;
    if constexpr (mode == Mode::PackedDWSnake) return 5;
    return 3;
  }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return mode == Mode::StateDW || mode == Mode::StateDWSnake ? 2 : 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** out) const noexcept {
    *out = nullptr;
    try { *out = new Kernel(api, info, mode); return nullptr; }
    catch (...) { return Error(api); }
  }
};
Op<Mode::StateDW> state_dw;
Op<Mode::StateDWSnake> state_snake;
Op<Mode::PackedSnake> packed_snake;
Op<Mode::PackedDWSnake> packed_dw_snake;
std::mutex registration_mutex;
const OrtApi* registered_api = nullptr;
std::unique_ptr<Ort::CustomOpDomain> domain;
}  // namespace

extern "C" NCC_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "ORT API29 required");
  try {
    std::lock_guard<std::mutex> lock(registration_mutex);
    Require(!registered_api || registered_api == api, "Mixed ORT runtimes unsupported");
    if (!registered_api) {
      Ort::InitApi(api);
      auto fresh = std::make_unique<Ort::CustomOpDomain>(kDomain);
      fresh->Add(&state_dw); fresh->Add(&state_snake); fresh->Add(&packed_snake); fresh->Add(&packed_dw_snake);
      domain = std::move(fresh); registered_api = api;
    }
    return api->AddCustomOpDomain(options, *domain);
  } catch (...) { return Error(*api); }
}
