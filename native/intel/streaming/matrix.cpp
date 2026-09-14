// Accepted one-thread BRGeMM64 projection arithmetic.
// Only mode4 is selected by the production bridge; comparison modes remain
// byte-identical to the qualification source to avoid arithmetic drift.
// Quantization follows experiments/intel-precision/native/precision.cpp.
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef DNNL_EXPERIMENTAL_UKERNEL
#error Build oneDNN with ONEDNN_EXPERIMENTAL_UKERNEL=ON
#endif
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SEQ || DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
#error This screen requires CPU SEQ and GPU NONE
#endif

using namespace dnnl;
using namespace dnnl::ukernel;
using DT = memory::data_type;
using TAG = memory::format_tag;
#define API extern "C" __attribute__((visibility("default")))
namespace {
thread_local std::string screen_error;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int quantize(float value, float scale) {
    float q = std::max(-127.f, std::min(127.f, value / scale));
    int n = int(std::floor(q));
    float fraction = q - float(n);
    if (fraction > .5f || (fraction == .5f && (n & 1))) ++n;
    return n;
}
float scale_of(float maximum) {
    if (maximum == 0.f) return 1.f;
    float scale = maximum / 127.f;
    return scale > 0.f ? scale : maximum;
}

// Match the existing core's 16-column conversion before changing its layout.
__attribute__((target("avx512f,avx512bw,avx512vnni")))
void quantize16(const float* x, int k, float* scales, int8_t* q) {
    const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.f);
    const __m512 limit = _mm512_set1_ps(127.f), negative = _mm512_set1_ps(-127.f);
    const __m512 largest = _mm512_set1_ps(std::numeric_limits<float>::max());
    const __m512i mask = _mm512_set1_epi32(0x7fffffff);
    __m512 maximum = zero;
    for (int row = 0; row < k; ++row) {
        __m512 value = _mm512_loadu_ps(x + size_t(row) * 16);
        __m512 absolute = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(value), mask));
        require(_mm512_cmp_ps_mask(absolute, largest, _CMP_LE_OQ) == 0xffff, "Nonfinite input");
        maximum = _mm512_max_ps(maximum, absolute);
    }
    __m512 scale = _mm512_div_ps(maximum, limit);
    scale = _mm512_mask_mov_ps(scale, _mm512_cmp_ps_mask(scale, zero, _CMP_EQ_OQ), maximum);
    scale = _mm512_mask_mov_ps(scale, _mm512_cmp_ps_mask(maximum, zero, _CMP_EQ_OQ), one);
    _mm512_storeu_ps(scales, scale);
    for (int row = 0; row < k; ++row) {
        __m512 value = _mm512_loadu_ps(x + size_t(row) * 16);
        __m512 rounded = _mm512_max_ps(negative, _mm512_min_ps(limit, _mm512_div_ps(value, scale)));
        __m512i integers = _mm512_cvt_roundps_epi32(rounded,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(q + size_t(row) * 16),
                _mm512_cvtsepi32_epi8(integers));
    }
}
struct Input {
    std::vector<float> scales;
    std::vector<uint8_t> tk;
    Input(const float* x, int k, int t) : scales(t), tk(size_t(k) * t) {
        std::vector<int8_t> kt(size_t(k) * t);
        if (t == 16) quantize16(x, k, scales.data(), kt.data());
        else {
            for (int col = 0; col < t; ++col) {
                float maximum = 0.f;
                for (int row = 0; row < k; ++row) {
                    float value = x[size_t(row) * t + col];
                    require(std::isfinite(value), "Nonfinite input");
                    maximum = std::max(maximum, std::abs(value));
                }
                scales[col] = scale_of(maximum);
            }
            for (int row = 0; row < k; ++row)
                for (int col = 0; col < t; ++col)
                    kt[size_t(row) * t + col] = int8_t(quantize(x[size_t(row) * t + col], scales[col]));
        }
        // Runtime transpose and the unsigned offset are part of measured work.
        for (int col = 0; col < t; ++col)
            for (int row = 0; row < k; ++row)
                tk[size_t(col) * k + row] = uint8_t(int(kt[size_t(row) * t + col]) + 128);
    }
};
struct Core {
    void* handle = nullptr;
    void* (*create)(int,int,const float*,int,int) = nullptr;
    void (*destroy_plan)(void*) = nullptr;
    void* (*prepare)(const void*,const float*,int) = nullptr;
    void (*destroy_input)(void*) = nullptr;
    int (*run_rows)(const void*,const void*,float*,int,int) = nullptr;
    size_t (*plan_bytes)(const void*) = nullptr;
    const char* (*last_error)() = nullptr;
    template<class T> T symbol(const char* name) {
        void* value = dlsym(handle, name);
        require(value != nullptr, "Precision core API symbol missing");
        return reinterpret_cast<T>(value);
    }
    explicit Core(const char* file) {
        handle = dlopen(file, RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(dlerror());
        try {
            create = symbol<decltype(create)>("ip_create");
            destroy_plan = symbol<decltype(destroy_plan)>("ip_destroy_plan");
            prepare = symbol<decltype(prepare)>("ip_prepare");
            destroy_input = symbol<decltype(destroy_input)>("ip_destroy_input");
            run_rows = symbol<decltype(run_rows)>("ip_run_rows");
            plan_bytes = symbol<decltype(plan_bytes)>("ip_plan_bytes");
            last_error = symbol<decltype(last_error)>("ip_last_error");
        } catch (...) { dlclose(handle); handle = nullptr; throw; }
    }
    ~Core() { if (handle) dlclose(handle); }
};
struct Weight {
    std::vector<float> scales;
    std::vector<int32_t> sums;
    std::vector<int8_t> packed;
    memory weights;
};
struct Plan {
    int m, k, t, mode, panel;
    engine cpu {engine::kind::cpu, 0};
    stream execution {cpu};
    matmul::primitive_desc pd;
    matmul primitive;
    brgemm kernel;
    memory::desc source_md, destination_md;
    std::array<Weight,2> weights;
    std::unique_ptr<Core> core;
    std::array<void*,2> core_plans {{nullptr, nullptr}};
    size_t scratch_bytes = 0, weight_bytes = 0;
    std::string info;
    bool profile = false;
    double prepare_ns = 0.;
    Plan(const char* core_file, int rows, int reduction, int time, int implementation,
            const float* w0, const float* w1)
        : m(rows), k(reduction), t(time), mode(implementation), panel(mode == 3 ? 32 : 64) {
        require(m == 3072 && k == 1024 && (t == 8 || t == 16), "Only verified second-pair shapes are enabled");
        require(mode == 4, "Only accepted BRGeMM64 mode is enabled");
        require(std::fegetround() == FE_TONEAREST, "Expected round-to-nearest environment");
        require(__builtin_cpu_supports("avx512vnni"), "AVX512 VNNI CPU/OS support required");
        std::array<const float*,2> source {{w0,w1}};
        if (mode < 2) {
            core = std::make_unique<Core>(core_file);
            try {
                for (int index = 0; index < 2; ++index) {
                    core_plans[index] = core->create(m, k, source[index], 8, 1);
                    if (!core_plans[index]) throw std::runtime_error(core->last_error());
                    weight_bytes += core->plan_bytes(core_plans[index]);
                }
            } catch (...) {
                for (void* pointer : core_plans) if (pointer) core->destroy_plan(pointer);
                throw;
            }
            info = mode == 0 ? "existing core: two independent preparations" : "existing core: one shared preparation";
            return;
        }
        source_md = memory::desc({t,k}, DT::u8, TAG::ab);
        destination_md = memory::desc({t,m}, DT::s32, TAG::ab);
        if (mode == 2) {
            primitive_attr attributes;
            attributes.set_scratchpad_mode(scratchpad_mode::user);
            pd = matmul::primitive_desc(cpu, source_md,
                    memory::desc({k,m}, DT::s8, TAG::any), destination_md, attributes);
            primitive = matmul(pd);
            scratch_bytes = pd.scratchpad_desc().get_size();
            info = pd.impl_info_str();
        } else {
            auto packing = brgemm::get_B_pack_type(DT::u8, DT::s8);
            require(packing != pack_type::undef, "BRGeMM unavailable on this CPU");
            kernel = brgemm(t, panel, k, 1, k, panel, panel, DT::u8, DT::s8, DT::s32);
            kernel.set_add_C(false);
            require(kernel.finalize(), "BRGeMM shape not supported");
            kernel.generate();
            scratch_bytes = kernel.get_scratchpad_size();
            info = "public BRGeMM: full K, panel=" + std::to_string(panel);
        }
        for (int index = 0; index < 2; ++index) {
            auto& weight = weights[index];
            weight.scales.resize(m);
            weight.sums.resize(m);
            std::vector<int8_t> dense(size_t(k) * m);
            for (int row = 0; row < m; ++row) {
                float maximum = 0.f;
                for (int inner = 0; inner < k; ++inner) {
                    float value = source[index][size_t(row) * k + inner];
                    require(std::isfinite(value), "Nonfinite weight");
                    maximum = std::max(maximum, std::abs(value));
                }
                weight.scales[row] = scale_of(maximum);
                for (int inner = 0; inner < k; ++inner) {
                    int q = quantize(source[index][size_t(row) * k + inner], weight.scales[row]);
                    dense[size_t(inner) * m + row] = int8_t(q);
                    weight.sums[row] += q;
                }
            }
            if (mode == 2) {
                auto plain = memory(memory::desc({k,m}, DT::s8, TAG::ab), cpu, dense.data());
                weight.weights = memory(pd.weights_desc(), cpu);
                reorder(plain, weight.weights).execute(execution, plain, weight.weights);
                execution.wait();
                weight_bytes += pd.weights_desc().get_size();
            } else {
                weight.packed.resize(size_t(k) * m);
                std::vector<int8_t> block(size_t(k) * panel);
                transform pack(k, panel, pack_type::no_trans, panel, panel, DT::s8, DT::s8);
                bool need_pack = brgemm::get_B_pack_type(DT::u8, DT::s8) != pack_type::no_trans;
                if (need_pack) pack.generate();
                for (int first = 0; first < m; first += panel) {
                    for (int inner = 0; inner < k; ++inner)
                        std::memcpy(block.data() + size_t(inner) * panel,
                                dense.data() + size_t(inner) * m + first, panel);
                    auto* destination = weight.packed.data() + size_t(first) * k;
                    if (need_pack) pack.execute(block.data(), destination);
                    else std::memcpy(destination, block.data(), block.size());
                }
                weight_bytes += weight.packed.size();
            }
            weight_bytes += size_t(m) * (sizeof(float) + sizeof(int32_t));
        }
    }
    ~Plan() { if (core) for (void* pointer : core_plans) if (pointer) core->destroy_plan(pointer); }
    void run_core(const float* input, float* y0, float* y1) {
        std::array<float*,2> outputs {{y0,y1}};
        void* prepared = nullptr;
        try {
            for (int index = 0; index < 2; ++index) {
                if (!prepared) {
                    auto begin = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    prepared = core->prepare(core_plans[index], input, t);
                    if (profile) prepare_ns += std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count();
                    if (!prepared) throw std::runtime_error(core->last_error());
                }
                if (core->run_rows(core_plans[index], prepared, outputs[index], 0, m))
                    throw std::runtime_error(core->last_error());
                if (mode == 0) { core->destroy_input(prepared); prepared = nullptr; }
            }
        } catch (...) { if (prepared) core->destroy_input(prepared); throw; }
        if (prepared) core->destroy_input(prepared);
    }
    void run(const float* input, float* y0, float* y1) {
        if (mode < 2) { run_core(input, y0, y1); return; }
        auto begin = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        Input prepared(input, k, t);
        if (profile) prepare_ns += std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count();
        std::vector<uint8_t> scratch(std::max<size_t>(scratch_bytes, 1));
        std::array<float*,2> outputs {{y0,y1}};
        if (mode == 2) {
            std::vector<int32_t> product(size_t(t) * m);
            auto source = memory(source_md, cpu, prepared.tk.data());
            auto destination = memory(destination_md, cpu, product.data());
            auto workspace = memory(pd.scratchpad_desc(), cpu, scratch.data());
            for (int index = 0; index < 2; ++index) {
                primitive.execute(execution, {{DNNL_ARG_SRC,source}, {DNNL_ARG_WEIGHTS,weights[index].weights},
                        {DNNL_ARG_DST,destination}, {DNNL_ARG_SCRATCHPAD,workspace}});
                execution.wait();
                for (int row = 0; row < m; ++row)
                    for (int col = 0; col < t; ++col) {
                        int32_t corrected = product[size_t(col) * m + row] - 128 * weights[index].sums[row];
                        outputs[index][size_t(row) * t + col] = float(corrected)
                                * (weights[index].scales[row] * prepared.scales[col]);
                    }
            }
        } else {
            std::vector<int32_t> product(size_t(t) * panel);
            const std::vector<std::pair<memory::dim,memory::dim>> offsets {{0,0}};
            kernel.set_hw_context();
            try {
                for (int index = 0; index < 2; ++index)
                    for (int first = 0; first < m; first += panel) {
                        kernel.execute(prepared.tk.data(), weights[index].packed.data() + size_t(first) * k,
                                offsets, product.data(), scratch.data());
                        for (int row = 0; row < panel; ++row)
                            for (int col = 0; col < t; ++col) {
                                int32_t corrected = product[size_t(col) * panel + row]
                                        - 128 * weights[index].sums[first + row];
                                outputs[index][size_t(first + row) * t + col] = float(corrected)
                                        * (weights[index].scales[first + row] * prepared.scales[col]);
                            }
                    }
            } catch (...) { brgemm::release_hw_context(); throw; }
            brgemm::release_hw_context();
        }
    }
};
}
API const char* iis_error() { return screen_error.c_str(); }
API const char* iis_version() {
    static std::string version = [] { const auto* v = dnnl_version();
        return std::to_string(v->major)+"."+std::to_string(v->minor)+"."+std::to_string(v->patch); }();
    return version.c_str();
}
API void* iis_create(const char* core, int m, int k, int t, int mode, const float* w0, const float* w1) {
    try { screen_error.clear(); return new Plan(core,m,k,t,mode,w0,w1); }
    catch (const std::exception& exception) { screen_error = exception.what(); return nullptr; }
}
API void iis_destroy(void* plan) { delete static_cast<Plan*>(plan); }
API const char* iis_info(const void* plan) { return static_cast<const Plan*>(plan)->info.c_str(); }
API size_t iis_weight_bytes(const void* plan) { return static_cast<const Plan*>(plan)->weight_bytes; }
API size_t iis_scratch_bytes(const void* plan) { return static_cast<const Plan*>(plan)->scratch_bytes; }
API int iis_run(void* plan, const float* x, float* y0, float* y1) {
    try { screen_error.clear(); static_cast<Plan*>(plan)->run(x,y0,y1); return 0; }
    catch (const std::exception& exception) { screen_error = exception.what(); return -1; }
}
// One attribution call after the timed screen. No per-phase clocks in normal runs.
// phases[0] is preparation, phases[1] is all remaining run work, in microseconds.
API int iis_run_phases(void* pointer, const float* x, float* y0, float* y1, double* phases) {
    auto* plan = static_cast<Plan*>(pointer);
    plan->profile = true;
    plan->prepare_ns = 0.;
    auto begin = std::chrono::steady_clock::now();
    int status = iis_run(pointer,x,y0,y1);
    double total = std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count();
    plan->profile = false;
    phases[0] = plan->prepare_ns / 1000.;
    phases[1] = (total-plan->prepare_ns) / 1000.;
    return status;
}
