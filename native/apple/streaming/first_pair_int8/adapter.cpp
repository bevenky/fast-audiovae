// Apple W8A8 projection kernel. Pinned upstream arithmetic is unmodified.
#include "adapter.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include <arm_neon.h>
#include <sys/sysctl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__FAST_MATH__)
#error "Apple ARM without fast math required"
#endif
#define KF(name) kai_##name##_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot
// Declaration is architecture-gated upstream; the baseline wrapper calls it only
// after sysctl feature checks. The official helper object supplies the ABI.
extern "C" uint64_t kai_get_sme_vector_length_u8(void);

namespace {
thread_local char error_text[256] = {};
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
void failed() noexcept {
    try { throw; }
    catch (const std::exception& e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); }
    catch (...) { std::snprintf(error_text, sizeof(error_text), "Unknown adapter error"); }
}
size_t bytes(size_t a, size_t b) {
    require(!b || a <= SIZE_MAX / b, "Size overflow"); return a * b;
}
bool overlap(const void* a, size_t na, const void* b, size_t nb) {
    const auto aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    require(aa <= UINTPTR_MAX - na && bb <= UINTPTR_MAX - nb, "Pointer range overflow");
    return na && nb && aa < bb + nb && bb < aa + na;
}
bool flag(const char* name) {
    int value = 0; size_t n = sizeof(value);
    return sysctlbyname(name, &value, &n, nullptr, 0) == 0 && n == sizeof(value) && value == 1;
}
size_t runtime(size_t expected = 0) {
    // Feature query precedes all SME instructions, including vector length query.
    static const bool supported = flag("hw.optional.arm.FEAT_SME") && flag("hw.optional.arm.FEAT_SME2");
    require(supported, "Apple SME and SME2 support required");
    uint64_t fpcr; __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    require((fpcr & ((uint64_t(3) << 22) | (uint64_t(1) << 24) | 3)) == 0,
            "Requires nearest FP32 rounding and unmodified denormal handling");
    const size_t vl = kai_get_sme_vector_length_u8();
    require(vl >= 16 && vl <= 256 && vl % 16 == 0 && (!expected || expected == vl),
            "Unsupported or changed SME streaming vector length");
    return vl;
}
void valid_q(const int8_t* q, size_t count) {
    const auto minimum = vdupq_n_s8(-128);
    size_t i = 0;
    for (; count - i >= 16; i += 16)
        require(vmaxvq_u8(vceqq_s8(vld1q_s8(q+i), minimum)) == 0, "Symmetric quantized values must be in [-127,127]");
    for (; i < count; ++i) require(q[i] != -128, "Symmetric quantized values must be in [-127,127]");
}
void valid_float_pointer(const float* x) {
    require(x && reinterpret_cast<uintptr_t>(x) % alignof(float) == 0, "Missing or unaligned FP32 buffer");
}
bool finite4(float32x4_t v) {
    return vmaxvq_u32(vcgtq_u32(vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x7fffffffu)),
                               vdupq_n_u32(0x7f7fffffu))) == 0;
}
struct Plan {
    size_t k, kp, n, vl;
    std::vector<uint32_t> rhs, lhs;
    std::vector<float> sw, dots;
    std::mutex mutex;
};
}

extern "C" {
const char* av8_error(void) { return error_text; }
void* av8_create(size_t K, size_t N, const int8_t* qw_NK, const float* sw_N) {
    error_text[0] = 0;
    try {
        // Bounds prevent INT32 accumulation overflow and unbounded accidental setup.
        require(K > 0 && K <= 16384 && N > 0 && N <= 1048576, "K must be 1..16384 and N 1..1048576");
        require(qw_NK, "Missing quantized weights"); valid_float_pointer(sw_N);
        const size_t count = bytes(K, N); valid_q(qw_NK, count);
        for (size_t n = 0; n < N; ++n)
            require(std::isfinite(sw_N[n]) && sw_N[n] > 0, "Weight scales must be finite and positive");
        auto p = std::make_unique<Plan>();
        p->k = K; p->kp = (K + 31) / 32 * 32; p->n = N; p->vl = runtime();
        require(KF(get_mr)() == 1 && KF(get_nr)() == p->vl && KF(get_kr)() == 4 && KF(get_sr)() == 1,
                "Unexpected upstream kernel geometry");
        p->sw.assign(sw_N, sw_N + N);
        const size_t nr = p->vl;
        const size_t padded_n = (N + nr - 1) / nr * nr;
        const size_t packed_bytes = bytes(padded_n, p->kp + 12);
        require(packed_bytes == kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(N, p->kp, nr, 4, 1),
                "Unexpected packed weight size");
        p->rhs.resize(packed_bytes / 4);
        // Unit scales return rounded FP32 integer dot values from upstream FMLA.
        std::vector<float> ones(N, 1.f), negative_zero(N, -0.f);
        std::vector<int8_t> padded;
        const int8_t* weights = qw_NK;
        if (p->kp != K) {
            padded.resize(bytes(N, p->kp), 0);
            for (size_t n = 0; n < N; ++n) std::memcpy(padded.data()+n*p->kp, qw_NK+n*K, K);
            weights = padded.data();
        }
        const kai_rhs_pack_qsi8cx_params params{0, 1.f};
        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, N, p->kp, nr, 4, 1, weights,
            negative_zero.data(), ones.data(), p->rhs.data(), 0, &params);
        p->lhs.resize((p->kp + 8) / 4, 0);
        const float one = 1.f;
        std::memcpy(reinterpret_cast<uint8_t*>(p->lhs.data()) + p->kp + 4, &one, 4);
        p->dots.resize(bytes(N, 2));
        return p.release();
    } catch (...) { failed(); return nullptr; }
}

int av8_run_q(void* raw, size_t T, const int8_t* qx_TK, const float* sx_T, float* y_NT, int variant) {
    error_text[0] = 0;
    try {
        require(raw, "Missing plan"); require(variant == 0, "Only variant 0 SME2 DOT supported");
        require(T == 1 || T == 2, "T must be 1 or 2");
        auto& p = *static_cast<Plan*>(raw);
        std::lock_guard<std::mutex> lock(p.mutex);
        runtime(p.vl);
        require(qx_TK, "Missing quantized activation"); valid_float_pointer(sx_T); valid_float_pointer(y_NT);
        const size_t xbytes = bytes(T, p.k), ybytes = bytes(bytes(T, p.n), sizeof(float));
        require(!overlap(y_NT, ybytes, qx_TK, xbytes) && !overlap(y_NT, ybytes, sx_T, T * sizeof(float)),
                "Output must not alias quantized input or scales");
        for (size_t t = 0; t < T; ++t)
            require(std::isfinite(sx_T[t]) && sx_T[t] > 0, "Activation scales must be finite and positive");
        valid_q(qx_TK, xbytes);
        for (size_t t = 0; t < T; ++t) {
            // Padding and unit metadata are persistent; only actual q bytes change.
            std::memcpy(p.lhs.data(), qx_TK+t*p.k, p.k);
            KF(run)(1, p.n, p.kp, p.lhs.data(), p.rhs.data(), p.dots.data()+t*p.n,
                    p.n*sizeof(float), sizeof(float), -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity());
        }
        // Scale product rounds first, then multiplication by rounded float(dot).
        // No contraction is allowed by build flags. T2 layout restoration is timed.
        const float32x4_t sx0 = vdupq_n_f32(sx_T[0]);
        const float32x4_t sx1 = vdupq_n_f32(sx_T[T-1]);
        size_t n = 0;
        for (; p.n-n >= 4; n += 4) {
            const float32x4_t sw = vld1q_f32(p.sw.data()+n);
            const float32x4_t s0 = vmulq_f32(sw, sx0);
            const float32x4_t y0 = vmulq_f32(vld1q_f32(p.dots.data()+n), s0);
            require(finite4(s0) && finite4(y0), "Nonfinite dequantized output or combined scale");
            if (T == 1) vst1q_f32(y_NT+n, y0);
            else {
                const float32x4_t s1 = vmulq_f32(sw, sx1);
                const float32x4_t y1 = vmulq_f32(vld1q_f32(p.dots.data()+p.n+n), s1);
                require(finite4(s1) && finite4(y1), "Nonfinite dequantized output or combined scale");
                const float32x4x2_t pair{{y0, y1}}; vst2q_f32(y_NT+2*n, pair);
            }
        }
        for (; n < p.n; ++n) for (size_t t = 0; t < T; ++t) {
            const float scale = p.sw[n] * sx_T[t];
            const float value = p.dots[t*p.n+n] * scale;
            require(std::isfinite(scale) && std::isfinite(value), "Nonfinite dequantized output or combined scale");
            y_NT[n*T+t] = value;
        }
        return 0;
    } catch (...) { failed(); return -1; }
}
void av8_destroy(void* raw) { delete static_cast<Plan*>(raw); }
}
