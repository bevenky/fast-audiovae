// Symmetric, causal INT8 packing for the pinned Arm complete-K SME2 kernel.
#include "sme_backend.h"
#include "upstream/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#include <sys/sysctl.h>
#else
#error "This isolated adapter requires Apple ARM"
#endif
#if defined(__FAST_MATH__)
#error "Exact INT8 dequantization must not use fast math"
#endif

extern "C" uint64_t kai_get_sme_vector_length_u8(void);
namespace {
thread_local char error_text[256] = {};
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
void failed() noexcept {
    try { throw; }
    catch (const std::exception& e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); }
    catch (...) { std::snprintf(error_text, sizeof(error_text), "Unknown SME adapter failure"); }
}
bool flag(const char* name) {
    int value = 0; size_t bytes = sizeof(value);
    return sysctlbyname(name, &value, &bytes, nullptr, 0) == 0 && bytes == sizeof(value) && value == 1;
}
size_t multiply(size_t a, size_t b) {
    require(!b || a <= SIZE_MAX / b, "SME byte size overflow");
    return a * b;
}
size_t rounded(size_t n, size_t multiple) {
    require(multiple && n <= SIZE_MAX - (multiple - 1), "SME rounding overflow");
    return ((n + multiple - 1) / multiple) * multiple;
}
void geometry(size_t vl) {
    require(vl >= 16 && vl <= 256 && vl % 16 == 0, "Invalid packed SME streaming vector length");
}
void shape(size_t m, size_t k, size_t t) {
    require(m && m <= INT32_MAX && k && k <= 16384 && t <= INT32_MAX, "Invalid SME matrix shape");
}
size_t lhs_size(size_t m, size_t k, size_t vl) {
    geometry(vl); shape(m, k, 0);
    return multiply(rounded(m, vl / 4), rounded(k, 32) + 8);
}
size_t rhs_size(size_t k, size_t t, size_t vl) {
    geometry(vl); shape(1, k, t);
    return multiply(rounded(t, vl), rounded(k, 32) + 12);
}
bool overlap(const void* a, size_t na, const void* b, size_t nb) {
    if (!na || !nb) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    require(aa <= UINTPTR_MAX - na && bb <= UINTPTR_MAX - nb, "SME pointer range overflow");
    return aa < bb + nb && bb < aa + na;
}
void buffer(const void* p, size_t bytes, size_t required) {
    require(bytes >= required && (p || !required), "SME packed buffer is too small or missing");
    require(!required || reinterpret_cast<uintptr_t>(p) % 4 == 0, "SME packed buffer must be 4-byte aligned");
}
void runtime(size_t vl) {
    require(ipa_sme_supported(), "Apple SME2 capability unavailable");
    require(kai_get_sme_vector_length_u8() == vl, "Worker SME vector length differs from packed layout");
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    // RMode, FZ, and FEAT_AFP FIZ/AH affect the exact FP32 edge contract.
    require((fpcr & ((uint64_t(3) << 22) | (uint64_t(1) << 24) | 3)) == 0,
            "SME exact FP32 requires nearest rounding and unmodified FP32 denormals");
}
void finite_output(const float* data, size_t count) {
    const uint32x4_t mask = vdupq_n_u32(0x7fffffffu), largest = vdupq_n_u32(0x7f7fffffu);
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        const uint32x4_t bits = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(data + i)), mask);
        require(vmaxvq_u32(vcgtq_u32(bits, largest)) == 0, "Nonfinite SME precision output");
    }
    for (; i < count; ++i) require(std::isfinite(data[i]), "Nonfinite SME precision output");
}
}

extern "C" {
int ipa_sme_supported(void) {
    static const bool supported = flag("hw.optional.arm.FEAT_SME") && flag("hw.optional.arm.FEAT_SME2");
    return supported ? 1 : 0;
}
const char* ipa_sme_last_error(void) { return error_text; }
int ipa_sme_geometry(size_t* vl, size_t* mr, size_t* nr) {
    error_text[0] = 0;
    try {
        require(vl && mr && nr, "Missing SME geometry output");
        require(ipa_sme_supported(), "Apple SME2 capability unavailable");
        const size_t value = kai_get_sme_vector_length_u8(); geometry(value);
        *vl = value; *mr = value / 4; *nr = value; return 0;
    } catch (...) { failed(); return -1; }
}
int ipa_sme_validate_runtime(size_t vl) {
    error_text[0] = 0;
    try { geometry(vl); runtime(vl); return 0; }
    catch (...) { failed(); return -1; }
}
size_t ipa_sme_lhs_bytes(size_t m, size_t k, size_t vl) {
    error_text[0] = 0;
    try { return lhs_size(m, k, vl); } catch (...) { failed(); return 0; }
}
size_t ipa_sme_rhs_bytes(size_t k, size_t t, size_t vl) {
    error_text[0] = 0;
    try { return rhs_size(k, t, vl); } catch (...) { failed(); return 0; }
}
int ipa_sme_pack_lhs(const int8_t* qw, const float* sw, size_t m, size_t k,
                     void* destination, size_t bytes, size_t vl) {
    error_text[0] = 0;
    try {
        const size_t required = lhs_size(m, k, vl), mr = vl / 4, kp = rounded(k, 32);
        buffer(destination, bytes, required); require(qw && sw, "Missing SME weight values/scales");
        require(!overlap(destination, required, qw, multiply(m, k)) &&
                !overlap(destination, required, sw, multiply(m, 4)), "SME LHS packing aliases its input");
        std::memset(destination, 0, required);
        auto* packed = static_cast<uint8_t*>(destination);
        for (size_t row = 0; row < m; ++row) {
            require(std::isfinite(sw[row]) && sw[row] > 0.f, "Invalid SME weight scale");
            const size_t lane = row % mr;
            auto* block = packed + (row / mr) * mr * (kp + 8);
            for (size_t q = 0; q < k; ++q) block[(q / 4) * mr * 4 + lane * 4 + q % 4] = uint8_t(qw[row * k + q]);
            std::memcpy(block + mr * kp + mr * 4 + lane * 4, sw + row, 4);
        }
        return 0;
    } catch (...) { failed(); return -1; }
}
int ipa_sme_pack_rhs(const int8_t* qx, const float* sx, size_t k, size_t t,
                     void* destination, size_t bytes, size_t vl) {
    error_text[0] = 0;
    try {
        const size_t required = rhs_size(k, t, vl), nr = vl, kp = rounded(k, 32);
        buffer(destination, bytes, required); if (!t) return 0;
        require(qx && sx, "Missing SME activation values/scales");
        require(!overlap(destination, required, qx, multiply(k, t)) &&
                !overlap(destination, required, sx, multiply(t, 4)), "SME RHS packing aliases its input");
        std::memset(destination, 0, required);
        auto* packed = static_cast<uint8_t*>(destination);
        const uint32_t negative_zero = 0x80000000u;
        for (size_t col = 0; col < t; ++col) {
            require(std::isfinite(sx[col]) && sx[col] > 0.f, "Invalid SME activation scale");
            const size_t lane = col % nr;
            auto* block = packed + (col / nr) * nr * (kp + 12);
            for (size_t q = 0; q < k; ++q) block[(q / 4) * nr * 4 + lane * 4 + q % 4] = uint8_t(qx[q * t + col]);
            std::memcpy(block + nr * kp + nr * 4 + lane * 4, sx + col, 4);
            std::memcpy(block + nr * kp + nr * 8 + lane * 4, &negative_zero, 4);
        }
        return 0;
    } catch (...) { failed(); return -1; }
}
int ipa_sme_run_rows(size_t m, size_t k, size_t t,
                     const void* lhs, size_t lhs_bytes,
                     const void* rhs, size_t rhs_bytes,
                     float* y, size_t first, size_t last, size_t vl) {
    return ipa_sme_run_panel(m, k, t, lhs, lhs_bytes, rhs, rhs_bytes,
                            y, t, 0, first, last, vl);
}
int ipa_sme_run_panel(size_t m, size_t k, size_t t,
                     const void* lhs, size_t lhs_bytes,
                     const void* rhs, size_t rhs_bytes,
                     float* y, size_t output_time, size_t output_offset,
                     size_t first, size_t last, size_t vl) {
    error_text[0] = 0;
    try {
        geometry(vl); shape(m, k, t);
        require(output_time <= INT32_MAX && output_offset <= output_time &&
                t <= output_time - output_offset, "SME output time region exceeds row stride");
        require(first <= last && last <= m && first % (vl / 4) == 0, "SME row range requires MR-aligned first row");
        if (!t || first == last) return 0;
        const size_t need_lhs = lhs_size(m, k, vl), need_rhs = rhs_size(k, t, vl);
        buffer(lhs, lhs_bytes, need_lhs); buffer(rhs, rhs_bytes, need_rhs);
        const size_t output_bytes = multiply(multiply(m, output_time), 4);
        require(y && reinterpret_cast<uintptr_t>(y) % 4 == 0, "Missing or unaligned SME output");
        require(!overlap(y, output_bytes, lhs, need_lhs) && !overlap(y, output_bytes, rhs, need_rhs),
                "SME output overlaps packed operands");
        runtime(vl);
        const auto* row_lhs = static_cast<const uint8_t*>(lhs) + first * (rounded(k, 32) + 8);
        float* output = y + first * output_time + output_offset;
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
            last - first, t, k, row_lhs, rhs, output, multiply(output_time, 4), 4,
            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        if (output_time == t) finite_output(output, multiply(last - first, t));
        else for (size_t row = first; row < last; ++row)
            finite_output(y + row * output_time + output_offset, t);
        return 0;
    } catch (...) { failed(); return -1; }
}
}
