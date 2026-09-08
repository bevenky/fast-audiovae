#ifndef FAST_AUDIO_ITERATION3_DIRECT_VNNI_H
#define FAST_AUDIO_ITERATION3_DIRECT_VNNI_H

// Internal CPU-only experiment. Build with -ffp-contract=off, without fast math.
// No quantization is performed here. All input bytes/scales come from the
// accepted preparation routine. No public symbols or internal thread pool.
// Wide candidate: MR4/NR64 by default, with exact NR16 tails. Source-only
// derivative of header SHA256 96792f107d35f11cc84b592384b93be531f9803e862ee153cc21248a0fbe928e.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__FAST_MATH__)
#error "Direct VNNI requires ordinary FP32 semantics"
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#define IP3_DIRECT_VNNI_X86 1
#define IP3_DIRECT_VNNI_TARGET __attribute__((target("avx512f,avx512bw,avx512vnni")))
#else
#define IP3_DIRECT_VNNI_X86 0
#endif

#ifndef IP3_VNNI_ROWS
#define IP3_VNNI_ROWS 4
#endif
static_assert(IP3_VNNI_ROWS == 4 || IP3_VNNI_ROWS == 8, "Choose four or eight register rows");
#ifndef IP3_VNNI_COLUMNS
#define IP3_VNNI_COLUMNS 64
#endif
static_assert(IP3_VNNI_COLUMNS == 32 || IP3_VNNI_COLUMNS == 64, "Choose 32 or 64 wide columns");
#if defined(__clang__)
#define IP3_VNNI_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define IP3_VNNI_UNROLL _Pragma("GCC unroll 8")
#else
#define IP3_VNNI_UNROLL
#endif

namespace ip3_direct_vnni {

static inline void check(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}

static inline bool host_available() {
#if IP3_DIRECT_VNNI_X86
    static const bool available = [] {
        unsigned a, b, c, d;
        uint32_t lo, hi;
        if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & bit_AVX) || !(c & bit_OSXSAVE)) return false;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        (void)hi;
        if ((lo & 0xe6) != 0xe6 || !__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
        return (b & (1u << 16)) && (b & (1u << 30)) && (c & (1u << 11));
    }();
    return available;
#else
    return false;
#endif
}

static inline bool pack_shape(int k, int t) {
    return k >= 4 && k <= 256 && k % 4 == 0 && t >= 16 && t <= 256 && t % 16 == 0;
}

static inline bool shape(int m, int k, int t, int first, int last) {
    return pack_shape(k, t) && m >= 4 && m <= 256 && m % 4 == 0
        && first >= 0 && last >= first && last <= m && first % 4 == 0 && last % 4 == 0;
}

static inline bool disjoint(const void* a, size_t an, const void* b, size_t bn) {
    if (!an || !bn) return true;
    check(a && b, "Null direct VNNI buffer");
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    check(av <= UINTPTR_MAX-an && bv <= UINTPTR_MAX-bn, "Direct VNNI pointer range overflow");
    return av + an <= bv || bv + bn <= av;
}

struct Arguments {
    int m, k, t, first, last;
    const uint8_t* weights;  // Original row-major U8(qweight+128), [M,K].
    const int8_t* packed;    // [K/4,T,4], full current K only.
    const float* sw;         // [M], unchanged weight scales.
    const float* sx;         // [T], unchanged per-column activation scales.
    const int32_t* sums;     // [T], sum of signed activation bytes over K.
    float* output;           // Full row-major [M,T].
    const float* bias;       // Both NULL or bias[M] plus skip[M,T].
    const float* skip;
};

#if IP3_DIRECT_VNNI_X86
IP3_DIRECT_VNNI_TARGET
static inline void pack_impl(const int8_t* source, int k, int t, int8_t* packed) {
    for (int row = 0; row < k; row += 4) {
        int8_t* destination = packed + size_t(row / 4) * t * 4;
        for (int column = 0; column < t; column += 16) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + size_t(row) * t + column));
            const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + size_t(row+1) * t + column));
            const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + size_t(row+2) * t + column));
            const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + size_t(row+3) * t + column));
            const __m128i ab0 = _mm_unpacklo_epi8(a, b), ab1 = _mm_unpackhi_epi8(a, b);
            const __m128i cd0 = _mm_unpacklo_epi8(c, d), cd1 = _mm_unpackhi_epi8(c, d);
            int8_t* out = destination + size_t(column) * 4;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm_unpacklo_epi16(ab0, cd0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out+16), _mm_unpackhi_epi16(ab0, cd0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out+32), _mm_unpacklo_epi16(ab1, cd1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out+48), _mm_unpackhi_epi16(ab1, cd1));
        }
    }
}

IP3_DIRECT_VNNI_TARGET
static inline void finite_vector(__m512 values, const char* message) {
    const __m512 absolute = _mm512_castsi512_ps(_mm512_and_si512(
        _mm512_castps_si512(values), _mm512_set1_epi32(0x7fffffff)));
    check(_mm512_cmp_ps_mask(absolute, _mm512_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ) == 0xffff,
          message);
}

template<int Rows, int Vectors, bool Residual>
IP3_DIRECT_VNNI_TARGET
__attribute__((always_inline)) static inline void block(const Arguments& a, int row, int column) {
    static_assert(Rows * Vectors <= 16, "At most sixteen integer accumulators");
    __m512i accumulators[Rows][Vectors];
    IP3_VNNI_UNROLL
    for (int r = 0; r < Rows; ++r) {
        IP3_VNNI_UNROLL
        for (int v = 0; v < Vectors; ++v) accumulators[r][v] = _mm512_setzero_si512();
    }
    // Complete K stays in INT32 registers. VPDPBUSD consumes unsigned weight
    // bytes first and signed activation bytes second. No INT16 saturation.
    for (int k = 0; k < a.k; k += 4) {
        __m512i activations[Vectors];
        IP3_VNNI_UNROLL
        for (int v = 0; v < Vectors; ++v)
            activations[v] = _mm512_loadu_si512(a.packed + (size_t(k / 4) * a.t + column + 16*v) * 4);
        IP3_VNNI_UNROLL
        for (int r = 0; r < Rows; ++r) {
            int32_t word;
            std::memcpy(&word, a.weights + size_t(row+r) * a.k + k, sizeof(word));
            const __m512i weights = _mm512_set1_epi32(word);
            IP3_VNNI_UNROLL
            for (int v = 0; v < Vectors; ++v)
                accumulators[r][v] = _mm512_dpbusd_epi32(accumulators[r][v], weights, activations[v]);
        }
    }
    IP3_VNNI_UNROLL
    for (int r = 0; r < Rows; ++r) {
        IP3_VNNI_UNROLL
        for (int v = 0; v < Vectors; ++v) {
            const int t = column + 16*v;
            const __m512i correction = _mm512_slli_epi32(_mm512_loadu_si512(a.sums + t), 7);
            const __m512 column_scale = _mm512_loadu_ps(a.sx + t);
            const __m512 scale = _mm512_mul_ps(_mm512_set1_ps(a.sw[row+r]), column_scale);
            const __m512 value = _mm512_mul_ps(_mm512_cvtepi32_ps(
                _mm512_sub_epi32(accumulators[r][v], correction)), scale);
            finite_vector(value, "Nonfinite direct VNNI product");
            float* destination = a.output + size_t(row+r) * a.t + t;
            if constexpr (Residual) {
                check(std::isfinite(a.bias[row+r]), "Nonfinite direct VNNI bias");
                const __m512 skip = _mm512_loadu_ps(a.skip + size_t(row+r) * a.t + t);
                finite_vector(skip, "Nonfinite direct VNNI skip");
                const __m512 biased = _mm512_add_ps(value, _mm512_set1_ps(a.bias[row+r]));
                const __m512 result = _mm512_add_ps(skip, biased);
                finite_vector(result, "Nonfinite direct VNNI result");
                _mm512_storeu_ps(destination, result);
            } else {
                _mm512_storeu_ps(destination, value);
            }
        }
    }
}

template<bool Residual>
IP3_DIRECT_VNNI_TARGET
static inline void run_impl(const Arguments& a, int register_rows) {
    int row = a.first;
    if (register_rows == 8) {
        // MR8 uses NR32, also sixteen accumulators. This retains the explicit
        // row-block option without creating a spill-prone MR8/NR64 variant.
        for (; row <= a.last - 8; row += 8) {
            int t = 0;
            for (; t <= a.t - 32; t += 32) block<8, 2, Residual>(a, row, t);
            for (; t < a.t; t += 16) block<8, 1, Residual>(a, row, t);
        }
    }
    for (; row < a.last; row += 4) {
        int t = 0;
        for (; t <= a.t - IP3_VNNI_COLUMNS; t += IP3_VNNI_COLUMNS)
            block<4, IP3_VNNI_COLUMNS/16, Residual>(a, row, t);
        for (; t < a.t; t += 16) block<4, 1, Residual>(a, row, t);
    }
}
#endif

// Unsupported shape/backend/ISA returns false without reading or writing data.
// Eligible calls with invalid storage throw, matching the parent checked API.
static inline bool pack(const int8_t* source, size_t source_bytes, int k, int t,
                        int8_t* packed, size_t packed_bytes, int backend, int capabilities) {
    if (backend != 1 || (capabilities & 3) != 3 || !pack_shape(k, t) || !host_available()) return false;
    const size_t bytes = size_t(k) * t;
    check(source_bytes >= bytes && packed_bytes >= bytes, "Direct VNNI packing capacity too small");
    check(disjoint(source, bytes, packed, bytes), "Direct VNNI pack source/output overlap");
#if IP3_DIRECT_VNNI_X86
    pack_impl(source, k, t, packed);
    return true;
#else
    return false;
#endif
}

static inline bool run(const Arguments& a, int backend, int capabilities,
                       int register_rows = IP3_VNNI_ROWS) {
    if (backend != 1 || (capabilities & 3) != 3 || !shape(a.m, a.k, a.t, a.first, a.last)
        || (register_rows != 4 && register_rows != 8) || !host_available()) return false;
    if (a.first == a.last) return true;
    check((a.bias == nullptr) == (a.skip == nullptr), "Direct VNNI bias/skip must both be supplied");
    const size_t output_bytes = size_t(a.m) * a.t * sizeof(float);
    check(disjoint(a.output, output_bytes, a.weights, size_t(a.m) * a.k), "Direct VNNI output/weight overlap");
    check(disjoint(a.output, output_bytes, a.packed, size_t(a.k) * a.t), "Direct VNNI output/activation overlap");
    check(disjoint(a.output, output_bytes, a.sw, size_t(a.m) * sizeof(float)), "Direct VNNI output/weight-scale overlap");
    check(disjoint(a.output, output_bytes, a.sx, size_t(a.t) * sizeof(float)), "Direct VNNI output/activation-scale overlap");
    check(disjoint(a.output, output_bytes, a.sums, size_t(a.t) * sizeof(int32_t)), "Direct VNNI output/sum overlap");
    if (a.bias) {
        check(disjoint(a.output, output_bytes, a.bias, size_t(a.m) * sizeof(float)), "Direct VNNI output/bias overlap");
        check(disjoint(a.output, output_bytes, a.skip, output_bytes), "Direct VNNI output/skip overlap");
    }
#if IP3_DIRECT_VNNI_X86
    if (a.bias) run_impl<true>(a, register_rows);
    else run_impl<false>(a, register_rows);
    return true;
#else
    return false;
#endif
}

}  // namespace ip3_direct_vnni
#undef IP3_VNNI_UNROLL
#endif
