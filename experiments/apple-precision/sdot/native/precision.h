#ifndef FAST_AUDIO_INTEL_PRECISION_H
#define FAST_AUDIO_INTEL_PRECISION_H
#define ip_abi ipa_abi
#define ip_capabilities ipa_capabilities
#define ip_last_error ipa_last_error
#define ip_create ipa_create
#define ip_destroy_plan ipa_destroy_plan
#define ip_prepare ipa_prepare
#define ip_destroy_input ipa_destroy_input
#define ip_run_rows ipa_run_rows
#define ip_plan_bytes ipa_plan_bytes
#define ip_input_bytes ipa_input_bytes
#include <stddef.h>
#include <stdint.h>
#if defined(__GNUC__) || defined(__clang__)
#define IP_EXPORT __attribute__((visibility("default")))
#else
#define IP_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental approximation, ABI1. All activation entry/output arrays are
 * contiguous FP32 BCT rows without a batch dimension. No bias or skip epilogue.
 * mode8: symmetric 8-bit row weights/column activations, INT32 dot, FP32 output.
 * Internal MKL packing uses U8(qweight+128), S8 activation and compensation.
 * mode16: round both operands to IEEE half then use FP32 accumulation/output.
 * Apple port: backend0 scalar reference; backend2 guarded ARM SDOT, mode8 only.
 * A plan is immutable and safe for concurrent independent invocations. */
/* This candidate uses ordinary oneMKL INT8 GEMM in row blocks up to2048,
 * time blocks up to512 and bounded4MiB INT32 scratch per callback.
 * No opaque INT8 packed panels are retained. mode16 uses aligned, positive-zero
 * padded64x64 output tiles with fixed leading dimensions. Its dense operand
 * panels use synchronized bounded caches with shared ownership during calls.
 * Padding never extends K or replaces real values. FP16 operands are rounded;
 * compute and output are FP32, not native FP16 GEMM. Reported byte counts include
 * cached buffers, excluding in-flight evicted handles, allocator/map overhead. */
IP_EXPORT int ip_abi(void);
/* Apple port bit8: Apple sysctl reports usable ARM DotProd.
 * bit1: sequential MKL compiled; bit2: CPU+OS usable AVX512 VNNI;
 * bit4: CPU+OS usable AVX/F16C conversion. */
IP_EXPORT int ip_capabilities(void);
IP_EXPORT const char *ip_last_error(void); /* thread-local, until next API call */
IP_EXPORT void *ip_create(int m,int k,const float *weights,int mode,int backend);
IP_EXPORT void ip_destroy_plan(void *plan);
/* A prepared input may be shared across ANY plans with equal K and mode,
 * regardless of M or weight values. It is immutable after construction.
 * Scale reductions are across K only, independently for each time column.
 * Caller retains original X memory until input destruction for overlap checks.
 * T==0 is supported. Nonfinite input/weights and half overflow are rejected. */
IP_EXPORT void *ip_prepare(const void *plan,const float *x,int time);
IP_EXPORT void ip_destroy_input(void *input);
/* Y is the full [M,T] destination, even when processing a row subset.
 * Concurrent calls must own disjoint output rows. Input/weights/output must
 * not overlap. No internal thread pool. Returns0 on success, -1 on failure. */
IP_EXPORT int ip_run_rows(const void *plan,const void *input,float *y,int first,int last);
IP_EXPORT size_t ip_plan_bytes(const void *plan);
IP_EXPORT size_t ip_input_bytes(const void *input);
/* Read-only test views; pointers remain valid only while their owner lives. */
IP_EXPORT int ipa_inspect_input(const void*,const int8_t**,const float**,const int32_t**);
#ifdef __cplusplus
}
#endif
#endif
