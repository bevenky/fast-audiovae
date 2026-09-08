#ifndef FAST_AUDIO_INTEL_PRECISION_H
#define FAST_AUDIO_INTEL_PRECISION_H
#include <stddef.h>
#ifdef IP_ITERATION3_NAMESPACE
#define ip_abi ip3_abi
#define ip_capabilities ip3_capabilities
#define ip_last_error ip3_last_error
#define ip_create ip3_create
#define ip_destroy_plan ip3_destroy_plan
#define ip_prepare ip3_prepare
#define ip_destroy_input ip3_destroy_input
#define ip_run_rows ip3_run_rows
#define ip_plan_bytes ip3_plan_bytes
#define ip_input_bytes ip3_input_bytes
#define ip_workspace_create ip3_workspace_create
#define ip_workspace_prepare ip3_workspace_prepare
#define ip_workspace_run_rows ip3_workspace_run_rows
#define ip_destroy_workspace ip3_destroy_workspace
#define ip_workspace_bytes ip3_workspace_bytes
#endif
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
 * backend0: scalar correctness reference; backend1: sequential oneMKL.
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
/* bit1: sequential MKL compiled; bit2: CPU+OS usable AVX512 VNNI;
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
/* Additive INT8 workspace interface. Existing entry points remain unchanged.
 * Each workspace belongs to one invocation's worker. Never destroy it during
 * a call. Concurrent prepare/run attempts on one workspace are rejected.
 * Creation allocates all activation, scale, sum and product capacity once.
 * max_rows controls the product row tile, capped at2048; time tiles are capped
 * at512. Bounds must satisfy 0<max_k<=16384, max_time>=0 and max_rows>0.
 * A workspace can switch K between completed preparations within its bounds.
 * It can share one preparation between any INT8 plans with the same K, even
 * when M/weights differ. No previous audio may be used by the next prepare. */
IP_EXPORT void *ip_workspace_create(int max_k,int max_time,int max_rows);
IP_EXPORT int ip_workspace_prepare(void *workspace,const void *plan,const float *x,int time);
/* Y is full[M,T]. With bias and skip both NULL, emit the original raw product.
 * With both non-NULL, use bias[M], skip[M,T] and emit exactly
 * skip + (float(corrected) * (weight_scale * column_scale) + bias), retaining
 * all intermediate FP32 roundings. No bias, skip, original input or original
 * weight memory may overlap Y. Nonfinite products/bias/skip/results fail.
 * A failed prepare invalidates prior prepared content. A successful prepare
 * borrows original X for overlap checks until the next prepare or destruction.
 * Prepared time may be0. Successful prepare/run_rows reuse all application
 * scratch; oneMKL may still allocate its internal packing/workspace. */
IP_EXPORT int ip_workspace_run_rows(void *workspace,const void *plan,float *y,
                                   int first,int last,const float *bias,const float *skip);
IP_EXPORT void ip_destroy_workspace(void *workspace);
IP_EXPORT size_t ip_workspace_bytes(const void *workspace);
#ifdef __cplusplus
}
#endif
#endif
