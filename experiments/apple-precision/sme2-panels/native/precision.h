#ifndef FAST_AUDIO_INTEL_PRECISION_H
#define FAST_AUDIO_INTEL_PRECISION_H
#define ip_abi ipc_abi
#define ip_capabilities ipc_capabilities
#define ip_last_error ipc_last_error
#define ip_create ipc_create
#define ip_destroy_plan ipc_destroy_plan
#define ip_prepare ipc_prepare
#define ip_destroy_input ipc_destroy_input
#define ip_run_rows ipc_run_rows
#define ip_plan_bytes ipc_plan_bytes
#define ip_input_bytes ipc_input_bytes
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
 * Apple port: backend0 scalar reference; backend2 guarded ARM SDOT and
 * backend3 guarded SME2, both mode8 only.
 * A plan is immutable and safe for concurrent independent invocations. */
/* SME weights are packed once. SME input preparation stores only packed RHS;
 * test inspection materializes separate views lazily. The panel workspace API
 * below never retains unpacked activation copies. All reductions retain K. */
IP_EXPORT int ip_abi(void);
/* Apple port bit8: Apple sysctl reports usable ARM DotProd, bit16: SME2.
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
IP_EXPORT int ipc_inspect_input(const void*,const int8_t**,const float**,const int32_t**);
/* Two-phase preparation. Each job owns disjoint time panels. Run rows only
 * after all jobs have completed successfully and the caller has joined. */
IP_EXPORT void* ipc_allocate_input(const void* plan,const float* x,int time,int jobs);
IP_EXPORT int ipc_prepare_jobs(const void* input);
IP_EXPORT int ipc_prepare_job(void* input,int job);
IP_EXPORT int ipc_row_step(const void* plan);
IP_EXPORT int ipc_check_packed_input(const void* input);
/* Invocation-owned packed-only SME workspace, with capacity max_time.
 * Compatible plans have identical K/mode8/SME layout; M may differ.
 * prepare gathers X[k*source_time+first+j], j in [0,count). It replaces the
 * previous panel and rejects concurrent preparation or active readers.
 * run writes raw product Y[row*output_time+output_offset+j], without bias or
 * residual. Complete K, original scale multiplication and FP32 conversion.
 * first_row is MR aligned; last_row may be a tail. Concurrent read-only runs
 * may share a prepared workspace but callers must own disjoint destinations.
 * Source memory stays alive until the next prepare or workspace destruction.
 * No internal threads. Failed preparation invalidates the previous panel. */
IP_EXPORT void* ipc_create_workspace(const void* plan,int max_time);
IP_EXPORT void ipc_destroy_workspace(void* workspace);
IP_EXPORT int ipc_prepare_panel(void* workspace,const float* x,int source_time,int first,int count);
IP_EXPORT int ipc_run_panel(const void* plan,const void* workspace,float* y,int output_time,int output_offset,int first_row,int last_row);
IP_EXPORT size_t ipc_workspace_bytes(const void* workspace);
/* Test-only copies of a prepared panel. All destinations have count columns;
 * no inspection caches or unpacked activations are retained in the workspace. */
IP_EXPORT int ipc_copy_panel(const void* workspace,int8_t* q,float* scales,int32_t* sums,int count);
#ifdef __cplusplus
}
#endif
#endif
