#ifndef APPLE_SWEEP_TASKS_EXPERIMENT_H
#define APPLE_SWEEP_TASKS_EXPERIMENT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int av_sweep_supported(void);
void* av_sweep_create(size_t k,size_t n,size_t max_m,const float* weight,size_t count);
void av_sweep_destroy(void* handle);
const char* av_sweep_error(void); /* Thread-local; copy on worker if detail is needed. */
int av_sweep_run(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc);
/* New lifecycle only: K1024,N3072,max_m16; M1..16; BCT X[K,M],Y[N,M].
 * Begin owns the existing busy guard, converts and packs LHS once. Grain is
 * output channels, a positive multiple of stat(11), at most3072. No per-call
 * weight packing or heap allocation. Caller owns handle/Y through finish. */
int av_sweep_begin(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc,
                   size_t channels_per_task);
/* Positive after successful begin; zero for missing or inactive handles. */
size_t av_sweep_task_count(void* handle);
/* Run each index exactly once in the shared ORT pool. Packed operands read-only;
 * disjoint row-output columns; each task checks runtime capabilities,FPCR,VL.
 * Return0/-1; no exception crosses this API. Failures must be collected by caller. */
int av_sweep_task(void* handle,size_t index);
/* ONLY after all callbacks join. Commit1 validates coverage then transposes;
 * commit0 skips Y publication. Both release guard. Failed commit1 also releases
 * after checking no task is still running. Failed begin requires no finish.
 * Never destroy/rebegin/finish a handle while callbacks can still use it. */
int av_sweep_finish(void* handle,int commit);
/* Original0..8 unchanged;9=tiled successes,10=completed tile kernel calls
 * (each includes worker runtime check),11=N-step alignment or0 unsupported.
 * Field7 counts original LHS packs, once per successful begin/serial run.
 * Read statistics after join/finish, not during a live lifecycle. */
uint64_t av_sweep_stat(void* handle,int field);
#ifdef __cplusplus
}
#endif
#endif
