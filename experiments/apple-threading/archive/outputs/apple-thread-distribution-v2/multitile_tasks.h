#ifndef APPLE_MULTITILE_TASKS_EXPERIMENT_H
#define APPLE_MULTITILE_TASKS_EXPERIMENT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void* av_multitile_create(size_t k,size_t n,size_t max_m,const float* weight,size_t count);
void av_multitile_destroy(void* handle);
const char* av_multitile_error(void); /* Thread-local; copy on the failing worker. */
int av_multitile_run(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc);
/* K2048/N8192/M1 or M2, original BCT X and Y. Begin owns the region guard and
 * packs LHS once. Grain is channels, a multiple of stat(11)=4*nr, at most N.
 * The caller owns the handle and Y lifetime until finish, and must not destroy
 * the region, begin another call, or call finish while callbacks are pending. */
int av_multitile_begin(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc,
                       size_t channels_per_task);
size_t av_multitile_task_count(void* handle);
/* Dispatch each index [0,task_count) exactly once in the shared ORT pool.
 * Duplicate/out-of-range tasks fail. Outputs stay in region-private row scratch. */
int av_multitile_task(void* handle,size_t index);
/* ONLY after join: commit1 requires every task successful, transposes to Y,
 * then releases the guard. Commit0 skips publication and releases the guard.
 * A failed commit1 also releases the guard after validating no task is running.
 * A failed begin owns no lifecycle and must not be followed by finish. */
int av_multitile_finish(void* handle,int commit);
/* Original fields0..8 unchanged;9 successful tiled calls,10 completed worker
 * checks,11 required channel alignment. Read statistics after join/finish. */
uint64_t av_multitile_stat(void* handle,int field);
#ifdef __cplusplus
}
#endif
#endif
