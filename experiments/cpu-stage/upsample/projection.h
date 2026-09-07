#ifndef UP_PROJECTION_H
#define UP_PROJECTION_H
#ifdef __cplusplus
extern "C" {
#endif

/* Fixed W[256,256] X[256,time] -> Y[256,time], time in 1..128.
 * Complete-K FP32 reductions, without bias or an add-zero epilogue.
 * Mode0 requires ISA512. Optional mode1 uses LIBXSMM with ISA0.
 * Plans are immutable and shareable. Output must not overlap either input.
 * No internal threads, copied weights, or mutable audio history.
 */
int up_projection_init(void);
int up_projection_has_xsmm(void);
void *up_projection_create(int time, int mode, int isa);
int up_projection_run(const void *plan, const float *w, const float *x, float *y);
void up_projection_destroy(void *plan);

#ifdef __cplusplus
}
#endif
#endif
