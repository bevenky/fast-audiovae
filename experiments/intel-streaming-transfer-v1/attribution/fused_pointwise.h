#ifndef FX_POINTWISE_H
#define FX_POINTWISE_H
#ifdef __cplusplus
extern "C" {
#endif
int fx_init(void);
int fx_has_xsmm(void);
void *fx_create(int c,int k,int t,int tt,int tc,int mode,int isa);
void *fx_create_ordered(int c,int k,int t,int tt,int tc,int mode,int isa,int skip_first);
int fx_blocks(const void *plan);
int fx_isa(const void *plan);
int fx_run_range(const void *plan,const float *w,const float *x,const float *bias,
                 const float *skip,float *y,int first,int last);
void fx_destroy(void *plan);
#ifdef __cplusplus
}
#endif
#endif
