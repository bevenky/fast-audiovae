#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#define API extern "C" __attribute__((visibility("default")))
#if defined(__FAST_MATH__)
#error No fast math
#endif
static bool environment_ok(){uint64_t f;asm volatile("mrs %0, fpcr":"=r"(f));return !(f&((3ULL<<22)|(1ULL<<24)|3));}
static float scale_of(float m){if(m==0)return 1;float s=m/127.f;return s>0?s:m;}
static int quant(float v,float s){float q=std::max(-127.f,std::min(127.f,v/s));int a=(int)std::floor(q);float r=q-a;return a+(r>.5f||(r==.5f&&(a&1)));}
API int av8_quant_scalar(size_t k,size_t t,const float* x,int8_t* q,float* s){
 if(!environment_ok()||!k||k>16384||(t!=1&&t!=2)||!x||!q||!s)return -1;
 for(size_t i=0;i<k*t;i++)if(!std::isfinite(x[i]))return -2;
 for(size_t a=0;a<t;a++){float m=0;for(size_t i=0;i<k;i++)m=std::max(m,std::fabs(x[i*t+a]));s[a]=scale_of(m);for(size_t i=0;i<k;i++)q[a*k+i]=quant(x[i*t+a],s[a]);}return 0;
}
API int av8_quant(size_t k,size_t t,const float* x,int8_t* q,float* s){
 if(!environment_ok()||!k||k>16384||(t!=1&&t!=2)||!x||!q||!s)return -1;
 // Deinterleave across channels for a two-frame streaming packet.
 float32x4_t m[2]={vdupq_n_f32(0),vdupq_n_f32(0)};uint32x4_t finite=vdupq_n_u32(~0U);size_t i=0;
 for(;i+4<=k;i+=4){float32x4_t v[2];if(t==2){auto z=vld2q_f32(x+2*i);v[0]=z.val[0];v[1]=z.val[1];}else v[0]=vld1q_f32(x+i);
  for(size_t a=0;a<t;a++){auto ab=vabsq_f32(v[a]);finite=vandq_u32(finite,vcltq_f32(ab,vdupq_n_f32(INFINITY)));m[a]=vmaxq_f32(m[a],ab);}}
 if(vminvq_u32(finite)!=~0U)return -2;
 float mx[2]={vmaxvq_f32(m[0]),vmaxvq_f32(m[1])};for(;i<k;i++)for(size_t a=0;a<t;a++){if(!std::isfinite(x[i*t+a]))return -2;mx[a]=std::max(mx[a],std::fabs(x[i*t+a]));}
 for(size_t a=0;a<t;a++)s[a]=scale_of(mx[a]);i=0;
 for(;i+4<=k;i+=4){float32x4_t v[2];if(t==2){auto z=vld2q_f32(x+2*i);v[0]=z.val[0];v[1]=z.val[1];}else v[0]=vld1q_f32(x+i);
  for(size_t a=0;a<t;a++){auto z=vdivq_f32(v[a],vdupq_n_f32(s[a]));z=vmaxq_f32(vdupq_n_f32(-127),vminq_f32(vdupq_n_f32(127),z));int32_t b[4];vst1q_s32(b,vcvtnq_s32_f32(z));for(size_t j=0;j<4;j++)q[a*k+i+j]=(int8_t)b[j];}}
 for(;i<k;i++)for(size_t a=0;a<t;a++)q[a*k+i]=quant(x[i*t+a],s[a]);return 0;
}
