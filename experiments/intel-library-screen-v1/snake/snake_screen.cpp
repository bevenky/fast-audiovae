// Isolated oneMKL HA Snake screen. No tensor values leave this process.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "mkl_vml.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#if defined(__FAST_MATH__) || !defined(__linux__) || !defined(__x86_64__)
#error Linux x86 CPU and ordinary floating point required
#endif
#define API __attribute__((visibility("default")))
#define AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma")))
extern "C" API int32_t ncc_snake_f32(const float*,const float*,const float*,float*,int64_t,int64_t,int64_t,int32_t,int32_t);
namespace {
using Snake=decltype(&ncc_snake_f32);
using Sine=decltype(&vmsSin);
static_assert(sizeof(MKL_INT)==4,"LP64 oneMKL required");
Snake original=nullptr;Sine sine=nullptr;void* native_handle=nullptr;void* mkl_handle=nullptr;
std::once_flag once;bool ready=false;thread_local char error[256]{};
constexpr uint64_t MODE_SHIFT=60,ACTIVE_MASK=(uint64_t(1)<<MODE_SHIFT)-1;
std::atomic<uint64_t> gate{0},intercepted{0},fallbacks{0};std::mutex mutex;
struct Sample {int64_t b,c,t;std::vector<float>x,a,r;uint64_t hits=1;};
std::vector<Sample> samples;size_t captured_bytes=0;
constexpr size_t MAX_SAMPLES=48,MAX_BYTES=32u<<20,MAX_ELEMENTS=1u<<20;
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int fail()noexcept{try{throw;}catch(const std::exception& e){std::snprintf(error,sizeof(error),"%s",e.what());}
 catch(...){std::snprintf(error,sizeof(error),"Unknown Snake screen error");}return -1;}
bool same_file(void* symbol,const char* path){Dl_info info{};struct stat x{},y{};
 return dladdr(symbol,&info)&&info.dli_fname&&stat(info.dli_fname,&x)==0&&stat(path,&y)==0&&x.st_dev==y.st_dev&&x.st_ino==y.st_ino;}
void* resolve(const char* env,const char* symbol,void*& handle,void* self){
 const char* path=std::getenv(env);require(path&&path[0]=='/',"Absolute pinned resolver path required");
 handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);require(handle,"Cannot load pinned CPU library");
 void* value=dlsym(handle,symbol);require(value&&value!=self&&same_file(value,path),"CPU symbol owner mismatch");return value;
}
bool initialize()noexcept{try{std::call_once(once,[]{
 original=reinterpret_cast<Snake>(resolve("SK_NATIVE_LIBRARY","ncc_snake_f32",native_handle,reinterpret_cast<void*>(&ncc_snake_f32)));
 sine=reinterpret_cast<Sine>(resolve("SK_MKL_LIBRARY","vmsSin",mkl_handle,nullptr));
 auto tile=reinterpret_cast<uint32_t(*)()>(dlsym(native_handle,"ncc_compiled_tile"));
 auto math=reinterpret_cast<uint32_t(*)(int32_t)>(dlsym(native_handle,"ncc_streaming_math_version"));
 auto available=reinterpret_cast<int32_t(*)(int32_t)>(dlsym(native_handle,"ncc_backend_available"));
 require(tile&&math&&available&&tile()==256&&math(5)==1&&available(5),"Canonical AVX512 streaming Snake required");
 ready=true;});return ready;}catch(...){fail();return false;}}
uint64_t now(){timespec t{};require(clock_gettime(CLOCK_MONOTONIC,&t)==0,"Clock failed");return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec;}
bool supported(int64_t b,int64_t c,int64_t t,int backend,int threads){
 return backend==5&&threads==1&&b>0&&c>0&&t>0&&b<=1024&&c<=2048&&t<=65536
        &&uint64_t(b)*c*t<=MAX_ELEMENTS&&(_mm_getcsr()&((1u<<15)|(1u<<6)))==0;
}
AVX512 void scale(const float*x,float*y,float a,int n){for(int i=0;i<n;++i)y[i]=a*x[i];}
AVX512 void finish(const float*x,const float*s,float*y,float r,int n){for(int i=0;i<n;++i){
 const float square=s[i]*s[i];const float correction=r*square;y[i]=x[i]+correction;}}
int candidate(const float*x,const float*a,const float*r,float*y,int64_t b,int64_t c,int64_t t,int mode){
 const auto vml_mode=VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE;
 if(mode==1){alignas(64) float temp[256];
  for(int64_t row=0;row<b*c;++row)for(int64_t begin=0;begin<t;begin+=256){
   const int n=int(std::min<int64_t>(256,t-begin));const auto offset=row*t+begin;
   scale(x+offset,temp,a[row%c],n);sine(n,temp,temp,vml_mode);finish(x+offset,temp,y+offset,r[row%c],n);
  }
 }else{thread_local std::vector<float> temp;temp.resize(size_t(b*c*t));
  for(int64_t row=0;row<b*c;++row)scale(x+row*t,temp.data()+row*t,a[row%c],int(t));
  sine(MKL_INT(b*c*t),temp.data(),temp.data(),vml_mode);
  for(int64_t row=0;row<b*c;++row)finish(x+row*t,temp.data()+row*t,y+row*t,r[row%c],int(t));
 }return 0;
}
void capture(const float*x,const float*a,const float*r,int64_t b,int64_t c,int64_t t){
 std::lock_guard<std::mutex> lock(mutex);
 for(auto& s:samples)if(s.b==b&&s.c==c&&s.t==t&&std::memcmp(s.a.data(),a,size_t(c)*4)==0&&std::memcmp(s.r.data(),r,size_t(c)*4)==0){++s.hits;return;}
 const size_t bytes=size_t(b*c*t+2*c)*4;if(samples.size()>=MAX_SAMPLES||captured_bytes+bytes>MAX_BYTES)return;
 samples.push_back({b,c,t,{x,x+b*c*t},{a,a+c},{r,r+c}});captured_bytes+=bytes;
}
struct Enter {uint64_t mode;Enter(){uint64_t value=gate.fetch_add(1,std::memory_order_acq_rel);mode=value>>MODE_SHIFT;}
 ~Enter(){gate.fetch_sub(1,std::memory_order_release);}};
void quiescent(){require(gate.load(std::memory_order_acquire)==0,"Disable wrapper mode at a quiescent boundary first");}
const Sample& sample(size_t i){quiescent();require(i<samples.size(),"Capture index out of range");return samples[i];}
}
extern "C" {
API const char* sk_error(){return error;}
API int sk_initialize(){return initialize()?0:-1;}
// 0: original, 1: VML HA in original 256 tiles, 2: VML HA one whole call,
// 3: original plus bounded in-memory capture. Calls outside supported geometry
// always retain the original implementation and are explicitly counted.
API int sk_set_mode(int mode){try{require(mode>=0&&mode<=3,"Invalid mode");if(!initialize())return -1;
 uint64_t expected=gate.load(std::memory_order_acquire);require((expected&ACTIVE_MASK)==0,"Active calls");
 require(gate.compare_exchange_strong(expected,uint64_t(mode)<<MODE_SHIFT),"Concurrent mode change");return 0;}catch(...){return fail();}}
API int sk_clear(){try{std::lock_guard<std::mutex>lock(mutex);quiescent();samples.clear();captured_bytes=0;intercepted=0;fallbacks=0;return 0;}catch(...){return fail();}}
API int sk_summary(uint64_t*out,size_t n){try{require(out&&n>=5,"Summary buffer too small");quiescent();
 out[0]=samples.size();out[1]=captured_bytes;out[2]=intercepted;out[3]=fallbacks;out[4]=ready;return 0;}catch(...){return fail();}}
API int sk_info(size_t i,int64_t*out,size_t n){try{require(out&&n>=4,"Info buffer too small");const auto&s=sample(i);out[0]=s.b;out[1]=s.c;out[2]=s.t;out[3]=s.hits;return 0;}catch(...){return fail();}}
API int sk_compare(size_t i,int mode,double*out,size_t n){try{require(out&&n>=7&&(mode==1||mode==2),"Bad compare arguments");const auto&s=sample(i);
 std::vector<float>ref(s.x.size()),y(s.x.size());require(original(s.x.data(),s.a.data(),s.r.data(),ref.data(),s.b,s.c,s.t,5,1)==0,"Original Snake failed");
 require(candidate(s.x.data(),s.a.data(),s.r.data(),y.data(),s.b,s.c,s.t,mode)==0,"Candidate failed");
 double max_abs=0,max_rel=0,squares=0,unequal=0,violations=0,nonfinite=0;
 for(size_t j=0;j<y.size();++j){uint32_t a,b;std::memcpy(&a,&ref[j],4);std::memcpy(&b,&y[j],4);unequal+=a!=b;
  if(!std::isfinite(ref[j])||!std::isfinite(y[j])){++nonfinite;continue;}
  const double d=std::abs(double(y[j])-ref[j]);max_abs=std::max(max_abs,d);max_rel=std::max(max_rel,d/std::max(1e-12,std::abs(double(ref[j]))));squares+=d*d;
  violations+=d>1e-5+1e-4*std::abs(double(ref[j]));
 }out[0]=y.size();out[1]=unequal;out[2]=max_abs;out[3]=max_rel;out[4]=std::sqrt(squares/y.size());out[5]=violations;out[6]=nonfinite;return 0;
 }catch(...){return fail();}}
API int sk_measure(size_t i,int mode,int repetitions,uint64_t*out,size_t n){try{
 require(out&&repetitions>=1&&repetitions<=9&&n>=size_t(2*repetitions)&&(mode==1||mode==2),"Bad timing arguments");const auto&s=sample(i);
 std::vector<float>ref(s.x.size()),y(s.x.size());
 auto original_call=[&](){require(original(s.x.data(),s.a.data(),s.r.data(),ref.data(),s.b,s.c,s.t,5,1)==0,"Original Snake failed");};
 auto candidate_call=[&](){require(candidate(s.x.data(),s.a.data(),s.r.data(),y.data(),s.b,s.c,s.t,mode)==0,"Candidate failed");};
 original_call();candidate_call();
 for(int r=0;r<repetitions;++r){auto measure=[&](bool changed){auto start=now();if(changed)candidate_call();else original_call();out[2*r+int(changed)]=now()-start;};
  if(r%2){measure(true);measure(false);}else{measure(false);measure(true);}}
 return 0;}catch(...){return fail();}}
API int32_t ncc_snake_f32(const float*x,const float*a,const float*r,float*y,int64_t b,int64_t c,int64_t t,int32_t backend,int32_t threads){
 if(!initialize()){const char*msg="Snake screen CPU symbol resolution failed\n";(void)!write(2,msg,std::strlen(msg));_exit(127);}
 Enter active;intercepted.fetch_add(1,std::memory_order_relaxed);
 const bool eligible=supported(b,c,t,backend,threads);
 if(!eligible){fallbacks.fetch_add(1,std::memory_order_relaxed);return original(x,a,r,y,b,c,t,backend,threads);}
 try{if(active.mode==1||active.mode==2)return candidate(x,a,r,y,b,c,t,int(active.mode));
  const int status=original(x,a,r,y,b,c,t,backend,threads);if(status==0&&active.mode==3)capture(x,a,r,b,c,t);return status;
 }catch(...){fail();return 1;}
}
}
