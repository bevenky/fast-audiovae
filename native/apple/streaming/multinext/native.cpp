// Six remaining selected SME nodes: original multi-tile MOPA API, with mechanical 1VL packing.

#include "onnxruntime_c_api.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <sys/sysctl.h>
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__FAST_MATH__)
#error Apple ARM64 without fast-math is required
#endif
#define EXPORT __attribute__((visibility("default")))
namespace {
thread_local char error_text[256]={};
void require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
void failed() noexcept {
  try { throw; }
  catch (const std::exception& e) { std::snprintf(error_text,sizeof(error_text),"%s",e.what()); }
  catch (...) { std::snprintf(error_text,sizeof(error_text),"Unknown SME2 failure"); }
}
bool flag(const char* name) {
  int value=0;size_t bytes=sizeof(value);
  return sysctlbyname(name,&value,&bytes,nullptr,0)==0 && bytes==sizeof(value) && value==1;
}
bool supported() {
  static const bool result=flag("hw.optional.arm.FEAT_SME") && flag("hw.optional.arm.FEAT_SME2");
  return result;
}
kai_matmul_uker_api get_api() {
  return kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
}
size_t streaming_bytes() {
  const kai_matmul_uker_config cfg{};
  return get_api().get_step(&cfg).m*sizeof(float);
}
void fpcr_check() {
  uint64_t fpcr;__asm__ volatile("mrs %0, fpcr":"=r"(fpcr));
  require((fpcr&((uint64_t(3)<<22)|(uint64_t(1)<<24)|3))==0,
          "Requires nearest FP32 rounding without FZ/FIZ/AH changes");
}
void runtime(size_t expected_vl=0) {
  require(supported(),"Apple SME and SME2 are required");
  fpcr_check();
  if (expected_vl) require(streaming_bytes()==expected_vl,"SME vector length changed");
}
bool overlaps(const float* a,size_t ac,const float* b,size_t bc) {
  auto aa=reinterpret_cast<uintptr_t>(a),bb=reinterpret_cast<uintptr_t>(b);
  require(aa<=UINTPTR_MAX-ac*4 && bb<=UINTPTR_MAX-bc*4,"Pointer range overflow");
  return aa<bb+bc*4 && bb<aa+ac*4;
}
struct Region {
  size_t k,n,max_m,vl,mr,nr,rhs_bytes,lhs_bytes;
  kai_matmul_uker_api api;
  kai_matmul_uker_config config{};
  kai_matmul_uker_lhs_stride_args lhs_stride;
  kai_matmul_uker_rhs_stride_args rhs_stride;
  kai_matmul_uker_dst_stride_args dst_stride;
  std::vector<float> rhs,lhs,row_output;
  uint64_t runs=0,api_calls=0;
  std::atomic_flag busy=ATOMIC_FLAG_INIT;
  Region(size_t K,size_t N,size_t M,const float* weight,size_t count):k(K),n(N),max_m(M) {
    runtime();
    require(((k==64 && n==2048 && max_m==2) || (k==1024 && (n==1024 || n==3072) && max_m==16)),"Only the three explicitly approved remaining SME geometries are supported");
    require(weight && count==n*k,"Expected original contiguous FP32 weight [N,K]");
    api=get_api();vl=streaming_bytes();
    require(vl>=16 && vl<=256 && vl%16==0,"Unexpected SME streaming vector length");
    const auto step=api.get_step(&config);mr=step.m;nr=step.n;
    require(mr==vl/4 && nr==vl/4 && max_m<=mr && n%(4*nr)==0,
            "Expected 1VL packing and complete 1VLx4VL bottom-edge dispatch");
    const kai_matmul_uker_lhs_dim_args ls{max_m,k};
    const kai_matmul_uker_rhs_dim_args rs{n,k};
    const kai_matmul_uker_dst_dim_args ds{max_m,n};
    lhs_stride=api.get_lhs_stride(&config,&ls);
    rhs_stride=api.get_rhs_stride(&config,&rs);
    dst_stride=api.get_dst_stride(&config,&ds);
    require(lhs_stride.m==mr*k*4 && rhs_stride.n==nr*(k+1)*4 && dst_stride.m==n*4,
            "Original API packing strides changed");
    lhs_bytes=lhs_stride.m;rhs_bytes=(n/nr)*rhs_stride.n;
    lhs.resize(lhs_bytes/4,0.0f);rhs.resize(rhs_bytes/4,0.0f);
    row_output.resize(api.get_dst_size(&config,&ds,&dst_stride)/4);
    // Exact layout required by the original kernel: for each 1VL N panel,
    // zero bias[nr], then k vectors of nr FP32 weights. Its assembly consumes
    // four such panels concurrently. No conversion or arithmetic on weights.
    for(size_t column=0;column<n;column+=nr) {
      const kai_matmul_uker_rhs_dim_args index{column,0};
      const size_t offset=api.get_rhs_offset(&config,&index,&rhs_stride);
      require(offset%4==0 && offset+rhs_stride.n<=rhs_bytes,"RHS offset out of bounds");
      float* packed=rhs.data()+offset/4+nr;
      for(size_t depth=0;depth<k;++depth) for(size_t lane=0;lane<nr;++lane) {
        float value=weight[(column+lane)*k+depth];
        require(std::isfinite(value),"Nonfinite original weight");
        packed[depth*nr+lane]=value;
      }
    }
  }
  void run(size_t m,const float* x,size_t xc,float* y,size_t yc) {
    require(m>0 && m<=max_m && x && y && xc==k*m && yc==n*m,"Invalid region input/output");
    require(!overlaps(x,xc,y,yc),"Region input and output must not overlap");
    runtime(vl);
    require(!busy.test_and_set(std::memory_order_acquire),"A region may not run concurrently");
    struct Release {std::atomic_flag& flag;~Release(){flag.clear(std::memory_order_release);}} release{busy};
    // Every timed call includes conversion of the original BCT input to the
    // K-major 1VL LHS buffer, including zero padding of unused rows.
    std::fill(lhs.begin(),lhs.end(),0.0f);
    for(size_t depth=0;depth<k;++depth) for(size_t row=0;row<m;++row)
      lhs[depth*mr+row]=x[depth*m+row];
    const float lower=-std::numeric_limits<float>::infinity();
    const float upper=std::numeric_limits<float>::infinity();
    kai_matmul_uker_args args{};
    args.flags=KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape={m,n,k};
    args.operand.lhs={lhs.data(),lhs_stride};
    args.operand.rhs={rhs.data(),rhs_stride};
    args.operand.dst={row_output.data(),dst_stride};
    args.activation.clamp={&lower,&upper};
    api.run(&config,&args);++api_calls;
    for(size_t column=0;column<n;++column) for(size_t row=0;row<m;++row)
      y[column*m+row]=row_output[row*n+column];
    ++runs;
  }
};
}
extern "C" {
// The existing bundle loader authenticates and registers every dependency.
// This core exports no ORT operators; the separate bridge owns the domain.
EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions*,const OrtApiBase*) { return nullptr; }
EXPORT int av_multinext_supported() { return supported()?1:0; }
EXPORT const char* av_multinext_error() { return error_text; }
EXPORT void* av_multinext_create(size_t k,size_t n,size_t max_m,const float* weight,size_t count) {
  error_text[0]=0;try { return new Region(k,n,max_m,weight,count); } catch (...) { failed();return nullptr; }
}
EXPORT void av_multinext_destroy(void* handle) { delete static_cast<Region*>(handle); }
EXPORT int av_multinext_run(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc) {
  error_text[0]=0;try {require(handle,"Missing region");static_cast<Region*>(handle)->run(m,x,xc,y,yc);return 0;}
  catch (...) {failed();return -1;}
}
EXPORT uint64_t av_multinext_stat(void* handle,int field) {
  if (!handle) return 0;
  auto& p=*static_cast<Region*>(handle);
  switch(field) {
    case 0:return p.vl;case 1:return p.mr;case 2:return p.nr;case 3:return p.rhs_bytes;
    case 4:return p.lhs_bytes;case 5:return p.row_output.size()*4;
    case 6:return p.runs;case 7:return p.api_calls;case 8:return 1;default:return 0;
  }
}
}
