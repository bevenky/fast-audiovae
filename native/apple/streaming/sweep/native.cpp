// Generalized only to the ten authenticated 80 ms matrix geometries.
#include "geometry.h"
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
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
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
size_t streaming_bytes() {
  // This original getter executes RDSVL and returns twice the FP32 vector
  // length. Keep the wrapper compiled without SME so ordinary C++ loops
  // cannot silently acquire a streaming/SVE calling convention.
  return 2*kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
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
  std::vector<float> rhs,lhs,row_input,row_output;
  uint64_t runs=0,lhs_packs=0;
  std::atomic_flag busy=ATOMIC_FLAG_INIT;
  Region(size_t K,size_t N,size_t M,const float* weight,size_t count):k(K),n(N),max_m(M) {
    runtime();
    require(approved_max_m(k,n)>0 && max_m==approved_max_m(k,n),"Unapproved matrix geometry");
    require(weight && count==n*k,"Expected original contiguous FP32 weight [N,K]");
    vl=streaming_bytes();
    require(vl>=16 && vl<=256 && vl%16==0,"Unexpected SME streaming vector length");
    mr=kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    nr=kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    rhs_bytes=kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n,k);
    lhs_bytes=kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(max_m,k,mr,1,1);
    require(rhs_bytes%4==0 && lhs_bytes%4==0,"Packed size alignment failed");
    rhs.resize(rhs_bytes/4,0.0f);lhs.resize(lhs_bytes/4,0.0f);
    row_input.resize(max_m*k);row_output.resize(max_m*n);
    std::vector<float> transposed(k*n),zero_bias(n,0.0f);
    for (size_t row=0;row<n;++row) for (size_t col=0;col<k;++col) {
      const float value=weight[row*k+col];require(std::isfinite(value),"Nonfinite weight");
      transposed[col*n+row]=value;
    }
    // Only construction touches the full original weight or packs the RHS.
    // The original KleidiAI wrapper calls kai_commit_za before its assembly.
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
      1,n,k,nr,1,1,n*4,transposed.data(),zero_bias.data(),nullptr,rhs.data(),0,nullptr);
  }
  void run(size_t m,const float* x,size_t xc,float* y,size_t yc) {
    require(m>0 && m<=max_m && x && y && xc==k*m && yc==n*m,"Invalid region input/output");
    require(!overlaps(x,xc,y,yc),"Region input and output must not overlap");
    runtime(vl);
    require(!busy.test_and_set(std::memory_order_acquire),"A region may not run concurrently");
    struct Release { std::atomic_flag& flag;~Release(){flag.clear(std::memory_order_release);} } release{busy};
    // This complete path is the timing contract: original [K,M] activation,
    // layout conversion, original LHS pack, SME matrix product, then [N,M].
    // All scratch is preallocated. No BLAS, OpenMP or worker pool is used.
    for (size_t t=0;t<m;++t) for (size_t c=0;c<k;++c) row_input[t*k+c]=x[c*m+t];
    kai_run_lhs_pack_f32p2vlx1_f32_sme(m,k,mr,1,1,0,row_input.data(),k*4,lhs.data());
    ++lhs_packs;
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
      m,n,k,lhs.data(),rhs.data(),row_output.data(),n*4,4,
      -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity());
    for (size_t c=0;c<n;++c) for (size_t t=0;t<m;++t) y[c*m+t]=row_output[t*n+c];
    ++runs;
  }
};
}
extern "C" {
// The existing bundle loader authenticates and registers every dependency.
// This core exports no ORT operators; the separate bridge owns the domain.
EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions*,const OrtApiBase*) { return nullptr; }
EXPORT int av_sweep_supported() { return supported()?1:0; }
EXPORT const char* av_sweep_error() { return error_text; }
EXPORT void* av_sweep_create(size_t k,size_t n,size_t max_m,const float* weight,size_t count) {
  error_text[0]=0;try { return new Region(k,n,max_m,weight,count); } catch (...) { failed();return nullptr; }
}
EXPORT void av_sweep_destroy(void* handle) { delete static_cast<Region*>(handle); }
EXPORT int av_sweep_run(void* handle,size_t m,const float* x,size_t xc,float* y,size_t yc) {
  error_text[0]=0;try {require(handle,"Missing region");static_cast<Region*>(handle)->run(m,x,xc,y,yc);return 0;}
  catch (...) {failed();return -1;}
}
EXPORT uint64_t av_sweep_stat(void* handle,int field) {
  if (!handle) return 0;
  auto& p=*static_cast<Region*>(handle);
  switch(field) {
    case 0:return p.vl;case 1:return p.mr;case 2:return p.nr;case 3:return p.rhs_bytes;
    case 4:return p.lhs_bytes;case 5:return (p.row_input.size()+p.row_output.size())*4+p.lhs_bytes;
    case 6:return p.runs;case 7:return p.lhs_packs;case 8:return 1;default:return 0;
  }
}
}
