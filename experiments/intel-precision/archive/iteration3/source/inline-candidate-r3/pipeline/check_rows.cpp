/* CPU-only exact parity/safety check. No timing or full model inference.
 * Link accepted native library and exactly pinned SLEEF archive; use the same
 * no-fast-math, no-FP-contraction flags as candidate libraries. */
#include "native_kernels.h"
#include "row_nonlinear.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr int Guard=16;
constexpr float Sentinel=1234567.0f;
uint32_t state=921735;
float random_value(){
    state^=state<<13;state^=state>>17;state^=state<<5;
    return (static_cast<int32_t>(state%200001)-100000)*0.00005f;
}
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
bool identical(const float* a,const float* b,int n){
    return !n||std::memcmp(a,b,static_cast<size_t>(n)*sizeof(float))==0;
}
void guard(const std::vector<float>& value,int offset,int n){
    for(int i=0;i<offset;++i)require(value[i]==Sentinel,"Prefix guard overwritten");
    for(size_t i=static_cast<size_t>(offset+n);i<value.size();++i)
        require(value[i]==Sentinel,"Suffix guard overwritten");
}
void reference(const float* x,const float* weights,float bias,float ap,float rp,float aq,float rq,
               float* history,float* y,int n,int d,int backend){
    if(!n)return;
    const int halo=6*d;
    alignas(64) float transformed[54+256],pre[256],depthwise[256];
    require(ncc_snake_f32(x,&ap,&rp,pre,1,1,n,backend,1)==0,"Reference pre-Snake rejected");
    std::memcpy(transformed,history,static_cast<size_t>(halo)*sizeof(float));
    std::memcpy(transformed+halo,pre,static_cast<size_t>(n)*sizeof(float));
    sp_dw(transformed,weights,bias,depthwise,n,d,backend);
    std::memcpy(history,transformed+n,static_cast<size_t>(halo)*sizeof(float));
    require(ncc_snake_f32(depthwise,&aq,&rq,y,1,1,n,backend,1)==0,"Reference post-Snake rejected");
}
void fill(std::vector<float>& values,int offset,int n,int pattern){
    const float special[]={0.f,-0.f,0x1p-125f,-0x1p-125f,0x1p25f,-0x1p25f,100000.f,-100000.f};
    for(int i=0;i<n;++i)values[offset+i]=pattern?special[i%8]:random_value();
}
}
int main(){
    try{
        require(ncc_abi_version()==1,"Native ABI mismatch");
        int rows=0,snakes=0,sequences=0,backends=0;
        for(int backend:{4,5}){
            require(ncc_backend_available(backend)&&ncc_vector_sine_available(backend),
                    "This Intel check requires accurate AVX2 and AVX512 backends");
            ++backends;
            // Null pointers are legal only for this already-validated helper's zero width.
            row_snake_dw_snake(nullptr,nullptr,0.f,1.f,1.f,1.f,1.f,nullptr,nullptr,0,9,backend);
            for(int d:{1,3,9})for(int n=0;n<=256;++n)for(int pattern:{0,1}){
                const int offset=Guard+n%4,halo=6*d;
                std::vector<float> x(offset+n+Guard,Sentinel),actual(x.size(),Sentinel),expected(x.size(),Sentinel);
                fill(x,offset,n,pattern);const auto original=x;
                std::array<float,7> weights{};for(float& v:weights)v=random_value()*.025f;
                std::vector<float> h(Guard+halo+Guard,Sentinel),href=h;
                for(int i=0;i<halo;++i)h[Guard+i]=href[Guard+i]=random_value();
                const float ap=1.37f,rp=.43f,aq=.67f,rq=.91f,bias=-.13f;
                row_snake_dw_snake(x.data()+offset,weights.data(),bias,ap,rp,aq,rq,
                                   h.data()+Guard,actual.data()+offset,n,d,backend);
                reference(x.data()+offset,weights.data(),bias,ap,rp,aq,rq,
                          href.data()+Guard,expected.data()+offset,n,d,backend);
                require(identical(actual.data(),expected.data(),static_cast<int>(actual.size())),"Row output differs from accepted operations");
                require(identical(h.data(),href.data(),static_cast<int>(h.size())),"Chronological history differs");
                require(identical(x.data(),original.data(),static_cast<int>(x.size())),"Input mutated");
                guard(actual,offset,n);guard(h,Guard,halo);++rows;
                // Separately compare the vector-register Snake against native checked Snake.
                std::fill(actual.begin(),actual.end(),Sentinel);std::fill(expected.begin(),expected.end(),Sentinel);
                row_snake(x.data()+offset,actual.data()+offset,ap,rp,n,backend);
                require(ncc_snake_f32(x.data()+offset,&ap,&rp,expected.data()+offset,1,1,n,backend,1)==0,"Native Snake rejected");
                require(identical(actual.data(),expected.data(),static_cast<int>(actual.size())),"Vector Snake tail/order differs");
                ++snakes;
            }
            for(int d:{1,3,9})for(int q:{64,128,256}){
                const int halo=6*d;std::array<float,54> h{},href{};
                const std::array<float,7> w={.07f,-.06f,.05f,-.04f,.03f,-.02f,.01f};
                // Widths below halo explicitly exercise carry over very short repeated calls.
                for(int n:{q,1,2,3,5,17,53,54,55,q-1,q}){
                    std::vector<float> x(n),actual(n),expected(n);for(float& v:x)v=random_value();
                    row_snake_dw_snake(x.data(),w.data(),.01f,.73f,.91f,1.31f,.4f,h.data(),actual.data(),n,d,backend);
                    reference(x.data(),w.data(),.01f,.73f,.91f,1.31f,.4f,href.data(),expected.data(),n,d,backend);
                    require(identical(actual.data(),expected.data(),n)&&identical(h.data(),href.data(),halo),
                            "Repeated row history differs");
                    ++sequences;
                }
            }
        }
        std::printf("{\"status\":\"complete\",\"row_cases\":%d,\"snake_cases\":%d,\"sequence_tiles\":%d,"
                    "\"backends\":%d,\"bitwise_required\":true,\"gpu_used\":false,\"timing_benchmark\":false}\n",
                    rows,snakes,sequences,backends);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Row check failed: %s\n",e.what());return 1;}
}
