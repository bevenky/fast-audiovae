// CPU-only bit parity, with no models or timings. Link the pinned old archive.
#include "sleef_inline_avx512.h"
#include <array>
#include <cmath>
#include <cpuid.h>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>

#define CHECK_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma")))
extern "C" CHECK_AVX512 __m512 Sleef_sinf16_u10avx512f(__m512);
struct RestoreCSR { unsigned old=_mm_getcsr(); ~RestoreCSR(){_mm_setcsr(old);} };
static bool available(){
    unsigned a,b,c,d;uint32_t lo,hi;
    if(!__get_cpuid(1,&a,&b,&c,&d)||!(c&bit_AVX)||!(c&bit_OSXSAVE)||!(c&bit_FMA))return false;
    __asm__ volatile("xgetbv":"=a"(lo),"=d"(hi):"c"(0));(void)hi;
    if((lo&0xe6)!=0xe6||!__get_cpuid_count(7,0,&a,&b,&c,&d))return false;
    const unsigned required=(1u<<5)|(1u<<16)|(1u<<17)|(1u<<30)|(1u<<31);
    return (b&required)==required;
}
static float from_bits(uint32_t b){float x;std::memcpy(&x,&b,sizeof(x));return x;}
CHECK_AVX512
static void compare(const std::array<float,16>& input){
    const __m512 x=_mm512_loadu_ps(input.data());
    const __m512 original=Sleef_sinf16_u10avx512f(x);
    const __m512 candidate=ip3_sleef_inline_sinf16_u10avx512f(x);
    if(_mm512_cmpeq_epi32_mask(_mm512_castps_si512(original),_mm512_castps_si512(candidate))!=0xffff)
        throw std::runtime_error("Inline sine differs from the pinned archive as uint32 values");
}
static uint64_t exercise(unsigned csr,unsigned seed){
    RestoreCSR restore;_mm_setcsr(csr);uint64_t count=0;
    std::array<float,16> v{};
    const std::array<uint32_t,24> special{{0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
        0x00800000,0x80800000,0x3f800000,0xbf800000,0x40490fdb,0xc0490fdb,
        0x42f9ffff,0x42fa0000,0x42fa0001,0xc2f9ffff,0xc2fa0000,0xc2fa0001,
        0x7f7fffff,0xff7fffff,0x4f000000,0xcf000000,0x3f000000,0xbf000000}};
    for(uint32_t bits:special){
        v.fill(from_bits(bits));compare(v);count+=16;
        for(int lane=0;lane<16;++lane){v.fill(.25f);v[lane]=from_bits(bits);compare(v);count+=16;}
    }
    std::mt19937 random(seed);
    for(int sample=0;sample<4096;++sample){
        for(float& x:v){
            uint32_t b=random();
            if((b&0x7fffffff)>=0x7f800000)b=(b&0x80000000)|0x7f7fffff;
            x=from_bits(b);
        }
        compare(v);count+=16;
    }
    // Every lane stays strictly inside the |x|<125 range-reduction branch.
    // Integer-to-float conversions and power-of-two products are exact in
    // every tested rounding mode. A separate generator preserves old cases.
    std::mt19937 dense(seed^0x36df719bu);
    for(int sample=0;sample<2048;++sample){
        for(float& x:v){
            const int coordinate=static_cast<int>(dense()%255999u)-127999;
            x=static_cast<float>(coordinate)*0x1p-10f;
        }
        compare(v);count+=16;
    }
    // Typical finite activation range, including both endpoints [-16,16].
    for(int sample=0;sample<2048;++sample){
        for(float& x:v){
            const int coordinate=static_cast<int>(dense()%32769u)-16384;
            x=static_cast<float>(coordinate)*0x1p-10f;
        }
        compare(v);count+=16;
    }
    // Dense bit patterns near zero, |x|<=2^-10, include all exponent bands
    // down through subnormals without arithmetic that could erase their bits.
    for(int sample=0;sample<2048;++sample){
        for(float& x:v){
            const uint32_t sign=dense()&0x80000000u;
            const uint32_t magnitude=dense()%0x3a800001u;
            x=from_bits(sign|magnitude);
        }
        compare(v);count+=16;
    }
    return count;
}
int main(){
    try{
        if(!available()){
            std::cout<<"{\"status\":\"skipped\",\"reason\":\"CPU/OS AVX512 target unavailable\",\"models_executed\":false}\n";
            return 0;
        }
        RestoreCSR restore;
        const unsigned base=(restore.old|0x1f80u)&~(_MM_ROUND_MASK|0x8040u|0x3fu);
        uint64_t checked=0;
        for(unsigned rounding:{0u,0x2000u,0x4000u,0x6000u})
            for(unsigned denormal:{0u,0x40u,0x8000u,0x8040u})
                checked+=exercise(base|rounding|denormal,191+rounding+denormal);
        auto first=std::async(std::launch::async,[base]{return exercise(base,77);});
        auto second=std::async(std::launch::async,[base]{return exercise(base|0x8040u,91);});
        const uint64_t concurrent=first.get()+second.get();
        std::cout<<"{\"status\":\"passed\",\"rounding_denormal_combinations\":16,\"serial_lanes\":"<<checked
                 <<",\"concurrent_lanes\":"<<concurrent
                 <<",\"dense_fast_vectors_per_environment\":2048,\"audio_range_vectors_per_environment\":2048"
                 <<",\"near_zero_vectors_per_environment\":2048,\"models_executed\":false,\"timings_collected\":false}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
