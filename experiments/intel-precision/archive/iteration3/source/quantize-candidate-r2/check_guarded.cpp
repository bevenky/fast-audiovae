// Native exact-byte checker. No model inference, external libraries or timing.
#include "guarded_quantize.h"
#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#if IP3_GQ_X86
struct Counts { uint64_t vectors=0,lanes=0,fast=0,fallback=0; };
struct RestoreCSR { unsigned old=_mm_getcsr(); ~RestoreCSR(){_mm_setcsr(old);} };
static float from_bits(uint32_t bits){float value;std::memcpy(&value,&bits,4);return value;}
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}

IP3_GQ_TARGET
static void compare(const std::array<float,16>& values,const std::array<float,16>& scales,Counts& counts,bool allow_fast){
    const __m512 x=_mm512_loadu_ps(values.data()),s=_mm512_loadu_ps(scales.data());
    const auto plan=ip3_guarded_quantize::make_plan(s);
    uint16_t mask=0;
    const __m512 candidate=ip3_guarded_quantize::quotient(x,plan,&mask);
    const __m512 reference=_mm512_div_ps(x,s);
    const __m512 high=_mm512_set1_ps(127.f),low=_mm512_set1_ps(-127.f);
    const __m512 candidate_clamped=_mm512_max_ps(low,_mm512_min_ps(high,candidate));
    const __m512 reference_clamped=_mm512_max_ps(low,_mm512_min_ps(high,reference));
    const __m512i actual=_mm512_cvt_roundps_epi32(candidate_clamped,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    const __m512i expected=_mm512_cvt_roundps_epi32(reference_clamped,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    check(_mm512_cmpeq_epi32_mask(actual,expected)==0xffff,"Quantized INT32 mismatch");
    const __m128i bytes_a=_mm512_cvtsepi32_epi8(actual),bytes_b=_mm512_cvtsepi32_epi8(expected);
    check(_mm_movemask_epi8(_mm_cmpeq_epi8(bytes_a,bytes_b))==0xffff,"Quantized S8 mismatch");
    check(allow_fast||!mask,"Directed-rounding mode dispatched fast quotient");
    ++counts.vectors;counts.lanes+=16;
    counts.fast+=__builtin_popcount(unsigned(mask));counts.fallback+=16-__builtin_popcount(unsigned(mask));
}

static Counts exercise(unsigned csr,int seed,bool boundaries){
    RestoreCSR restore;_mm_setcsr(csr);
    Counts counts;std::mt19937 random(seed);
    std::array<float,16> x{},s{};
    const bool allow_fast=(csr&_MM_ROUND_MASK)==_MM_ROUND_NEAREST;
    const std::array<uint32_t,24> scale_bits{{0,1,2,0x007fffff,0x00800000,0x00800001,
        0x01000000,0x0d800000,0x20000000,0x30000000,0x3b000000,0x3e800000,
        0x3f000000,0x3f7fffff,0x3f800000,0x3f800001,0x40000000,0x50000000,
        0x60000000,0x70000000,0x7e7fffff,0x7e800000,0x7e800001,0x7f7fffff}};
    const std::array<uint32_t,16> value_bits{{0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
        0x00800000,0x80800000,0x3f000000,0xbf000000,0x3f800000,0xbf800000,
        0x7f7fffff,0xff7fffff,0x43000000,0xc3000000}};
    for(uint32_t bits:scale_bits){
        s.fill(from_bits(bits));
        for(int lane=0;lane<16;++lane)x[lane]=from_bits(value_bits[lane]);
        compare(x,s,counts,allow_fast);
    }
    // Explicit ordinary lanes must exercise the fast branch in RN, including
    // when FTZ/DAZ are enabled. Values are exact integers, far from ties.
    s.fill(1.f);for(int lane=0;lane<16;++lane)x[lane]=float(lane-8);
    const uint64_t fast_before=counts.fast;compare(x,s,counts,allow_fast);
    if(allow_fast)check(counts.fast==fast_before+16,"Ordinary fast path was not exercised");
    // Every half integer at unit scale must take exact division.
    for(int base=-127;base<=127;base+=16){
        for(int lane=0;lane<16;++lane)x[lane]=float(base+lane)+0.5f;
        const uint64_t before=counts.fast;compare(x,s,counts,allow_fast);
        check(counts.fast==before,"Half-integer guard missed a tie");
    }
    if(boundaries){
        const std::array<float,16> offset{{0.f,0.f,0.f,0x1p-12f,-0x1p-12f,0x1p-13f,-0x1p-13f,
            0x1p-11f,-0x1p-11f,0x1p-10f,-0x1p-10f,0.03125f,-0.03125f,0.125f,-0.125f,0.f}};
        for(uint32_t bits:scale_bits)for(int half=-129;half<=129;++half){
            s.fill(from_bits(bits));
            for(int lane=0;lane<16;++lane){
                float value=(float(half)+0.5f+offset[lane])*s[lane];
                if(!std::isfinite(value))value=std::copysign(std::numeric_limits<float>::max(),value);
                if(lane%3==0)value=std::nextafter(value,-std::numeric_limits<float>::infinity());
                if(lane%3==1)value=std::nextafter(value,std::numeric_limits<float>::infinity());
                if(!std::isfinite(value))value=std::copysign(std::numeric_limits<float>::max(),value);
                x[lane]=value;
            }
            compare(x,s,counts,allow_fast);
        }
    }
    for(int sample=0;sample<4096;++sample){
        for(int lane=0;lane<16;++lane){
            uint32_t sb=random()&0x7fffffff,xb=random();
            if(sb>=0x7f800000)sb=0x7f7fffff;
            if((xb&0x7fffffff)>=0x7f800000)xb=(xb&0x80000000)|0x7f7fffff;
            s[lane]=from_bits(sb);x[lane]=from_bits(xb);
        }
        compare(x,s,counts,allow_fast);
    }
    return counts;
}
#endif

int main(){
    try{
        if(!ip3_guarded_quantize::host_available()){
            std::cout<<"{\"status\":\"skipped\",\"reason\":\"CPU and OS AVX512F/BW unavailable\",\"timings_collected\":false}\n";
            return 0;
        }
#if IP3_GQ_X86
        RestoreCSR restore;
        // Mask FP exceptions, clear sticky flags, and explicitly enumerate all
        // four rounding modes and the four FTZ/DAZ settings. The original
        // quantizer's final integer conversion remains nearest-even throughout.
        const unsigned base=(restore.old|0x1f80u)&~(_MM_ROUND_MASK|0x8040u|0x3fu);
        Counts all;
        for(unsigned rounding:{0u,0x2000u,0x4000u,0x6000u})for(unsigned denormal:{0u,0x40u,0x8000u,0x8040u}){
            const auto c=exercise(base|rounding|denormal,917+rounding+denormal,true);
            all.vectors+=c.vectors;all.lanes+=c.lanes;all.fast+=c.fast;all.fallback+=c.fallback;
        }
        auto first=std::async(std::launch::async,[base]{return exercise(base,441,false);});
        auto second=std::async(std::launch::async,[base]{return exercise(base|0x8040u,442,false);});
        const auto one=first.get(),two=second.get();
        check(all.fast>0&&all.fallback>0,"Both branches must execute");
        std::cout<<"{\"status\":\"passed\",\"gpu_used\":false,\"models_executed\":false,\"timings_collected\":false,"
                 <<"\"rounding_denormal_combinations\":16,\"vectors\":"<<all.vectors<<",\"quantized_lanes\":"<<all.lanes
                 <<",\"fast_lanes\":"<<all.fast<<",\"exact_division_lanes\":"<<all.fallback
                 <<",\"concurrent_quantized_lanes\":"<<one.lanes+two.lanes<<"}\n";
#endif
        return 0;
    }catch(const std::exception& error){std::cerr<<"Guarded quantization failed: "<<error.what()<<'\n';return 1;}
}
