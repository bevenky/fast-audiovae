#include "sme_backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

void need(bool value,const char* text){if(!value)throw std::runtime_error(text);}
void ok(int value){need(value==0,ipa_sme_last_error());}
void equal(float a,float b){need(std::memcmp(&a,&b,4)==0,"Full-K SME bit mismatch");}
int main(){
    try{
        size_t vl,mr,nr;ok(ipa_sme_geometry(&vl,&mr,&nr));
        constexpr size_t k=16384,t=17;size_t checked=0;
        for(int ws:{-127,127})for(int xs:{-127,127}){
            std::vector<int8_t>w(k,int8_t(ws)),x(k*t,int8_t(xs));
            const float sw=0.001237f;std::vector<float>sx(t),ref(t);
            for(size_t i=0;i<t;++i){
                sx[i]=i%3==0?0.013f:i%3==1?std::numeric_limits<float>::min():std::numeric_limits<float>::denorm_min();
                const float scale=sw*sx[i];ref[i]=float(int32_t(k)*ws*xs)*scale;
            }
            std::vector<uint32_t>lhs(ipa_sme_lhs_bytes(1,k,vl)/4),rhs(ipa_sme_rhs_bytes(k,t,vl)/4);
            ok(ipa_sme_pack_lhs(w.data(),&sw,1,k,lhs.data(),lhs.size()*4,vl));
            ok(ipa_sme_pack_rhs(x.data(),sx.data(),k,t,rhs.data(),rhs.size()*4,vl));
            std::vector<std::future<std::vector<float>>> jobs;
            for(int j=0;j<4;++j)jobs.push_back(std::async(std::launch::async,[&](){
                std::vector<float> y(t);ok(ipa_sme_run_rows(1,k,t,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,y.data(),0,1,vl));return y;
            }));
            for(auto&job:jobs){const auto y=job.get();for(size_t i=0;i<t;++i){equal(y[i],ref[i]);++checked;}}
        }
        int8_t q=1;float scale=std::numeric_limits<float>::max(),y=0;
        std::vector<uint32_t>lhs(ipa_sme_lhs_bytes(1,1,vl)/4),rhs(ipa_sme_rhs_bytes(1,1,vl)/4);
        ok(ipa_sme_pack_lhs(&q,&scale,1,1,lhs.data(),lhs.size()*4,vl));
        ok(ipa_sme_pack_rhs(&q,&scale,1,1,rhs.data(),rhs.size()*4,vl));
        need(ipa_sme_run_rows(1,1,1,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,&y,0,1,vl)==-1,"Overflow output must fail");
        need(ipa_sme_run_rows(1,1,1,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,reinterpret_cast<float*>(lhs.data()),0,1,vl)==-1,"Packed operand overlap must fail");
        need(ipa_sme_run_rows(1,1,1,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,&y,0,1,16)==-1,"Changed worker vector length must fail");
        float invalid=0;need(ipa_sme_pack_lhs(&q,&invalid,1,1,lhs.data(),lhs.size()*4,vl)==-1,"Zero scale must fail");
        std::printf("{\"status\":\"passed\",\"K\":16384,\"sign_combinations\":4,\"concurrent_callers\":4,\"bitwise_values\":%zu,\"overflow_alias_scale_and_SVL_errors\":4,\"timings_collected\":false}\n",checked);return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"SME extremes failed: %s\n",e.what());return 1;}
}
