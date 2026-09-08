#include "sme_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

void need(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
void ok(int value) { need(value == 0, ipa_sme_last_error()); }
void exact(float a, float b) { need(std::memcmp(&a, &b, 4) == 0, "SME scalar bits differ"); }

int main() {
    try {
        size_t vl, mr, nr; ok(ipa_sme_geometry(&vl, &mr, &nr));
        const size_t shapes[][3] = {{1,1,1},{3,7,5},{17,31,65},{32,128,129},{128,256,65}};
        size_t checked = 0;
        for (const auto& shape : shapes) {
            const size_t m=shape[0], k=shape[1], t=shape[2];
            std::vector<int8_t> w(m*k), x(k*t);
            std::vector<float> sw(m), sx(t), ref(m*t), y(m*t);
            for (size_t i=0; i<w.size(); ++i) w[i]=int8_t(int((i*31+13)%255)-127);
            for (size_t i=0; i<x.size(); ++i) x[i]=int8_t(int((i*47+17)%255)-127);
            for (size_t i=0; i<m; ++i) sw[i]=float(i%7+1)*0.00037f;
            for (size_t i=0; i<t; ++i) sx[i]=float(i%11+1)*0.013f;
            std::vector<uint32_t> lhs(ipa_sme_lhs_bytes(m,k,vl)/4), rhs(ipa_sme_rhs_bytes(k,t,vl)/4);
            ok(ipa_sme_pack_lhs(w.data(),sw.data(),m,k,lhs.data(),lhs.size()*4,vl));
            ok(ipa_sme_pack_rhs(x.data(),sx.data(),k,t,rhs.data(),rhs.size()*4,vl));
            for (size_t r=0;r<m;++r) for(size_t c=0;c<t;++c){
                int32_t acc=0;for(size_t q=0;q<k;++q)acc+=int32_t(w[r*k+q])*int32_t(x[q*t+c]);
                const float scale=sw[r]*sx[c];ref[r*t+c]=float(acc)*scale;
            }
            ok(ipa_sme_run_rows(m,k,t,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,y.data(),0,m,vl));
            for(size_t i=0;i<y.size();++i){exact(y[i],ref[i]);++checked;}
            const size_t blocks=(m+mr-1)/mr;
            std::vector<std::future<void>> jobs;
            for(size_t j=0;j<3;++j)jobs.push_back(std::async(std::launch::async,[&,j](){
                const size_t first=std::min(m,(blocks*j/3)*mr),last=std::min(m,(blocks*(j+1)/3)*mr);
                if(first!=last)ok(ipa_sme_run_rows(m,k,t,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,y.data(),first,last,vl));
            }));
            for(auto& job:jobs)job.get();
            for(size_t i=0;i<y.size();++i){exact(y[i],ref[i]);++checked;}
        }
        // Positive zero, negative zero, gradual underflow, and scale underflow.
        const float scales[][2]={{1.f,1.f},{std::numeric_limits<float>::denorm_min(),1.f},
                                  {std::numeric_limits<float>::min(),std::numeric_limits<float>::min()}};
        for(const auto& scale:scales)for(int dot:{-1,0,1}){
            int8_t w=int8_t(dot),x=1;float y=0;
            std::vector<uint32_t> lhs(ipa_sme_lhs_bytes(1,1,vl)/4),rhs(ipa_sme_rhs_bytes(1,1,vl)/4);
            ok(ipa_sme_pack_lhs(&w,&scale[0],1,1,lhs.data(),lhs.size()*4,vl));
            ok(ipa_sme_pack_rhs(&x,&scale[1],1,1,rhs.data(),rhs.size()*4,vl));
            ok(ipa_sme_run_rows(1,1,1,lhs.data(),lhs.size()*4,rhs.data(),rhs.size()*4,&y,0,1,vl));
            const float combined=scale[0]*scale[1];exact(y,float(dot)*combined);++checked;
        }
        need(ipa_sme_rhs_bytes(128,0,vl)==0,"T0 RHS bytes");
        ok(ipa_sme_run_rows(128,128,0,nullptr,0,nullptr,0,nullptr,0,128,vl));
        need(ipa_sme_lhs_bytes(SIZE_MAX,128,vl)==0,"M overflow rejected");
        need(ipa_sme_rhs_bytes(16385,128,vl)==0,"K limit rejected");
        need(ipa_sme_lhs_bytes(1,1,3)==0,"Invalid vector length rejected");
        std::printf("{\"status\":\"passed\",\"vector_bytes\":%zu,\"mr\":%zu,\"nr\":%zu,\"shapes\":5,\"bitwise_values\":%zu,\"parallel_row_partitions\":5,\"zero_subnormal_fixtures\":9,\"timings_collected\":false}\n",vl,mr,nr,checked);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SME check failed: %s\n",e.what()); return 1; }
}
