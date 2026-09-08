#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "precision.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

// An isolated experiment. Existing quantization and full-K integer arithmetic
// are retained. The two projections share input preparation, not outputs.
#if defined(__FAST_MATH__)
#error "Ordinary FP32 semantics required"
#endif
#define VNNI __attribute__((target("avx512f,avx512bw,avx512vnni")))
namespace {
void require(bool v,const char* text){if(!v)throw std::invalid_argument(text);}
OrtStatus* fail(const OrtApi& api) noexcept {
    try{throw;}catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
    catch(...){return api.CreateStatus(ORT_FAIL,"Paired projection failed");}
}
float scale_of(float mx){if(mx==0.f)return 1.f;const float s=mx/127.f;return s>0.f?s:mx;}
int quantize(float value,float scale){
    const float q=std::max(-127.f,std::min(127.f,value/scale));
    int n=int(std::floor(q));const float fraction=q-float(n);
    if(fraction>.5f||(fraction==.5f&&(n&1)))++n;
    return n;
}
using Plan=std::unique_ptr<void,decltype(&ip_destroy_plan)>;
using Input=std::unique_ptr<void,decltype(&ip_destroy_input)>;
struct Weights {
    int m,k;std::vector<uint8_t> packed;std::vector<float> scales;
    Plan fallback{nullptr,ip_destroy_plan};
    Weights(const float* w,int rows,int columns):m(rows),k(columns),packed(size_t(m)*k),scales(m){
        // 64-row panels, with four 16-lane vectors for each group of four K.
        for(int r=0;r<m;++r){
            float mx=0.f;
            for(int c=0;c<k;++c){float v=w[size_t(r)*k+c];require(std::isfinite(v),"Nonfinite weight");mx=std::max(mx,std::fabs(v));}
            const float s=scales[r]=scale_of(mx);
            for(int c=0;c<k;++c){
                const size_t offset=size_t(r/64)*64*k+size_t(c/4)*256+size_t((r%64)/16)*64+(r%16)*4+c%4;
                packed[offset]=uint8_t(quantize(w[size_t(r)*k+c],s)+128);
            }
        }
        // Only used for T>4. Keep the established implementation intact.
        fallback.reset(ip_create(m,k,w,8,1));require(bool(fallback),ip_last_error());
    }
};
struct Prepared {
    int t,k;float scales[4]={};int32_t sums[4]={};std::vector<uint32_t> words;
    Prepared(const float* x,int columns,int time):t(time),k(columns),words(size_t(time)*columns/4){
        // Exact same per-column scan and nearest-even quantizer as ip_prepare.
        for(int c=0;c<k;++c)for(int j=0;j<t;++j){
            const float v=x[size_t(c)*t+j];require(std::isfinite(v),"Nonfinite activation");
            scales[j]=std::max(scales[j],std::fabs(v));
        }
        for(int j=0;j<t;++j)scales[j]=scale_of(scales[j]);
        for(int c=0;c<k;++c)for(int j=0;j<t;++j){
            const int q=quantize(x[size_t(c)*t+j],scales[j]);sums[j]+=q;
            words[size_t(j)*(k/4)+c/4]|=uint32_t(uint8_t(int8_t(q)))<<(8*(c%4));
        }
    }
};
template<int T> VNNI void project(const Weights& w,const Prepared& x,float* output){
    constexpr int GROUPS=4;
    for(int row=0;row<w.m;row+=64){
        __m512i acc[GROUPS][T];
        for(int g=0;g<GROUPS;++g)for(int t=0;t<T;++t)acc[g][t]=_mm512_setzero_si512();
        const uint8_t* weights=w.packed.data()+size_t(row)*w.k;
        for(int c=0;c<w.k/4;++c){
            __m512i qx[T];for(int t=0;t<T;++t)qx[t]=_mm512_set1_epi32(int(x.words[size_t(t)*(w.k/4)+c]));
            for(int g=0;g<GROUPS;++g){
                const __m512i qw=_mm512_loadu_si512(weights+size_t(c)*256+g*64);
                for(int t=0;t<T;++t)acc[g][t]=_mm512_dpbusd_epi32(acc[g][t],qw,qx[t]);
            }
        }
        for(int g=0;g<GROUPS;++g){
            const __m512 sw=_mm512_loadu_ps(w.scales.data()+row+g*16);
            alignas(64) float y[T][16];
            for(int t=0;t<T;++t){
                const __m512i corrected=_mm512_sub_epi32(acc[g][t],_mm512_set1_epi32(128*x.sums[t]));
                const __m512 scale=_mm512_mul_ps(sw,_mm512_set1_ps(x.scales[t]));
                _mm512_store_ps(y[t],_mm512_mul_ps(_mm512_cvtepi32_ps(corrected),scale));
            }
            for(int r=0;r<16;++r)for(int t=0;t<T;++t){
                require(std::isfinite(y[t][r]),"Nonfinite output");output[size_t(row+g*16+r)*T+t]=y[t][r];
            }
        }
    }
}
void project(const Weights& w,const Prepared& x,float* y){
    switch(x.t){case 1:project<1>(w,x,y);break;case 2:project<2>(w,x,y);break;
    case 3:project<3>(w,x,y);break;case 4:project<4>(w,x,y);break;default:throw std::invalid_argument("Small T required");}
}
struct Kernel {
    const OrtApi& api;std::unique_ptr<Weights> a,b;int m,k;
    Kernel(const OrtApi& api_,const OrtKernelInfo* raw):api(api_){
        Ort::ConstKernelInfo info(raw);
        require(info.GetAttribute<int64_t>("native_abi")==1&&ip_abi()==1,"ABI1 required");
        require((ip_capabilities()&3)==3,"CPU/OS AVX512-VNNI and sequential MKL required");
        const int64_t mm=info.GetAttribute<int64_t>("M"),kk=info.GetAttribute<int64_t>("K");
        require(mm==8192&&kk==2048,"Only the first 8192x2048 projection pair is supported");m=int(mm);k=int(kk);
        for(int i=0;i<2;++i){
            int constant=0;auto value=info.GetTensorConstantInput(i,&constant);
            require(constant&&value,"Constant weights required");auto dims=value.GetTensorTypeAndShapeInfo();
            require(dims.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT&&dims.GetShape()==std::vector<int64_t>({m,k}),"Weight shape/type mismatch");
            auto w=std::make_unique<Weights>(value.GetTensorData<float>(),m,k);if(i==0)a=std::move(w);else b=std::move(w);
        }
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try{
            Ort::KernelContext context(raw);auto value=context.GetInput(2);auto shape=value.GetTensorTypeAndShapeInfo();const auto dims=shape.GetShape();
            require(shape.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT&&dims.size()==3&&dims[0]==1&&dims[1]==k&&dims[2]>=0&&dims[2]<=INT32_MAX,"Expected [1,2048,T] FP32");
            const int t=int(dims[2]);auto oa=context.GetOutput(0,{1,m,t}),ob=context.GetOutput(1,{1,m,t});if(!t)return nullptr;
            const float* x=value.GetTensorData<float>();float* ya=oa.GetTensorMutableData<float>();float* yb=ob.GetTensorMutableData<float>();
            if(t<=4){Prepared input(x,k,t);project(*a,input,ya);project(*b,input,yb);}
            else{
                Input input(ip_prepare(a->fallback.get(),x,t),ip_destroy_input);require(bool(input),ip_last_error());
                require(ip_run_rows(a->fallback.get(),input.get(),ya,0,m)==0,ip_last_error());
                require(ip_run_rows(b->fallback.get(),input.get(),yb,0,m)==0,ip_last_error());
            }
            return nullptr;
        }catch(...){return fail(api);}
    }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
    Op(){start_ver_=1;end_ver_=1;}
    const char* GetName()const{return "PackedProjectionPairF32";}
    const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
    size_t GetInputTypeCount()const{return 3;}
    ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    size_t GetOutputTypeCount()const{return 2;}
    ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept {
        *out=nullptr;try{*out=new Kernel(api,info);return nullptr;}catch(...){return fail(api);}
    }
};
std::mutex mutex;const OrtApi* initialized=nullptr;Op op;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
    const auto* api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
    try{std::lock_guard<std::mutex> lock(mutex);require(!initialized||initialized==api,"Multiple runtimes unsupported");
        if(!initialized){Ort::InitApi(api);domain=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.streaming.matrix.experimental");domain->Add(&op);initialized=api;}
        return api->AddCustomOpDomain(options,*domain);
    }catch(...){return fail(*api);}
}
