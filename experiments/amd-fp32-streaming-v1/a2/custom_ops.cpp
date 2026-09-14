#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "native_kernels.h"
#include <cmath>
#include <cstring>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__)
#error FP32 requires fast math off
#endif
extern "C" int32_t a2_raw_history_triple_f32(const float*,const float*,const float*,const float*,
    const float*,const float*,const float*,const float*,float*,int64_t,int64_t,int64_t,int32_t,int32_t,int32_t);
namespace {
constexpr const char* domain_name="fast.audiovae.amd.rawhistory.a2.v1";
void Require(bool v,const char* s){if(!v)throw std::invalid_argument(s);}
OrtStatus* Error(const OrtApi& api)noexcept{
    try{throw;}catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
    catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
    catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
    catch(...){return api.CreateStatus(ORT_FAIL,"A2 unknown error");}
}
std::vector<int64_t> Shape(Ort::ConstValue value){
    Require(value!=nullptr && value.IsTensor(),"Dense tensor required");
    auto info=value.GetTensorTypeAndShapeInfo();
    Require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 required");
    Require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU required");
    return info.GetShape();
}
std::vector<float> Constant(Ort::ConstKernelInfo info,size_t i,const std::vector<int64_t>& shape){
    int fixed=0;auto v=info.GetTensorConstantInput(i,&fixed);
    Require(fixed && v!=nullptr && Shape(v)==shape,"Fixed coefficient shape required");
    auto n=v.GetTensorTypeAndShapeInfo().GetElementCount();const float* p=v.GetTensorData<float>();
    for(size_t j=0;j<n;++j)Require(std::isfinite(p[j]),"Finite coefficient required");
    return {p,p+n};
}
void Disjoint(const float* a,size_t an,const float* b,size_t bn){
    if(!an || !bn)return;
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    Require(a && b && x%4==0 && y%4==0,"Aligned buffers required");
    Require(an<=(std::numeric_limits<uintptr_t>::max()-x)/4 &&
            bn<=(std::numeric_limits<uintptr_t>::max()-y)/4,"Pointer overflow");
    Require(x+an*4<=y || y+bn*4<=x,"Read/output buffers overlap");
}
struct Kernel{
    const OrtApi& api;int64_t channels;int32_t dilation,backend;size_t row_batches;
    std::vector<float> w,b,ap,rp,aq,rq;
    Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
        Ort::ConstKernelInfo info(raw);
        Require(ncc_abi_version()==1 && ncc_compiled_tile()==256 && info.GetAttribute<int64_t>("candidate_abi")==1,"ABI/tile mismatch");
        channels=info.GetAttribute<int64_t>("channels");
        Require(channels==512 || channels==1024,"A2 supports audited C512/C1024 residual regions only");
        auto d=info.GetAttribute<int64_t>("dilation");Require(d==1 || d==3 || d==9,"Invalid dilation");dilation=static_cast<int32_t>(d);
        Require(info.GetAttribute<int64_t>("backend")==NCC_AVX512,"Audited AVX512 backend required");backend=NCC_AVX512;
        Require(info.GetAttribute<int64_t>("require_vector_sine")==1 && ncc_backend_available(backend) &&
                ncc_vector_sine_available(backend) && ncc_streaming_math_version(backend)==1,"Original streaming SLEEF AVX512 required");
        auto rows=info.GetAttribute<int64_t>("row_batches");Require(rows>=0,"Invalid row batches");row_batches=static_cast<size_t>(rows);
        w=Constant(info,2,{channels,1,7});b=Constant(info,3,{channels});
        ap=Constant(info,4,{channels});rp=Constant(info,5,{channels});
        aq=Constant(info,6,{channels});rq=Constant(info,7,{channels});
    }
    struct Work{const Kernel* k;const float* x;const float* h;float* y;float* next;int64_t t;
        std::atomic<int32_t> error{NCC_OK};};
    static void Row(void* raw,size_t row)noexcept{
        auto& v=*static_cast<Work*>(raw);const auto& k=*v.k;const size_t h=6*k.dilation;
        if(v.t){
            auto s=a2_raw_history_triple_f32(v.x+row*v.t,v.h+row*h,k.w.data()+row*7,k.b.data()+row,
                k.ap.data()+row,k.rp.data()+row,k.aq.data()+row,k.rq.data()+row,v.y+row*v.t,
                1,1,v.t,k.dilation,k.backend,1);
            if(s!=NCC_OK){v.error.store(s,std::memory_order_relaxed);return;}
        }
        if(static_cast<uint64_t>(v.t)>=h)std::memcpy(v.next+row*h,v.x+row*v.t+v.t-h,h*4);
        else{
            std::memcpy(v.next+row*h,v.h+row*h+v.t,(h-v.t)*4);
            if(v.t)std::memcpy(v.next+row*h+h-v.t,v.x+row*v.t,v.t*4);
        }
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw)noexcept{
        try{
            Ort::KernelContext c(raw);auto x=c.GetInput(0),h=c.GetInput(1);auto s=Shape(x);
            Require(s.size()==3 && s[0]==1 && s[1]==channels && s[2]>=0,"Expected [1,C,T]");
            const int64_t halo=6*dilation;Require(Shape(h)==std::vector<int64_t>{1,channels,halo},"Raw history shape mismatch");
            Require(static_cast<uint64_t>(s[2])<=std::numeric_limits<size_t>::max()/(4*static_cast<uint64_t>(channels)),"Size overflow");
            auto y=c.GetOutput(0,s),next=c.GetOutput(1,std::vector<int64_t>{1,channels,halo});
            Work v{this,x.GetTensorData<float>(),h.GetTensorData<float>(),y.GetTensorMutableData<float>(),next.GetTensorMutableData<float>(),s[2]};
            auto n=static_cast<size_t>(channels)*s[2],hn=static_cast<size_t>(channels)*halo;
            Disjoint(v.y,n,v.next,hn);Disjoint(v.y,n,v.x,n);Disjoint(v.y,n,v.h,hn);
            Disjoint(v.next,hn,v.x,n);Disjoint(v.next,hn,v.h,hn);
            c.ParallelFor(Row,static_cast<size_t>(channels),row_batches,&v);
            if(v.error.load()!=NCC_OK)return api.CreateStatus(ORT_FAIL,ncc_status_string(v.error.load()));
            return nullptr;
        }catch(...){return Error(api);}
    }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
    Op(){start_ver_=1;end_ver_=1;}
    const char* GetName()const{return "RawHistorySnakeDW7SnakeF32";}
    const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
    size_t GetInputTypeCount()const{return 8;}size_t GetOutputTypeCount()const{return 2;}
    ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    OrtStatus* CreateKernelV2(const OrtApi& a,const OrtKernelInfo* i,void** out)const noexcept{
        *out=nullptr;try{*out=new Kernel(a,i);return nullptr;}catch(...){return Error(a);}}
};
Op op;std::mutex lock;const OrtApi* registered=nullptr;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
    const auto* api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
    try{std::lock_guard<std::mutex> held(lock);Require(!registered || registered==api,"Mixed runtimes");
        if(!registered){Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>(domain_name);d->Add(&op);domain=std::move(d);registered=api;}
        return api->AddCustomOpDomain(options,*domain);
    }catch(...){return Error(*api);}
}
