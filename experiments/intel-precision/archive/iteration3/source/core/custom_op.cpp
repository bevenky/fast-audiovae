#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "precision.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
OrtStatus* error(const OrtApi& api) noexcept {
    try{throw;}catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
    catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
    catch(...){return api.CreateStatus(ORT_FAIL,"Unknown precision matrix error");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
    require(value&&value.IsTensor(),"Dense tensor required");
    require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
    auto info=value.GetTensorTypeAndShapeInfo();
    require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 activation interface required");
    return info.GetShape();
}
using Plan=std::unique_ptr<void,decltype(&ip_destroy_plan)>;
using Input=std::unique_ptr<void,decltype(&ip_destroy_input)>;
struct Kernel{
    const OrtApi& api;int m,k,shards;Plan plan{nullptr,ip_destroy_plan};
    Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
        Ort::ConstKernelInfo info(raw);
        auto integer=[&](const char* name){return info.GetAttribute<int64_t>(name);};
        require(integer("native_abi")==1&&ip_abi()==1,"Precision ABI1 required");
        const int64_t mm=integer("M"),kk=integer("K"),ss=integer("shards");
        require(mm>0&&mm<=INT32_MAX&&kk>0&&kk<=16384,"Invalid matrix shape");
        require(static_cast<uint64_t>(mm)<=SIZE_MAX/static_cast<uint64_t>(kk)/4,"Weight size overflow");
        require(ss>=1&&ss<=64,"Shards must be in1..64");m=int(mm);k=int(kk);shards=int(ss);
        const auto mode=integer("precision_mode"),backend=integer("backend");
        require(mode==8||mode==16,"Precision mode must be8 or16");
        require(backend==0||backend==1,"Backend must be scalar0 or oneMKL1");
        int constant=0;auto weights=info.GetTensorConstantInput(0,&constant);
        require(constant&&weights,"Weights must be constant initializer");
        require(shape(weights)==std::vector<int64_t>({m,k}),"Unexpected constant weight shape");
        plan.reset(ip_create(m,k,weights.GetTensorData<float>(),int(mode),int(backend)));
        require(bool(plan),ip_last_error());
    }
    struct Work{
        const Kernel* kernel;const void* input;float* output;int jobs;
        std::atomic<bool> failed{false};std::mutex mutex;std::string message;
    };
    static void run(void* raw,size_t job) noexcept {
        auto& w=*static_cast<Work*>(raw);const auto& k=*w.kernel;
        const int first=int(int64_t(k.m)*job/w.jobs),last=int(int64_t(k.m)*(job+1)/w.jobs);
        if(ip_run_rows(k.plan.get(),w.input,w.output,first,last)){
            w.failed=true;
            try{std::lock_guard<std::mutex> lock(w.mutex);if(w.message.empty())w.message=ip_last_error();}catch(...){}
        }
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try{
            Ort::KernelContext context(raw);auto value=context.GetInput(1);const auto dims=shape(value);
            require(dims.size()==3&&dims[0]==1&&dims[1]==k&&dims[2]>=0&&dims[2]<=INT32_MAX,
                    "Expected contiguous [1,K,T] with nonnegative LP64 time");
            const int t=int(dims[2]);
            require(static_cast<uint64_t>(m)<=SIZE_MAX/4/static_cast<uint64_t>(std::max(1,t)),"Output size overflow");
            auto output=context.GetOutput(0,{1,m,t});
            Input input(ip_prepare(plan.get(),value.GetTensorData<float>(),t),ip_destroy_input);
            require(bool(input),ip_last_error());
            if(!t)return nullptr;
            Work work{this,input.get(),output.GetTensorMutableData<float>(),std::min(m,shards),false,{},{}};
            if(work.jobs==1)run(&work,0);else context.ParallelFor(run,work.jobs,work.jobs,&work);
            require(!work.failed.load(),work.message.empty()?"Precision worker failure":work.message.c_str());
            return nullptr;
        }catch(...){return error(api);}
    }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
    Op(){start_ver_=1;end_ver_=1;}
    const char* GetName()const{return "PrecisionMatMulF32";}
    const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
    size_t GetInputTypeCount()const{return 2;}
    ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    size_t GetOutputTypeCount()const{return 1;}
    ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** result)const noexcept {
        *result=nullptr;try{*result=new Kernel(api,info);return nullptr;}catch(...){return error(api);}
    }
};
std::mutex mutex;const OrtApi* initialized=nullptr;Op op;std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" IP_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
    const auto* api=base->GetApi(ORT_API_VERSION);
    if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
    try{
        std::lock_guard<std::mutex> lock(mutex);
        require(!initialized||initialized==api,"Multiple ORT runtimes unsupported");
        if(!initialized){Ort::InitApi(api);
#ifdef IP_ITERATION3_NAMESPACE
            domain=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.precision.iteration3");
#else
            domain=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.precision.matrix.experimental");
#endif
            domain->Add(&op);initialized=api;}
        return api->AddCustomOpDomain(options,*domain);
    }catch(...){return error(*api);}
}
