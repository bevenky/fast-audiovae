// Accepted one-thread second projection pair; unchanged oneMKL fallback.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

extern "C" {
void* iis_create(const char*,int,int,int,int,const float*,const float*);
void iis_destroy(void*);
int iis_run(void*,const float*,float*,float*);
const char* iis_error();
const char* iis_version();
void* ip_create(int,int,const float*,int,int);
void ip_destroy_plan(void*);
void* ip_prepare(const void*,const float*,int);
void ip_destroy_input(void*);
int ip_run_rows(const void*,const void*,float*,int,int);
const char* ip_last_error();
}
namespace {
void require(bool value,const char* message) { if (!value) throw std::invalid_argument(message); }
OrtStatus* failure(const OrtApi& api) noexcept {
    try { throw; }
    catch (const std::exception& exception) { return api.CreateStatus(ORT_FAIL,exception.what()); }
    catch (...) { return api.CreateStatus(ORT_FAIL,"Intel matrix screen failed"); }
}
using Plan = std::unique_ptr<void,decltype(&iis_destroy)>;
using CorePlan = std::unique_ptr<void,decltype(&ip_destroy_plan)>;
using CoreInput = std::unique_ptr<void,decltype(&ip_destroy_input)>;
struct Kernel {
    const OrtApi& api;
    Plan t8 {nullptr,iis_destroy}, t16 {nullptr,iis_destroy};
    CorePlan fallback0 {nullptr,ip_destroy_plan}, fallback1 {nullptr,ip_destroy_plan};
    Kernel(const OrtApi& api_,const OrtKernelInfo* raw) : api(api_) {
        Ort::ConstKernelInfo info(raw);
        require(info.GetAttribute<int64_t>("native_abi")==1,"ABI1 required");
        require(info.GetAttribute<int64_t>("M")==3072 && info.GetAttribute<int64_t>("K")==1024,
                "Only the second projection pair is supported");
        int mode=int(info.GetAttribute<int64_t>("mode"));
        require(mode==4,"A qualified oneDNN matrix mode is required");
        require(std::string(iis_version())=="3.13.2","oneDNN3.13.2 required");
        std::array<const float*,2> weights{};
        for (int index=0;index<2;++index) {
            int constant=0;
            auto value=info.GetTensorConstantInput(index,&constant);
            require(constant && value,"Fixed weight initializers required");
            auto shape=value.GetTensorTypeAndShapeInfo();
            require(shape.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    && shape.GetShape()==std::vector<int64_t>({3072,1024}),"Weight shape/type changed");
            weights[index]=value.GetTensorData<float>();
        }
        // Prepare tested packet shapes and the unchanged oneMKL fallback before
        // streaming. Mixed partitions exercise the fallback, not an untested
        // BRGeMM shape; normal40/80ms packets use only the candidate.
        t8.reset(iis_create("",3072,1024,8,mode,weights[0],weights[1]));
        require(bool(t8),iis_error());
        t16.reset(iis_create("",3072,1024,16,mode,weights[0],weights[1]));
        require(bool(t16),iis_error());
        fallback0.reset(ip_create(3072,1024,weights[0],8,1));
        require(bool(fallback0),ip_last_error());
        fallback1.reset(ip_create(3072,1024,weights[1],8,1));
        require(bool(fallback1),ip_last_error());
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try {
            Ort::KernelContext context(raw);
            auto value=context.GetInput(2);
            auto info=value.GetTensorTypeAndShapeInfo();
            auto shape=info.GetShape();
            require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && shape.size()==3
                    && shape[0]==1 && shape[1]==1024 && shape[2]>=0 && shape[2]<=INT32_MAX,
                    "Expected [1,1024,T] FP32");
            auto output0=context.GetOutput(0,{1,3072,shape[2]});
            auto output1=context.GetOutput(1,{1,3072,shape[2]});
            if (shape[2]==8 || shape[2]==16) {
                void* plan=shape[2]==8?t8.get():t16.get();
                int status=iis_run(plan,value.GetTensorData<float>(),output0.GetTensorMutableData<float>(),
                                   output1.GetTensorMutableData<float>());
                if (status) throw std::runtime_error(iis_error());
            } else if (shape[2]) {
                CoreInput prepared(ip_prepare(fallback0.get(),value.GetTensorData<float>(),int(shape[2])),ip_destroy_input);
                require(bool(prepared),ip_last_error());
                int status=ip_run_rows(fallback0.get(),prepared.get(),output0.GetTensorMutableData<float>(),0,3072);
                if (status) throw std::runtime_error(ip_last_error());
                status=ip_run_rows(fallback1.get(),prepared.get(),output1.GetTensorMutableData<float>(),0,3072);
                if (status) throw std::runtime_error(ip_last_error());
            }
            return nullptr;
        } catch (...) { return failure(api); }
    }
};
struct Op : Ort::CustomOpBase<Op,Kernel,true> {
    Op() { start_ver_=1; end_ver_=1; }
    const char* GetName() const { return "SecondProjectionPairF32"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    OrtStatus* CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out) const noexcept {
        *out=nullptr;
        try { *out=new Kernel(api,info); return nullptr; }
        catch (...) { return failure(api); }
    }
};
std::mutex registration_mutex;
const OrtApi* initialized=nullptr;
Op operation;
std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" __attribute__((visibility("default")))
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
    const auto* api=base->GetApi(ORT_API_VERSION);
    if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
    try {
        std::lock_guard<std::mutex> lock(registration_mutex);
        require(!initialized || initialized==api,"Multiple ORT runtimes unsupported");
        if (!initialized) {
            Ort::InitApi(api);
            domain=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.intel.streaming.matrix.v1");
            domain->Add(&operation);
            initialized=api;
        }
        return api->AddCustomOpDomain(options,*domain);
    } catch (...) { return failure(*api); }
}
