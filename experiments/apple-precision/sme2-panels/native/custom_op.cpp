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
using Workspace=std::unique_ptr<void,decltype(&ipc_destroy_workspace)>;
struct Kernel{
    const OrtApi& api;int m,k,shards,backend=0,row_step=1;Plan plan{nullptr,ip_destroy_plan};
    Kernel(const OrtApi& a,const OrtKernelInfo* raw):api(a){
        Ort::ConstKernelInfo info(raw);
        auto integer=[&](const char* name){return info.GetAttribute<int64_t>(name);};
        require(integer("native_abi")==1&&ip_abi()==1,"Precision ABI1 required");
        const int64_t mm=integer("M"),kk=integer("K"),ss=integer("shards");
        require(mm>0&&mm<=INT32_MAX&&kk>0&&kk<=16384,"Invalid matrix shape");
        require(static_cast<uint64_t>(mm)<=SIZE_MAX/static_cast<uint64_t>(kk)/4,"Weight size overflow");
        require(ss>=1&&ss<=64,"Shards must be in1..64");m=int(mm);k=int(kk);shards=int(ss);
        const auto mode=integer("precision_mode"),chosen_backend=integer("backend");backend=int(chosen_backend);
        require(mode==8,"Apple port supports INT8 mode8 only");
        require(chosen_backend==0||chosen_backend==2||chosen_backend==3,"Apple backend must be scalar0, SDOT2 or SME3");
        int constant=0;auto weights=info.GetTensorConstantInput(0,&constant);
        require(constant&&weights,"Weights must be constant initializer");
        require(shape(weights)==std::vector<int64_t>({m,k}),"Unexpected constant weight shape");
        plan.reset(ip_create(m,k,weights.GetTensorData<float>(),int(mode),int(backend)));
        require(bool(plan),ip_last_error());
        row_step=ipc_row_step(plan.get());require(row_step>0,ip_last_error());
    }
    struct Work{
        const Kernel* kernel;void* input;float* output;int jobs;bool prepare;
        std::atomic<bool> failed{false};std::mutex mutex;std::string message;
        Work(const Kernel* k,void* x,float* y,int n,bool prep=false):kernel(k),input(x),output(y),jobs(n),prepare(prep){}
    };
    static void run(void* raw,size_t job) noexcept {
        auto& w=*static_cast<Work*>(raw);const auto& k=*w.kernel;
        const int64_t blocks=(int64_t(k.m)+k.row_step-1)/k.row_step;
        const int first=int(std::min<int64_t>(k.m,blocks*job/w.jobs*k.row_step)),last=int(std::min<int64_t>(k.m,blocks*(job+1)/w.jobs*k.row_step));
        const int status=w.prepare?ipc_prepare_job(w.input,int(job)):ip_run_rows(k.plan.get(),w.input,w.output,first,last);
        if(status){
            w.failed=true;
            try{std::lock_guard<std::mutex> lock(w.mutex);if(w.message.empty())w.message=ip_last_error();}catch(...){}
        }
    }
    struct PanelWork{
        const Kernel* kernel;const float* input;float* output;int time,tile,jobs;
        std::atomic<bool> failed{false};std::mutex mutex;std::string message;
        PanelWork(const Kernel* k,const float* x,float* y,int t,int q,int n):kernel(k),input(x),output(y),time(t),tile(q),jobs(n){}
    };
    static void run_panel(void* raw,size_t job) noexcept {
        auto& w=*static_cast<PanelWork*>(raw);
        try{
            Workspace panel(ipc_create_workspace(w.kernel->plan.get(),w.tile),ipc_destroy_workspace);
            require(bool(panel),ip_last_error());
            const int64_t tiles=(int64_t(w.time)+w.tile-1)/w.tile;
            const int64_t begin=tiles*int64_t(job)/w.jobs,end=tiles*int64_t(job+1)/w.jobs;
            for(int64_t tile=begin;tile<end;++tile){
                const int first=int(tile*w.tile),length=std::min(w.tile,w.time-first);
                require(ipc_prepare_panel(panel.get(),w.input,w.time,first,length)==0,ip_last_error());
                require(ipc_run_panel(w.kernel->plan.get(),panel.get(),w.output,w.time,first,0,w.kernel->m)==0,ip_last_error());
            }
        }catch(const std::exception& e){
            w.failed=true;
            try{std::lock_guard<std::mutex> lock(w.mutex);if(w.message.empty())w.message=e.what();}catch(...){}
        }catch(...){w.failed=true;}
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try{
            Ort::KernelContext context(raw);auto value=context.GetInput(1);const auto dims=shape(value);
            require(dims.size()==3&&dims[0]==1&&dims[1]==k&&dims[2]>=0&&dims[2]<=INT32_MAX,
                    "Expected contiguous [1,K,T] with nonnegative LP64 time");
            const int t=int(dims[2]);
            require(static_cast<uint64_t>(m)<=SIZE_MAX/4/static_cast<uint64_t>(std::max(1,t)),"Output size overflow");
            auto output=context.GetOutput(0,{1,m,t});
            if(backend==3&&m<=1024&&t>=2048){
                // Keep one packed time panel per worker. K>=512 uses512
                // columns; smaller K uses1024. No split-K or inner thread pool.
                const int tile=k>=512?512:1024;
                const int jobs=int(std::min<int64_t>((int64_t(t)+tile-1)/tile,shards));
                PanelWork work{this,value.GetTensorData<float>(),output.GetTensorMutableData<float>(),t,tile,jobs};
                if(jobs==1)run_panel(&work,0);else context.ParallelFor(run_panel,jobs,jobs,&work);
                require(!work.failed.load(),work.message.empty()?"Panel worker failure":work.message.c_str());
                return nullptr;
            }
            Input input(backend==3?ipc_allocate_input(plan.get(),value.GetTensorData<float>(),t,shards):
                        ip_prepare(plan.get(),value.GetTensorData<float>(),t),ip_destroy_input);
            require(bool(input),ip_last_error());
            if(backend==3){
                const int jobs=ipc_prepare_jobs(input.get());require(jobs>0,ip_last_error());
                Work prep{this,input.get(),nullptr,jobs,true};
                if(jobs==1)run(&prep,0);else context.ParallelFor(run,jobs,jobs,&prep);
                require(!prep.failed.load(),prep.message.empty()?"Parallel preparation failed":prep.message.c_str());
            }
            if(!t)return nullptr;
            Work work{this,input.get(),output.GetTensorMutableData<float>(),int(std::min<int64_t>((int64_t(m)+row_step-1)/row_step,shards))};
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
        if(!initialized){Ort::InitApi(api);domain=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.precision.apple.r4.experimental");domain->Add(&op);initialized=api;}
        return api->AddCustomOpDomain(options,*domain);
    }catch(...){return error(*api);}
}
