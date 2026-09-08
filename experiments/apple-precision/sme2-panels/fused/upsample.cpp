// Apple r4 shared packed SME2 projections, ordered phase and C128 residual stack.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "native_kernels.h"
#include "stage_dw.h"
#include "precision.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__FAST_MATH__)
#error "Upsample stage requires exact ordered network operations"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif

namespace {
constexpr int Channels=128, ProjectionChannels=256, Stride=2;
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
OrtStatus* error(const OrtApi& api) noexcept {
    try{throw;}
    catch(const Ort::Exception& e){return api.CreateStatus(e.GetOrtErrorCode(),e.what());}
    catch(const std::invalid_argument& e){return api.CreateStatus(ORT_INVALID_ARGUMENT,e.what());}
    catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
    catch(...){return api.CreateStatus(ORT_FAIL,"Upsample stage failed");}
}
std::vector<int64_t> shape(Ort::ConstValue value){
    require(value!=nullptr&&value.IsTensor(),"Expected dense tensor");
    auto info=value.GetTensorTypeAndShapeInfo();
    require(info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,"FP32 tensor required");
    require(value.GetTensorMemoryInfo().GetDeviceType()==OrtMemoryInfoDeviceType_CPU,"CPU tensor required");
    return info.GetShape();
}
std::vector<float> constant(Ort::ConstKernelInfo info,size_t index,const std::vector<int64_t>& dims){
    int fixed=0;auto value=info.GetTensorConstantInput(index,&fixed);
    require(fixed&&shape(value)==dims,"Immutable constant has wrong shape");
    const float* data=value.GetTensorData<float>();
    const size_t size=value.GetTensorTypeAndShapeInfo().GetElementCount();
    for(size_t i=0;i<size;++i)require(std::isfinite(data[i]),"Nonfinite upsample or stage constant");
    return {data,data+size};
}

/* This helper accepts one real previous column. Local tile starts do not reset
 * causal history. The two additions are deliberately separate. */
#if SP_X86
__attribute__((target("avx512f,avx2,fma")))
#endif
void phase_tile(const float* current,const float* previous,const float* carry,
                const float* bias,float* output,int low_n,int high_n){
    for(int c=0;c<Channels;++c){
        const float* cur0=current+static_cast<size_t>(2*c)*low_n;
        const float* cur1=cur0+low_n;
        const float* prev0=previous+static_cast<size_t>(2*c)*low_n;
        const float* prev1=prev0+low_n;
        float* out=output+static_cast<size_t>(c)*high_n;
        int i=0;
#if SP_X86
        const __m512 bb=_mm512_set1_ps(bias[c]);
        const __m512i shift=_mm512_setr_epi32(0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
        const __m512i front=_mm512_setr_epi32(0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
        const __m512i back=_mm512_setr_epi32(8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
        for(;i+16<=low_n&&2*(i+16)<=high_n;i+=16){
            const __m512 p0=i?_mm512_loadu_ps(prev0+i-1):
                _mm512_mask_permutexvar_ps(_mm512_set1_ps(carry[2*c]),0xfffe,shift,_mm512_loadu_ps(prev0));
            const __m512 p1=i?_mm512_loadu_ps(prev1+i-1):
                _mm512_mask_permutexvar_ps(_mm512_set1_ps(carry[2*c+1]),0xfffe,shift,_mm512_loadu_ps(prev1));
            const __m512 even=_mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(cur0+i),p0),bb);
            const __m512 odd=_mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(cur1+i),p1),bb);
            _mm512_storeu_ps(out+2*i,_mm512_permutex2var_ps(even,front,odd));
            _mm512_storeu_ps(out+2*i+16,_mm512_permutex2var_ps(even,back,odd));
        }
#endif
#if SP_NEON
        const float32x4_t bb=vdupq_n_f32(bias[c]);
        for(;i+4<=low_n&&2*(i+4)<=high_n;i+=4){
            float32x4_t p0,p1;
            if(i){p0=vld1q_f32(prev0+i-1);p1=vld1q_f32(prev1+i-1);}
            else{
                p0=vextq_f32(vdupq_n_f32(carry[2*c]),vld1q_f32(prev0),3);
                p1=vextq_f32(vdupq_n_f32(carry[2*c+1]),vld1q_f32(prev1),3);
            }
            const float32x4x2_t result={{vaddq_f32(vaddq_f32(vld1q_f32(cur0+i),p0),bb),
                                       vaddq_f32(vaddq_f32(vld1q_f32(cur1+i),p1),bb)}};
            vst2q_f32(out+2*i,result);
        }
#endif
        for(;i<low_n;++i){
            const float p0=i?prev0[i-1]:carry[2*c];
            const float sum0=cur0[i]+p0;
            out[2*i]=sum0+bias[c];
            if(2*i+1<high_n){
                const float p1=i?prev1[i-1]:carry[2*c+1];
                const float sum1=cur1[i]+p1;
                out[2*i+1]=sum1+bias[c];
            }
        }
    }
}

struct Unit{std::vector<float> w,b,ap,rp,aq,rq,pw,pb;};
struct Kernel{
    const OrtApi& api;
    int q,segments,backend;
    bool debug;
    std::vector<float> current_w,previous_w,phase_bias;
    std::array<Unit,3> units;
    int precision_mode;
    std::shared_ptr<void> current_precision,previous_precision;
    std::array<std::shared_ptr<void>,3> precision;

    Kernel(const OrtApi& a,const OrtKernelInfo* raw,bool dbg):api(a),debug(dbg){
        Ort::ConstKernelInfo info(raw);
        auto integer=[&](const char* name){
            const auto value=info.GetAttribute<int64_t>(name);
            require(value>=0&&value<=INT32_MAX,"Invalid integer attribute");
            return static_cast<int>(value);
        };
        require(ncc_abi_version()==1&&integer("native_abi")==1,"Native ABI mismatch");
        require(integer("channels")==Channels&&integer("stride")==Stride,"Only stage4 C128 stride2 is supported");
        q=integer("tile_time");segments=integer("segments");backend=integer("backend");
        require(q==64||q==128||q==256||q==512,"Output tile must be 64,128,256,512");
        require(segments>=1&&segments<=64,"Invalid contiguous segment count");
        require(backend==2&&ncc_backend_available(2)&&(ncc_capabilities() & NCC_CAP_VFORCE),"Accurate Apple backend2 required");
        require(integer("matrix_mode")==0&&integer("matrix_isa")==128,"Apple NEON FP32 surroundings required");
        require(integer("projection_mode")==8&&integer("projection_isa")==0,"Packed SME2 INT8 projections required");
        current_w=constant(info,1,{ProjectionChannels,ProjectionChannels});
        previous_w=constant(info,2,{ProjectionChannels,ProjectionChannels});
        phase_bias=constant(info,3,{Channels});
        for(int u=0;u<3;++u){
            auto& z=units[u];size_t i=4+8*u;
            z.w=constant(info,i++,{Channels,1,7});z.b=constant(info,i++,{Channels});
            z.ap=constant(info,i++,{Channels});z.rp=constant(info,i++,{Channels});
            z.aq=constant(info,i++,{Channels});z.rq=constant(info,i++,{Channels});
            z.pw=constant(info,i++,{Channels,Channels});z.pb=constant(info,i++,{Channels});
        }
        precision_mode=integer("precision_mode");
        require(precision_mode==8,"SME2 INT8 precision mode8 required");
        current_precision=std::shared_ptr<void>(ip_create(ProjectionChannels,ProjectionChannels,current_w.data(),precision_mode,3),ip_destroy_plan);
        require(bool(current_precision),ip_last_error());
        previous_precision=std::shared_ptr<void>(ip_create(ProjectionChannels,ProjectionChannels,previous_w.data(),precision_mode,3),ip_destroy_plan);
        require(bool(previous_precision),ip_last_error());
        for(int u=0;u<3;++u){
            precision[u]=std::shared_ptr<void>(ip_create(Channels,Channels,units[u].pw.data(),precision_mode,3),ip_destroy_plan);
            require(bool(precision[u]),ip_last_error());
        }
    }
    struct Work{
        const Kernel* self;const float* x;std::array<float*,6> out;
        int64_t input_time,output_time;int parts;
        std::mutex failure_mutex;std::exception_ptr failure;
    };
    static void publish_high(const Work& work,size_t output,const float* tile,int n,
                             int64_t t,int64_t first,size_t batch){
        if(!work.out[output])return;
        const int skip=static_cast<int>(std::max<int64_t>(0,first-t));
        if(skip>=n)return;
        for(int c=0;c<Channels;++c)
            std::memcpy(work.out[output]+(batch*Channels+c)*work.output_time+t+skip,
                        tile+static_cast<size_t>(c)*n+skip,(n-skip)*sizeof(float));
    }
    static void segment(void* opaque,size_t task) noexcept {
        auto& work=*static_cast<Work*>(opaque);const auto& k=*work.self;
        try{
            const int part=static_cast<int>(task%work.parts);const size_t batch=task/work.parts;
            const int64_t first=work.output_time*part/work.parts,last=work.output_time*(part+1)/work.parts;
            const int64_t warm=(std::max<int64_t>(0,first-78)/Stride)*Stride;
            const size_t tile_size=static_cast<size_t>(Channels)*k.q;
            std::vector<float> scratch(4*tile_size);
            std::vector<float> history(Channels*78,0.0f);
            std::array<float,ProjectionChannels> carry{};
            const float* input=work.x+batch*ProjectionChannels*work.input_time;
            alignas(64) float row[54+512];
            std::unique_ptr<void,decltype(&ipc_destroy_workspace)> projections(
                ipc_create_workspace(k.current_precision.get(),k.q/Stride),ipc_destroy_workspace);
            std::unique_ptr<void,decltype(&ipc_destroy_workspace)> residual(
                ipc_create_workspace(k.precision[0].get(),k.q),ipc_destroy_workspace);
            require(bool(projections)&&bool(residual),ip_last_error());
            if(warm){
                float* seeded=scratch.data()+2*tile_size;
                require(ipc_prepare_panel(projections.get(),input,static_cast<int>(work.input_time),static_cast<int>(warm/Stride-1),1)==0,ip_last_error());
                require(ipc_run_panel(k.previous_precision.get(),projections.get(),seeded,1,0,0,ProjectionChannels)==0,ip_last_error());
                std::copy_n(seeded,ProjectionChannels,carry.begin());
            }
            for(int64_t t=warm;t<last;t+=k.q){
                const int n=static_cast<int>(std::min<int64_t>(k.q,last-t));
                const int low_n=(n+Stride-1)/Stride;const int64_t low_t=t/Stride;
                float* a=scratch.data();float* b=a+tile_size;
                float* pre=b+tile_size;float* post=pre+tile_size;
                // Direct BCT preparation avoids a separate gather-copy pass.
                // Projection outputs still fit one C128*Q scratch tile each.
                {
                    require(ipc_prepare_panel(projections.get(),input,static_cast<int>(work.input_time),static_cast<int>(low_t),low_n)==0,ip_last_error());
                    require(ipc_run_panel(k.current_precision.get(),projections.get(),b,low_n,0,0,ProjectionChannels)==0,ip_last_error());
                    require(ipc_run_panel(k.previous_precision.get(),projections.get(),pre,low_n,0,0,ProjectionChannels)==0,ip_last_error());
                }
                if(k.debug){
                    // Ownership follows phase0. Odd segment boundaries never
                    // give two workers permission to write the same column.
                    const int64_t begin=std::max(low_t,(first+1)/Stride);
                    const int64_t end=std::min(low_t+low_n,(last+1)/Stride);
                    if(begin<end)for(int c=0;c<ProjectionChannels;++c){
                        const size_t dst=(batch*ProjectionChannels+c)*work.input_time+begin;
                        const size_t src=static_cast<size_t>(c)*low_n+begin-low_t;
                        std::memcpy(work.out[0]+dst,b+src,(end-begin)*sizeof(float));
                        std::memcpy(work.out[1]+dst,pre+src,(end-begin)*sizeof(float));
                    }
                }
                phase_tile(b,pre,carry.data(),k.phase_bias.data(),a,low_n,n);
                for(int c=0;c<ProjectionChannels;++c)carry[c]=pre[static_cast<size_t>(c)*low_n+low_n-1];
                publish_high(work,2,a,n,t,first,batch);
                int history_offset=0;
                for(int u=0;u<3;++u){
                    const auto& z=k.units[u];const int d=u==0?1:u==1?3:9,halo=6*d;
                    require(ncc_snake_f32(a,z.ap.data(),z.rp.data(),pre,1,Channels,n,k.backend,1)==0,"Pre-Snake failed");
                    for(int c=0;c<Channels;++c){
                        float* h=history.data()+history_offset+static_cast<size_t>(c)*halo;
                        std::memcpy(row,h,halo*sizeof(float));
                        std::memcpy(row+halo,pre+static_cast<size_t>(c)*n,n*sizeof(float));
                        sp_dw(row,z.w.data()+7*c,z.b[c],post+static_cast<size_t>(c)*n,n,d,k.backend);
                        std::memcpy(h,row+n,halo*sizeof(float));
                    }
                    history_offset+=Channels*halo;
                    require(ncc_snake_f32(post,z.aq.data(),z.rq.data(),pre,1,Channels,n,k.backend,1)==0,"Post-Snake failed");
                    require(ipc_prepare_panel(residual.get(),pre,n,0,n)==0,ip_last_error());
                    require(ipc_run_panel(k.precision[u].get(),residual.get(),b,n,0,0,Channels)==0,ip_last_error());
                    for(int oc=0;oc<Channels;++oc)for(int it=0;it<n;++it){
                        const size_t offset=static_cast<size_t>(oc)*n+it;
                        const float dot_bias=b[offset]+z.pb[oc];
                        b[offset]=a[offset]+dot_bias;
                    }
                    std::swap(a,b);publish_high(work,3+u,a,n,t,first,batch);
                }
            }
        }catch(...){
            std::lock_guard<std::mutex> lock(work.failure_mutex);
            if(!work.failure)work.failure=std::current_exception();
        }
    }
    OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
        try{
            Ort::KernelContext context(raw);auto input=context.GetInput(0);const auto dims=shape(input);
            require(dims.size()==3&&dims[0]>=0&&dims[1]==ProjectionChannels&&dims[2]>=0&&dims[2]<=INT32_MAX/Stride,
                    "Expected FP32 [B,256,T] with bounded doubled time");
            const uint64_t limit=static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max())/sizeof(float)/ProjectionChannels;
            require(static_cast<uint64_t>(dims[0])<=limit/static_cast<uint64_t>(std::max<int64_t>(1,dims[2])),
                    "Input/output tensor size overflow");
            const int64_t high_time=Stride*dims[2];
            const std::vector<int64_t> high_shape{dims[0],Channels,high_time};
            std::array<float*,6> outputs{};
            if(debug){
                for(size_t i=0;i<2;++i)outputs[i]=context.GetOutput(i,dims).GetTensorMutableData<float>();
                for(size_t i=2;i<6;++i)outputs[i]=context.GetOutput(i,high_shape).GetTensorMutableData<float>();
            }else outputs[5]=context.GetOutput(0,high_shape).GetTensorMutableData<float>();
            if(!dims[0]||!dims[2])return nullptr;
            const int parts=static_cast<int>(std::min<int64_t>(segments,high_time));
            Work work{this,input.GetTensorData<float>(),outputs,dims[2],high_time,parts,{},{}};
            const size_t tasks=static_cast<size_t>(dims[0])*parts;
            if(tasks==1)segment(&work,0);else context.ParallelFor(segment,tasks,0,&work);
            if(work.failure)std::rethrow_exception(work.failure);
            return nullptr;
        }catch(...){return error(api);}
    }
};

template<bool debug>struct Op:Ort::CustomOpBase<Op<debug>,Kernel,true>{
    Op(){this->start_ver_=1;this->end_ver_=1;}
    const char* GetName()const{return debug?"UpsampleStageDebugF32":"UpsampleStageF32";}
    const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
    size_t GetInputTypeCount()const{return 28;}
    ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    size_t GetOutputTypeCount()const{return debug?6:1;}
    ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
    OrtStatus* CreateKernelV2(const OrtApi& a,const OrtKernelInfo* info,void** out)const noexcept{
        *out=nullptr;try{*out=new Kernel(a,info,debug);return nullptr;}catch(...){return error(a);}
    }
    static OrtStatus* InferOutputShape(Ort::ShapeInferContext& context)noexcept{
        try{
            const auto& low=context.GetInputShape(0);
            require(low.size()==3,"Upsample input must have rank three");
            require(context.GetAttrInt("channels")==Channels&&context.GetAttrInt("stride")==Stride,
                    "Only C128 stride2 output is supported");
            if(low[1].IsInt()&&low[1].AsInt()>=0)require(low[1].AsInt()==ProjectionChannels,"Wrong projection input channels");
            auto high=low;high[1]=Channels;std::string symbol;
            if(low[2].IsInt()&&low[2].AsInt()>=0){
                require(low[2].AsInt()<=INT32_MAX/Stride,"Doubled output length overflow");high[2]=low[2].AsInt()*Stride;
            }else{
                symbol=(low[2].IsInt()?"unknown_upsample_T":std::string(low[2].AsSym()))+"_times_2";
                high[2]=symbol.c_str();
            }
            if(debug){
                for(size_t i=0;i<2;++i){auto status=context.SetOutputShape(i,low);if(status)return status.release();}
                for(size_t i=2;i<6;++i){auto status=context.SetOutputShape(i,high);if(status)return status.release();}
            }else return context.SetOutputShape(0,high).release();
            return nullptr;
        }catch(...){return error(Ort::GetApi());}
    }
};
std::mutex registration;const OrtApi* registered=nullptr;
std::unique_ptr<Ort::CustomOpDomain> domain;Op<false> op;Op<true> debug_op;
}

extern "C" NCC_API OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
    const OrtApi* api=base->GetApi(ORT_API_VERSION);
    if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API29 required");
    try{
        std::lock_guard<std::mutex> lock(registration);
        require(!registered||registered==api,"Multiple ORT runtimes unsupported");
        if(!registered){
            Ort::InitApi(api);auto d=std::make_unique<Ort::CustomOpDomain>("fast.audiovae.precision.apple.r4.fused.upsample.experimental");
            d->Add(&op);d->Add(&debug_op);domain=std::move(d);registered=api;
        }
        return api->AddCustomOpDomain(options,*domain);
    }catch(...){return error(*api);}
}
