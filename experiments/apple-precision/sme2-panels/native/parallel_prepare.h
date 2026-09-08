// Included in the private core namespace. One callback owns complete NR panels.
std::unique_ptr<Input> allocate_sme_input(const Plan& p,const float* values,int time,int jobs){
    require(time>=0&&jobs>=1&&jobs<=64,"Invalid parallel preparation geometry");
    require(values||!time,"Missing input");
    require(p.nr>=16&&p.nr<=256&&p.nr%16==0,"Unsupported SME panel width");
    const auto n=count(p.k,time,4);auto x=std::make_unique<Input>();
    x->k=p.k;x->t=time;x->source_time=time;x->mode=8;x->original=values;x->svl=p.svl;x->nr=p.nr;
    (void)n; // Only the packed RHS is allocated on the production path.
    const size_t bytes=ipa_sme_rhs_bytes(p.k,time,p.svl);require(!time||bytes,"Invalid SME RHS allocation");
    require(bytes%4==0,"SME RHS alignment");x->sme_rhs.resize(bytes/4);
    const size_t panels=(size_t(time)+p.nr-1)/p.nr;
    x->prep_jobs=std::min(jobs,int(std::max<size_t>(1,panels)));
    x->prep_states=std::make_unique<std::atomic<int>[]>(x->prep_jobs);
    for(int j=0;j<x->prep_jobs;++j)x->prep_states[j].store(0);
    return x;
}
#if IP_APPLE_ARM
// Four vectors remain independent time columns throughout the complete K max.
void prepare_sme_sixteen(Input& x,int time,uint8_t* panel,int lane,size_t kp){
    const float32x4_t zero=vdupq_n_f32(0),one=vdupq_n_f32(1),limit=vdupq_n_f32(127),negative=vdupq_n_f32(-127);
    const float32x4_t largest=vdupq_n_f32(std::numeric_limits<float>::max());
    float32x4_t mx0=zero,mx1=zero,mx2=zero,mx3=zero;
    for(int k=0;k<x.k;++k){
        const auto* row=x.original+size_t(k)*x.source_time+x.source_first+time;
        const float32x4_t a0=vabsq_f32(vld1q_f32(row)),a1=vabsq_f32(vld1q_f32(row+4)),
                          a2=vabsq_f32(vld1q_f32(row+8)),a3=vabsq_f32(vld1q_f32(row+12));
        const uint32x4_t ok=vandq_u32(vandq_u32(vcleq_f32(a0,largest),vcleq_f32(a1,largest)),
                                    vandq_u32(vcleq_f32(a2,largest),vcleq_f32(a3,largest)));
        require(vminvq_u32(ok)==UINT32_MAX,"Nonfinite INT8 activation");
        mx0=vmaxq_f32(mx0,a0);mx1=vmaxq_f32(mx1,a1);mx2=vmaxq_f32(mx2,a2);mx3=vmaxq_f32(mx3,a3);
    }
    auto scale=[&](float32x4_t mx){auto s=vdivq_f32(mx,limit);s=vbslq_f32(vceqq_f32(s,zero),mx,s);return vbslq_f32(vceqq_f32(mx,zero),one,s);};
    const float32x4_t s0=scale(mx0),s1=scale(mx1),s2=scale(mx2),s3=scale(mx3);
    alignas(16) float scales[16];
    vst1q_f32(scales,s0);vst1q_f32(scales+4,s1);
    vst1q_f32(scales+8,s2);vst1q_f32(scales+12,s3);
    // Metadata is copied as bytes so the packed buffer never violates C++ aliasing.
    std::memcpy(panel+kp*x.nr+x.nr*4+size_t(lane)*4,scales,64);
    for(int first=0;first<x.k;first+=4){
        int8x16x4_t bytes;
        for(int r=0;r<4;++r){
            if(first+r>=x.k){bytes.val[r]=vdupq_n_s8(0);continue;}
            const auto* row=x.original+size_t(first+r)*x.source_time+x.source_first+time;
            auto q=[&](const float* v,float32x4_t s){return vcvtnq_s32_f32(vmaxq_f32(negative,vminq_f32(limit,vdivq_f32(vld1q_f32(v),s))));};
            const int32x4_t q0=q(row,s0),q1=q(row+4,s1),q2=q(row+8,s2),q3=q(row+12,s3);
            const int16x8_t h0=vcombine_s16(vmovn_s32(q0),vmovn_s32(q1)),h1=vcombine_s16(vmovn_s32(q2),vmovn_s32(q3));
            bytes.val[r]=vcombine_s8(vmovn_s16(h0),vmovn_s16(h1));
        }
        // Store interleaved K4 words directly, without a separate repacking pass.
        vst4q_s8(reinterpret_cast<int8_t*>(panel)+size_t(first/4)*x.nr*4+size_t(lane)*4,bytes);
    }
}
#endif
void prepare_sme_job(Input& x,int job){
    require(job>=0&&job<x.prep_jobs,"Invalid preparation job");int expected=0;
    require(x.prep_states[job].compare_exchange_strong(expected,1),"Preparation job already claimed");
    try{
        require(ipa_sme_validate_runtime(x.svl)==0,ipa_sme_last_error());
        const size_t panels=(size_t(x.t)+x.nr-1)/x.nr,kp=(size_t(x.k)+31)/32*32;
        const size_t begin=panels*size_t(job)/x.prep_jobs,end=panels*size_t(job+1)/x.prep_jobs;
        for(size_t tile=begin;tile<end;++tile){
            auto* panel=reinterpret_cast<uint8_t*>(x.sme_rhs.data())+tile*x.nr*(kp+12);
            std::memset(panel,0,x.nr*(kp+12));
            const int time=int(tile*x.nr),cols=int(std::min(x.nr,size_t(x.t-time)));int lane=0;
            const uint32_t minus_zero=0x80000000u;
            for(int c=0;c<cols;++c)std::memcpy(panel+kp*x.nr+x.nr*8+size_t(c)*4,&minus_zero,4);
#if IP_APPLE_ARM
            for(;cols-lane>=16;lane+=16)prepare_sme_sixteen(x,time+lane,panel,lane,kp);
#endif
            for(;lane<cols;++lane){
                const int t=time+lane;float mx=0;
                for(int k=0;k<x.k;++k){const float v=x.original[size_t(k)*x.source_time+x.source_first+t];require(std::isfinite(v),"Nonfinite INT8 activation");mx=std::max(mx,std::fabs(v));}
                const float s=scale_of(mx);
                for(int k=0;k<x.k;++k){const int q=quantize(x.original[size_t(k)*x.source_time+x.source_first+t],s);
                    panel[size_t(k/4)*x.nr*4+size_t(lane)*4+k%4]=static_cast<uint8_t>(static_cast<int8_t>(q));}
                std::memcpy(panel+kp*x.nr+x.nr*4+size_t(lane)*4,&s,4);
            }
        }
        x.prep_states[job].store(2);x.prep_done.fetch_add(1);
    }catch(...){x.prep_states[job].store(3);throw;}
}

// Inspection and scalar fallback decode packed bytes without retaining a second
// activation representation on the production path.
const uint8_t* sme_panel(const Input& x,int col){
    const size_t kp=(size_t(x.k)+31)/32*32;
    return reinterpret_cast<const uint8_t*>(x.sme_rhs.data())+(size_t(col)/x.nr)*x.nr*(kp+12);
}
int8_t sme_value(const Input& x,int k,int col){
    const auto* block=sme_panel(x,col);const size_t lane=size_t(col)%x.nr;
    int8_t value;std::memcpy(&value,block+size_t(k/4)*x.nr*4+lane*4+k%4,1);return value;
}
float sme_scale(const Input& x,int col){
    const size_t kp=(size_t(x.k)+31)/32*32,lane=size_t(col)%x.nr;
    float value;std::memcpy(&value,sme_panel(x,col)+kp*x.nr+x.nr*4+lane*4,4);return value;
}
void copy_sme_panel(const Input& x,int8_t* q,float* scales,int32_t* sums){
    for(int t=0;t<x.t;++t){
        scales[t]=sme_scale(x,t);int32_t sum=0;
        for(int k=0;k<x.k;++k){const int8_t value=sme_value(x,k,t);q[size_t(k)*x.t+t]=value;sum+=value;}
        sums[t]=sum;
    }
}
void inspect_sme_input(const Input& x){
    std::lock_guard<std::mutex> lock(x.inspect_mutex);
    if(x.inspection_ready)return;
    x.x8.resize(count(x.k,x.t,1));x.scales.resize(x.t);x.sums.resize(x.t);
    copy_sme_panel(x,x.x8.data(),x.scales.data(),x.sums.data());
    x.inspection_ready=true;
}
void scalar_sme(const Plan& p,const Input& x,float* y,int first,int last){
    for(int row=first;row<last;++row)for(int t=0;t<x.t;++t){
        int32_t dot=0;
        for(int k=0;k<p.k;++k)dot+=(int(p.w8[size_t(row)*p.k+k])-128)*int(sme_value(x,k,t));
        const float scale=p.scales[row]*sme_scale(x,t);
        const float value=float(dot)*scale;require(std::isfinite(value),"Nonfinite precision output");
        y[size_t(row)*x.t+t]=value;
    }
}
