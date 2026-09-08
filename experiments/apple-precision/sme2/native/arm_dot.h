// Included inside precision.cpp's private namespace after Plan and Input.
// This is an integer dot backend, not a different quantization scheme.
void pack_dot_panels(Input& x){
    const int groups=(x.k+3)/4;
    const size_t panels=(size_t(x.t)+7)/8;
    require(!panels||size_t(groups)<=SIZE_MAX/panels/32,"SDOT panel size overflow");
    x.dot_panels.assign(panels*size_t(groups)*32,0);
    for(size_t tile=0;tile<panels;++tile)for(int group=0;group<groups;++group){
        auto* out=x.dot_panels.data()+(tile*groups+group)*32;
        for(int lane=0;lane<8;++lane){
            const size_t t=tile*8+lane;if(t>=size_t(x.t))continue;
            for(int j=0;j<4;++j){const int k=group*4+j;
                if(k<x.k)out[lane*4+j]=x.x8[size_t(k)*x.t+t];
            }
        }
    }
}
#if IP_APPLE_ARM
int prepare_columns_neon(const float* values,Input& x){
    const float32x4_t zero=vdupq_n_f32(0.f),one=vdupq_n_f32(1.f);
    const float32x4_t limit=vdupq_n_f32(127.f),negative=vdupq_n_f32(-127.f);
    const float32x4_t largest=vdupq_n_f32(std::numeric_limits<float>::max());
    int begin=0;
    for(;x.t-begin>=4;begin+=4){
        float32x4_t mx=zero;
        for(int k=0;k<x.k;++k){
            const float32x4_t ab=vabsq_f32(vld1q_f32(values+size_t(k)*x.t+begin));
            require(vminvq_u32(vcleq_f32(ab,largest))==UINT32_MAX,"Nonfinite INT8 activation");
            mx=vmaxq_f32(mx,ab);
        }
        float32x4_t scale=vdivq_f32(mx,limit);
        scale=vbslq_f32(vceqq_f32(scale,zero),mx,scale);
        scale=vbslq_f32(vceqq_f32(mx,zero),one,scale);
        vst1q_f32(x.scales.data()+begin,scale);int32x4_t sum=vdupq_n_s32(0);
        for(int k=0;k<x.k;++k){
            const float32x4_t v=vld1q_f32(values+size_t(k)*x.t+begin);
            const float32x4_t q=vmaxq_f32(negative,vminq_f32(limit,vdivq_f32(v,scale)));
            const int32x4_t qi=vcvtnq_s32_f32(q);sum=vaddq_s32(sum,qi);
            const int8x8_t bytes=vmovn_s16(vcombine_s16(vmovn_s32(qi),vdup_n_s16(0)));
            const uint32_t word=vget_lane_u32(vreinterpret_u32_s8(bytes),0);
            std::memcpy(x.x8.data()+size_t(k)*x.t+begin,&word,4);
        }
        vst1q_s32(x.sums.data()+begin,sum);
    }
    return begin;
}
uint32_t signed_weight_word(const Plan& p,int row,int group){
    return p.dot_weights[size_t(row)*((p.k+3)/4)+group];
}
IP_DOT void apple_dot(const Plan& p,const Input& x,float* y,int first,int last){
    const int groups=(p.k+3)/4;
    const size_t panels=(size_t(x.t)+7)/8;
    require(x.dot_panels.size()==panels*size_t(groups)*32,"Missing SDOT input panels");
    // Four rows by eight time columns. SDOT accumulates exact signed INT32.
    // 127^2*16384=264257536, so all prefixes and complete dots fit INT32.
    for(int m=first;m<last;){
        const int rows=std::min(4,last-m);
        for(size_t tile=0;tile<panels;++tile){
            int32x4_t a0=vdupq_n_s32(0),a1=a0,b0=a0,b1=a0,c0=a0,c1=a0,d0=a0,d1=a0;
            for(int group=0;group<groups;++group){
                const auto* panel=x.dot_panels.data()+(tile*groups+group)*32;
                const int8x16_t x0=vld1q_s8(panel),x1=vld1q_s8(panel+16);
                const int8x16_t wa=vreinterpretq_s8_u32(vdupq_n_u32(signed_weight_word(p,m,group)));
                a0=vdotq_s32(a0,x0,wa);a1=vdotq_s32(a1,x1,wa);
                if(rows>1){const int8x16_t w=vreinterpretq_s8_u32(vdupq_n_u32(signed_weight_word(p,m+1,group)));
                    b0=vdotq_s32(b0,x0,w);b1=vdotq_s32(b1,x1,w);}
                if(rows>2){const int8x16_t w=vreinterpretq_s8_u32(vdupq_n_u32(signed_weight_word(p,m+2,group)));
                    c0=vdotq_s32(c0,x0,w);c1=vdotq_s32(c1,x1,w);}
                if(rows>3){const int8x16_t w=vreinterpretq_s8_u32(vdupq_n_u32(signed_weight_word(p,m+3,group)));
                    d0=vdotq_s32(d0,x0,w);d1=vdotq_s32(d1,x1,w);}
            }
            const int32x4_t acc[8]={a0,a1,b0,b1,c0,c1,d0,d1};
            const int time=int(tile*8),cols=std::min(8,x.t-time);
            for(int r=0;r<rows;++r){
                int j=0;
                for(;cols-j>=4;j+=4){
                    const float32x4_t scale=vmulq_f32(vdupq_n_f32(p.scales[m+r]),vld1q_f32(x.scales.data()+time+j));
                    const float32x4_t out=vmulq_f32(vcvtq_f32_s32(acc[2*r+j/4]),scale);
                    require(vminvq_u32(vcleq_f32(vabsq_f32(out),vdupq_n_f32(std::numeric_limits<float>::max())))==UINT32_MAX,"Nonfinite precision output");
                    vst1q_f32(y+size_t(m+r)*x.t+time+j,out);
                }
                if(j<cols){int32_t tail[4];vst1q_s32(tail,acc[2*r+j/4]);
                    for(int lane=0;j<cols;++j,++lane){
                        const float out=float(tail[lane])*(p.scales[m+r]*x.scales[time+j]);
                        require(std::isfinite(out),"Nonfinite precision output");y[size_t(m+r)*x.t+time+j]=out;
                    }
                }
            }
        }
        m+=rows;
    }
}
#endif
