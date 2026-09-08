"""Derive one AMD-only integer GEMM backend without changing quantization or FP32 epilogues."""
import argparse,hashlib,json
from pathlib import Path
PIN='95eeddad1153679fc0582896b965ad088100f7d1c0abc43b33587421adf4b9e3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert sha(a.source/'precision.cpp')==PIN
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'precision.cpp').exists():raise ValueError('Fresh derivative required')
    s=(a.source/'precision.cpp').read_text()
    anchor='#if defined(IP_WITH_MKL)\n#include "mkl_cblas.h"'
    assert s.count(anchor)==1
    s=s.replace(anchor,'#if defined(IP_WITH_AOCL)\nextern "C" {\n#include "classic/aocl_gemm_interface_apis.h"\n#include "classic/aocl_lib_interface_apis.h"\n#include "classic/dlp_errors.h"\n}\n#endif\n'+anchor)
    anchor='#if defined(IP_WITH_MKL)\n    return cpu|(mkl_get_max_threads()==1?1:0);'
    assert s.count(anchor)==1
    s=s.replace(anchor,'#if defined(IP_WITH_AOCL)\n    return cpu|1; // compiled sequential AOCL backend; checked on every call\n#elif defined(IP_WITH_MKL)\n    return cpu|(mkl_get_max_threads()==1?1:0);')
    start=s.index('    constexpr int MR=2048,NT=512;');end=s.index('\n}\n#endif\n}\nextern "C"',start)
    body=s[start:end].replace('MKL_INT32','int32_t').replace('    const int32_t zero=0;\n','')
    old='''            cblas_gemm_s8u8s32(CblasRowMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,
                rows,cols,p.k,1.f,p.w8.data()+size_t(m)*p.k,p.k,0,
                x.x8.data()+t,x.t,0,0.f,product.data(),cols,&zero);'''
    new='''            dlp_metadata_t metadata{};
            aocl_gemm_u8s8s32os32('R','N','N',rows,cols,p.k,1,
                p.w8.data()+size_t(m)*p.k,p.k,'N',x.x8.data()+t,x.t,'N',
                0,product.data(),cols,&metadata);
            require(metadata.error_hndl.error_code==DLP_CLSC_SUCCESS,"AOCL integer GEMM failed");'''
    assert body.count(old)==1;body=body.replace(old,new)
    addition='''
#if defined(IP_WITH_AOCL)
bool cpu_amd(){
#if IP_X86
    unsigned a,b,c,d;if(!__get_cpuid(0,&a,&b,&c,&d))return false;
    char vendor[13]={0};std::memcpy(vendor,&b,4);std::memcpy(vendor+4,&d,4);std::memcpy(vendor+8,&c,4);
    return std::strcmp(vendor,"AuthenticAMD")==0;
#else
    return false;
#endif
}
void aocl(const Plan& p,const Input& x,float* y,int first,int last){
    require(p.mode==8,"AOCL port supports INT8 only");
    // These are C TLS settings for this ORT callback, not a global thread pool.
    // Reject higher-priority ways-based settings that could create extra workers.
    dlp_thread_set_num_threads_local(1);
    require(dlp_thread_get_num_threads_active()==1&&dlp_thread_get_ic_ways_active()<=1&&
            dlp_thread_get_jc_ways_active()<=1,"AOCL must use exactly one inner worker");
'''+body+'''
}
#endif
'''
    anchor='\n}\nextern "C" {';assert s.count(anchor)==1;s=s.replace(anchor,addition+anchor)
    anchor='        require(backend==0||mode!=8||(capabilities()&2),"W8A8 oneMKL requires CPU and OS AVX512 VNNI");'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'''
#if defined(IP_WITH_AOCL)
        require(backend==0||(mode==8&&cpu_amd()),"AOCL backend requires AMD CPU and INT8 mode8");
        const char* forced=std::getenv("AOCL_DLP_ENABLE_INSTRUCTIONS");
        require(!forced||!*forced,"AOCL instruction overrides are not accepted");
#endif''')
    anchor='#if defined(IP_WITH_MKL)\n        else mkl(p,x,y,first,last);'
    assert s.count(anchor)==1;s=s.replace(anchor,'#if defined(IP_WITH_AOCL)\n        else aocl(p,x,y,first,last);\n#elif defined(IP_WITH_MKL)\n        else mkl(p,x,y,first,last);')
    s=s.replace('Backend must be scalar0 or oneMKL1','Backend must be scalar0 or compiled CPU GEMM1').replace('Sequential oneMKL backend unavailable','Compiled CPU GEMM backend unavailable').replace('W8A8 oneMKL requires CPU and OS AVX512 VNNI','W8A8 native backend requires CPU and OS AVX512 VNNI')
    (out/'precision.cpp').write_text(s)
    for name in ('precision.h','custom_op.cpp'):(out/name).write_bytes((a.source/name).read_bytes())
    r={'parent_precision_sha256':PIN,'sources':{n:sha(out/n) for n in ('precision.cpp','precision.h','custom_op.cpp')},'backend':'AMD AOCL-DLP direct U8xS8->S32, row-major, N/N, alpha1 beta0, full K','unchanged':['weight quantization','activation per-time-column quantization and scales','MR2048 NT512 geometry','128*sum activation compensation','ordered FP32 dequantization','all fused-stage arithmetic'],'threads':'Each ORT row callback requests one AOCL TLS thread and rejects higher-priority ways above one; ORT owns outer parallelism.','gpu_used':False,'native_execution':False,'script_sha256':sha(Path(__file__))}
    (out/'derivation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
