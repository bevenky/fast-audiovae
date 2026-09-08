"""Build the isolated Apple CPU INT8 port using pinned ORT29 headers."""
import argparse,hashlib,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--ort-include',type=Path,required=True)
    p.add_argument('--ort-pins',type=Path,required=True)
    p.add_argument('--cxx',default='c++');p.add_argument('--cc',default='cc');a=p.parse_args()
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise ValueError('Apple ARM build required')
    out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh output directory required')
    pins=json.loads(a.ort_pins.read_text())['ort_headers']['sha256']
    assert {n:sha(a.ort_include/n) for n in pins}==pins
    sources={str(x.relative_to(ROOT)):sha(x) for x in sorted((ROOT/'native').rglob('*'))
             if x.is_file() and (x.suffix in ('.h','.cpp','.c','.S','.json') or 'LICENSES' in x.parts)}
    sources['build.py']=sha(__file__)
    tag=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    core=out/('libapple_int8_core_'+tag+'.dylib');ops=out/('libapple_int8_ops_'+tag+'.dylib')
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    flags=['-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden',
           '-Wall','-Wextra','-Werror','-isysroot',sdk,'-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    sme=ROOT/'native/sme';upstream=sme/'upstream'
    kernel=upstream/'kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa'
    base=['-O3','-fPIC','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden','-Wall','-Wextra','-Werror',
          '-isysroot',sdk,'-I'+str(upstream)]
    objects=[out/n for n in ('wrapper.o','kernel_c.o','kernel_asm.o','common_sme.o')]
    commands=[
        [a.cxx,*base,'-std=c++17','-isystem',str(Path(sdk)/'usr/include/c++/v1'),'-c',str(sme/'sme_backend.cpp'),'-o',str(objects[0])],
        [a.cc,*base,'-std=c11','-march=armv9.2-a+sme2','-c',str(kernel)+'.c','-o',str(objects[1])],
        [a.cc,*base,'-march=armv9.2-a+sme2','-c',str(kernel)+'_asm.S','-o',str(objects[2])],
        [a.cc,*base,'-march=armv9.2-a+sme2','-c',str(upstream/'kai/kai_common_sme_asm.S'),'-o',str(objects[3])],
        [a.cxx,*flags,str(ROOT/'native/precision.cpp'),*[str(x) for x in objects],'-Wl,-install_name,@rpath/'+core.name,'-o',str(core)],
        [a.cxx,*flags,'-I'+str(a.ort_include.resolve()),str(ROOT/'native/custom_op.cpp'),str(core),
         '-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+ops.name,'-o',str(ops)]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',HIP_VISIBLE_DEVICES='-1',
             ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    out.mkdir(parents=True)
    for command in commands:subprocess.run(command,env=env,check=True)
    assert {n:sha(ROOT/n) for n in sources}==sources
    assert {n:sha(a.ort_include/n) for n in pins}==pins
    record={'sources':sources,'commands':commands,'core_path':str(core),'ops_path':str(ops),
            'libraries':{core.name:sha(core),ops.name:sha(ops)},'ORT_API':29,'GPU_used':False,
            'ort_headers':pins,'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],
            'domain':'fast.audiovae.precision.apple.r3.experimental','C_API_prefix':'ipb_',
            'backend':3,'capability_guard':'Apple SME2 plus matching streaming vector length and FPCR',
            'objects':{x.name:sha(x) for x in objects},
            'scope':'Parallel direct preparation followed by full-K SME2 INT32 products; no nested thread pools or default change'}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
