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
    p.add_argument('--cxx',default='c++');a=p.parse_args()
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise ValueError('Apple ARM build required')
    out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh output directory required')
    pins=json.loads(a.ort_pins.read_text())['ort_headers']['sha256']
    assert {n:sha(a.ort_include/n) for n in pins}==pins
    sources={str(x.relative_to(ROOT)):sha(x) for x in sorted((ROOT/'native').glob('*')) if x.is_file()}
    sources['build.py']=sha(__file__)
    tag=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
    core=out/('libapple_int8_core_'+tag+'.dylib');ops=out/('libapple_int8_ops_'+tag+'.dylib')
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    flags=['-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden',
           '-Wall','-Wextra','-Werror','-isysroot',sdk,'-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    commands=[
        [a.cxx,*flags,str(ROOT/'native/precision.cpp'),'-Wl,-install_name,@rpath/'+core.name,'-o',str(core)],
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
            'domain':'fast.audiovae.precision.apple.experimental','C_API_prefix':'ipa_',
            'backend':2,'capability_guard':'hw.optional.arm.FEAT_DotProd == 1',
            'scope':'Standalone Apple W8A8 matrix experiment; full-K INT32, FP32 output; defaults unchanged'}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
