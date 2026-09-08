"""Build isolated Apple r4 fused CPU operators against explicitly supplied dependencies."""
import argparse,hashlib,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ort-include','native-include','native-library','core-library','output-dir'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--cc',default='clang');p.add_argument('--cxx',default='clang++');p.add_argument('--deployment-target',default='13.5')
    a=p.parse_args()
    if platform.system()!='Darwin' or platform.machine() not in ('arm64','aarch64'):raise RuntimeError('Apple arm64 host required')
    destination=a.output_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
    for f in destination.glob('*.dylib'):raise RuntimeError('Preserve existing build directory')
    native=a.native_library.resolve();core=a.core_library.resolve();inc=a.ort_include.resolve();ni=a.native_include.resolve()
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    flags=['-O3','-fPIC','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-isysroot',sdk,'-mmacosx-version-min='+a.deployment_target]
    cpp=[a.cxx,*flags,'-isystem',str(Path(sdk)/'usr/include/c++/v1'),'-std=c++17','-I'+str(inc),'-I'+str(ni),'-I'+str(ROOT.parent/'native'),'-I'+str(ROOT)]
    commands=[[a.cc,*flags,'-std=c11','-c',str(ROOT/'fused_pointwise.c'),'-o',str(destination/'matrix.o')]]
    for name in ('stage','upsample'):commands.append([*cpp,'-c',str(ROOT/(name+'.cpp')),'-o',str(destination/(name+'.o'))])
    libraries={}
    for name in ('stage','upsample'):
        output=destination/f'libapple_r4_{name}.dylib';libraries[name]=str(output)
        objects=[str(destination/(name+'.o'))]+([str(destination/'matrix.o')] if name=='stage' else [])
        commands.append([a.cxx,'-dynamiclib','-mmacosx-version-min='+a.deployment_target,*objects,str(native),str(core),
            '-Wl,-rpath,'+str(native.parent),'-Wl,-rpath,'+str(core.parent),'-lm','-o',str(output)])
    files=[ROOT/n for n in ('stage.cpp','upsample.cpp','stage_dw.h','fused_pointwise.c','fused_pointwise.h','build_fused.py')]
    files += [ROOT.parent/'native/precision.h',ni/'native_kernels.h',native,core]
    files += [inc/n for n in ('onnxruntime_c_api.h','onnxruntime_cxx_api.h','onnxruntime_cxx_inline.h','onnxruntime_float16.h')]
    before={str(f):sha(f) for f in files}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for command in commands:subprocess.run(command,env=env,check=True)
    if before!={str(f):sha(f) for f in files}:raise RuntimeError('Sources/dependencies changed during build')
    record={'sources_dependencies_sha256':before,'commands':commands,'libraries':libraries,'library_sha256':{k:sha(v) for k,v in libraries.items()},
        'status':'built_unvalidated','gpu_used':False,'inference_executed':False,'ort_api':29,'tile_capacity':512,'no_internal_threads':True}
    (destination/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
