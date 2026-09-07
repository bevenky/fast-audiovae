"""Build the fixed sequential CPU oneMKL adapter from existing dependencies.

Requires pinned Linux oneMKL 2026.1.0 and ORT1.29 headers. No download, install,
GPU runtime, native-library probe or model execution is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verified_files(directory,pins):
    directory=Path(directory).resolve()
    actual={}
    for name,expected in pins.items():
        path=directory/name
        if not path.is_file():raise ValueError('Required pinned dependency missing: '+str(path))
        actual[name]=sha(path)
        if actual[name]!=expected:raise ValueError('Dependency checksum mismatch: '+str(path))
    return actual


def build_plan(ort_include,mkl_include,mkl_library_dir,output_dir,compiler='g++'):
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise ValueError('This adapter requires Linux x86-64')
    includes=Path(ort_include).resolve();mkl_headers=Path(mkl_include).resolve()
    libraries=Path(mkl_library_dir).resolve();destination=Path(output_dir).resolve()
    pins=json.loads((ROOT/'dependency-pins.json').read_text())
    ort_pins=json.loads((ROOT.parent/'dependency-pins.json').read_text())['ort_headers']['sha256']
    headers=verified_files(includes,ort_pins)
    mkl_header_hashes=verified_files(mkl_headers,pins['header_sha256'])
    mkl_hashes=verified_files(libraries,pins['cpu_library_sha256'])
    source=ROOT/'intel_decoder.cpp'
    reference=json.loads((ROOT/'reference-hashes.json').read_text())
    if sha(source)!=reference['native_cpp_sha256']:
        raise ValueError('Measured native adapter source changed')
    if not compiler:raise ValueError('Compiler is required')
    library=destination/'libintel_decoder.so'
    if library.exists() or (destination/'build_decoder.json').exists():
        raise ValueError('Use a fresh output directory for the adapter build')
    linked=[libraries/name for name in pins['direct_link_libraries']]
    command=[compiler,'-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off',
             '-fvisibility=hidden','-I'+str(includes),'-I'+str(mkl_headers),str(source),
             '-Wl,--no-as-needed',*map(str,linked),'-Wl,-rpath,'+str(libraries),
             '-Wl,--disable-new-dtags','-lpthread','-lm','-ldl','-o',str(library)]
    record={'library':str(library),'command':command,
            'domain':'venky.audio.intel.decoder.experimental','operator':'IntelPlainMatMulF32',
            'sources':{'intel_decoder.cpp':sha(source),'build_decoder.py':sha(__file__)},
            'mkl_version':pins['version'],'mkl_include':str(mkl_headers),'mkl_library_dir':str(libraries),
            'mkl_header_sha256':mkl_header_hashes,'mkl_library_sha256':mkl_hashes,
            'direct_link_libraries':pins['direct_link_libraries'],'ORT_API':29,'ort_header_sha256':headers,
            'blas_threading':'sequential direct linked layer; ORT owns one or two row workers',
            'scope':'Fixed Intel VM experiment, not a generally redistributable CPU runtime bundle',
            'native_or_model_execution':False,'downloads_or_installs':False,'gpu_used':False}
    return command,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ort-include',required=True,type=Path)
    parser.add_argument('--mkl-include',required=True,type=Path)
    parser.add_argument('--mkl-library-dir',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent/'.build/mkl')
    parser.add_argument('--cxx',default='g++')
    args=parser.parse_args()
    command,record=build_plan(args.ort_include,args.mkl_include,args.mkl_library_dir,args.output_dir,args.cxx)
    destination=args.output_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
             HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run(command,check=True,env=env)
    record['sha256']=sha(record['library'])
    record['compiler']=subprocess.check_output([args.cxx,'--version'],text=True,env=env).splitlines()[0]
    # Record source/library identity again after the compiler finishes. No ldd
    # or loading of the new library is required for build provenance.
    dependencies=json.loads((ROOT/'dependency-pins.json').read_text())
    if verified_files(args.mkl_library_dir,dependencies['cpu_library_sha256'])!=record['mkl_library_sha256']:
        raise ValueError('MKL dependencies changed during the build')
    if sha(ROOT/'intel_decoder.cpp')!=record['sources']['intel_decoder.cpp']:
        raise ValueError('Adapter source changed during the build')
    (destination/'build_decoder.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
