"""Compile the isolated CPU stage experiment. No installs or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT.parent/'.build/stage')
    p.add_argument('--cc',default='cc');p.add_argument('--cxx',default='c++')
    p.add_argument('--ort-include',required=True,type=Path)
    p.add_argument('--native-library',required=True,type=Path)
    p.add_argument('--native-build-manifest',type=Path,help='Optional existing native build.json for linked SLEEF/source provenance')
    p.add_argument('--native-include',type=Path,default=ROOT.parents[2]/'native/x86')
    p.add_argument('--matrix-source',type=Path,default=ROOT.parent/'matrix')
    p.add_argument('--xsmm-source',type=Path,help='Optional existing pinned source with lib/libxsmm.a; never rebuild')
    a=p.parse_args()
    destination=a.output_dir.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    mac=platform.system()=='Darwin'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',
             HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1')
    if 'LIBXSMM_TARGET' in env:raise RuntimeError('Remove LIBXSMM_TARGET override')
    inc=a.ort_include.resolve();native=a.native_library.resolve();matrix=a.matrix_source.resolve()
    if '#define ORT_API_VERSION 29' not in (inc/'onnxruntime_c_api.h').read_text():raise ValueError('ORT API29 required')
    pins=json.loads((ROOT.parent/'dependency-pins.json').read_text())['ort_headers']['sha256']
    header_hashes={name:hashlib.sha256((inc/name).read_bytes()).hexdigest() for name in pins}
    if header_hashes!=pins:raise ValueError('ORT1.29 header pin hash mismatch')
    if not native.is_file():raise ValueError('Native library missing')
    native_record=json.loads(a.native_build_manifest.read_text()) if a.native_build_manifest else None
    if native_record and native_record.get('library_sha256')!=hashlib.sha256(native.read_bytes()).hexdigest():
        raise ValueError('Native library does not match its supplied build manifest')
    common=['-O3','-fPIC','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror']
    cpp=[]
    if mac:
        sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
        common+=['-isysroot',sdk];cpp+=['-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    c=[a.cc,*common,'-std=c11','-c',str(matrix/'fused_pointwise.c'),'-o',str(destination/'matrix.o')]
    link=[a.cxx,'-dynamiclib' if mac else '-shared',str(destination/'matrix.o'),str(destination/'custom_op.o'),str(native)]
    if a.xsmm_source:
        xs=a.xsmm_source.resolve()
        if not (xs/'lib/libxsmm.a').is_file():raise ValueError('Existing static LIBXSMM archive required')
        c+=['-DFX_WITH_LIBXSMM=1','-I'+str(xs/'include')];link+=[str(xs/'lib/libxsmm.a')]
        if not mac:link+=['-ldl','-pthread']
    bridge=[a.cxx,*common,*cpp,'-std=c++17','-I'+str(inc),'-I'+str(a.native_include.resolve()),
            '-I'+str(matrix),'-c',str(ROOT/'custom_op.cpp'),'-o',str(destination/'custom_op.o')]
    output=destination/('libstage_pipeline.dylib' if mac else 'libstage_pipeline.so')
    link+=['-Wl,-rpath,'+str(native.parent),'-lm','-o',str(output)]
    commands=[c,bridge,link]
    for command in commands:subprocess.run(command,check=True,env=env)
    hashed=[ROOT/'build.py',ROOT/'custom_op.cpp',ROOT/'stage_dw.h',matrix/'fused_pointwise.c',matrix/'fused_pointwise.h',
            a.native_include.resolve()/'native_kernels.h',native,output]
    if a.xsmm_source:
        hashed.extend(a.xsmm_source.resolve()/name for name in ('lib/libxsmm.a','include/libxsmm.h','include/libxsmm_typedefs.h'))
    if a.native_build_manifest:hashed.append(a.native_build_manifest.resolve())
    report={'commands':commands,'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in hashed},
            'native_dependency':str(native),'library':str(output),'ort_api':29,'gpu_used':False,
            'kernel_or_model_execution':False,'matrix_explicit_fp32_fma':True,
            'network_fp_contraction':False,'default_promoted':False,'internal_threadpool':False,
            'native_build_manifest':native_record,'ort_header_sha256':header_hashes,
            'linked_native_includes_sine_implementation':True}
    (destination/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
