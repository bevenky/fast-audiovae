"""Build isolated CPU precision core and optional ORT29 bridge from pinned dependencies.

No downloads, device queries, native execution, model inference or benchmarks.
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
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify(directory,pins):
    actual={name:sha(Path(directory)/name) for name in pins}
    if actual!=pins:raise ValueError('Dependency hash mismatch: '+str(directory))
    return actual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mkl-include',type=Path)
    p.add_argument('--mkl-library-dir',type=Path)
    p.add_argument('--mkl-pins',type=Path)
    p.add_argument('--ort-include',type=Path)
    p.add_argument('--ort-pins',type=Path)
    p.add_argument('--without-mkl',action='store_true')
    p.add_argument('--cxx',default='c++')
    a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise ValueError('Use a fresh build output directory')
    suffix='.dylib' if platform.system()=='Darwin' else '.so'
    core=out/('libintel_precision_core'+suffix);ops=out/('libintel_precision_ops'+suffix)
    records={'scope':'Larger ordinary-W8A8 GEMM and fixed64 FP16-rounded operand candidate',
             'integer_geometry':'MR2048/NT512 ordinary GEMM, <=4MiB INT32 scratch per callback',
             'half_compute':'64x64 aligned padded FP32 SGEMM; half-rounded operands, FP32 accumulation/output',
             'GPU_used':False,'native_execution':False,'ORT_API':29,
             'sources':{name:sha(ROOT/name) for name in ('precision.h','precision.cpp','custom_op.cpp','build.py','check_micro.py','check_candidate.py','check_half_fixed.py')}}
    flags=['-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden']
    if platform.system()=='Darwin':
        sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
        flags+=['-isysroot',sdk,'-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    mkl_flags=[]
    if not a.without_mkl:
        if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
            raise ValueError('Pinned MKL experiment requires Linux x86-64')
        if not all((a.mkl_include,a.mkl_library_dir,a.mkl_pins)):raise ValueError('Explicit MKL headers, library directory and pins required')
        pins=json.loads(a.mkl_pins.read_text())
        if pins['version']!='2026.1.0':raise ValueError('Expected pinned oneMKL2026.1.0')
        records['mkl_header_sha256']=verify(a.mkl_include,pins['header_sha256'])
        records['mkl_library_sha256']=verify(a.mkl_library_dir,pins['cpu_library_sha256'])
        records['mkl_pins_sha256']=sha(a.mkl_pins)
        mkl_flags=['-DIP_WITH_MKL=1','-I'+str(a.mkl_include.resolve()),'-Wl,--no-as-needed',
                   *[str((a.mkl_library_dir/name).resolve()) for name in pins['direct_link_libraries']],
                   '-Wl,-rpath,'+str(a.mkl_library_dir.resolve()),'-Wl,--disable-new-dtags','-lpthread','-lm','-ldl']
    if a.ort_include:
        if not a.ort_pins:raise ValueError('ORT pinned-header manifest required')
        raw=json.loads(a.ort_pins.read_text());opins=raw.get('ort_headers',raw)
        opins=opins.get('sha256',opins)
        records['ort_headers']=verify(a.ort_include,opins)
        if '#define ORT_API_VERSION 29' not in (a.ort_include/'onnxruntime_c_api.h').read_text():
            raise ValueError('ORT API29 required')
    out.mkdir(parents=True)
    commands=[[a.cxx,*flags,str(ROOT/'precision.cpp'),*mkl_flags,'-o',str(core)]]
    if a.ort_include:
        link=['-Wl,-rpath,@loader_path'] if platform.system()=='Darwin' else ['-Wl,-rpath,$ORIGIN','-Wl,-Bsymbolic-functions']
        commands.append([a.cxx,*flags,'-I'+str(a.ort_include.resolve()),str(ROOT/'custom_op.cpp'),str(core),*link,'-o',str(ops)])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for cmd in commands:subprocess.run(cmd,check=True,env=env)
    for name,expected in records['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source changed during build: '+name)
    if not a.without_mkl:
        verify(a.mkl_include,pins['header_sha256']);verify(a.mkl_library_dir,pins['cpu_library_sha256'])
    if a.ort_include:verify(a.ort_include,opins)
    records.update(commands=commands,compiler=subprocess.check_output([a.cxx,'--version'],text=True,env=env).splitlines()[0],
                   libraries={core.name:sha(core)},core_path=str(core),ops_path=str(ops) if a.ort_include else None)
    if a.ort_include:records['libraries'][ops.name]=sha(ops)
    (out/'build.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))

if __name__=='__main__':main()
