"""Compile the isolated CPU callable kernel. No installation or model execution."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
PIN = '55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1'
HEADERS = {
    'onnxruntime_c_api.h':'acc0cf4b3f28d39339c76770d76164bb7a0637dc89f5fde764b4017b632f6743',
    'onnxruntime_cxx_api.h':'9c63ed5bf0427ddf1c10a99aca0beeec77318ce77d2ea6f7ae76d7d933660f67',
    'onnxruntime_cxx_inline.h':'c508dbb7d8203003a2584b8712118cb88e5b4ef1e1c7876a9974156fb7489301',
    'onnxruntime_ep_c_api.h':'e6c986c9e98583f8113b2c6bc3864814883b806d501cf24da4d239c45753e235',
    'onnxruntime_error_code.h':'5ce3b054e798eced8d14f5b86e98692fd33470463f96194ce0700a2d53dd8721',
    'onnxruntime_float16.h':'88b242845d25981633a0bbd1c148e273cf8bfb016ea3f57c4af41a06530f72b0',
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT.parent/'.build/matrix')
    p.add_argument('--cc', default='cc')
    p.add_argument('--cxx', default='c++')
    p.add_argument('--ort-include', type=Path, help='ORT1.29 include directory; adds the custom-op bridge')
    p.add_argument('--xsmm-source', type=Path,
                   help='Existing pinned source tree with include/ and lib/libxsmm')
    p.add_argument('--build-xsmm', action='store_true',
                   help='Build the supplied source with two jobs; never install it')
    p.add_argument('--xsmm-static', action='store_true', help='Link existing lib/libxsmm.a instead of shared library')
    a = p.parse_args()
    destination = a.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    if 'LIBXSMM_TARGET' in env:
        raise RuntimeError('Remove forced LIBXSMM_TARGET')
    is_mac = platform.system() == 'Darwin'
    out = destination / ('libfused_pointwise.dylib' if is_mac else 'libfused_pointwise.so')
    common = ['-O3','-fPIC','-fno-fast-math','-ffp-contract=off']
    cpp_extra=[]
    if is_mac:
        sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
        common+=['-isysroot',sdk]
        libcxx=Path(sdk)/'usr/include/c++/v1'
        if (libcxx/'cmath').exists():cpp_extra+=['-isystem',str(libcxx)]
    command = [a.cc,*common,'-std=c11','-c',str(ROOT/'fused_pointwise.c'),'-o',str(destination/'fused_pointwise.o')]
    link = [a.cxx if a.ort_include else a.cc, '-dynamiclib' if is_mac else '-shared',str(destination/'fused_pointwise.o')]
    commands = []
    dependency=None;header_hashes={}
    if a.xsmm_source:
        src = a.xsmm_source.resolve()
        if not (src/'include/libxsmm.h').is_file():
            raise RuntimeError('LIBXSMM header missing')
        if a.build_xsmm:
            with (destination/'xsmm-build.log').open('w') as log:
                subprocess.run(['make', '-j2', 'STATIC='+('1' if a.xsmm_static else '0'), 'lib'], cwd=src, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        command += ['-DFX_WITH_LIBXSMM=1', '-I'+str(src/'include')]
        if a.xsmm_static:
            archive=src/'lib/libxsmm.a'
            if not archive.exists(): raise RuntimeError('Static archive missing: '+str(archive))
            link += [str(archive)]
            if not is_mac: link += ['-ldl','-pthread']
        else:
            link += ['-L'+str(src/'lib'),'-lxsmm','-Wl,-rpath,'+str(src/'lib')]
            archive=src/'lib'/('libxsmm.dylib' if is_mac else 'libxsmm.so')
            if not archive.exists():raise RuntimeError('Shared LIBXSMM missing: '+str(archive))
        dependency={'linked_path':str(archive.resolve()),'linked_binary_sha256':sha(archive),
                    'linkage':'static' if a.xsmm_static else 'shared',
                    'api_header_sha256':sha(src/'include/libxsmm.h'),
                    'typedef_header_sha256':sha(src/'include/libxsmm_typedefs.h')}
    elif a.build_xsmm:
        raise RuntimeError('--build-xsmm requires --xsmm-source')
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    commands.append(command)
    if a.ort_include:
        include=a.ort_include.resolve()
        header_hashes={name:sha(include/name) for name in HEADERS}
        if header_hashes!=HEADERS:raise RuntimeError('ORT1.29 header pin hash mismatch')
        header=(include/'onnxruntime_c_api.h').read_text()
        if '#define ORT_API_VERSION 29' not in header: raise RuntimeError('Expected ORT API29 headers')
        cpp=[a.cxx,*common,*cpp_extra,'-std=c++17','-I'+str(include),'-c',str(ROOT/'custom_op.cpp'),'-o',str(destination/'custom_op.o')]
        subprocess.run(cpp,cwd=ROOT,env=env,check=True);commands.append(cpp)
        link += [str(destination/'custom_op.o')]
        if not is_mac: link += ['-pthread']
    link += ['-lm','-o',str(out)]
    subprocess.run(link,cwd=ROOT,env=env,check=True);commands.append(link)
    report = {'commands': commands, 'platform': platform.platform(),
              'compiler': subprocess.check_output([a.cc, '--version'], text=True).splitlines()[0],
              'library': str(out), 'library_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT/'fused_pointwise.c').read_bytes()).hexdigest(),
              'source_hashes': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ('fused_pointwise.c','fused_pointwise.h','custom_op.cpp','build.py')},
              'libxsmm_source': str(a.xsmm_source.resolve()) if a.xsmm_source else None,
              'expected_libxsmm_pin': PIN, 'gpu_used': False,
              'ort_bridge': bool(a.ort_include), 'xsmm_static': a.xsmm_static,
              'libxsmm_linked_dependency':dependency,'ort_header_sha256':header_hashes,
              'ort_header_source':'https://github.com/microsoft/onnxruntime/tree/v1.29.0/include/onnxruntime/core/session',
              'explicit_fma_in_dot': True, 'post_dot_add_contraction': False,
              'internal_threads': 0, 'kernel_or_model_execution': False}
    (destination/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
