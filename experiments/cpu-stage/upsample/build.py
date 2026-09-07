"""Build the isolated Intel upsample/stage experiment; no model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT.parent / '.build/upsample')
    p.add_argument('--ort-include', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--native-build-manifest', type=Path, required=True)
    p.add_argument('--native-include', type=Path, default=ROOT.parents[2] / 'native/x86')
    p.add_argument('--xsmm-source', type=Path)
    p.add_argument('--cc', default='gcc')
    p.add_argument('--cxx', default='g++')
    a = p.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('This experimental builder targets Linux x86 only')
    if os.environ.get('LIBXSMM_TARGET') is not None:
        raise ValueError('Remove LIBXSMM_TARGET before building')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    inc, native = a.ort_include.resolve(), a.native_library.resolve()
    pins = json.loads((ROOT.parent / 'dependency-pins.json').read_text())['ort_headers']['sha256']
    if {name: sha(inc / name) for name in pins} != pins:
        raise ValueError('ORT 1.29 header pins mismatch')
    manifest = json.loads(a.native_build_manifest.read_text())
    if manifest['library_sha256'] != sha(native):
        raise ValueError('Linked native library does not match its build manifest')
    dest = a.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / 'libupsample_stage.so'
    if output.exists():
        raise ValueError('Use a new build directory to preserve previous evidence')
    matrix = ROOT.parent / 'matrix'
    stage = ROOT.parent / 'stage'
    common = ['-O3', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']
    defs, extra = [], []
    if a.xsmm_source:
        xs = a.xsmm_source.resolve()
        if not (xs / 'lib/libxsmm.a').is_file():
            raise ValueError('Existing pinned LIBXSMM static archive required')
        defs = ['-DUP_WITH_LIBXSMM=1', '-DFX_WITH_LIBXSMM=1', '-I' + str(xs / 'include')]
        extra = [str(xs / 'lib/libxsmm.a'), '-ldl', '-pthread']
    commands = [
        [a.cc, *common, '-std=c11', *defs, '-c', str(ROOT / 'projection.c'), '-o', str(dest / 'projection.o')],
        [a.cc, *common, '-std=c11', *defs, '-c', str(matrix / 'fused_pointwise.c'), '-o', str(dest / 'matrix.o')],
        [a.cxx, *common, '-std=c++17', '-I' + str(inc), '-I' + str(a.native_include.resolve()),
         '-I' + str(matrix), '-I' + str(stage), '-c', str(ROOT / 'custom_op.cpp'), '-o', str(dest / 'custom_op.o')],
        [a.cxx, '-shared', str(dest / 'projection.o'), str(dest / 'matrix.o'), str(dest / 'custom_op.o'),
         str(native), *extra, '-Wl,-rpath,' + str(native.parent), '-lm', '-o', str(output)],
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    for command in commands:
        subprocess.run(command, check=True, env=env)
    files = [ROOT / name for name in ('build.py', 'projection.c', 'projection.h', 'custom_op.cpp')]
    files += [matrix / 'fused_pointwise.c', matrix / 'fused_pointwise.h', stage / 'stage_dw.h',
              a.native_include.resolve() / 'native_kernels.h', a.native_build_manifest.resolve(), native, output]
    if a.xsmm_source:
        files += [a.xsmm_source.resolve() / n for n in ('lib/libxsmm.a', 'include/libxsmm.h', 'include/libxsmm_typedefs.h')]
    report = dict(commands=commands, hashes={str(f): sha(f) for f in files}, library=str(output),
                  library_sha256=sha(output), native_dependency=str(native),
                  native_build_manifest=manifest, ort_header_sha256=pins, ort_api=29,
                  gpu_used=False, model_execution=False, network_fp_contraction=False,
                  complete_k_fp32=True, default_promoted=False, internal_threadpool=False)
    (dest / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'library': str(output), 'sha256': sha(output), 'gpu_used': False}))


if __name__ == '__main__':
    main()
