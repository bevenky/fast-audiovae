"""Compile the isolated Apple CPU stage experiment. No installs or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_native(record, library, header):
    """Check the linked Apple ABI and inherit its minimum supported OS."""
    if (record.get('domain'), record.get('native_abi'), record.get('ort_api_version')) != ('venky.audio.cpu', 1, 29):
        raise ValueError('Apple native ABI1 and ORT API29 required')
    if record.get('library_sha256') != sha(library):
        raise ValueError('Native library does not match its build manifest')
    fingerprint = record.get('fingerprint', {})
    if fingerprint.get('architecture') != 'arm64' or not record.get('accelerate'):
        raise ValueError('Existing Apple arm64 Accelerate build required')
    if fingerprint.get('source_sha256', {}).get('native/apple/native_kernels.h') != sha(header):
        raise ValueError('Native header does not match its build manifest')
    deployment = fingerprint.get('deployment_target', '')
    if not isinstance(deployment, str) or not re.fullmatch(r'[0-9]+\.[0-9]+(?:\.[0-9]+)?', deployment):
        raise ValueError('Native deployment target is missing or malformed')
    return deployment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT.parent/'.build/apple')
    p.add_argument('--cc', default='cc'); p.add_argument('--cxx', default='c++')
    p.add_argument('--ort-include', required=True, type=Path)
    p.add_argument('--native-library', required=True, type=Path)
    p.add_argument('--native-build-manifest', required=True, type=Path)
    p.add_argument('--native-include', required=True, type=Path)
    a = p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() not in ('arm64', 'aarch64'):
        raise RuntimeError('This builder requires Apple Silicon')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    destination = a.output_dir.resolve(); destination.mkdir(parents=True, exist_ok=True)
    inc = a.ort_include.resolve(); native = a.native_library.resolve()
    header = a.native_include.resolve()/'native_kernels.h'
    matrix = ROOT.parent/'matrix'
    pins = json.loads((ROOT.parent/'dependency-pins.json').read_text())['ort_headers']['sha256']
    header_hashes = {name: sha(inc/name) for name in pins}
    if header_hashes != pins:
        raise ValueError('ORT1.29 header pin hash mismatch')
    native_record = json.loads(a.native_build_manifest.read_text())
    deployment = validate_native(native_record, native, header)
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    common = ['-O3', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
              '-isysroot', sdk, '-mmacosx-version-min='+deployment]
    c = [a.cc, *common, '-std=c11', '-c', str(matrix/'fused_pointwise.c'), '-o', str(destination/'matrix.o')]
    bridge = [a.cxx, *common, '-isystem', str(Path(sdk)/'usr/include/c++/v1'), '-std=c++17',
              '-I'+str(inc), '-I'+str(header.parent), '-I'+str(matrix), '-c', str(ROOT/'custom_op.cpp'),
              '-o', str(destination/'custom_op.o')]
    output = destination/'libstage_pipeline.dylib'
    link = [a.cxx, '-dynamiclib', '-mmacosx-version-min='+deployment,
            str(destination/'matrix.o'), str(destination/'custom_op.o'), str(native),
            '-Wl,-rpath,'+str(native.parent), '-lm', '-o', str(output)]
    hashed = [ROOT/'build.py', ROOT/'custom_op.cpp', ROOT/'stage_dw.h', matrix/'fused_pointwise.c',
              matrix/'fused_pointwise.h', header, native, a.native_build_manifest.resolve()]
    before = {str(path): sha(path) for path in hashed}
    commands = [c, bridge, link]
    for command in commands:
        subprocess.run(command, check=True, env=env)
    if before != {str(path): sha(path) for path in hashed}:
        raise RuntimeError('A source or linked dependency changed during compilation')
    report = {'commands': commands, 'hashes': {**before, str(output): sha(output)},
              'library': str(output), 'ort_api': 29, 'ort_header_sha256': header_hashes,
              'architecture': 'arm64', 'deployment_target': deployment,
              'native_build_manifest': native_record, 'gpu_used': False,
              'kernel_or_model_execution': False, 'matrix_explicit_fp32_fma': True,
              'network_fp_contraction': False, 'default_promoted': False,
              'internal_threadpool': False, 'linked_native_provides_vforce': True}
    (destination/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
