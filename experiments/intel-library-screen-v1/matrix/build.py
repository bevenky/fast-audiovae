#!/usr/bin/env python3
"""Compile an isolated pair-matrix screen against an existing CPU oneDNN build."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onednn-source', required=True, type=Path)
    parser.add_argument('--onednn-build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This helper targets the Intel Linux experiment only')
    source, build, output = (path.resolve() for path in
                             (args.onednn_source, args.onednn_build, args.output))
    library = (build / 'src/libdnnl.so').resolve(strict=True)
    headers = [source / 'include/oneapi/dnnl/dnnl.hpp',
               source / 'include/oneapi/dnnl/dnnl_ukernel.hpp',
               build / 'include/oneapi/dnnl/dnnl_config.h',
               build / 'include/oneapi/dnnl/dnnl_version.h']
    output.mkdir(parents=True, exist_ok=False)
    result_library = output / 'libintel_matrix_screen.so'
    command = ['g++', '-std=c++17', '-O3', '-shared', '-fPIC', '-fno-fast-math',
               '-ffp-contract=off', '-fvisibility=hidden', '-Wall', '-Wextra',
               '-I' + str(build / 'include'), '-I' + str(source / 'include'),
               str(HERE / 'screen.cpp'), str(library), '-ldl',
               '-Wl,-z,defs', '-Wl,-rpath,' + str(library.parent), '-o', str(result_library)]
    process = subprocess.run(command, text=True, capture_output=True)
    pins = {str(path): sha(path) for path in [HERE / 'screen.cpp', HERE / 'build.py',
                                            HERE / 'run.py', library, *headers]}
    receipt = {'status': 'built' if process.returncode == 0 else 'failed',
               'command': command, 'returncode': process.returncode,
               'stdout': process.stdout, 'stderr': process.stderr, 'pins': pins,
               'library': str(result_library), 'onednn_library': str(library),
               'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
               'scope': 'CPU SEQ / GPU NONE; second projection pair only; no inference run'}
    if process.returncode == 0:
        receipt['pins'][str(result_library)] = sha(result_library)
    (output / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if process.returncode:
        raise RuntimeError('Build failed: ' + str(output / 'build.json'))
    print(json.dumps({'build': str(output / 'build.json')}))


if __name__ == '__main__':
    main()
