#!/usr/bin/env python3
"""Build a timing-only Intel preload shim; never run model inference here."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = Path('/dev/shm/fast-audiovae-projection-candidate-20260908-r1')
GRAPH_SHA = '59a937ee76d494896561b994c111b708603fb2f3c8e0fdf53df1380f2df41516'
LATENTS = Path('/var/tmp/fast-audiovae-20260907/assets/multilingual/fast_audiovae2.npz')
LATENTS_SHA = '9b709b787c040591de57e094a5ba5a4515cc74e256e8014530ff8c7bb54d2d24'
LIBRARIES = {
    'libfast_audiovae_x86_8eed60d94b7f.so': '0cb2b82857e5c0c9c2054000b19c670f02b25c92c54d7225422a9b0f7d99a7c2',
    'libstage_pipeline.so': '33ed06712852546af72d209904e135fe151ffedc545fa392fe9b42bbaaa069e5',
    'precision_ops.so': '41209af17f5f3e5ac1173bcfa32c9e014e5f6257fd8ad7e446a9d7872d196858',
    'libprecision_stage.so': '04a19020e7c9a763a357b528bef8d8b29f0ec3fffa96a52f2971e612012df7e5',
    'libprecision_upsample.so': '066e31b1a04f25a6a9ff7db21ab033081b62accab2f1d74ad20d8bd853a739eb',
    'libpaired_projection.so': '5f8f40e62ca3d2357290fba4630fc4410ac26d9ce68f030b30ed498283287f58',
}
CORE = Path('/var/tmp/fast-audiovae-intel-precision/native-large-build-r1/libintel_precision_core.so')
MKL = Path('/var/tmp/fast-audiovae-20260907/mkl_candidate/dependency/lib/libmkl_intel_lp64.so.3')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mkl_include(explicit):
    if explicit:
        root = Path(explicit).resolve()
        if not (root / 'mkl_cblas.h').is_file():
            raise ValueError('--mkl-include must contain mkl_cblas.h')
        return root
    candidates = sorted({p.parent.resolve() for p in
                         Path('/var/tmp/fast-audiovae-intel-precision/mkl').rglob('mkl_cblas.h')})
    if len(candidates) != 1:
        raise ValueError('Pass --mkl-include: expected exactly one existing mkl_cblas.h, found '
                         + str(len(candidates)))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mkl-include', type=Path)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--core', type=Path, default=CORE)
    parser.add_argument('--mkl-library', type=Path, default=MKL)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Intel Linux x86-64 build required')
    if 'GenuineIntel' not in Path('/proc/cpuinfo').read_text():
        raise RuntimeError('This diagnostic is for the verified Intel host')
    include = mkl_include(args.mkl_include)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = args.bundle.resolve() / 'streaming/decoder_1.onnx'
    libraries = [args.bundle.resolve() / 'libs' / name for name in LIBRARIES]
    expected = {str(graph): GRAPH_SHA, str(LATENTS): LATENTS_SHA,
                **{str(path): LIBRARIES[path.name] for path in libraries}}
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError('Verified Intel baseline changed: ' + path)
    # These dependency owners were present in the live process map. Pin the exact
    # files used for this diagnostic rather than inventing earlier hash evidence.
    resolvers = {'IP_PROBE_CORE_LIBRARY': str(args.core.resolve()),
                 'IP_PROBE_GEMM_LIBRARY': str(args.mkl_library.resolve()),
                 'IP_PROBE_SNAKE_LIBRARY': str(libraries[0]),
                 'IP_PROBE_FX_LIBRARY': str(libraries[1])}
    source_names = ('build.py', 'run.py', 'precision_probe.cpp', 'precision.h',
                    'native_kernels.h', 'fused_pointwise.h')
    pins = {**expected, **{str(HERE / name): sha(HERE / name) for name in source_names},
            **{path: sha(path) for path in resolvers.values()}}
    for name in ('mkl_cblas.h', 'mkl_types.h'):
        pins[str(include / name)] = sha(include / name)
    library = output / 'libintel_stage_attribution.so'
    command = ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', '-fno-fast-math',
               '-ffp-contract=off', '-fvisibility=hidden', '-I' + str(HERE),
               '-I' + str(include), str(HERE / 'precision_probe.cpp'),
               '-Wl,-z,defs', '-ldl', '-pthread', '-o', str(library)]
    result = subprocess.run(command, text=True, capture_output=True)
    receipt = {'version': 1, 'status': 'built' if result.returncode == 0 else 'failed',
               'command': command, 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr, 'ort': '1.29.0',
               'graph': str(graph), 'latents': str(LATENTS), 'libraries': list(map(str, libraries)),
               'resolvers': resolvers, 'pins': pins, 'shim': str(library),
               'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
               'scope': 'Attribution only; unchanged model kernels and frozen tensors. No inference run.'}
    if result.returncode == 0:
        receipt['pins'][str(library)] = sha(library)
    (output / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('Shim build failed; see ' + str(output / 'build.json'))
    print(json.dumps({'build': str(output / 'build.json'), 'shim': str(library)}))


if __name__ == '__main__':
    main()
