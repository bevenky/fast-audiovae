"""Build the AOCL experiment from source and supplied pinned CPU dependencies.

This packaging adapter has offline checks only. Every fresh build is unvalidated.
No downloads, device queries, inference or performance measurements are performed.
"""
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
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expand(value, roots):
    for key, path in roots.items():
        value = value.replace('${' + key + '}', str(path))
    if '${' in value:
        raise ValueError('Unresolved recipe variable: ' + value)
    return value


def compile_commands(recipe, roots, native_library, cc='gcc', cxx='g++'):
    commands = []
    native_token = '${NATIVE_BUILD}/libfast_audiovae_x86_728052551e0b.so'
    for index, original in enumerate(recipe['object_compile_commands'] + recipe['aocl_link_commands']):
        command = [str(native_library) if item == native_token else expand(item, roots)
                   for item in original]
        command[0] = cc if index < 2 else cxx
        commands.append(command)
    return commands


def validated_inputs(recipe, roots, native_library, library_pins):
    expected = {}
    for group in ('source_header_closure', 'dependency_and_library_headers'):
        expected.update({Path(expand(name, roots)): digest for name, digest in recipe[group].items()})
    for name, digest in recipe['AOCL_core_source'].items():
        expected[Path(roots['AOCL_CORE_SOURCE']) / name] = digest
    expected[Path(roots['AOCL_ROOT']) / 'install/lib/libaocl-dlp.so'] = library_pins['aocl']
    expected[Path(roots['LIBXSMM']) / 'lib/libxsmm.a'] = library_pins['libxsmm']
    expected[Path(native_library)] = library_pins['native']
    for path, digest in expected.items():
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Expected SHA256 must be 64 lowercase hexadecimal characters')
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Source or dependency digest mismatch: ' + str(path))
    header = Path(roots['ORT_INCLUDE']) / 'onnxruntime_c_api.h'
    if '#define ORT_API_VERSION 29' not in header.read_text():
        raise ValueError('ORT API29 required')
    return {str(path): digest for path, digest in expected.items()}


def main():
    pins = json.loads((ROOT / 'pins/dependencies.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aocl-root', 'libxsmm-root', 'ort-include', 'native-library', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--aocl-library-sha256', default=pins['AOCL_library_sha256'])
    parser.add_argument('--libxsmm-library-sha256', default=pins['LIBXSMM_library_sha256'])
    parser.add_argument('--native-library-sha256', default=pins['native_library_sha256'])
    parser.add_argument('--cc', default='gcc')
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--plan-only', action='store_true', help='Verify inputs and print commands without compiling')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('Linux x86-64 build required; runtime separately requires AMD AVX512 VNNI')
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    recipe_path = ROOT / 'pins/rebuild.json'
    if sha(recipe_path) != pins['recipe_sha256']:
        raise ValueError('Frozen compilation recipe changed')
    recipe = json.loads(recipe_path.read_text())
    roots = {'RELEASE': ROOT / 'common', 'AOCL_CORE_SOURCE': ROOT / 'aocl/source',
             'AOCL_ROOT': args.aocl_root.resolve(), 'LIBXSMM': args.libxsmm_root.resolve(),
             'ORT_INCLUDE': args.ort_include.resolve(), 'NATIVE_BUILD': args.native_library.resolve().parent,
             'OBJECTS': output, 'OUTPUT': output}
    library_pins = {'aocl': args.aocl_library_sha256, 'libxsmm': args.libxsmm_library_sha256,
                    'native': args.native_library_sha256}
    inputs = validated_inputs(recipe, roots, args.native_library.resolve(), library_pins)
    inputs[str(recipe_path)] = sha(recipe_path)
    inputs[str(ROOT / 'pins/dependencies.json')] = sha(ROOT / 'pins/dependencies.json')
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    commands = compile_commands(recipe, roots, args.native_library.resolve(), args.cc, args.cxx)
    if args.plan_only:
        print(json.dumps({'status': 'inputs_verified_not_built', 'commands': commands,
                          'sha256': inputs, 'native_execution': False}, indent=2))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
               HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output.mkdir(parents=True)
    for command in commands:
        subprocess.run(command, env=env, check=True)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError('Input changed during compilation: ' + path)
    libraries = {str(output / ('libamd_aocl_precision_' + name + '.so')):
                 sha(output / ('libamd_aocl_precision_' + name + '.so'))
                 for name in ('core', 'ops', 'stage', 'upsample')}
    measured_dependencies = (library_pins == {'aocl': pins['AOCL_library_sha256'],
        'libxsmm': pins['LIBXSMM_library_sha256'], 'native': pins['native_library_sha256']})
    record = {'status': 'built_unvalidated', 'native_execution': False, 'gpu_used': False,
              'defaults_changed': False, 'commands': commands, 'sha256': inputs,
              'objects': {name + '.o': sha(output / (name + '.o')) for name in ('matrix', 'projection', 'stage', 'upsample')},
              'libraries': libraries, 'core_path': str(output / 'libamd_aocl_precision_core.so'),
              'ops_path': str(output / 'libamd_aocl_precision_ops.so'),
              'aocl_commit': pins['AOCL_commit'] if library_pins['aocl'] == pins['AOCL_library_sha256'] else None,
              'expected_aocl_source_commit': pins['AOCL_commit'],
              'aocl_library_matches_recorded_source_build': library_pins['aocl'] == pins['AOCL_library_sha256'],
              'aocl_library_sha256': library_pins['aocl'],
              'compiler': {'cc': subprocess.check_output([args.cc, '--version'], text=True).splitlines()[0],
                           'cxx': subprocess.check_output([args.cxx, '--version'], text=True).splitlines()[0]},
              'measured_dependency_digests_match': measured_dependencies,
              'binary_identity_claim': 'Actual output hashes only; changed paths, compilers or libraries require new validation',
              'dependency_origin_limit': 'Digest verification checks supplied files; it is not an attestation of how a replacement library was built'}
    (output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'libraries': libraries}, indent=2))


if __name__ == '__main__':
    main()
