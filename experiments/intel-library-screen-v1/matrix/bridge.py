#!/usr/bin/env python3
"""Build an optional bridge and replace only the verified second projection pair."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
SOURCE_SHA = 'ada257599e8f40c57203fa9dbcfd3954476f34807a7328fe0511b49641b61a40'
DOMAIN = 'fast.audiovae.intel.library.screen'
OUTPUTS = ('ncc_up_2_split_matmul_current', 'ncc_up_2_split_matmul_previous_unshifted')
WEIGHTS = ('ncc_up_2_split_matmul_current_w', 'ncc_up_2_split_matmul_previous_w')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rewrite(original, mode):
    if mode not in (2, 3, 4):
        raise ValueError('Select a oneDNN candidate mode')
    if any(item.domain == DOMAIN for item in original.opset_import):
        raise ValueError('Matrix screen is already present')
    if original.functions or original.graph.sparse_initializer:
        raise ValueError('Flat dense graph required')
    weights = {weight.name: weight for weight in original.graph.initializer}
    selected = []
    for output, weight in zip(OUTPUTS, WEIGHTS):
        found = [node for node in original.graph.node if list(node.output) == [output]]
        if len(found) != 1:
            raise ValueError('Expected a single original second-pair output: ' + output)
        node = found[0]
        attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        if (node.op_type != 'PrecisionMatMulF32' or list(node.input) != [weight, 'view_15']
                or attrs.get('M') != 3072 or attrs.get('K') != 1024
                or attrs.get('precision_mode') != 8 or attrs.get('backend') != 1 or attrs.get('native_abi') != 1):
            raise ValueError('Original matrix contract changed: ' + node.name)
        if (weight not in weights or list(weights[weight].dims) != [3072, 1024]
                or weights[weight].data_type != onnx.TensorProto.FLOAT):
            raise ValueError('Original weight geometry changed')
        selected.append(node)
    if selected[0].domain != selected[1].domain:
        raise ValueError('Original matrix domains differ')
    selected_names = {node.name for node in selected}
    if len(selected_names) != 2:
        raise ValueError('Distinct node names required')
    graph = copy.deepcopy(original)
    replacement = onnx.helper.make_node('SecondProjectionPairF32', [*WEIGHTS, 'view_15'], list(OUTPUTS),
            name='intel_second_projection_pair_screen', domain=DOMAIN, native_abi=1, M=3072, K=1024, mode=mode)
    nodes, inserted = [], False
    for node in graph.graph.node:
        if node.name in selected_names:
            if not inserted:
                nodes.append(replacement)
                inserted = True
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    graph.opset_import.append(onnx.helper.make_opsetid(DOMAIN, 1))
    checks = {field + '_byte_identical': [value.SerializeToString() for value in getattr(original.graph, field)]
              == [value.SerializeToString() for value in getattr(graph.graph, field)]
              for field in ('initializer', 'input', 'output')}
    checks['remaining_nodes_byte_identical'] = (
        [node.SerializeToString() for node in original.graph.node if node.name not in selected_names]
        == [node.SerializeToString() for node in graph.graph.node if node.name != replacement.name])
    if not all(checks.values()):
        raise RuntimeError('An unselected payload changed')
    return graph, {'checks': checks, 'removed_nodes': sorted(selected_names), 'inserted_node': replacement.name,
                   'mode': mode, 'scope': 'Only two second-pair matrix nodes; existing first VNNI pair is unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--matrix-build', type=Path, required=True)
    parser.add_argument('--ort-include', type=Path, required=True)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--mode', type=int, choices=(2, 3, 4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Intel Linux experiment required')
    source, output = args.source.resolve(), args.output.resolve()
    if sha(source) != SOURCE_SHA:
        raise ValueError('Expected the combined Intel baseline graph')
    if output.exists():
        raise FileExistsError('Use a fresh bridge/graph output directory')
    build = json.loads(args.matrix_build.read_text())
    if build['status'] != 'built':
        raise ValueError('Matrix library did not build')
    for file, expected in build['pins'].items():
        if sha(file) != expected:
            raise ValueError('Matrix build input changed: ' + file)
    matrix_library = Path(build['library']).resolve()
    source_manifest = json.loads((source.parent / 'manifest.json').read_text())
    shared = source_manifest['shared_weights']
    weight_file = (source.parent / shared['file']).resolve()
    if sha(weight_file) != shared['sha256']:
        raise ValueError('Baseline external weights changed')
    original = onnx.load(source, load_external_data=False)
    candidate, audit = rewrite(original, args.mode)
    locations = {dict((item.key, item.value) for item in weight.external_data)['location']
                 for weight in original.graph.initializer if weight.data_location == onnx.TensorProto.EXTERNAL}
    if locations != {shared['file']} or Path(shared['file']).name != shared['file']:
        raise ValueError('Expected one simple relative weights file')
    include = args.ort_include.resolve()
    header = include / 'onnxruntime_c_api.h'
    if '#define ORT_API_VERSION 29' not in header.read_text():
        raise ValueError('ORT1.29 headers required')
    output.mkdir(parents=True)
    library = output / 'libintel_matrix_screen_ort.so'
    command = ['g++', '-std=c++17', '-shared', '-O3', '-fPIC', '-fno-fast-math', '-ffp-contract=off',
               '-fvisibility=hidden', '-pthread', '-I' + str(include), str(HERE / 'bridge.cpp'),
               str(matrix_library), str(args.core.resolve()), '-Wl,-z,defs',
               '-Wl,-rpath,' + str(matrix_library.parent) + ':' + str(args.core.resolve().parent), '-o', str(library)]
    process = subprocess.run(command, text=True, capture_output=True)
    receipt = {'status': 'built' if process.returncode == 0 else 'failed', 'command': command,
               'stdout': process.stdout, 'stderr': process.stderr, 'matrix_library': str(matrix_library),
               'matrix_build': str(args.matrix_build.resolve()), 'matrix_build_sha256': sha(args.matrix_build),
               'source_sha256': sha(source), 'bridge_source_sha256': sha(HERE / 'bridge.cpp'),
               'fallback_core_sha256': sha(args.core),
               'fallback_core': str(args.core.resolve()),
               'header_sha256': sha(header), 'mode': args.mode}
    (output / 'bridge-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if process.returncode:
        raise RuntimeError('Bridge compile failed; see bridge-build.json')
    # Share immutable external coefficients; no audio, weight copying or edits.
    # ORT confines external-data resolution to the graph directory. A hardlink
    # shares immutable bytes on this filesystem without escaping that boundary.
    (output / shared['file']).hardlink_to(weight_file)
    (output / 'baseline.onnx').write_bytes(source.read_bytes())
    (output / 'matrix.onnx').write_bytes(candidate.SerializeToString())
    onnx.checker.check_model(str(output / 'matrix.onnx'), check_custom_domain=False)
    manifest = {'baseline': {'model': 'baseline.onnx', 'sha256': sha(output / 'baseline.onnx')},
                'variants': {'matrix': {'model': 'matrix.onnx', 'sha256': sha(output / 'matrix.onnx'), **audit}},
                'shared_weights': shared,
                'additional_library': {'path': str(library), 'sha256': sha(library)},
                'scope': 'Experimental one-thread 40/80ms only; no automatic serving selection or promotion'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'graph': str(output / 'matrix.onnx'), 'library': str(library), 'audit': audit}))


if __name__ == '__main__':
    main()
