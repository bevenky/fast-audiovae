"""Pin and relocate a four-model Apple experiment config without model execution.

Models, weights and corpus files are user-supplied and remain in place. Paths in
the new config are relative to its destination. Importing this module performs
no I/O or numerical imports. The CLI reuses the pinned public config validator,
including its static ONNX external-tensor inventory.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_SHA = '593dc1df8b7ac7334add9211da869eac931aaf61d360656b15a3196467cecf01'
COMMON_SHA = '7f42ba2e2ae9f1c9b311e2b3b90eb1f3f6e911f0934c25b32570f36f60efb202'
SOURCE_GRAPH_SHA = 'fa7992825e807cac7ab1be405912ae3735031fb941a18dd81006005dc597bde3'
MODELS = ('audio_stock', 'fast_fp32', 'int8_large', 'mimi')
DOMAINS = ('fast.audiovae.precision.apple.r4.experimental',
           'fast.audiovae.precision.apple.r4.fused.stage.experimental',
           'fast.audiovae.precision.apple.r4.fused.upsample.experimental')
CONTRACTS = {'audio': {'sample_rate': 48000, 'hop': 1920, 'latent_channels': 64},
             'mimi': {'sample_rate': 24000, 'hop': 1920, 'latent_channels': 32}}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, expected):
    require(sha(path) == expected, 'Input JSON hash mismatch: ' + str(path))
    result = json.loads(Path(path).read_text())
    require(isinstance(result, dict), 'Expected a JSON object')
    return result


def libraries(model):
    require(not (model.get('custom_library') and model.get('custom_libraries')), 'Ambiguous library declaration')
    return model.get('custom_libraries', []) or ([model['custom_library']] if model.get('custom_library') else [])


def model_identity(model, base, helper):
    value = copy.deepcopy(model)
    value['path'] = sha(helper.relative_path(base, value['path']))
    value['custom_libraries'] = [sha(helper.relative_path(base, p)) for p in libraries(model)]
    value.pop('custom_library', None)
    value['external_data'] = [sha(helper.relative_path(base, p)) for p in model.get('external_data', [])]
    return value


def verify_cohort(source, reference, source_base, reference_base, helper):
    for config in (source, reference):
        require(tuple(m['name'] for m in config['models']) == MODELS, 'Frozen four-model order required')
        require(config['expected_case_count'] == 60 and len(config['expected_uids']) == 60,
                'Frozen 60-clip validation cohort required')
        require(len(set(config['expected_uids'])) == 60 and len(config['timing_uids']) == 10
                and len(set(config['timing_uids'])) == 10
                and set(config['timing_uids']) <= set(config['expected_uids']), 'Invalid frozen timing subset')
        require(config['kind_contracts'] == CONTRACTS, 'Codec interface changed')
        require([m['kind'] for m in config['models']] == ['audio', 'audio', 'audio', 'mimi'], 'Model family changed')
        require(all(m.get('causal') is True and m.get('backend', 'ort') == 'ort' for m in config['models']),
                'Causal ORT models required')
        require([m['name'] for m in config['models'] if m.get('approximate')] == ['int8_large'],
                'Only selective INT8 may declare approximation')
        require(config['models'][2].get('approximate') is True, 'INT8 approximation must be explicit')
        require(not config.get('custom_library'), 'Use explicit per-model native libraries')
    for field in ('expected_uids', 'timing_uids', 'kind_contracts', 'frozen_manifest_sha256'):
        require(source[field] == reference[field], 'Reference cohort/contract changed: ' + field)
    for kind in ('audio', 'mimi'):
        for suffix in (None, '.json'):
            paths = [helper.relative_path(base, config[kind + '_cases'])
                     for config, base in ((source, source_base), (reference, reference_base))]
            if suffix:
                paths = [p.with_suffix(suffix) for p in paths]
            require(sha(paths[0]) == sha(paths[1]), 'Reference case archive/metadata bytes changed')
    for index in (0, 1, 3):
        require(model_identity(source['models'][index], source_base, helper)
                == model_identity(reference['models'][index], reference_base, helper),
                'Baseline model definition changed: ' + MODELS[index])


def prepare(args, helper, package_root=ROOT):
    """The injectable helper is used only by small metadata fixtures in tests."""
    output = args.output.resolve()
    require(not output.exists(), 'Preserve existing output config')
    source_path, reference_path = args.source_config.resolve(), args.reference_config.resolve()
    source, _ = helper.read_config(source_path, args.source_config_sha256)
    reference, _ = helper.read_config(reference_path, args.reference_config_sha256)
    verify_cohort(source, reference, source_path.parent, reference_path.parent, helper)
    core_build = pinned_json(args.core_build, args.core_build_sha256)
    fused_build = pinned_json(args.fused_build, args.fused_build_sha256)
    graph_manifest = pinned_json(args.graph_manifest, args.graph_manifest_sha256)
    require(core_build['ORT_API'] == 29 and core_build['GPU_used'] is False
            and core_build['backend'] == 3 and core_build['domain'] == DOMAINS[0]
            and core_build['C_API_prefix'] == 'ipc_', 'Unexpected core build contract')
    require(fused_build['ort_api'] == 29 and fused_build['gpu_used'] is False
            and fused_build['tile_capacity'] == 512 and fused_build['no_internal_threads'] is True,
            'Unexpected fused build contract')
    require(graph_manifest['source_sha256'] == SOURCE_GRAPH_SHA
            and graph_manifest['output_sha256'] == sha(args.graph)
            and graph_manifest['selected_products'] == 22 and graph_manifest['retained_FP32_products'] == 9
            and graph_manifest['tile_time'] == 512 and graph_manifest['segments'] == 4
            and graph_manifest['shards'] == 4 and tuple(graph_manifest['domains']) == DOMAINS
            and graph_manifest['original_initializer_bytes_unchanged'] is True,
            'Graph manifest does not match the frozen graph contract')
    base_model = source['models'][1]
    require(sha(helper.relative_path(source_path.parent, base_model['path'])) == SOURCE_GRAPH_SHA,
            'Exact accepted Apple FP32 source graph required')
    base_libraries = libraries(base_model)
    require(len(base_libraries) == 1, 'One accepted Apple FP32 native library required')
    native = helper.relative_path(source_path.parent, base_libraries[0])
    def build_path(value, manifest_path):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (manifest_path.resolve().parent / path).resolve()
    core = build_path(core_build['core_path'], args.core_build)
    ops = build_path(core_build['ops_path'], args.core_build)
    require(set(core_build['libraries']) == {core.name, ops.name}, 'Unexpected core library inventory')
    require(set(fused_build['libraries']) == {'stage', 'upsample'}, 'Unexpected fused library inventory')
    fused = {name: build_path(value, args.fused_build) for name, value in fused_build['libraries'].items()}
    paths = {}
    def add(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'Artifact hash mismatch: ' + str(path))
        require(path not in paths or paths[path] == actual, 'Conflicting artifact identities')
        paths[path] = actual
        return os.path.relpath(path, output.parent).replace(os.sep, '/')
    for config, path in ((source, source_path), (reference, reference_path)):
        for name, expected in config['artifact_sha256'].items():
            add(helper.relative_path(path.parent, name), expected)
    for name, expected in core_build['sources'].items():
        add(helper.relative_path(package_root, name), expected)
    add(core, core_build['libraries'][core.name])
    add(ops, core_build['libraries'][ops.name])
    dependencies = {}
    for name, expected in fused_build['sources_dependencies_sha256'].items():
        path = build_path(name, args.fused_build)
        add(path, expected)
        dependencies[path] = expected
    require(dependencies.get(core) == sha(core) and dependencies.get(native) == sha(native),
            'Fused build is not linked to the selected core and FP32 library')
    for name, path in fused.items():
        add(path, fused_build['library_sha256'][name])
    for path, expected in ((source_path, args.source_config_sha256), (reference_path, args.reference_config_sha256),
                           (args.core_build, args.core_build_sha256), (args.fused_build, args.fused_build_sha256),
                           (args.graph_manifest, args.graph_manifest_sha256)):
        add(path, expected)
    graph_path = add(args.graph, graph_manifest['output_sha256'])
    external = sorted(helper.model_external_data(args.graph.resolve()))
    graph_external = [add(p) for p in external]
    config = copy.deepcopy(source)
    for model in config['models']:
        model['path'] = add(helper.relative_path(source_path.parent, model['path']))
        if model.get('custom_library'):
            model['custom_library'] = add(helper.relative_path(source_path.parent, model['custom_library']))
        for field in ('custom_libraries', 'external_data'):
            if field in model:
                model[field] = [add(helper.relative_path(source_path.parent, name)) for name in model[field]]
    for kind in ('audio', 'mimi'):
        config[kind + '_cases'] = add(helper.relative_path(source_path.parent, config[kind + '_cases']))
    model = config['models'][2]
    model.pop('custom_library', None)
    model['path'], model['external_data'] = graph_path, graph_external
    model['custom_libraries'] = [add(native), add(ops), add(fused['stage']), add(fused['upsample'])]
    config['transitive_core_library'] = add(core)
    scripts = ('campaign_portable.py', 'prepare_config_portable.py')
    for name in scripts:
        add(package_root / 'validation' / name)
    config['port_provenance'] = {
        'source_config': add(source_path), 'source_config_sha256': args.source_config_sha256,
        'reference_config': add(reference_path), 'reference_config_sha256': args.reference_config_sha256,
        'graph_manifest': add(args.graph_manifest), 'build': add(args.core_build), 'fused_build': add(args.fused_build),
        'runner': add(package_root / 'validation/campaign_portable.py'),
        'runner_sha256': sha(package_root / 'validation/campaign_portable.py'),
        'scope': 'Path-only packaging of the frozen experiment; new builds require fresh validation; no default change'}
    config['artifact_sha256'] = {os.path.relpath(p, output.parent).replace(os.sep, '/'): digest
                                 for p, digest in sorted(paths.items())}
    # Recheck every input after constructing the config. No graph/session is executed.
    require(all(sha(path) == digest for path, digest in paths.items()), 'Input changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write('\n')
    helper.read_config(output, sha(output))
    return {'config': str(output), 'sha256': sha(output), 'artifacts': len(paths),
            'reference_config_sha256': args.reference_config_sha256,
            'model_execution': False, 'runtime_defaults_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-config', 'reference-config', 'core-build', 'fused-build', 'graph-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harness', type=Path, default=ROOT.parent / 'tools/compare_decoders.py')
    args = parser.parse_args()
    require(sha(args.harness) == HARNESS_SHA, 'Pinned public config validator required')
    common = ROOT.parent / 'tools/platform_campaign.py'
    require(sha(common) == COMMON_SHA, 'Pinned common runner required')
    spec = importlib.util.spec_from_file_location('portable_config_helper', args.harness)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    print(json.dumps(prepare(args, helper), indent=2))


if __name__ == '__main__':
    main()
