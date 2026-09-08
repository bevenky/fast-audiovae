"""Freeze candidate artifacts without modifying the original r3 reference."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    reference = root.parent / 'candidate-r3/config.json'
    assert sha(reference) == 'e3dd01f5a39d17bce7582fc8c726be4d8b43847e955e47a655c01cdacda28e20'
    config = json.loads(reference.read_text())
    core_build = json.loads((root / 'build/build.json').read_text())
    fused_build = json.loads((root / 'build-fused-r2/build.json').read_text())
    relative = lambda p: str(Path(p).relative_to(root))
    core = relative(core_build['core_path'])
    ops = relative(core_build['ops_path'])
    fused = [relative(fused_build['libraries'][key]) for key in ('stage', 'upsample')]
    model = next(m for m in config['models'] if m['name'] == 'int8_large')
    model['path'] = 'graphs/fused_int8.onnx'
    model['custom_libraries'] = [model['custom_libraries'][0], ops, *fused]
    artifacts = {p: h for p, h in config['artifact_sha256'].items() if p.startswith('../../../')}
    files = ['graphs/fused_int8.onnx', 'graphs/fused_int8.json',
             'graphs/selective_int8.onnx', 'graphs/selective_int8.json',
             'build/build.json', 'build-fused-r2/build.json', core, ops, *fused,
             'validation/campaign.py', 'validation/prepare_config.py']
    for folder in ('native', 'fused'):
        files += [str(p.relative_to(root)) for p in (root / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    for p in files:
        artifacts[p] = sha(root / p)
    for p, expected in artifacts.items():
        assert sha(root / p) == expected, p
    config['artifact_sha256'] = artifacts
    config['transitive_core_library'] = core
    config['port_provenance'] = {
        'source_config': '../candidate-r3/config.json',
        'source_config_sha256': sha(reference),
        'graph_manifest': 'graphs/fused_int8.json',
        'build': 'build/build.json', 'fused_build': 'build-fused-r2/build.json',
        'scope': 'Apple CPU packed time panels and four fused regions; validation pending; no default change'}
    output = root / 'config.json'
    with output.open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'config': str(output), 'sha256': sha(output), 'artifacts': len(artifacts)}))


if __name__ == '__main__':
    main()
