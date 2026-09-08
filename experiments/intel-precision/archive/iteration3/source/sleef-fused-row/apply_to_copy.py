"""Fuse ordered AVX512 DW output into accurate post-Snake in a pinned copy."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inline-source', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.inline_source.resolve()
    manifest_path = source_dir/'source-provenance.json'
    manifest_bytes = manifest_path.read_bytes()
    if sha(manifest_bytes) != args.source_manifest_sha256:
        raise ValueError('Explicit parent source-manifest hash mismatch')
    parent = json.loads(manifest_bytes)
    required = {'row_nonlinear.h', 'sleefinline_avx512f.h',
                'sleef_inline_avx512.h', 'check_sine.cpp'}
    if set(parent.get('outputs', {})) != required or parent.get('sleef_version') != '3.9.0':
        raise ValueError('Expected the isolated inline-SLEEF source manifest')
    original = {}
    for name, expected in parent['outputs'].items():
        path = source_dir/name
        if path.resolve().parent != source_dir:
            raise ValueError('Parent artifact escapes source directory')
        data = path.read_bytes()
        if sha(data) != expected:
            raise ValueError('Parent source artifact hash mismatch: '+name)
        original[name] = data
    row = original['row_nonlinear.h'].decode('utf-8')
    if row.count('#include "sleef_inline_avx512.h"') != 1 or row.count(
            'const __m512 s=ip3_sleef_inline_sinf16_u10avx512f(_mm512_mul_ps(av,v));') != 1:
        raise ValueError('Expected the existing inline 16-lane pre/post Snake')
    anchor = '/* n is in [0,256], dilation is 1/3/9, backend is 4/5. History contains exactly'
    sequence = ('    row_snake(x,transformed+halo,ap,rp,n,backend);\n'
                '    sp_dw(transformed,weights,bias,depthwise,n,d,backend);')
    if row.count(anchor) != 1 or row.count(sequence) != 1:
        raise ValueError('Expected one unchanged row helper and operation sequence')
    include = '#include "row_dw_post_snake.h"\n\n'
    branch = ('    row_snake(x,transformed+halo,ap,rp,n,backend);\n'
              '    if(backend==5){\n'
              '        row_dw_post_snake_avx512(transformed,weights,bias,aq,rq,y,n,d);\n'
              '        std::memcpy(history,transformed+n,static_cast<size_t>(halo)*sizeof(float));\n'
              '        return;\n'
              '    }\n'
              '    sp_dw(transformed,weights,bias,depthwise,n,d,backend);')
    modified = row.replace(anchor, include+anchor, 1).replace(sequence, branch, 1)
    restored = modified.replace(include+anchor, anchor, 1).replace(branch, sequence, 1)
    if restored != row:
        raise ValueError('Unplanned row source modification')
    here = Path(__file__).resolve().parent
    helper_path = here/'row_dw_post_snake.h'
    helper, script = helper_path.read_bytes(), Path(__file__).read_bytes()
    if (manifest_path.read_bytes() != manifest_bytes or
            any((source_dir/name).read_bytes() != data for name, data in original.items())):
        raise ValueError('Parent source changed during preparation')
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh isolated output directory')
    output.mkdir(parents=True)
    results = {**original, 'row_nonlinear.h': modified.encode(), 'row_dw_post_snake.h': helper}
    for name, data in results.items():
        (output/name).write_bytes(data)
    if helper_path.read_bytes() != helper or Path(__file__).read_bytes() != script:
        raise ValueError('Candidate helper or patch script changed during preparation')
    manifest = {
        'status': 'source_prepared', 'compiled': False, 'inference_executed': False,
        'parent_source_manifest_sha256': sha(manifest_bytes),
        'parent_outputs': parent['outputs'], 'sleef_version': parent['sleef_version'],
        'sleef_commit': parent['sleef_commit'],
        'sleef_source_archive_sha256': parent['sleef_source_archive_sha256'],
        'generated_header_sha256': sha(original['sleefinline_avx512f.h']),
        'patch_script_sha256': sha(script), 'helper_sha256': sha(helper),
        'outputs': {name: sha(data) for name, data in results.items()},
        'changes': ['Include the fused AVX512 DW/post-Snake helper',
                    'Dispatch only backend5 to its ordered vector path'],
        'unchanged': ['AVX2 execution branch', 'Pre-Snake and all sine grouping',
                      'Upstream generated sine and wrapper bytes',
                      'Seven DW products, six additions, bias, and Snake order',
                      'Chronological 6*d history bytes', 'Weights and model graphs'],
        'tail_policy': 'Scalar DW for fewer than 16 remaining values, then original 8/4/padded4 Snake',
        'history_note': 'Backend5 copies the same transformed[n:n+6*d] interval after post-Snake; buffers are disjoint and no external history is read during the fused operation.',
        'required_flags': ['-fno-fast-math', '-ffp-contract=off'],
        'required_validation': ['Original versus inline sine parity',
                                'Existing check_rows all widths/dilations/backends and repeated histories',
                                'Exact full-decoder waveform, repeated-call and causal checks'],
        'performance': 'Unmeasured. Removes the vector depthwise row store/reload; register spills or instruction-cache pressure may erase savings.',
    }
    (output/'source-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
