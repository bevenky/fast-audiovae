"""Prepare a separate hash-pinned core with guarded exact-byte quantization."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--header-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).resolve().with_name('guarded_quantize.h')
    originals = {str(args.source): args.source.read_bytes(), str(args.header): args.header.read_bytes(),
                 str(helper): helper.read_bytes(), str(Path(__file__).resolve()): Path(__file__).read_bytes()}
    if digest(originals[str(args.source)]) != args.source_sha256 or digest(originals[str(args.header)]) != args.header_sha256:
        raise ValueError('Explicit source or public-header hash mismatch')
    source = originals[str(args.source)].decode('utf-8')
    changes = []

    def replace(old, new, label):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Expected one unchanged source region: ' + label)
        source = source.replace(old, new, 1)
        changes.append(label)

    replace('#include "precision.h"', '#include "precision.h"\n#include "guarded_quantize.h"', 'private helper include')
    replace('        _mm512_storeu_ps(x.scales.data()+begin,scale);__m512i sums=_mm512_setzero_si512();',
            '        _mm512_storeu_ps(x.scales.data()+begin,scale);__m512i sums=_mm512_setzero_si512();\n'
            '        const auto quotient_plan=ip3_guarded_quantize::make_plan(scale);',
            'one guarded reciprocal plan per sixteen time columns')
    replace('const __m512 q=_mm512_max_ps(negative,_mm512_min_ps(limit,_mm512_div_ps(v,scale)));',
            'const __m512 q=_mm512_max_ps(negative,_mm512_min_ps(limit,ip3_guarded_quantize::quotient(v,quotient_plan)));',
            'guarded quotient before unchanged clamp and RN-even conversion')
    for path, data in originals.items():
        if Path(path).read_bytes() != data:
            raise ValueError('Source changed during preparation')
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh isolated output directory')
    output.mkdir(parents=True)
    (output / 'precision.cpp').write_text(source)
    (output / 'precision.h').write_bytes(originals[str(args.header)])
    (output / helper.name).write_bytes(originals[str(helper)])
    manifest = {
        'status': 'source_prepared', 'compiled': False, 'inference_executed': False,
        'baseline_core_sha256': args.source_sha256, 'public_header_sha256': args.header_sha256,
        'patch_script_sha256': digest(originals[str(Path(__file__).resolve())]),
        'guarded_helper_sha256': digest(originals[str(helper)]), 'changes': changes,
        'outputs': {name: digest((output / name).read_bytes()) for name in ('precision.cpp', 'precision.h', helper.name)},
        'scope': 'Existing AVX512 INT8 activation preparation only; scales, clamp, integer rounding, sums, scalar tails, FP16 and GEMM remain unchanged',
        'extra_per_column_state': 'Sixteen scale values, reciprocal values and a mask in the local plan; no persistent buffer',
    }
    (output / 'source-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
