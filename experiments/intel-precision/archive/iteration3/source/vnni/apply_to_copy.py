"""Create a separate hash-pinned core copy with a guarded small VNNI workspace path."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--header', required=True, type=Path)
    parser.add_argument('--header-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    header_bytes = args.header.read_bytes()
    kernel = Path(__file__).resolve().with_name('direct_vnni.h')
    kernel_bytes = kernel.read_bytes()
    kernel_sha256 = hashlib.sha256(kernel_bytes).hexdigest()
    script_sha256 = sha(__file__)
    if hashlib.sha256(source_bytes).hexdigest() != args.source_sha256 or hashlib.sha256(header_bytes).hexdigest() != args.header_sha256:
        raise ValueError('Source core or public header differs from explicit pin')
    source = source_bytes.decode('utf-8')
    changes = []

    def replace(old, new, label):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Expected one unchanged source region: ' + label)
        source = source.replace(old, new, 1)
        changes.append(label)

    replace('#include "precision.h"', '#include "precision.h"\n#include "direct_vnni.h"', 'internal header')
    replace('    std::vector<int32_t> product;\n    std::atomic_flag busy',
            '    std::vector<int32_t> product;\n    std::vector<int8_t> vnni_packed;\n    bool vnni_packed_valid=false;\n    std::atomic_flag busy', 'workspace private capacity')
    replace('        workspace->input.sums.resize(max_time);',
            '        workspace->input.sums.resize(max_time);\n'
            '        workspace->vnni_packed.resize(count(std::min(max_k,256),std::min(max_time,256),sizeof(int8_t)));',
            'bounded packed activation allocation')
    replace('        add(workspace->product.capacity()*sizeof(int32_t));',
            '        add(workspace->product.capacity()*sizeof(int32_t));\n'
            '        add(workspace->vnni_packed.capacity());', 'allocation accounting')
    replace('        WorkspaceUse use(workspace);workspace.prepared=false;',
            '        WorkspaceUse use(workspace);workspace.prepared=false;workspace.vnni_packed_valid=false;',
            'invalidate packed state before preparation')
    replace('        prepare_reused_int8(p,values,time,workspace.input);\n        workspace.prepared=true;return 0;',
            '        prepare_reused_int8(p,values,time,workspace.input);\n'
            '        workspace.vnni_packed_valid=ip3_direct_vnni::pack(\n'
            '            workspace.input.x8.data(),workspace.input.x8.size(),p.k,time,\n'
            '            workspace.vnni_packed.data(),workspace.vnni_packed.size(),p.backend,capabilities());\n'
            '        workspace.prepared=true;return 0;', 'pack unchanged quantized activation once')
    replace('        else mkl_workspace(p,workspace,y,first,last,bias,skip);',
            '        else {\n'
            '            require(mkl_get_max_threads()==1,"Sequential oneMKL required");\n'
            '            const ip3_direct_vnni::Arguments args{p.m,p.k,x.t,first,last,\n'
            '                p.w8.data(),workspace.vnni_packed.data(),p.scales.data(),\n'
            '                x.scales.data(),x.sums.data(),y,bias,skip};\n'
            '            if(!(workspace.vnni_packed_valid&&ip3_direct_vnni::run(args,p.backend,capabilities())))\n'
            '                mkl_workspace(p,workspace,y,first,last,bias,skip);\n'
            '        }', 'guarded workspace dispatch with unchanged MKL fallback')
    if (sha(args.source) != args.source_sha256 or sha(args.header) != args.header_sha256
            or sha(kernel) != kernel_sha256 or sha(__file__) != script_sha256):
        raise ValueError('Source changed during preparation')
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh isolated output directory')
    output.mkdir(parents=True)
    (output / 'precision.cpp').write_text(source)
    (output / 'precision.h').write_bytes(header_bytes)
    (output / 'direct_vnni.h').write_bytes(kernel_bytes)
    manifest = {'status': 'source_prepared', 'compiled': False, 'inference_executed': False,
                'baseline_core_sha256': args.source_sha256, 'public_header_sha256': args.header_sha256,
                'patch_script_sha256': script_sha256, 'direct_kernel_sha256': kernel_sha256, 'changes': changes,
                'output_sha256': {name: sha(output / name) for name in ('precision.cpp', 'precision.h', 'direct_vnni.h')},
                'scope': 'Only small INT8 workspace calls; ordinary prepare/run_rows, FP16 and fallback kernels are unchanged',
                'extra_workspace_capacity_bytes_maximum': 65536}
    (output / 'source-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
