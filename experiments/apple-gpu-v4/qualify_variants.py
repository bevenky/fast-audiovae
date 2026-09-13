"""Unchanged bounded state/waveform gates for the numerical repair candidate."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
for key in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH', 'TORCHINDUCTOR_USE_FAST_MATH'):
    if os.environ.get(key, '0') != '0':
        raise RuntimeError(key+' must be disabled')
    os.environ[key] = '0'
os.environ.pop('PYTORCH_MPS_PREFER_METAL', None)
sys.path.insert(0, str(HERE.parent / 'apple-gpu-v2'))
import experiment as exp
import qualify_candidates as qualifier
import candidates
import repairs
import numpy as np


def main(args):
    output = HERE / (args.name+'.json')
    if output.exists():
        raise FileExistsError(output)
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__).resolve(), HERE/'candidates.py', HERE/'repairs.py')}
    cpu_math = candidates.verify_cpu_algebra()+repairs.verify_cpu_algebra()
    original_build, original_compare = exp.build, qualifier.compare
    failures = []

    def build(arm):
        if arm == 'before':
            return original_build(arm)
        if arm != 'before_compile':
            raise ValueError(arm)
        model = exp.legacy()
        assert repairs.apply(model, args.variant) == (5 if args.variant == 'retain_first' else 6)
        if args.pointwise_mm:
            import torch, types
            def forward(self, x):
                return torch.mm(self.weight[:, :, 0], x[0]).unsqueeze(0)+self.bias[None, :, None]
            changed = 0
            for module in model.modules():
                if module.__class__.__name__ == '_Pointwise':
                    module.forward = types.MethodType(forward, module)
                    changed += 1
            assert changed == 19
        return exp.CompiledModel(model)

    def compare(value, reference):
        result = original_compare(value, reference)
        if not result['passed']:
            diff = np.abs(value.astype(np.float64)-reference)
            threshold = 1e-5+1e-4*np.abs(reference.astype(np.float64))
            failures.append(dict(elements=value.size, failing_elements=int(np.count_nonzero(diff>threshold)),
                max_abs=float(diff.max()), rms=float(np.sqrt(np.mean(diff**2))),
                maximum_tolerance_ratio=float((diff/threshold).max())))
        return result

    exp.build, qualifier.compare = build, compare
    try:
        qualifier.main(output, ['before_compile'])
    finally:
        exp.build, qualifier.compare = original_build, original_compare
        if output.exists():
            result = json.loads(output.read_text())
            result['repair'] = dict(candidate=args.variant, pointwise_mm=args.pointwise_mm,
                raw_alias='before_compile', reference='Original eager input-history decoder',
                cpu_algebra=cpu_math, tolerance_failures=failures, unchanged_tolerances=True)
            result.setdefault('files', {}).update(source_hashes)
            unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                            for path, digest in source_hashes.items())
            result['repair']['sources_unchanged'] = unchanged
            result['protocol']['state_reference'] = 'Original eager model; each path evolves its own original activated-input history.'
            preparation = result.get('preparation', [])
            if result['status'] == 'passed':
                assert len(preparation) == 1
                select = lambda key: {k:v for k,v in preparation[0][key].items() if k in ('stats','frames','graph_break','unimplemented')}
                first, last = select('compiler_counters'), select('final_compiler_counters')
                if first != last or first.get('stats', {}).get('unique_graphs', 0) != 2:
                    result.update(status='failed', error='Unexpected recompilation or graph count')
                result['compile_counters_after_preparation'] = first
                result['compile_counters_after_qualification'] = last
            if not unchanged:
                result.update(status='failed', error='Candidate source changed during qualification')
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result = json.loads(output.read_text())
    if result['status'] != 'passed':
        raise RuntimeError(result.get('error', 'Qualification failed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--pointwise-mm', action='store_true')
    parser.add_argument('--variant', choices=['hybrid','interleaved','retain_first'], required=True)
    main(parser.parse_args())
