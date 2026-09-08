"""Verify published Apple source/evidence offline. No native or model execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/apple-precision'
EVIDENCE = ROOT / 'benchmarks/apple-precision'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((EVIDENCE / 'publication.json').read_text())
    for row in manifest['sources']:
        path = ROOT / row['published']
        assert path.is_relative_to(SOURCE) and sha(path) == row['sha256'], row['published']
    for row in manifest['normalized_records']:
        path = ROOT / row['published']
        assert path.is_relative_to(EVIDENCE) and sha(path) == row['published_sha256'], row['published']
    for label in ('sdot', 'sme2'):
        build = json.loads((EVIDENCE / label / 'build.json').read_text())
        for name, expected in build['sources'].items():
            assert sha(SOURCE / label / name) == expected, (label, name)
    upstream = SOURCE / 'sme2/native/sme'
    pins = json.loads((upstream / 'upstream-provenance.json').read_text())
    assert pins['commit'] == '02f7b3df98c39df1884eead7461c3269feca4ddc'
    for name, row in pins['files'].items():
        content = (upstream / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row['sha256']
        blob = b'blob ' + str(len(content)).encode() + b'\0' + content
        assert hashlib.sha1(blob).hexdigest() == row['git_blob_sha1']
    for label in ('sdot', 'sme2'):
        result = json.loads((EVIDENCE / label / 'screen-results.json').read_text())
        assert result['status'] == 'complete' and not result['failures']
        assert len(result['checks']) == 84 and all(x['passed'] for x in result['checks'])
        assert len(result['measurements']) == 60
        assert all(x['providers'] == ['CPUExecutionProvider'] for x in result['providers'])
    final = json.loads((EVIDENCE / 'sme2/screen-results.json').read_text())
    summary = json.loads((EVIDENCE / 'sme2/screen-summary.json').read_text())
    assert final['summary'] == summary['summary']
    ratio = 1 - final['summary']['int8_large']['decoder_rtf'] / final['summary']['fast_fp32']['decoder_rtf']
    assert abs(ratio - summary['aggregate_time_reduction']) < 1e-15
    native = json.loads((EVIDENCE / 'sme2/native-checks.json').read_text())
    assert native['status'] == 'passed' and len(native['native_shapes']) == 20
    assert native['invalid_cases'] == 17 and native['ORT_micro_cases'] == 8
    assert native['concurrent_shared_plan_cases'] == 6
    assert json.loads((EVIDENCE / 'sme2/parallel-checks.json').read_text())['status'] == 'passed'
    assert json.loads((EVIDENCE / 'sme2/helper-build.json').read_text())['status'] == 'passed'
    extremes = json.loads((EVIDENCE / 'sme2/helper-extremes.json').read_text())
    assert extremes['check']['status'] == 'passed'
    assert extremes['source_sha256'] == sha(upstream / 'check_extremes.cpp')
    retained = json.loads((EVIDENCE / 'retained-fp32/run/results.json').read_text())
    assert retained['status'] == 'complete' and not retained['failures']
    assert len(retained['checks']) == 156 and all(x['passed'] for x in retained['checks'])
    assert len(retained['exports']) == 120 and not retained['measurements']
    assert retained['script_sha256'] == sha(SOURCE / 'retained-fp32/validate_fp32.py')
    audit = json.loads((EVIDENCE / 'retained-fp32/audit.json').read_text())
    assert audit['audit_script_sha256'] == sha(SOURCE / 'retained-fp32/audit_fp32_validation.py')
    assert all(v['passed'] == 60 and v['failed'] == 0 for v in audit['summary'].values())
    quality = json.loads((EVIDENCE / 'retained-fp32/quality/comparison.json').read_text())
    assert quality['status'] == 'complete' and quality['scorer_exit_code'] == 0 and not quality['errors']
    assert quality['wrapper_sha256'] == sha(SOURCE / 'quality/score_quality.py')
    assert all(x == ['CPUExecutionProvider'] for x in quality['provenance']['cpu']['ort_providers'])
    deltas = quality['aggregate_candidate_minus_baseline']['fast_fp32']
    assert len(deltas) == 12 and all(x['pairs'] == 60 and x['status'] == 'complete' for x in deltas.values())
    for name, expected in quality['provenance']['code_sha256'].items():
        assert sha(SOURCE / 'quality/scorer' / name) == expected
    checked = 0
    for base in (SOURCE, EVIDENCE):
        for path in base.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            assert path.suffix not in ('.onnx', '.npz', '.wav', '.o', '.so', '.dylib', '.a', '.whl')
            text = path.read_text()
            assert '/' + 'Users/' not in text and chr(0x2014) not in text, path
            assert 'cod' + 'ex' not in text.lower(), path
            if path.suffix == '.json':
                json.loads(text)
            if path.suffix == '.py':
                ast.parse(text, filename=str(path))
            checked += 1
    print(json.dumps({'status': 'passed', 'text_files': checked,
                      'unchanged_sources': len(manifest['sources']),
                      'normalized_records': len(manifest['normalized_records']),
                      'original_build_source_maps': 2, 'exact_upstream_files': 6,
                      'native_execution': False, 'model_execution': False}))


if __name__ == '__main__':
    main()
