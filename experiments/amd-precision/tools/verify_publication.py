"""Offline source, provenance and completed-runtime checks. No native execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/amd-precision'
EVIDENCE = ROOT / 'benchmarks/amd-precision'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def main():
    publication = read('publication.json')
    for row in publication['sources']:
        path = ROOT / row['published']
        assert path.is_relative_to(SOURCE) and sha(path) == row['sha256']
    for row in publication['normalized_records']:
        path = ROOT / row['published']
        assert path.is_relative_to(EVIDENCE) and sha(path) == row['published_sha256']
    recipe = json.loads((SOURCE / 'pins/rebuild.json').read_text())
    for name, expected in recipe['source_header_closure'].items():
        if name.startswith('${RELEASE}/'):
            assert sha(SOURCE / 'common' / name[len('${RELEASE}/'):]) == expected
    for name, expected in recipe['AOCL_core_source'].items():
        assert sha(SOURCE / 'aocl/source' / name) == expected
    for name, expected in read('builds/aocl-build-r2/build.json')['sources'].items():
        assert sha(SOURCE / 'aocl/source' / name) == expected
    for name in ('screen-r1-results.json', 'screen-aocl4-r1-results.json'):
        screen = read(name)
        assert screen['status'] == 'complete' and not screen['failures']
        assert len(screen['checks']) == 84 and all(x['passed'] for x in screen['checks'])
        assert len(screen['measurements']) == 60
        assert all(x['providers'] == ['CPUExecutionProvider'] for x in screen['providers'])
    final = read('screen-aocl4-r1-results.json')
    stats = read('screen-aocl4-r1-statistics.json')
    reduction = 100 * (1 - final['summary']['int8_large']['decoder_rtf'] / final['summary']['fast_fp32']['decoder_rtf'])
    assert abs(reduction - stats['decoder_time_reduction_percent']) < 1e-12
    assert stats['paired_wins'] == stats['pair_count'] == 15
    schedule = read('checks-schedule4-r1.json')
    assert schedule['passed'] and schedule['record_count'] == len(schedule['records']) == 299
    assert all(x['passed'] for x in schedule['records'])
    assert schedule['script_sha256'] == sha(SOURCE / 'aocl/check_schedule.py')
    assert schedule['reference_helper_sha256'] == sha(SOURCE / 'aocl/stage_reference.py')
    full = read('quality-aocl4-r1-results.json')
    assert full['status'] == 'complete' and not full['failures']
    assert len(full['checks']) == 234 and all(x['passed'] for x in full['checks'])
    assert len(full['exports']) == 180 and not full['measurements']
    assert full['script_sha256'] == sha(SOURCE / 'tools/quality_only.py')
    identity = read('quality-identity.json')
    assert identity['script_sha256'] == sha(SOURCE / 'tools/verify_quality_identity.py')
    assert identity['summary']['int8_large']['identical_waveforms'] == 52
    quality = read('quality/comparison.json')
    assert quality['status'] == 'complete' and quality['scorer_exit_code'] == 0 and not quality['errors']
    assert quality['wrapper_sha256'] == sha(SOURCE / 'quality/score_quality.py')
    assert all(x == ['CPUExecutionProvider'] for x in quality['provenance']['cpu']['ort_providers'])
    for model in ('audio_stock', 'fast_fp32', 'int8_large'):
        metrics = quality['absolute_quality']['models'][model]['metrics']
        assert len(metrics) == 12 and all(x['count'] == 60 for x in metrics.values())
    deltas = quality['aggregate_candidate_minus_baseline']['int8_large']
    assert len(deltas) == 12 and all(x['pairs'] == 60 and x['status'] == 'complete' for x in deltas.values())
    for name, expected in quality['provenance']['code_sha256'].items():
        assert sha(SOURCE / 'quality/scorer' / name) == expected
    files = 0
    for base in (SOURCE, EVIDENCE):
        for path in base.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            assert path.suffix not in ('.onnx', '.npz', '.wav', '.so', '.dylib', '.o', '.a', '.whl')
            text = path.read_text()
            assert '/' + 'Users/' not in text and '/' + 'workspace/' not in text, path
            assert 'cod' + 'ex' not in text.lower() and chr(0x2014) not in text, path
            if path.suffix == '.py':
                ast.parse(text)
            if path.suffix == '.json':
                json.loads(text)
            files += 1
    print(json.dumps({'status': 'passed', 'text_files': files,
                      'unchanged_sources': len(publication['sources']),
                      'normalized_records': len(publication['normalized_records']),
                      'schedule_checks': 299, 'full_runtime_checks': 234,
                      'native_execution': False, 'model_execution': False}))


if __name__ == '__main__':
    main()
