"""Read-only manifest, split, label and availability audit, executed on Runpod."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
from datetime import datetime, timezone

BASE = Path('/workspace/fast-audiovae-convnext-20260908-r1')
SEL = Path('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1')

def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def group(rows, key):
    out = {}
    for row in rows:
        value = str(row.get(key) or 'unknown')
        record = out.setdefault(value, {'rows': 0, 'seconds': 0.0})
        record['rows'] += 1
        record['seconds'] += row['duration_seconds']
    for record in out.values():
        record['hours'] = record['seconds'] / 3600
    return dict(sorted(out.items()))

def identity_set(rows, key):
    return {row[key] for row in rows if row.get(key)}

def summary(rows, check_paths=True):
    normalized_rows = [dict(row, language=row['language'].split('_')[0]) for row in rows]
    return {
        'rows': len(rows), 'hours': sum(row['duration_seconds'] for row in rows)/3600,
        'by_dataset': group(rows, 'dataset'), 'by_language': group(rows, 'language'),
        'speakers_with_ids': len(identity_set(rows, 'speaker_id')),
        'rows_missing_speaker': sum(not row.get('speaker_id') for row in rows),
        'source_splits': dict(Counter(row['source_split'] for row in rows)),
        'by_split': group(rows, 'split'),
        'by_normalized_language': group(normalized_rows, 'language'),
        'missing_audio_paths': [row['source_id'] for row in rows if not Path(row['audio_path']).is_file()] if check_paths else 'not checked',
    }

def overlap(a, b):
    return {key: len(identity_set(a, key) & identity_set(b, key)) for key in
            ['source_id', 'audio_sha256', 'parent_recording_id', 'speaker_id', 'session_id']}

cache = {}
def provenance(row):
    path = row.get('access_record')
    if path not in cache:
        cache[path] = json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}
    return cache[path]

def events(row):
    data = provenance(row)
    labels = data.get('labels', {})
    result = list(data.get('selected_metadata', {}).get('target_labels', []))
    if row['dataset'].startswith('freesound_human_whistle_'):
        result.append('human_whistling_source_description')
    if labels.get('style') == 'whisper' or labels.get('intended_emotion') == 'whisper' or 'whisper' in str(labels).lower():
        result.append('explicit_whisper_style')
    return result

def expressive_summary(rows):
    rows = [r for r in rows if r['dataset'] not in {'fleurs', 'indicvoices', 'librispeech'}]
    counts = {}
    for row in rows:
        for event in events(row):
            item = counts.setdefault(event, {'rows': 0, 'whole_file_seconds': 0.0, 'source_ids': []})
            item['rows'] += 1
            item['whole_file_seconds'] += row['duration_seconds']
            item['source_ids'].append(row['source_id'])
    return {'rows': len(rows), 'hours': sum(r['duration_seconds'] for r in rows)/3600,
            'verified_source_event_labels': counts,
            'rows_with_no_verified_event_type': sum(not events(r) for r in rows),
            'note': 'Event labels refer to files, not all seconds within the file; emotion labels never establish a vocal event.'}

corpus_path = BASE/'expanded-pilot/corpus/source-manifest.jsonl'
corpus = load_rows(corpus_path)
ledger = json.loads((SEL/'ledger.json').read_text())
seen = [source['row'] for source in ledger['sources']]
seen_sources = identity_set(seen, 'source_id')
untouched = [r for r in corpus if r['source_id'] not in seen_sources]
untouched_train = [r for r in untouched if r['split'] == 'train']
scopes = {name: load_rows(SEL/(name+'.jsonl')) for name in
          ['train-manifest', 'dev-reserve', 'test-reserve', 'diagnostic', 'sentinel']}
expressive_dev = []
for name in ['data-expressive-originals', 'data-emogator', 'data-fsd-vocal', 'data-human-whistling']:
    expressive_dev += load_rows(BASE/name/'prepared/dev.jsonl')
legacy_dev_path = BASE/'expanded-pilot/corpus/candidate-dev.jsonl'
legacy_dev = load_rows(legacy_dev_path) if legacy_dev_path.exists() else []
windows = load_rows(SEL/'windows.jsonl')
window_counts = {}
for key in ['dataset', 'language', 'condition']:
    groups = defaultdict(lambda: {'windows': 0, 'input_samples': 0})
    for window in windows:
        record = groups[str(window.get(key))]
        record['windows'] += 1
        record['input_samples'] += window['valid_input_samples16k']
    for record in groups.values():
        record['scored_hours'] = record['input_samples'] / 16000 / 3600
    window_counts[key] = dict(sorted(groups.items()))

report = {
    'audited_utc': datetime.now(timezone.utc).isoformat(),
    'method': 'Live Runpod read-only metadata and file-existence audit; no audio scan, training, GPU inference or download.',
    'manifest_paths': {'corpus': str(corpus_path), 'selection': str(SEL), 'legacy_dev': str(legacy_dev_path)},
    'manifest_sha256': {name: hashlib.sha256((SEL/(name+'.jsonl')).read_bytes()).hexdigest() for name in scopes},
    'full_assembled_corpus': summary(corpus),
    'ledger': {'consumed_unique_source_rows': len(seen_sources),
               'previous_scored_hours': ledger['saved_sampler_state']['emitted_input_samples'] / 16000 / 3600},
    'whole_untouched_corpus_before_new_reserves': summary(untouched),
    'whole_untouched_training_before_new_reserves': summary(untouched_train),
    'selection_scopes': {name: summary(rows) for name, rows in scopes.items()},
    'pilot_scored_windows': window_counts,
    'expressive_coverage': {
       'full_assembled': expressive_summary(corpus),
       'untouched_before_reserves': expressive_summary(untouched),
       'untouched_training_before_reserves': expressive_summary(untouched_train),
       'pilot': expressive_summary(scopes['train-manifest']),
       'existing_dev': expressive_summary(expressive_dev),
    },
    'expressive_dev': summary(expressive_dev),
    'legacy_candidate_dev': summary(legacy_dev),
    'overlaps': {
       'pilot_vs_dev_reserve': overlap(scopes['train-manifest'], scopes['dev-reserve']),
       'pilot_vs_test_reserve': overlap(scopes['train-manifest'], scopes['test-reserve']),
       'pilot_vs_diagnostic': overlap(scopes['train-manifest'], scopes['diagnostic']),
       'pilot_vs_seen': overlap(scopes['train-manifest'], seen),
       'expressive_dev_vs_pilot': overlap(expressive_dev, scopes['train-manifest']),
       'expressive_dev_vs_seen': overlap(expressive_dev, seen),
       'dev_reserve_vs_seen': overlap(scopes['dev-reserve'], seen),
       'test_reserve_vs_seen': overlap(scopes['test-reserve'], seen),
    },
    'fleurs_regional_configurations': dict(sorted(Counter(r['source_id'].split(':')[0] for r in corpus if r['dataset']=='fleurs').items())),
    'limitations': [
        'Prepared audio files were checked for existence but not decoded or rehashed in this audit.',
        'No loudness or 20 ms quiet coverage can be inferred from language, emotion or source metadata.',
        'Uploader grouping is not verified speaker identity; missing speaker identity is not evidence of separation.',
        'Diagnostic fitting outputs are not independent validation of language or nonverbal generalization.',
        'Existing development sets have informed prior audits and are not new blind tests.',
    ]
}
print(json.dumps(report, indent=2, sort_keys=True))
