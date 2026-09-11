"""Prepare the approved metadata-only r6 comparison on the existing Runpod corpus."""
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PILOT = Path('/workspace/fast-audiovae-convnext-20260909-r5')
SEL = Path('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1')
PANEL = Path('/workspace/fast-audiovae-convnext-20260909-r4/data/dev-panel-v1')
BASE = Path('/workspace/fast-audiovae-convnext-20260909-r6')
sys.path.insert(0, str(PILOT))
spec = importlib.util.spec_from_file_location('audiovae_student.comparison_data', BASE / 'planning-code/comparison_data.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
from audiovae_student.data import load_manifest
from audiovae_student.restart_data import file_sha, canonical, identity


def unique(rows):
    found = {}
    for row in rows:
        if row.source_id in found:
            a, b = identity(row), identity(found[row.source_id])
            a.pop('split'); b.pop('split')
            if a != b:
                raise ValueError('Conflicting source identity: ' + row.source_id)
        else:
            found[row.source_id] = row
    return list(found.values())


def main():
    train_path = OLD / 'expanded-pilot/corpus/source-manifest.jsonl'
    supplement = PILOT / 'data-expressive-topup-v1/versions/v3-jsonl'
    hashes = {}
    def read(path):
        hashes[str(path)] = file_sha(path)
        return load_manifest(path)
    full = read(train_path)
    rows = unique([r for r in full if r.split == 'train'] + read(supplement / 'train.jsonl'))
    reserved = [r for r in full if r.split != 'train']
    heldout_paths = [OLD / 'expanded-pilot/corpus/candidate-dev.jsonl',
                    SEL / 'dev-reserve.jsonl', SEL / 'test-reserve.jsonl',
                    SEL / 'sentinel.jsonl', PANEL / 'manifest.jsonl', supplement / 'dev.jsonl']
    heldout_paths += [OLD / name / 'prepared/dev.jsonl' for name in
                     ('data-expressive-originals', 'data-emogator', 'data-fsd-vocal', 'data-human-whistling')]
    for path in heldout_paths:
        reserved.extend(read(path))
    reserved = unique(reserved)
    excluded = read(SEL / 'diagnostic.jsonl')
    parent_rows = read(SEL / 'train-manifest.jsonl')
    parent_counts_path = SEL / 'input-sample-counts.json'
    counts_doc = json.loads(parent_counts_path.read_text())
    hashes[str(parent_counts_path)] = file_sha(parent_counts_path)
    parent_counts = {r.source_id: counts_doc[r.source_id] for r in parent_rows}
    checkpoint = PILOT / 'training-runs/representative-reconstruction-v1/latest.pt'
    checkpoint_sha = file_sha(checkpoint)
    if checkpoint_sha != '955d12a84245765331b8697747e1565a3a90f3f2e1f6d8819e1c4f62d99ac104':
        raise ValueError('Approved parent checkpoint changed')
    ledger = mod.parent_student_ledger(parent_rows, parent_counts,
        checkpoint_sha256=checkpoint_sha,
        provenance={'parent_manifest_sha256': hashes[str(SEL / 'train-manifest.jsonl')],
                    'parent_exposure_sha256': file_sha(checkpoint.parent / 'exposure.jsonl'),
                    'parent_status_sha256': file_sha(checkpoint.parent / 'status.json')})
    labels = {}
    provenance_cache = {}
    for row in rows:
        if row.dataset in {'fleurs', 'librispeech', 'indicvoices', 'emogator', 'crema_d'}:
            continue
        path = row.access_record
        if path not in provenance_cache:
            provenance_cache[path] = json.loads(Path(path).read_text())
            hashes[path] = file_sha(path)
        record = provenance_cache[path]
        events = list(record.get('selected_metadata', {}).get('target_labels', []))
        if row.dataset.startswith('freesound_human_whistle_'):
            events.append('human_whistling_source_description')
        if 'whisper' in str(record.get('labels', {})).lower():
            events.append('explicit_whisper_style')
        labels[row.source_id] = sorted(set(events))
    counts = {r.source_id: round(r.duration_seconds * 16000) for r in rows}
    print(json.dumps({'phase': 'planning', 'training_pool_rows': len(rows), 'reserved_rows': len(reserved),
                      'parent_excluded_rows': len(parent_rows)}), flush=True)
    plan = mod.plan_comparison(rows, counts, ledger, event_labels=labels, reserved_rows=reserved,
                              excluded_sources=excluded, windows_count=16000)
    missing = [r.source_id for r in plan['rows'] if not Path(r.audio_path).is_file()]
    if missing:
        raise ValueError('Missing selected source files: ' + str(missing[:10]))
    events_summary = defaultdict(lambda: {'sources': 0, 'scored_input_samples': 0})
    row_samples = defaultdict(int)
    for window in plan['windows']:
        row_samples[window.source_id] += window.valid_input_samples16k
    for row in plan['rows']:
        for label in labels.get(row.source_id, ()):
            events_summary[label]['sources'] += 1
            events_summary[label]['scored_input_samples'] += row_samples[row.source_id]
    plan['metadata']['overlapping_event_label_coverage'] = dict(events_summary)
    plan['metadata']['existing_selected_files_checked'] = len(plan['rows'])
    plan['metadata']['source_sample_count_basis'] = 'Previously prepared manifest duration times 16000, verified integral; decoded length must be checked before teacher inference'
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'source_files_sha256': hashes, 'planner_sha256': file_sha(Path(__file__)),
                  'planner_module_sha256': file_sha(BASE / 'planning-code/comparison_data.py'),
                  'parent_checkpoint_sha256': checkpoint_sha,
                  'data_reuse_policy': 'Independent diagnostics may reuse downloaded training data; both arms independently inherit only current parent student exposure and never repeat new scored windows'}
    ready = mod.write_comparison_plan(plan, BASE / 'data/comparison-v1', provenance=provenance)
    loaded = mod.load_comparison_plan(BASE / 'data/comparison-v1')
    report = {'ready': ready, 'metadata': loaded['metadata'], 'parent_excluded_sources': len(parent_rows),
              'reserved_sources': len(reserved), 'excluded_diagnostic_sources': len(excluded)}
    (BASE / 'data-plan.json').write_bytes(canonical(report))
    print(json.dumps({'phase': 'ready', 'path': str(BASE / 'data/comparison-v1'),
                      'identity': ready['identity_sha256'], **loaded['metadata']}), flush=True)


if __name__ == '__main__':
    main()
