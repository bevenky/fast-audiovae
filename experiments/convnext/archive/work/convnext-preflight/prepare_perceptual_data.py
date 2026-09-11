"""Prepare the future perceptual comparison only after an explicitly selected r7 parent is verified."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys

OLD = Path('/workspace/fast-audiovae-convnext-20260908-r1')
PILOT = Path('/workspace/fast-audiovae-convnext-20260909-r5')
PREVIOUS = Path('/workspace/fast-audiovae-convnext-20260909-r7')
BASE = Path('/workspace/fast-audiovae-convnext-20260909-r8')
EXPECTED_PLAN = 'bbd230387be870bda39f677339238ec2ab7e9466a8a84ea32daf058a2ed9677b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=BASE)
    parser.add_argument('--previous-plan', type=Path, default=PREVIOUS / 'data/quiet-phase-v1')
    parser.add_argument('--parent-checkpoint', type=Path, default=BASE / 'selected-parent.pt')
    parser.add_argument('--source-checkpoint', type=Path, default=PREVIOUS / 'training-runs/quiet-phase-v1/latest.pt')
    parser.add_argument('--parent-selection', type=Path, default=BASE / 'selected-parent.json')
    parser.add_argument('--selection-sha256', required=True, help='Independently verified SHA256 of selected-parent.json')
    parser.add_argument('--output', type=Path, default=BASE / 'data/perceptual-v1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--planner-sha256', help='Only needed when running a read-only in-memory preview')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from audiovae_student.comparison_data import (EVENTS, known_identities, load_comparison_plan,
                                                 plan_comparison, write_comparison_plan)
    from audiovae_student.continuation_data import continuation_ledger
    from audiovae_student.data import load_manifest
    from audiovae_student.restart_data import canonical, digest, file_sha, identity

    previous = load_comparison_plan(args.previous_plan)
    if previous['identity']['identity_sha256'] != EXPECTED_PLAN:
        raise ValueError('Expected exact completed r7 comparison plan')
    selection_sha = file_sha(args.parent_selection)
    if not re.fullmatch(r'[a-f0-9]{64}', args.selection_sha256) or selection_sha != args.selection_sha256:
        raise ValueError('Selected-parent record does not match its independently verified SHA256')
    selection = json.loads(args.parent_selection.read_text())
    parent_arm = selection.get('selected_arm')
    checkpoint_sha = file_sha(args.parent_checkpoint)
    source_sha = file_sha(args.source_checkpoint)
    if (selection.get('parent_checkpoint_sha256') != checkpoint_sha
            or selection.get('source_checkpoint_sha256') != source_sha
            or parent_arm not in ('control', 'quiet-phase')
            or not isinstance(selection.get('selected_engine_sha256'), str)
            or not re.fullmatch(r'[a-f0-9]{64}', selection['selected_engine_sha256'])):
        raise ValueError('Selected parent provenance differs from the completed comparison')
    status_path = args.source_checkpoint.parent / 'status.json'
    status = json.loads(status_path.read_text())
    if (status.get('state') != 'completed_awaiting_review' or status.get('step') != 250
            or status.get('consumed_windows_per_arm') != 8000 or status.get('remaining_windows') != 0):
        raise ValueError('r7 has not completed its full shared comparison exposure')
    exposure_path = args.source_checkpoint.parent / 'exposure.jsonl'
    if selection.get('source_exposure_sha256') != file_sha(exposure_path):
        raise ValueError('Selected parent binds a different source exposure journal')
    exposure = [json.loads(line) for line in exposure_path.read_text().splitlines()]
    if len(exposure) != 250:
        raise ValueError('r7 exposure journal is not complete')
    for step, record in enumerate(exposure, 1):
        if (record.get('step') != step or record.get('cursor') != step * 32
                or record.get('window_identity') != previous['identity']['fixed_sampler_identity']):
            raise ValueError('r7 exposure journal differs from the exact shared window plan')
    ledger = continuation_ledger(previous['ledger'], previous['rows'], previous['counts'],
        parent_checkpoint_sha256=checkpoint_sha, used_plan_identity_sha256=EXPECTED_PLAN,
        provenance={'parent_exposure_sha256': file_sha(exposure_path),
                    'parent_status_sha256': file_sha(status_path), 'parent_arm': parent_arm,
                    'parent_selection_sha256': file_sha(args.parent_selection),
                    'source_comparison_checkpoint_sha256': source_sha,
                    'common_AB_exposure_counted_once': True})
    hashes = {}
    pinned = previous['identity']['provenance']['source_files_sha256']
    def read(path):
        digest_now = file_sha(path)
        if digest_now != pinned.get(str(path)):
            raise ValueError('Source manifest changed since r7: ' + str(path))
        hashes[str(path)] = digest_now
        return load_manifest(path)
    full = read(OLD / 'expanded-pilot/corpus/source-manifest.jsonl')
    supplement = read(PILOT / 'data-expressive-topup-v1/versions/v3-jsonl/train.jsonl')
    by_id = {}
    for row in full + supplement:
        if row.split != 'train':
            continue
        if row.source_id in by_id and identity(row) != identity(by_id[row.source_id]):
            raise ValueError('Duplicated source changed its immutable identity')
        by_id[row.source_id] = row
    rows = list(by_id.values())
    counts = {r.source_id: round(r.duration_seconds * 16000) for r in rows}
    labels, provenance_cache = {}, {}
    for row in rows:
        if row.dataset in {'fleurs', 'librispeech', 'indicvoices', 'emogator', 'crema_d'}:
            continue
        if row.access_record not in provenance_cache:
            path = Path(row.access_record)
            if file_sha(path) != pinned.get(str(path)):
                raise ValueError('Pinned event provenance changed: ' + str(path))
            hashes[str(path)] = file_sha(path)
            provenance_cache[row.access_record] = json.loads(path.read_text())
        record = provenance_cache[row.access_record]
        values = list(record.get('selected_metadata', {}).get('target_labels', []))
        if row.dataset.startswith('freesound_human_whistle_'):
            values.append('human_whistling_source_description')
        if 'whisper' in str(record.get('labels', {})).lower():
            values.append('explicit_whisper_style')
        labels[row.source_id] = sorted(set(values))
    reserved = set().union(*(known_identities(r) for r in previous['reserved']))
    excluded = set().union(*(known_identities(r, people=False) for r in previous['excluded']))
    available_events = {name: {'sources': 0, 'whole_source_seconds': 0.0} for name in EVENTS}
    for row in rows:
        if (not ledger.whole_untouched(row) or known_identities(row) & reserved
                or known_identities(row, people=False) & excluded):
            continue
        for event in labels.get(row.source_id, ()):
            if event in available_events:
                available_events[event]['sources'] += 1
                available_events[event]['whole_source_seconds'] += row.duration_seconds
    plan = plan_comparison(rows, counts, ledger, event_labels=labels, reserved_rows=previous['reserved'],
        excluded_sources=previous['excluded'], windows_count=32000, seed=43, minimum_input_samples=3040)
    missing = [r.source_id for r in plan['rows'] if not Path(r.audio_path).is_file()]
    if missing:
        raise ValueError('Selected source audio no longer exists: ' + str(missing[:10]))
    selected_events = {name: {'sources': 0, 'scored_input_samples': 0} for name in EVENTS}
    row_samples = defaultdict(int)
    for window in plan['windows']:
        row_samples[window.source_id] += window.valid_input_samples16k
    for row in plan['rows']:
        for event in labels.get(row.source_id, ()):
            if event in selected_events:
                selected_events[event]['sources'] += 1
                selected_events[event]['scored_input_samples'] += row_samples[row.source_id]
    plan['metadata'].update(
        available_event_inventory=available_events, overlapping_selected_event_coverage=selected_events,
        unavailable_event_classes=[k for k, v in available_events.items() if not v['sources']],
        inherited_excluded_sources=len(previous['ledger'].entries),
        newly_excluded_r7_sources=len(previous['rows']),
        total_student_lineage_excluded_sources=len(ledger.entries),
        existing_selected_audio_files=len(plan['rows']),
        mixed_japanese_scored_seconds=sum(w.valid_input_samples16k for w in plan['windows']
                                        if w.condition in {'jnv_nonverbal', 'jvnv_verbal_and_nonverbal'}) / 16000,
        event_shortage_policy='Do not replay unavailable rare events; disclose absent classes and actual achieved fractions. An exhausted event bucket transfers its remaining weight to English.',
        perceptual_crop_guarantee='Every selected crop has at least 9120 valid output samples, including qualifying partial tails',
        minimum_input_samples=3040,
        requires_explicit_phase_to_perceptual_transition=parent_arm == 'quiet-phase')
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'previous_plan_identity': EXPECTED_PLAN, 'source_files_sha256': hashes,
                  'parent_checkpoint_sha256': checkpoint_sha, 'parent_arm': parent_arm,
                  'source_comparison_checkpoint_sha256': source_sha,
                  'parent_selection_sha256': file_sha(args.parent_selection),
                  'parent_exposure_sha256': file_sha(exposure_path),
                  'planner_sha256': args.planner_sha256 or file_sha(Path(__file__)),
                  'source_root': str(args.source_root), 'ledger_identity': ledger.identity_sha256,
                  'preview_only': args.dry_run}
    report = {'metadata': plan['metadata'], 'provenance': provenance,
              'window_identity': digest([w.to_dict() for w in plan['windows']]),
              'planned_output': str(args.output), 'published': False}
    if not args.dry_run:
        ready = write_comparison_plan(plan, args.output, provenance=provenance)
        checked = load_comparison_plan(args.output)
        if checked['metadata'] != plan['metadata']:
            raise ValueError('Published metadata did not roundtrip')
        report.update(ready=ready, published=True)
        (args.output.parent / 'perceptual-data-plan.json').write_bytes(canonical(report))
    print(json.dumps({'published': report['published'], 'plan_identity': report.get('ready', {}).get('identity_sha256'),
        'planned_output': report['planned_output'], 'parent_arm': parent_arm,
        'windows': plan['metadata']['windows'], 'rows': plan['metadata']['rows'],
        'scored_hours': plan['metadata']['scored_hours'], 'by_bucket': plan['metadata']['by_bucket'],
        'unavailable_event_classes': plan['metadata']['unavailable_event_classes'],
        'drained_capacity': plan['metadata']['drained_capacity'],
        'student_lineage_excluded_sources': len(ledger.entries),
        'requires_explicit_phase_to_perceptual_transition': parent_arm == 'quiet-phase'},
        sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
