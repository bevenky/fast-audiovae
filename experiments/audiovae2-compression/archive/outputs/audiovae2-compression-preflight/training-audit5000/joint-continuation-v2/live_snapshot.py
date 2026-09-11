"""Collect existing continuation evidence without running inference."""
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('/tmp/fast-audiovae-joint-recovery-v2')
segment = root / 'segment-5625-6625'
records = []
for line in (segment / 'train.jsonl').read_text().splitlines():
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        # A final record may be in the middle of an append.
        if line != (segment / 'train.jsonl').read_text().splitlines()[-1]:
            raise
checks = [check for row in records for check in row['teacher_cache_checks']]
source_ids = [source for row in records for source in row['source_ids']]
snapshot = {
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'restore': json.loads((segment / 'restore-check.json').read_text()),
    'launcher': json.loads((root / 'launch-step5625.json').read_text()),
    'display': json.loads((root / 'display-launch.json').read_text()),
    'updates_observed': len(records),
    'last_step': records[-1]['step'],
    'fresh_cursor': records[-1]['fresh_cursor'],
    'elapsed_seconds': records[-1]['elapsed_seconds'],
    'teacher_cache_checks': len(checks),
    'teacher_cache_all_bitwise_equal': all(c['bitwise_equal'] for c in checks),
    'teacher_cache_all_within_tolerance': all(c['allclose_original_tolerance'] for c in checks),
    'sources_consumed': len(source_ids),
    'sources_unique': len(set(source_ids)),
    'failure_file_present': (segment / 'failure.json').exists(),
    'segment_complete': (segment / 'completed.json').exists(),
}
(root / 'launch-verification.json').write_text(json.dumps(snapshot, indent=2) + '\n')
print(json.dumps({k: v for k, v in snapshot.items() if k not in ('launcher', 'display')}))
